// MLXEngine `textToImage` package over the BooguImage core — Boogu-Image-0.1
// (Apache-2.0): Qwen3-VL-8B-conditioned OmniGen2-lineage DiT + FLUX VAE.
//
// The Swift core is parity-locked against the Python-MLX port (DiT bit-exact, VAE
// bit-exact, scheduler ~6e-8, Qwen3-VL text conditioning cos 1.0); this wrapper is a
// thin conformance layer. Variants: Base (30 step / g 3.5, int4) and Turbo (4 step /
// g 1.0, int8 — distilled, quant-sensitive). imageEdit (Edit variant) is added once
// the Qwen3-VL vision-conditioning parity lands.

import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import BooguImage
import MLX
import MLXToolKit

/// Init-time configuration (C9): the variant snapshot (transformer/vae/scheduler), the
/// Qwen3-VL conditioner snapshot, and generation defaults.
public struct BooguImageConfiguration: PackageConfiguration, ModelStorable {
    /// Variant snapshot root with `transformer/`, `vae/`, `scheduler/`.
    public var snapshotPath: String
    /// Stock Qwen3-VL-8B-Instruct snapshot (conditioner + tokenizer).
    public var qwenPath: String
    public var quant: Quant
    /// Run the DiT forward in fp32 (avoids the bf16 large-seqLen NaN at >=512²).
    public var useFP32DiT: Bool
    public var defaultSteps: Int
    public var defaultGuidance: Double
    public var defaultSize: Int
    public var defaultEditSteps: Int
    public var defaultEditGuidance: Double
    public var defaultEditSize: Int
    public var modelsRootDirectory: URL?

    public init(
        snapshotPath: String = "",
        qwenPath: String = "",
        quant: Quant = .int4,
        useFP32DiT: Bool = false,
        defaultSteps: Int = 30,
        defaultGuidance: Double = 3.5,
        defaultSize: Int = 1024,
        defaultEditSteps: Int = 50,
        defaultEditGuidance: Double = 4.0,
        defaultEditSize: Int = 768,
        modelsRootDirectory: URL? = nil
    ) {
        self.snapshotPath = snapshotPath
        self.qwenPath = qwenPath
        self.quant = quant
        self.useFP32DiT = useFP32DiT
        self.defaultSteps = defaultSteps
        self.defaultGuidance = defaultGuidance
        self.defaultSize = defaultSize
        self.defaultEditSteps = defaultEditSteps
        self.defaultEditGuidance = defaultEditGuidance
        self.defaultEditSize = defaultEditSize
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case snapshotPath, qwenPath, quant, useFP32DiT, defaultSteps, defaultGuidance, defaultSize
        case defaultEditSteps, defaultEditGuidance, defaultEditSize
    }
}

public enum BooguImagePackageError: Error, LocalizedError {
    case unreadableSnapshot(String)
    case pngEncode

    public var errorDescription: String? {
        switch self {
        case .unreadableSnapshot(let p): return "Boogu-Image snapshot not readable at \(p)."
        case .pngEncode: return "PNG encoding failed."
        }
    }
}

@InferenceActor
public final class BooguImagePackage: ModelPackage {
    public typealias Configuration = BooguImageConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            provenance: Provenance(
                sourceRepo: "mlx-community/Boogu-Image-0.1-Base-4bit", revision: "main", tier: 3),
            requirements: RequirementsManifest(
                // Resident = DiT + Qwen3-VL-8B (bf16) + FLUX VAE at 1024². int4/int8 quantize
                // only the DiT; the conditioner stays bf16. (Estimates — re-ground via
                // MemoryProbe in app validation.)
                footprints: [
                    QuantFootprint(quant: .int4, residentBytes: 24_000_000_000),
                    QuantFootprint(quant: .int8, residentBytes: 29_000_000_000),
                    QuantFootprint(quant: .bf16, residentBytes: 36_000_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: .max
            ),
            specialties: [],
            surfaces: [
                T2IContract.descriptor(
                    name: "boogu-image",
                    summary: "Boogu-Image-0.1 text-to-image (OmniGen2-lineage DiT + Qwen3-VL "
                        + "instruction conditioning + FLUX VAE): photorealistic generation, "
                        + "Base (30-step CFG) and Turbo (4-step distilled) tiers."),
                IEditContract.descriptor(
                    name: "boogu-image-edit",
                    summary: "Boogu-Image-0.1 instruction image editing (Edit variant): Qwen3-VL "
                        + "vision+text conditioning over the input image + a VAE ref-latent "
                        + "branch, structure-preserving edits at 50-step true CFG."),
            ]
        )
    }

    private let configuration: Configuration
    private var encoder: BooguPromptEncoder?
    private var generator: BooguImageGenerator?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard generator == nil else { return }
        let snapshot = URL(fileURLWithPath: configuration.snapshotPath)
        let transformerDir = snapshot.appendingPathComponent("transformer")
        guard FileManager.default.fileExists(atPath: transformerDir.path) else {
            throw BooguImagePackageError.unreadableSnapshot(snapshot.path)
        }

        // Conditioner (Qwen3-VL) + tokenizer.
        let enc = try await BooguPromptEncoder.load(
            qwenDir: URL(fileURLWithPath: configuration.qwenPath), dtype: .bfloat16)

        // DiT (quantized if a quant_config.json is present) + VAE + scheduler — loaded on
        // the CPU stream (a multi-GB read on the GPU stream trips the Metal watchdog).
        var dit: BooguImageTransformer2DModel!
        var vae: AutoencoderKL!
        try Device.withDefaultDevice(.cpu) {
            dit = try BooguWeights.loadDiTAuto(
                transformerDir: transformerDir, fp32: configuration.useFP32DiT)
            eval(dit)
            vae = try BooguWeights.loadVAE(
                directory: snapshot.appendingPathComponent("vae"), dtype: .float32)
        }
        let scheduler = try FlowMatchEulerDiscreteScheduler(
            directory: snapshot.appendingPathComponent("scheduler"))

        encoder = enc
        generator = BooguImageGenerator(dit: dit, vae: vae, scheduler: scheduler)
    }

    public func unload() async {
        encoder = nil
        generator = nil
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        guard let encoder, let generator else { throw PackageError.notLoaded }
        try Task.checkCancellation()

        switch request.capability {
        case .textToImage:
            guard let t2i = request as? T2IRequest else {
                throw PackageError.unsupportedCapability(request.capability)
            }
            let size = t2i.width ?? t2i.height ?? configuration.defaultSize
            let guidance = Float(t2i.guidanceScale ?? configuration.defaultGuidance)
            let pos = try encoder.encodeText(t2i.prompt).asType(generator.dit.dtype)
            let neg = guidance > 1.0
                ? try encoder.encodeText(t2i.negativePrompt ?? "").asType(generator.dit.dtype)
                : nil
            try Task.checkCancellation()
            let (pixels, w, h) = generator.generate(
                posCond: pos, negCond: neg,
                height: t2i.height ?? size, width: t2i.width ?? size,
                steps: t2i.steps ?? configuration.defaultSteps,
                guidance: guidance, seed: t2i.seed ?? 0)
            try Task.checkCancellation()
            let png = try Self.encodePNG(pixels: pixels, width: w, height: h)
            return T2IResponse(image: Image(format: .png, data: png, width: w, height: h))

        case .imageEdit:
            guard let edit = request as? IEditRequest, let first = edit.images.first else {
                throw PackageError.unsupportedCapability(request.capability)
            }
            let input = try Self.decodeRGB(first.data)
            let tw = edit.width ?? configuration.defaultEditSize
            let th = edit.height ?? configuration.defaultEditSize
            let textG = Float(edit.guidanceScale ?? configuration.defaultEditGuidance)
            // Vision+text conditioning over the input image; ref latent at the target size.
            let pos = try encoder.encodeImage(
                rgb: input.rgb, width: input.width, height: input.height, instruction: edit.prompt)
                .asType(generator.dit.dtype)
            let neg = try encoder.encodeImage(
                rgb: input.rgb, width: input.width, height: input.height,
                instruction: edit.negativePrompt ?? "").asType(generator.dit.dtype)
            let refLatent = generator.encodeRefLatent(
                rgb: input.rgb, width: input.width, height: input.height,
                targetWidth: tw, targetHeight: th)
            try Task.checkCancellation()
            let (pixels, w, h) = generator.generateEdit(
                posCond: pos, negCond: neg, refLatent: refLatent, height: th, width: tw,
                steps: edit.steps ?? configuration.defaultEditSteps,
                textGuidance: textG, seed: edit.seed ?? 0)
            try Task.checkCancellation()
            let png = try Self.encodePNG(pixels: pixels, width: w, height: h)
            return IEditResponse(image: Image(format: .png, data: png, width: w, height: h))

        default:
            throw PackageError.unsupportedCapability(request.capability)
        }
    }

    /// PNG/JPEG Data -> interleaved RGB8 (sRGB).
    nonisolated static func decodeRGB(_ data: Data) throws -> (rgb: [UInt8], width: Int, height: Int) {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw BooguImagePackageError.pngEncode }
        let (w, h) = (cg.width, cg.height)
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(
            data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw BooguImagePackageError.pngEncode }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            rgb[i * 3] = rgba[i * 4]; rgb[i * 3 + 1] = rgba[i * 4 + 1]; rgb[i * 3 + 2] = rgba[i * 4 + 2]
        }
        return (rgb, w, h)
    }

    /// Interleaved RGB8 -> PNG (canonical serialized artifact form, C3).
    nonisolated static func encodePNG(pixels: [UInt8], width: Int, height: Int) throws -> Data {
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let ctx = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: width * 4, space: cs,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw BooguImagePackageError.pngEncode }
        let buf = ctx.data!.bindMemory(to: UInt8.self, capacity: width * height * 4)
        for i in 0..<(width * height) {
            buf[i * 4] = pixels[i * 3]; buf[i * 4 + 1] = pixels[i * 3 + 1]
            buf[i * 4 + 2] = pixels[i * 3 + 2]; buf[i * 4 + 3] = 255
        }
        guard let image = ctx.makeImage() else { throw BooguImagePackageError.pngEncode }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            out, UTType.png.identifier as CFString, 1, nil)
        else { throw BooguImagePackageError.pngEncode }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw BooguImagePackageError.pngEncode }
        return out as Data
    }
}

extension BooguImagePackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(BooguImagePackage.self)
    }
}
