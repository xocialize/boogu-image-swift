// MLXEngine `textToImage` package over the BooguImage core — Boogu-Image-0.1
// (Apache-2.0): Qwen3-VL-8B-conditioned OmniGen2-lineage DiT + FLUX VAE.
//
// The Swift core is parity-locked against the Python-MLX port (DiT bit-exact, VAE
// bit-exact, scheduler ~6e-8, Qwen3-VL text conditioning cos 1.0); this wrapper is a
// thin conformance layer. Variants: Base (30 step / g 3.5) and Turbo (4 step / g 1.0 —
// distilled, quant-sensitive, so int8/bf16 only) back textToImage; Edit (50 step / true CFG 4.0)
// backs imageEdit. Each is published per quant on mlx-community (`BooguImageConfiguration.repo`).

import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import BooguImage
import MLX
import MLXToolKit

/// The three Boogu-Image-0.1 checkpoints. Base and Turbo back `textToImage`; Edit backs `imageEdit`
/// (an edit on the Base weights returns noise).
public enum BooguVariant: String, Codable, Sendable, CaseIterable {
    case base = "Base", turbo = "Turbo", edit = "Edit"
}

/// Init-time configuration (C9): the variant snapshot (transformer/vae/scheduler), the
/// Qwen3-VL conditioner snapshot, and generation defaults.
///
/// Weights resolve one of two ways:
/// - explicit local paths (`snapshotPath`, `qwenPath`), read in place;
/// - empty paths: `repo` and `qwenRepo` under the engine's store (`modelsRootDirectory`), which the
///   engine materializes first from `weightSources`. Use `.base(quant:)`, `.turbo(quant:)` or
///   `.edit(quant:)` to get a published tier's repo and defaults.
public struct BooguImageConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// Variant snapshot root with `transformer/`, `vae/`, `scheduler/`. Empty ⇒ `repo` in the store.
    public var snapshotPath: String
    /// Stock Qwen3-VL-8B-Instruct snapshot (conditioner + tokenizer). Empty ⇒ `qwenRepo` in the store.
    public var qwenPath: String
    /// DiT quant. It must match the snapshot (`transformer/quant_config.json` bits, or none for bf16):
    /// the governor charges this tier's footprint, so `load()` refuses a mismatch.
    public var quant: Quant
    /// The variant snapshot's hub repo (a published tier, `BooguImageConfiguration.repo(_:_:)`).
    public var repo: String
    public var revision: String?
    /// The conditioner's hub repo: the stock upstream checkpoint, shared with Qwen-Image-2.1.
    public var qwenRepo: String
    /// Run the DiT forward in fp32. No longer needed for correctness: the old "bf16
    /// large-seqLen NaN" was the mlx-swift NAX split-K GEMM bug (ml-explore/mlx#3797),
    /// fixed in mlx-swift 0.32.3 (the manifest floor) — bf16 runs clean and ~2× faster than fp32.
    /// Kept for parity work; the `BOOGU_FP32` env var forces it on at load time.
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
        modelsRootDirectory: URL? = nil,
        repo: String = "mlx-community/Boogu-Image-0.1-Base-4bit",
        revision: String? = nil,
        qwenRepo: String = BooguImageConfiguration.qwenRepo
    ) {
        self.snapshotPath = snapshotPath
        self.qwenPath = qwenPath
        self.quant = quant
        self.repo = repo
        self.revision = revision
        self.qwenRepo = qwenRepo
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
        case snapshotPath, qwenPath, quant, repo, revision, qwenRepo, useFP32DiT
        case defaultSteps, defaultGuidance, defaultSize
        case defaultEditSteps, defaultEditGuidance, defaultEditSize
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            snapshotPath: try c.decodeIfPresent(String.self, forKey: .snapshotPath) ?? "",
            qwenPath: try c.decodeIfPresent(String.self, forKey: .qwenPath) ?? "",
            quant: try c.decode(Quant.self, forKey: .quant),
            useFP32DiT: try c.decodeIfPresent(Bool.self, forKey: .useFP32DiT) ?? false,
            defaultSteps: try c.decodeIfPresent(Int.self, forKey: .defaultSteps) ?? 30,
            defaultGuidance: try c.decodeIfPresent(Double.self, forKey: .defaultGuidance) ?? 3.5,
            defaultSize: try c.decodeIfPresent(Int.self, forKey: .defaultSize) ?? 1024,
            defaultEditSteps: try c.decodeIfPresent(Int.self, forKey: .defaultEditSteps) ?? 50,
            defaultEditGuidance: try c.decodeIfPresent(Double.self, forKey: .defaultEditGuidance) ?? 4.0,
            defaultEditSize: try c.decodeIfPresent(Int.self, forKey: .defaultEditSize) ?? 768,
            repo: try c.decodeIfPresent(String.self, forKey: .repo) ?? "mlx-community/Boogu-Image-0.1-Base-4bit",
            revision: try c.decodeIfPresent(String.self, forKey: .revision),
            qwenRepo: try c.decodeIfPresent(String.self, forKey: .qwenRepo) ?? Self.qwenRepo)
    }

    /// The stock Qwen3-VL-8B-Instruct checkpoint the conditioner runs on (bit-identical upstream).
    public static let qwenRepo = "Qwen/Qwen3-VL-8B-Instruct"

    /// The mlx-community repo of a published tier, or nil for one that isn't published. Turbo ships
    /// int8 and bf16 only: the 4-step distilled DiT is quant-sensitive at int4.
    public static func repo(_ variant: BooguVariant, _ quant: Quant) -> String? {
        let suffix: String
        switch quant {
        case .int4: suffix = "4bit"
        case .int8: suffix = "8bit"
        case .bf16: suffix = "bf16"
        default: return nil
        }
        if variant == .turbo, quant == .int4 { return nil }
        return "mlx-community/Boogu-Image-0.1-\(variant.rawValue)-\(suffix)"
    }

    /// Base: 30 steps at CFG 3.5, int4 by default.
    public static func base(quant: Quant = .int4, modelsRootDirectory: URL? = nil) -> BooguImageConfiguration {
        BooguImageConfiguration(quant: quant, modelsRootDirectory: modelsRootDirectory,
                                repo: repo(.base, quant) ?? "")
    }

    /// Turbo: the 4-step distilled tier at guidance 1.0 (no CFG), int8 by default.
    public static func turbo(quant: Quant = .int8, modelsRootDirectory: URL? = nil) -> BooguImageConfiguration {
        BooguImageConfiguration(quant: quant, defaultSteps: 4, defaultGuidance: 1.0,
                                modelsRootDirectory: modelsRootDirectory, repo: repo(.turbo, quant) ?? "")
    }

    /// Edit: 50 steps at true CFG 4.0, int4 by default.
    public static func edit(quant: Quant = .int4, modelsRootDirectory: URL? = nil) -> BooguImageConfiguration {
        BooguImageConfiguration(quant: quant, modelsRootDirectory: modelsRootDirectory,
                                repo: repo(.edit, quant) ?? "")
    }

    /// The variant snapshot load() reads: the explicit path, else `repo` in the store.
    public func resolvedSnapshotDirectory() -> URL? {
        if !snapshotPath.isEmpty { return URL(fileURLWithPath: snapshotPath) }
        return repo.isEmpty ? nil : ModelStore(root: modelsRootDirectory).directory(for: repo)
    }

    /// The conditioner snapshot load() reads: the explicit path, else `qwenRepo` in the store.
    public func resolvedQwenDirectory() -> URL? {
        if !qwenPath.isEmpty { return URL(fileURLWithPath: qwenPath) }
        return ModelStore(root: modelsRootDirectory).directory(for: qwenRepo)
    }
}

extension BooguImageConfiguration: WeightSourcing {
    /// Two sources: the tier's snapshot (DiT + FLUX VAE + scheduler) and the shared conditioner.
    public var weightSources: [WeightSource] {
        var sources: [WeightSource] = []
        if !repo.isEmpty {
            sources.append(WeightSource(role: "snapshot", repo: repo, revision: revision,
                                        matching: ["transformer/*", "vae/*", "scheduler/*"]))
        }
        sources.append(WeightSource(role: "conditioner", repo: qwenRepo, revision: "main",
                                    matching: ["*.safetensors", "*.json", "merges.txt"]))
        return sources
    }

    /// An explicit local path satisfies its source; the rest are probed in the store.
    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let fm = FileManager.default
        let haveSnapshot = !snapshotPath.isEmpty
            && fm.fileExists(atPath: URL(fileURLWithPath: snapshotPath).appendingPathComponent("transformer").path)
        let haveQwen = !qwenPath.isEmpty
            && fm.fileExists(atPath: URL(fileURLWithPath: qwenPath).appendingPathComponent("config.json").path)
        return defaultMissingWeightSources(storeRoot: storeRoot).filter {
            !($0.role == "snapshot" && haveSnapshot) && !($0.role == "conditioner" && haveQwen)
        }
    }
}

extension BooguImageConfiguration: WorkloadDeclaring {
    /// A request's output pixels, sized exactly as run() will size it (the manifest's
    /// `ActivationScaling` axis), so the engine reserves for this job's size and refuses one past the cap
    /// before admission. An out-of-envelope size answers its floored area; run() refuses it (an edit
    /// past its smaller cap is still under the line's text-to-image ceiling, so run() is what refuses it).
    public func workloadUnits(for request: any CapabilityRequest) -> Double? {
        let width: Int, height: Int
        switch request {
        case let t2i as T2IRequest:
            let size = t2i.width ?? t2i.height ?? defaultSize
            (width, height) = (t2i.width ?? size, t2i.height ?? size)
        case let edit as IEditRequest:
            (width, height) = (edit.width ?? defaultEditSize, edit.height ?? defaultEditSize)
        default:
            return nil
        }
        let (w, h) = (try? BooguImagePackage.outputSize(width, height)) ?? (width, height)
        return Double(w * h)
    }
}

public enum BooguImagePackageError: Error, LocalizedError {
    case unreadableSnapshot(String)
    case pngEncode
    /// The configuration's quant doesn't match the snapshot's DiT (its `quant_config.json` bits).
    case quantMismatch(declared: Quant, snapshot: Quant)
    /// Output size outside the measured envelope: a side outside `minOutputSide…maxOutputSide`, or an
    /// area above the surface's cap (`maxOutputPixels` for text-to-image, `maxEditOutputPixels` for edit).
    case sizeOutOfEnvelope(width: Int, height: Int, maxPixels: Int)

    public var errorDescription: String? {
        switch self {
        case .unreadableSnapshot(let p): return "Boogu-Image snapshot not readable at \(p)."
        case .pngEncode: return "PNG encoding failed."
        case .quantMismatch(let declared, let snapshot):
            return "Boogu-Image is configured as \(declared.rawValue) but the snapshot's DiT is \(snapshot.rawValue)."
        case .sizeOutOfEnvelope(let w, let h, let maxPixels):
            let side = Int(Double(maxPixels).squareRoot())
            return "Boogu-Image output \(w)×\(h) is outside the supported envelope (each side "
                + "\(BooguImagePackage.minOutputSide)–\(BooguImagePackage.maxOutputSide) px, area ≤ \(side)×\(side))."
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
                // Split footprint, MEASURED 2026-10-29 (AB-A-0224, AB-T-0204): `BooguGate --pkg-measure` drives
                // load() + run() as a Gaea worker does — Release, MLX pool capped at 2 GiB (the engine's
                // automatic rule), process phys_footprint sampled every 20 ms and checked against the kernel's
                // lifetime peak, M5 Max 128 GB. The Qwen3-VL-8B conditioner is loaded per request, encodes,
                // and is evicted before the denoise, so it is activation, not residency.
                //   resident = phys after load() (DiT + fp32 FLUX VAE): int4 8.27 · int8 12.89 · bf16 20.95 GB
                //     (Base, Turbo and Edit tiers of a quant are the same size).
                //   activation = peak − resident, the same on every quant:
                //     t2i 512²/768²/1024²: 16.57 GB — the conditioner's transient sets it
                //     edit 1024²: 18.80 GB (target + reference latent tokens; int4 and bf16 identical)
                //     t2i 1536²: 23.34 GB (int8, bf16; CFG on — Turbo's real guidance 1.0 runs one pass)
                //   The scalar is the 1024² case covering edit: 19.8 GB. The `outputPixels` line,
                //   16.6 GB + 3000 B/px with its ceiling at the T2I cap (1536²), sits on or above every
                //   measured point (19.75 GB at 1024², 23.68 GB at 1536²); the engine reserves
                //   max(scalar, line) per run (WorkloadDeclaring) and refuses past the ceiling.
                //   Edit stops at 1024² (`maxEditOutputPixels`): at 1536² it measured 53.0 GB of
                //   activation and 108 s/step, so it is refused in run().
                //   Worst admitted case, bf16 t2i 1536²: declared 21.0 + 23.68 = 44.7 GB vs 44.29 measured.
                //   (The pre-evict in-app 67.4 / 70.5 GB of 2026-07 is gone.)
                footprints: [Quant.int4, .int8, .bf16].map { quant in
                    QuantFootprint(
                        quant: quant, residentBytes: Self.residentBytes[quant]!,
                        peakActivationBytes: 19_800_000_000,
                        activationScaling: ActivationScaling(
                            axis: .outputPixels, baseBytes: 16_600_000_000, bytesPerUnit: 3000,
                            measuredCeiling: Double(maxOutputPixels)))
                },
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

    /// Measured post-load phys per DiT quant (manifest comment above), rounded up.
    nonisolated static let residentBytes: [Quant: UInt64] = [
        .int4: 8_300_000_000, .int8: 12_900_000_000, .bf16: 21_000_000_000,
    ]

    private let configuration: Configuration
    /// Per-stage residency (efficiency contract 1.14.0): the Qwen3-VL-8B conditioner (~16 GB
    /// bf16) is NOT held resident — it is loaded on demand via this provider, used to encode
    /// the conditioning, then EVICTED before the DiT denoise peak (see `run(_:)`). Only the
    /// DiT + FLUX VAE (the `generator`) stay resident.
    private var encoderProvider: (() async throws -> BooguPromptEncoder)?
    private var generator: BooguImageGenerator?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard generator == nil else { return }
        guard let snapshot = configuration.resolvedSnapshotDirectory() else {
            throw BooguImagePackageError.unreadableSnapshot(configuration.repo)
        }
        let transformerDir = snapshot.appendingPathComponent("transformer")
        guard FileManager.default.fileExists(atPath: transformerDir.path) else {
            throw BooguImagePackageError.unreadableSnapshot(snapshot.path)
        }
        let snapshotQuant = try Self.snapshotQuant(transformerDir: transformerDir)
        guard snapshotQuant == configuration.quant else {
            throw BooguImagePackageError.quantMismatch(declared: configuration.quant, snapshot: snapshotQuant)
        }
        guard let qwenDir = configuration.resolvedQwenDirectory(),
              FileManager.default.fileExists(atPath: qwenDir.appendingPathComponent("config.json").path)
        else {
            throw BooguImagePackageError.unreadableSnapshot(
                configuration.resolvedQwenDirectory()?.path ?? configuration.qwenRepo)
        }

        // Conditioner (Qwen3-VL) + tokenizer: captured as a per-request loader, NOT loaded
        // resident. Each request loads it, encodes, then evicts it before the denoise loop —
        // so the ~16 GB conditioner is never co-resident with the DiT denoise activation peak.
        encoderProvider = { try await BooguPromptEncoder.load(qwenDir: qwenDir, dtype: .bfloat16) }

        // DiT (quantized if a quant_config.json is present) + VAE + scheduler — loaded on
        // the CPU stream (a multi-GB read on the GPU stream trips the Metal watchdog).
        var dit: BooguImageTransformer2DModel!
        var vae: AutoencoderKL!
        // BOOGU_FP32 escape hatch: force the fp32 DiT regardless of configuration
        // (mirrors MAGEFLOW_FP32 — for parity work / bisecting a suspect bf16 render).
        let fp32 = configuration.useFP32DiT
            || ProcessInfo.processInfo.environment["BOOGU_FP32"] != nil
        try Device.withDefaultDevice(.cpu) {
            dit = try BooguWeights.loadDiTAuto(
                transformerDir: transformerDir, fp32: fp32)
            eval(dit)
            vae = try BooguWeights.loadVAE(
                directory: snapshot.appendingPathComponent("vae"), dtype: .float32)
        }
        let scheduler = try FlowMatchEulerDiscreteScheduler(
            directory: snapshot.appendingPathComponent("scheduler"))

        generator = BooguImageGenerator(dit: dit, vae: vae, scheduler: scheduler)
    }

    public func unload() async {
        encoderProvider = nil
        generator = nil
        MLX.Memory.clearCache()  // release the retained MLX pool so eviction frees RSS (not just drop refs)
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before notLoaded validation
        // (engine ≥ 0.27.0). Mid-run cadence: post-encode-evict checkpoints below, per-denoise-
        // step `Task.isCancelled` break in BooguImageGenerator.denoise (cancelled-task decode
        // skip), and the post-generate checkpoints rethrow CancellationError unchanged.
        try Task.checkCancellation()
        guard let encoderProvider, let generator else { throw PackageError.notLoaded }

        switch request.capability {
        case .textToImage:
            guard let t2i = request as? T2IRequest else {
                throw PackageError.unsupportedCapability(request.capability)
            }
            let size = t2i.width ?? t2i.height ?? configuration.defaultSize
            let (width, height) = try Self.outputSize(t2i.width ?? size, t2i.height ?? size)
            let guidance = Float(t2i.guidanceScale ?? configuration.defaultGuidance)
            // PER-STAGE EVICTION: load the Qwen3-VL conditioner, encode pos/neg, force-materialize
            // the conditioning (`eval`), then drop the encoder + clear the cache BEFORE the denoise
            // loop so the ~16 GB conditioner is not co-resident with the DiT activation peak. The
            // `last_hidden_state` tensors are eval'd here, so the math the DiT consumes is identical
            // (parity-preserved: only WHEN the encoder frees changes).
            var encoderRef: BooguPromptEncoder? = try await encoderProvider()
            let pos = try encoderRef!.encodeText(t2i.prompt).asType(generator.dit.dtype)
            let neg = guidance > 1.0
                ? try encoderRef!.encodeText(t2i.negativePrompt ?? "").asType(generator.dit.dtype)
                : nil
            eval([pos] + (neg.map { [$0] } ?? []))  // materialize off the encoder graph
            encoderRef = nil                         // release the conditioner (last strong ref)
            MLX.Memory.clearCache()                  // reclaim the ~16 GB before the denoise peak
            try Task.checkCancellation()
            let (pixels, w, h) = generator.generate(
                posCond: pos, negCond: neg,
                height: height, width: width,
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
            let (tw, th) = try Self.outputSize(edit.width ?? configuration.defaultEditSize,
                                               edit.height ?? configuration.defaultEditSize,
                                               maxPixels: Self.maxEditOutputPixels)
            let textG = Float(edit.guidanceScale ?? configuration.defaultEditGuidance)
            // Vision+text conditioning over the input image; ref latent at the target size.
            // PER-STAGE EVICTION (as T2I): load the Qwen3-VL conditioner, encode pos/neg, eval the
            // conditioning, then evict the encoder before the ref-latent + denoise (parity-preserved
            // — only WHEN the encoder frees changes; the VAE ref-latent is on the resident generator).
            var encoderRef: BooguPromptEncoder? = try await encoderProvider()
            let pos = try encoderRef!.encodeImage(
                rgb: input.rgb, width: input.width, height: input.height, instruction: edit.prompt)
                .asType(generator.dit.dtype)
            let neg = try encoderRef!.encodeImage(
                rgb: input.rgb, width: input.width, height: input.height,
                instruction: edit.negativePrompt ?? "").asType(generator.dit.dtype)
            eval([pos, neg])         // materialize off the encoder graph
            encoderRef = nil         // release the conditioner (last strong ref)
            MLX.Memory.clearCache()  // reclaim the ~16 GB before the ref-latent + denoise peak
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

    /// Largest text-to-image output area the declared footprints cover: 1536², measured. Memory grows
    /// with area, so the cap is on area and a non-square request of the same area is admitted
    /// (AB-L-0202, AB-T-0204). The manifest's `outputPixels` activation line reaches exactly this far.
    public nonisolated static let maxOutputPixels = 1536 * 1536
    /// Largest edit output area: 1024². An edit runs the DiT over the target AND the reference
    /// latent (twice the tokens), and at 1536² it measured 53 GB of activation at 108 s/step (int4).
    public nonisolated static let maxEditOutputPixels = 1024 * 1024
    /// Each side, floored to multiples of 16 (the DiT's token grid); 512 is the smallest size measured
    /// and rendered. (A 512² render needs its default steps: the checkpoint's static time-shift is tuned
    /// for 1024², and at 8 steps a 512² image comes out undeveloped, in fp32 as in bf16.)
    public nonisolated static let minOutputSide = 512, maxOutputSide = 2048

    /// Request size → the generated size: floored to multiples of 16, then checked against the
    /// envelope. Throws `sizeOutOfEnvelope` instead of running past the declared footprint.
    public nonisolated static func outputSize(_ width: Int, _ height: Int,
                                              maxPixels: Int = maxOutputPixels) throws -> (Int, Int) {
        let w = (width / 16) * 16, h = (height / 16) * 16
        guard min(w, h) >= minOutputSide, max(w, h) <= maxOutputSide, w * h <= maxPixels else {
            throw BooguImagePackageError.sizeOutOfEnvelope(width: w, height: h, maxPixels: maxPixels)
        }
        return (w, h)
    }

    /// The snapshot DiT's quant: `transformer/quant_config.json` bits, else bf16.
    nonisolated static func snapshotQuant(transformerDir: URL) throws -> Quant {
        let url = transformerDir.appendingPathComponent("quant_config.json")
        guard FileManager.default.fileExists(atPath: url.path) else { return .bf16 }
        struct QC: Decodable { var bits: Int }
        let bits = try JSONDecoder().decode(QC.self, from: Data(contentsOf: url)).bits
        switch bits {
        case 4: return .int4
        case 8: return .int8
        default: throw BooguImagePackageError.unreadableSnapshot(url.path)
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
