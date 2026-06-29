// CLI gate runner for the Boogu-Image Swift port. Each gate is a `swift run BooguGate
// --<gate> ...` mode so it runs in a real Metal context (the SPM test product's
// metallib is unreliable; see the integration skill).
//
//   --s0-keys <baseDir> <fixturesDir>   structural key contract (no eval, no weights)
//
// More gates (component / e2e / quant parity) append here as phases land.

import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import BooguImage
import MLX
import MLXNN

/// Interleaved RGB8 -> PNG on disk.
func writePNG(pixels: [UInt8], width: Int, height: Int, to url: URL) {
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    let ctx = CGContext(
        data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
        space: cs, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
    let buf = ctx.data!.bindMemory(to: UInt8.self, capacity: width * height * 4)
    for i in 0..<(width * height) {
        buf[i * 4] = pixels[i * 3]; buf[i * 4 + 1] = pixels[i * 3 + 1]
        buf[i * 4 + 2] = pixels[i * 3 + 2]; buf[i * 4 + 3] = 255
    }
    let image = ctx.makeImage()!
    let dest = CGImageDestinationCreateWithData(
        NSMutableData() as CFMutableData, UTType.png.identifier as CFString, 1, nil)
    _ = dest
    if let d = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) {
        CGImageDestinationAddImage(d, image, nil)
        CGImageDestinationFinalize(d)
    }
}

func err(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }

/// Run async `work` to completion from sync code while keeping the main run loop alive,
/// then exit with the returned code. `AutoTokenizer.from` hops to the main actor; a bare
/// `sem.wait()` on main deadlocks (the app host works because it has a live run loop).
func runBlocking(_ work: @Sendable @escaping () async -> Int32) -> Never {
    final class Box: @unchecked Sendable { var rc: Int32 = 0 }
    let box = Box()
    let sem = DispatchSemaphore(value: 0)
    Task { box.rc = await work(); sem.signal() }
    while sem.wait(timeout: .now() + 0.02) == .timedOut {
        RunLoop.main.run(mode: .default, before: Date(timeIntervalSinceNow: 0.02))
    }
    exit(box.rc)
}

/// Decode a PNG/JPEG file to interleaved RGB8 (sRGB).
func decodeRGB(_ url: URL) -> (rgb: [UInt8], width: Int, height: Int)? {
    guard let data = try? Data(contentsOf: url),
          let src = CGImageSourceCreateWithData(data as CFData, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
    let (w, h) = (cg.width, cg.height)
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    guard let ctx = CGContext(
        data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
        space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    var rgb = [UInt8](repeating: 0, count: w * h * 3)
    for i in 0..<(w * h) {
        rgb[i * 3] = rgba[i * 4]; rgb[i * 3 + 1] = rgba[i * 4 + 1]; rgb[i * 3 + 2] = rgba[i * 4 + 2]
    }
    return (rgb, w, h)
}

func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
    abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
    let x = a.asType(.float32).flattened(); let y = b.asType(.float32).flattened()
    return (x * y).sum().item(Float.self)
        / (sqrt((x * x).sum()).item(Float.self) * sqrt((y * y).sum()).item(Float.self))
}

/// Report a parity result against a threshold; returns pass/fail.
func report(_ label: String, _ value: Float, threshold: Float) -> Bool {
    let ok = value <= threshold
    err("[\(label)] max_abs \(value) (threshold \(threshold)) -> \(ok ? "PASS" : "FAIL")")
    return ok
}

let args = Array(CommandLine.arguments.dropFirst())
guard let gate = args.first else {
    err("usage: BooguGate --<gate> [args...]")
    exit(2)
}

switch gate {
case "--s0-keys":
    guard args.count >= 3 else { err("--s0-keys <baseDir> <fixturesDir>"); exit(2) }
    let base = URL(fileURLWithPath: args[1])
    let fixtures = URL(fileURLWithPath: args[2])

    func loadKeyFixture(_ name: String) throws -> Set<String> {
        let data = try Data(contentsOf: fixtures.appendingPathComponent(name))
        return Set(try JSONDecoder().decode([String].self, from: data))
    }

    func compare(label: String, module keys: Set<String>, expected: Set<String>) -> Bool {
        let missing = expected.subtracting(keys).sorted()
        let extra = keys.subtracting(expected).sorted()
        if missing.isEmpty && extra.isEmpty {
            err("[\(label)] OK — \(keys.count) keys match exactly")
            return true
        }
        err("[\(label)] MISMATCH — module \(keys.count), expected \(expected.count)")
        if !missing.isEmpty { err("  missing from module (in ckpt only): \(missing.prefix(12))") }
        if !extra.isEmpty { err("  extra in module (not in ckpt): \(extra.prefix(12))") }
        return false
    }

    do {
        let ditCfg = try BooguTransformerConfig.load(
            base.appendingPathComponent("transformer/config.json"))
        let dit = BooguImageTransformer2DModel(ditCfg)
        let ditKeys = Set(dit.parameters().flattened().map(\.0))
        let okDiT = compare(label: "DiT", module: ditKeys, expected: try loadKeyFixture("dit_keys.json"))

        let vaeCfg = try BooguVAEConfig.load(base.appendingPathComponent("vae/config.json"))
        let vae = AutoencoderKL(vaeCfg)
        let vaeKeys = Set(vae.parameters().flattened().map(\.0))
        let okVAE = compare(label: "VAE", module: vaeKeys, expected: try loadKeyFixture("vae_keys.json"))

        exit(okDiT && okVAE ? 0 : 1)
    } catch {
        err("error: \(error)")
        exit(3)
    }

case "--s1-vae":
    guard args.count >= 3 else { err("--s1-vae <baseDir> <goldenDir>"); exit(2) }
    let base = URL(fileURLWithPath: args[1])
    let g = try! MLX.loadArrays(url: URL(fileURLWithPath: args[2]).appendingPathComponent("vae_golden.safetensors"))
    var ok = true
    Device.withDefaultDevice(.cpu) {
        do {
            let vae = try BooguWeights.loadVAE(directory: base.appendingPathComponent("vae"), dtype: .float32)
            let dec = vae.decode(g["z"]!); eval(dec)
            ok = report("VAE.decode", maxAbs(dec, g["decode_out"]!), threshold: 1e-3) && ok
            let mom = vae.encodeMoments(g["img"]!); eval(mom)
            ok = report("VAE.encode", maxAbs(mom, g["encode_out"]!), threshold: 1e-3) && ok
        } catch { err("error: \(error)"); ok = false }
    }
    exit(ok ? 0 : 1)

case "--s1-sched":
    guard args.count >= 3 else { err("--s1-sched <baseDir> <goldenDir>"); exit(2) }
    let base = URL(fileURLWithPath: args[1])
    let g = try! MLX.loadArrays(url: URL(fileURLWithPath: args[2]).appendingPathComponent("scheduler_golden.safetensors"))
    do {
        let sch = try FlowMatchEulerDiscreteScheduler(directory: base.appendingPathComponent("scheduler"))
        let ts28 = MLXArray(sch.setTimesteps(28))
        let ts4 = MLXArray(sch.setTimesteps(4))
        let ok = report("sched.28", maxAbs(ts28, g["ts_static_28"]!), threshold: 1e-6)
            && report("sched.4", maxAbs(ts4, g["ts_static_4"]!), threshold: 1e-6)
        exit(ok ? 0 : 1)
    } catch { err("error: \(error)"); exit(3) }

case "--s2-dit":
    guard args.count >= 3 else { err("--s2-dit <baseDir> <goldenDir>"); exit(2) }
    let base = URL(fileURLWithPath: args[1])
    let g = try! MLX.loadArrays(url: URL(fileURLWithPath: args[2]).appendingPathComponent("dit_golden.safetensors"))
    var ok = true
    Device.withDefaultDevice(.cpu) {
        do {
            let dit = try BooguWeights.loadDiT(directory: base.appendingPathComponent("transformer"), dtype: .float32)
            let outT2I = dit(latent: g["latent"]!, timestep: g["timestep"]!,
                             instructionHiddenStates: g["instruction"]!)
            eval(outT2I)
            ok = report("DiT.t2i", maxAbs(outT2I, g["out_t2i"]!), threshold: 5e-3) && ok
            let outEdit = dit(latent: g["latent"]!, timestep: g["timestep"]!,
                              instructionHiddenStates: g["instruction"]!, refLatent: g["ref_latent"]!)
            eval(outEdit)
            ok = report("DiT.edit", maxAbs(outEdit, g["out_edit"]!), threshold: 5e-3) && ok
        } catch { err("error: \(error)"); ok = false }
    }
    exit(ok ? 0 : 1)

case "--s6-quant":
    // Quantized DiT forward (load CPU stream, forward GPU) vs the fp32 golden, cosine.
    guard args.count >= 4 else { err("--s6-quant <configDir> <quantDir> <goldenDir>"); exit(2) }
    let configDir = URL(fileURLWithPath: args[1])
    let quantDir = URL(fileURLWithPath: args[2])
    let g = try! MLX.loadArrays(url: URL(fileURLWithPath: args[3]).appendingPathComponent("dit_golden.safetensors"))
    do {
        var dit: BooguImageTransformer2DModel!
        try Device.withDefaultDevice(.cpu) {
            dit = try BooguWeights.loadDiTQuantized(configDir: configDir, quantDir: quantDir)
        }
        // Forward on GPU (CPU-pinned quantized graph grinds — skill lesson).
        let out = dit(latent: g["latent"]!, timestep: g["timestep"]!,
                      instructionHiddenStates: g["instruction"]!)
        eval(out)
        let cos = cosine(out, g["out_t2i"]!)
        err("[s6-quant] DiT.t2i cosine \(cos) (threshold 0.99)")
        exit(cos >= 0.99 ? 0 : 1)
    } catch { err("error: \(error)"); exit(3) }

case "--e2e-golden":
    // DiT(bf16) + VAE + golden conditioning -> a real image. Proves DiT+VAE+scheduler
    // compose, independent of the live Qwen3-VL encoder.
    guard args.count >= 4 else { err("--e2e-golden <baseDir> <goldenDir> <out.png> [steps] [size]"); exit(2) }
    let base = URL(fileURLWithPath: args[1])
    let g = try! MLX.loadArrays(url: URL(fileURLWithPath: args[2]).appendingPathComponent("cond_e2e.safetensors"))
    let outURL = URL(fileURLWithPath: args[3])
    let steps = args.count > 4 ? Int(args[4])! : 30
    let size = args.count > 5 ? Int(args[5])! : 768
    let ditDType: DType = (args.count > 6 && args[6] == "fp32") ? .float32 : .bfloat16
    do {
        var dit: BooguImageTransformer2DModel!
        try Device.withDefaultDevice(.cpu) {
            dit = try BooguWeights.loadDiT(directory: base.appendingPathComponent("transformer"), dtype: ditDType)
            eval(dit)
        }
        let vae = try Device.withDefaultDevice(.cpu) {
            try BooguWeights.loadVAE(directory: base.appendingPathComponent("vae"), dtype: .float32)
        }
        let sched = try FlowMatchEulerDiscreteScheduler(directory: base.appendingPathComponent("scheduler"))
        let gen = BooguImageGenerator(dit: dit, vae: vae, scheduler: sched)
        let pos = g["feats"]!.asType(dit.dtype)
        let neg = g["feats_neg"]!.asType(dit.dtype)
        let (px, w, h) = gen.generate(
            posCond: pos, negCond: neg, height: size, width: size, steps: steps, guidance: 3.5, seed: 0,
            progress: { i, n in if i % 5 == 0 || i == n { err("  step \(i)/\(n)") } })
        writePNG(pixels: px, width: w, height: h, to: outURL)
        err("[e2e-golden] wrote \(outURL.path) (\(w)x\(h), \(steps) steps)")
        exit(0)
    } catch { err("error: \(error)"); exit(3) }

case "--e2e":
    // Full live textToImage: prompt -> Qwen3-VL conditioning -> DiT -> VAE -> PNG.
    guard args.count >= 5 else { err("--e2e <baseDir> <qwenDir> <prompt> <out.png> [steps] [guidance] [size]"); exit(2) }
    let base = URL(fileURLWithPath: args[1])
    let qwenDir = URL(fileURLWithPath: args[2])
    let prompt = args[3]
    let outURL = URL(fileURLWithPath: args[4])
    let steps = args.count > 5 ? Int(args[5])! : 30
    let guidance = args.count > 6 ? Float(args[6])! : 3.5
    let size = args.count > 7 ? Int(args[7])! : 768
    runBlocking {
        do {
            let encoder = try await BooguPromptEncoder.load(qwenDir: qwenDir, dtype: .bfloat16)
            var dit: BooguImageTransformer2DModel!
            try Device.withDefaultDevice(.cpu) {
                dit = try BooguWeights.loadDiT(directory: base.appendingPathComponent("transformer"), dtype: .bfloat16)
            }
            let vae = try Device.withDefaultDevice(.cpu) {
                try BooguWeights.loadVAE(directory: base.appendingPathComponent("vae"), dtype: .float32)
            }
            let sched = try FlowMatchEulerDiscreteScheduler(directory: base.appendingPathComponent("scheduler"))
            let gen = BooguImageGenerator(dit: dit, vae: vae, scheduler: sched)
            let pos = try encoder.encodeText(prompt).asType(dit.dtype)
            let neg = guidance > 1.0 ? try encoder.encodeText("").asType(dit.dtype) : nil
            err("[e2e] cond pos \(pos.shape)")
            let (px, w, h) = gen.generate(
                posCond: pos, negCond: neg, height: size, width: size, steps: steps,
                guidance: guidance, seed: 0,
                progress: { i, n in if i % 5 == 0 || i == n { err("  step \(i)/\(n)") } })
            writePNG(pixels: px, width: w, height: h, to: outURL)
            err("[e2e] wrote \(outURL.path) (\(w)x\(h))")
            return 0
        } catch { err("error: \(error)"); return 3 }
    }

case "--e2e-edit":
    // Full live imageEdit: input image + instruction -> Qwen3-VL vision+text conditioning
    // + VAE ref latent -> DiT edit denoise -> PNG.
    guard args.count >= 6 else {
        err("--e2e-edit <editSnapshot> <qwenDir> <inImage> <instruction> <out.png> [steps] [size] [dtype]")
        exit(2)
    }
    let base = URL(fileURLWithPath: args[1])
    let qwenDir = URL(fileURLWithPath: args[2])
    let inImage = URL(fileURLWithPath: args[3])
    let instruction = args[4]
    let outURL = URL(fileURLWithPath: args[5])
    let steps = args.count > 6 ? Int(args[6])! : 28
    let size = args.count > 7 ? Int(args[7])! : 512
    let ditDType: DType = (args.count > 8 && args[8] == "fp32") ? .float32 : .bfloat16
    runBlocking {
        do {
            guard let img = decodeRGB(inImage) else { err("cannot read \(inImage.path)"); return 3 }
            let encoder = try await BooguPromptEncoder.load(qwenDir: qwenDir, dtype: .bfloat16)
            var dit: BooguImageTransformer2DModel!
            var vae: AutoencoderKL!
            try Device.withDefaultDevice(.cpu) {
                dit = try BooguWeights.loadDiTAuto(
                    transformerDir: base.appendingPathComponent("transformer"), fp32: ditDType == .float32)
                eval(dit)
                vae = try BooguWeights.loadVAE(directory: base.appendingPathComponent("vae"), dtype: .float32)
            }
            let sched = try FlowMatchEulerDiscreteScheduler(directory: base.appendingPathComponent("scheduler"))
            let gen = BooguImageGenerator(dit: dit, vae: vae, scheduler: sched)
            let pos = try encoder.encodeImage(rgb: img.rgb, width: img.width, height: img.height,
                                              instruction: instruction).asType(dit.dtype)
            let neg = try encoder.encodeImage(rgb: img.rgb, width: img.width, height: img.height,
                                              instruction: "").asType(dit.dtype)
            let refLatent = gen.encodeRefLatent(rgb: img.rgb, width: img.width, height: img.height,
                                                targetWidth: size, targetHeight: size)
            err("[e2e-edit] cond \(pos.shape) ref \(refLatent.shape)")
            let (px, w, h) = gen.generateEdit(
                posCond: pos, negCond: neg, refLatent: refLatent, height: size, width: size,
                steps: steps, textGuidance: 4.0, seed: 0,
                progress: { i, n in if i % 5 == 0 || i == n { err("  step \(i)/\(n)") } })
            writePNG(pixels: px, width: w, height: h, to: outURL)
            err("[e2e-edit] wrote \(outURL.path) (\(w)x\(h))")
            return 0
        } catch { err("error: \(error)"); return 3 }
    }

default:
    err("unknown gate: \(gate)")
    exit(2)
}
