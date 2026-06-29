// Boogu-Image T2I / Edit generation — Swift mirror of boogu_image_mlx/pipeline_mlx.py
// (the denoise + decode half). Conditioning (Qwen3-VL last_hidden_state) is produced by
// BooguPromptEncoder and passed in, keeping this generator independent of the encoder.

import Foundation
import MLX
import MLXNN
import MLXRandom
import Qwen3VL

public final class BooguImageGenerator {
    public let dit: BooguImageTransformer2DModel
    public let vae: AutoencoderKL
    public let scheduler: FlowMatchEulerDiscreteScheduler

    public init(dit: BooguImageTransformer2DModel, vae: AutoencoderKL,
                scheduler: FlowMatchEulerDiscreteScheduler) {
        self.dit = dit
        self.vae = vae
        self.scheduler = scheduler
    }

    private var condDType: DType { dit.xEmbedder.weight.dtype }

    /// Decode a denoised latent [1,16,hl,wl] to interleaved RGB8 + dims.
    private func decodeToRGB(_ latent: MLXArray, height: Int, width: Int)
        -> (pixels: [UInt8], width: Int, height: Int)
    {
        if ProcessInfo.processInfo.environment["BOOGU_DEBUG"] != nil {
            let lf = latent.asType(.float32)
            let msg = "latent min \(lf.min().item(Float.self)) max \(lf.max().item(Float.self)) "
                + "mean \(lf.mean().item(Float.self))\n"
            FileHandle.standardError.write(Data(msg.utf8))
        }
        let z = latent.asType(.float32) / vae.scalingFactor + vae.shiftFactor
        let img = vae.decode(z)  // [1,3,H,W]
        if ProcessInfo.processInfo.environment["BOOGU_DEBUG"] != nil {
            let msg = "decoded min \(img.min().item(Float.self)) max \(img.max().item(Float.self))\n"
            FileHandle.standardError.write(Data(msg.utf8))
        }
        let arr = clip(img[0] / 2 + 0.5, min: 0, max: 1)  // [3,H,W]
        let hwc = (arr.transposed(1, 2, 0) * 255).asType(.uint8)  // [H,W,3]
        eval(hwc)
        return (hwc.asArray(UInt8.self), width, height)
    }

    /// Run the CFG denoise loop and return RGB8. `refLatent` (Edit) is optional.
    private func denoise(
        posCond: MLXArray, negCond: MLXArray?, refLatent: MLXArray?,
        height: Int, width: Int, steps: Int, guidance: Float, seed: UInt64,
        progress: ((Int, Int) -> Void)?
    ) -> (pixels: [UInt8], width: Int, height: Int) {
        let dtype = condDType
        let (hl, wl) = (height / 8, width / 8)
        scheduler.setTimesteps(steps, numTokens: hl * wl)
        var latent = MLXRandom.normal([1, 16, hl, wl], key: MLXRandom.key(seed)).asType(dtype)

        for i in 0..<steps {
            let t = MLXArray([scheduler.timesteps[i]]).asType(dtype)
            var pred = dit(latent: latent, timestep: t, instructionHiddenStates: posCond,
                           refLatent: refLatent)
            if let negCond, guidance > 1.0 {
                let pu = dit(latent: latent, timestep: t, instructionHiddenStates: negCond,
                             refLatent: refLatent)
                pred = pu + guidance * (pred - pu)
            }
            latent = scheduler.step(pred, stepIndex: i, sample: latent)
            eval(latent)
            MLX.GPU.clearCache()
            progress?(i + 1, steps)
        }
        return decodeToRGB(latent, height: height, width: width)
    }

    /// textToImage. Base: steps 30 / guidance 3.5; Turbo: steps 4 / guidance 1.0.
    public func generate(
        posCond: MLXArray, negCond: MLXArray?, height: Int = 1024, width: Int = 1024,
        steps: Int = 30, guidance: Float = 3.5, seed: UInt64 = 0,
        progress: ((Int, Int) -> Void)? = nil
    ) -> (pixels: [UInt8], width: Int, height: Int) {
        denoise(posCond: posCond, negCond: negCond, refLatent: nil, height: height, width: width,
                steps: steps, guidance: guidance, seed: seed, progress: progress)
    }

    /// imageEdit. ref latent = (vae mean(image) - shift) * scale; text_guidance default 4.0.
    public func generateEdit(
        posCond: MLXArray, negCond: MLXArray, refLatent: MLXArray, height: Int, width: Int,
        steps: Int = 50, textGuidance: Float = 4.0, seed: UInt64 = 0,
        progress: ((Int, Int) -> Void)? = nil
    ) -> (pixels: [UInt8], width: Int, height: Int) {
        denoise(posCond: posCond, negCond: negCond, refLatent: refLatent, height: height,
                width: width, steps: steps, guidance: textGuidance, seed: seed, progress: progress)
    }

    /// Encode an RGB image to the scaled ref latent the Edit DiT consumes.
    public func encodeRefLatent(rgb: [UInt8], width: Int, height: Int) -> MLXArray {
        var chw = [Float](repeating: 0, count: 3 * height * width)
        let plane = height * width
        for i in 0..<plane {
            let p = i * 3
            chw[i] = Float(rgb[p]) / 255 * 2 - 1
            chw[plane + i] = Float(rgb[p + 1]) / 255 * 2 - 1
            chw[2 * plane + i] = Float(rgb[p + 2]) / 255 * 2 - 1
        }
        let x = MLXArray(chw, [1, 3, height, width])
        let moments = vae.encodeMoments(x)  // [1,32,H/8,W/8]
        let mean = moments[0..., 0..<16]
        return ((mean - vae.shiftFactor) * vae.scalingFactor).asType(condDType)
    }

    /// Resize an arbitrary-size RGB image to (targetWidth, targetHeight) — PIL bicubic,
    /// matching the Python pipeline's `image.resize((width, height))` — then encode the
    /// ref latent.
    public func encodeRefLatent(
        rgb: [UInt8], width: Int, height: Int, targetWidth: Int, targetHeight: Int
    ) -> MLXArray {
        let resized = PILResize.resize(
            rgb: rgb, width: width, height: height, outWidth: targetWidth, outHeight: targetHeight)
        return encodeRefLatent(rgb: resized, width: targetWidth, height: targetHeight)
    }
}
