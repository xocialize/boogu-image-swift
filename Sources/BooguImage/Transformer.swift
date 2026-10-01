// BooguImageTransformer2DModel (OmniGen2 / Lumina2 lineage) — Swift mirror of
// boogu_image_mlx/models/transformer.py. Module / parameter names mirror the PyTorch
// state_dict so the 942-key checkpoint maps 1:1 (all Linear / RMSNorm + the bare
// `image_index_embedding` parameter — no transposes).
//
// Stream topology: x_embedder -> context_refiner(instruct) + noise_refiner(img)
// -> 8 double-stream (img<->instruct joint attn) -> fuse -> 32 single-stream
// -> norm_out -> unpatchify. Base T2I omits the ref-image branches; Edit activates
// ref_image_patch_embedder + ref_image_refiner + image_index_embedding.

import Foundation
import MLX
import MLXFast
import MLXNN

private func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

// MARK: - RoPE (3-axis, Lumina complex form expressed as real arithmetic)

/// position_ids: [L,3] -> (cos, sin) each [L, sum(axesDim)/2]. Frequencies are built
/// in Double (numpy-float64-then-astype) before the Float cast, per the parity doctrine.
func ropeCosSin(positionIds: [[Int]], axesDim: [Int], theta: Int) -> (MLXArray, MLXArray) {
    let L = positionIds.count
    let halfTotal = axesDim.reduce(0) { $0 + $1 / 2 }
    var cos = [Float](repeating: 0, count: L * halfTotal)
    var sin = [Float](repeating: 0, count: L * halfTotal)
    let thetaD = Double(theta)
    for l in 0..<L {
        var col = 0
        for (a, dim) in axesDim.enumerated() {
            let pos = Double(positionIds[l][a])
            var d = 0
            while d < dim {
                let invFreq = 1.0 / Foundation.pow(thetaD, Double(d) / Double(dim))
                let ang = pos * invFreq
                cos[l * halfTotal + col] = Float(Foundation.cos(ang))
                sin[l * halfTotal + col] = Float(Foundation.sin(ang))
                col += 1
                d += 2
            }
        }
    }
    return (MLXArray(cos, [L, halfTotal]), MLXArray(sin, [L, halfTotal]))
}

/// x: [B,L,H,D]; cos/sin: [1,L,1,D/2]. Complex-pair rotation.
func applyRope(_ x: MLXArray, cos cosT: MLXArray, sin sinT: MLXArray) -> MLXArray {
    let shape = x.shape  // [B,L,H,D]
    let c = cosT.asType(x.dtype)
    let s = sinT.asType(x.dtype)
    let xp = x.reshaped(shape[0], shape[1], shape[2], shape[3] / 2, 2)
    let x0 = xp[.ellipsis, 0]
    let x1 = xp[.ellipsis, 1]
    let out0 = x0 * c - x1 * s
    let out1 = x0 * s + x1 * c
    return stacked([out0, out1], axis: -1).reshaped(shape)
}

/// q:[B,L,H,d] k,v:[B,L,kvH,d] -> [B,L,H*d]. GQA expand, no mask (batch=1).
private func sdpa(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray,
                  heads: Int, kvHeads: Int, scale: Float) -> MLXArray {
    let (B, L, d) = (q.dim(0), q.dim(1), q.dim(3))
    var qT = q.transposed(0, 2, 1, 3)  // [B,H,L,d]
    var kT = k.transposed(0, 2, 1, 3)
    var vT = v.transposed(0, 2, 1, 3)
    _ = qT  // qT used below
    if heads != kvHeads {
        let rep = heads / kvHeads
        kT = repeated(kT, count: rep, axis: 1)
        vT = repeated(vT, count: rep, axis: 1)
    }
    let out = MLXFast.scaledDotProductAttention(
        queries: qT, keys: kT, values: vT, scale: scale, mask: .none)
    return out.transposed(0, 2, 1, 3).reshaped(B, L, heads * d)
}

// MARK: - Norms / FFN / embeddings

final class LuminaRMSNormZero: Module {
    @ModuleInfo var linear: Linear
    @ModuleInfo var norm: RMSNorm

    init(dim: Int, normEps: Float) {
        self._linear.wrappedValue = Linear(min(dim, 1024), 4 * dim, bias: true)
        self._norm.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
        super.init()
    }

    /// Returns (x_normed_scaled, gate_msa, scale_mlp, gate_mlp).
    func callAsFunction(_ x: MLXArray, _ temb: MLXArray)
        -> (MLXArray, MLXArray, MLXArray, MLXArray)
    {
        let emb = linear(silu(temb))  // [B, 4*dim]
        let parts = split(emb, parts: 4, axis: -1)
        let scaleMsa = parts[0], gateMsa = parts[1], scaleMlp = parts[2], gateMlp = parts[3]
        let xn = norm(x) * (1 + scaleMsa.expandedDimensions(axis: 1))
        return (xn, gateMsa, scaleMlp, gateMlp)
    }
}

/// elementwise_affine=False LayerNorm + AdaLN scale + output projection.
final class LuminaLayerNormContinuous: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    let eps: Float

    init(dim: Int, condDim: Int, outDim: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._linear1.wrappedValue = Linear(condDim, dim, bias: true)
        self._linear2.wrappedValue = Linear(dim, outDim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ cond: MLXArray) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let variance = x.variance(axis: -1, keepDims: true)
        var y = (x - mean) * rsqrt(variance + eps)
        let scale = linear1(silu(cond))
        y = y * (1 + scale).expandedDimensions(axis: 1)
        return linear2(y)
    }
}

final class LuminaFeedForward: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    @ModuleInfo(key: "linear_3") var linear3: Linear

    init(dim: Int, innerDim innerDimIn: Int, multipleOf: Int = 256) {
        let innerDim = multipleOf * ((innerDimIn + multipleOf - 1) / multipleOf)
        self._linear1.wrappedValue = Linear(dim, innerDim, bias: false)
        self._linear2.wrappedValue = Linear(innerDim, dim, bias: false)
        self._linear3.wrappedValue = Linear(dim, innerDim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(x)) * linear3(x))
    }
}

/// diffusers Timesteps: flip_sin_to_cos=True, downscale_freq_shift=0.0 -> [cos, sin].
func getTimestepEmbedding(_ timesteps: MLXArray, dim: Int, scale: Float,
                          maxPeriod: Int = 10000) -> MLXArray {
    let half = dim / 2
    let exponent = -Foundation.log(Float(maxPeriod))
        * MLXArray(0..<half).asType(.float32) / Float(half)
    var emb = exp(exponent)
    emb = (timesteps.expandedDimensions(axis: 1).asType(.float32) * emb.expandedDimensions(axis: 0)) * scale
    return concatenated([cos(emb), sin(emb)], axis: -1)
}

final class TimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(inDim: Int, timeDim: Int) {
        self._linear1.wrappedValue = Linear(inDim, timeDim, bias: true)
        self._linear2.wrappedValue = Linear(timeDim, timeDim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
}

final class Lumina2CombinedTimestepCaptionEmbedding: Module {
    @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: TimestepEmbedding
    // Sequential(RMSNorm, Linear) -> checkpoint keys caption_embedder.0 / .1; mirror the
    // Python list as a heterogeneous [Module] so unflatten maps the numeric indices.
    @ModuleInfo(key: "caption_embedder") var captionEmbedder: [Module]
    let frequencyEmbeddingSize: Int
    let timestepScale: Float

    init(hiddenSize: Int, instructionFeatDim: Int, frequencyEmbeddingSize: Int = 256,
         normEps: Float = 1e-5, timestepScale: Float = 1000.0) {
        self.frequencyEmbeddingSize = frequencyEmbeddingSize
        self.timestepScale = timestepScale
        self._timestepEmbedder.wrappedValue = TimestepEmbedding(
            inDim: frequencyEmbeddingSize, timeDim: min(hiddenSize, 1024))
        self._captionEmbedder.wrappedValue = [
            RMSNorm(dimensions: instructionFeatDim, eps: normEps),
            Linear(instructionFeatDim, hiddenSize, bias: true),
        ]
        super.init()
    }

    func callAsFunction(_ timestep: MLXArray, _ caption: MLXArray) -> (MLXArray, MLXArray) {
        let tProj = getTimestepEmbedding(timestep, dim: frequencyEmbeddingSize, scale: timestepScale)
        let temb = timestepEmbedder(tProj.asType(caption.dtype))
        let norm = captionEmbedder[0] as! RMSNorm
        let linear = captionEmbedder[1] as! Linear
        let cap = linear(norm(caption))
        return (temb, cap)
    }
}

// MARK: - Attention

/// Self-attention with GQA, per-head RMSNorm q/k, 3-axis RoPE.
final class Attention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    let heads: Int, kvHeads: Int, headDim: Int, scale: Float

    init(dim: Int, heads: Int, kvHeads: Int, eps: Float = 1e-5) {
        self.heads = heads
        self.kvHeads = kvHeads
        self.headDim = dim / heads
        self.scale = Foundation.pow(Float(headDim), -0.5)
        self._toQ.wrappedValue = Linear(dim, heads * headDim, bias: false)
        self._toK.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._toV.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._normK.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._toOut.wrappedValue = [Linear(heads * headDim, dim, bias: false)]
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var q = normQ(toQ(x).reshaped(B, L, heads, headDim))
        var k = normK(toK(x).reshaped(B, L, kvHeads, headDim))
        let v = toV(x).reshaped(B, L, kvHeads, headDim)
        q = applyRope(q, cos: cos, sin: sin)
        k = applyRope(k, cos: cos, sin: sin)
        let out = sdpa(q, k, v, heads: heads, kvHeads: kvHeads, scale: scale)
        return toOut[0](out)
    }
}

/// Separate img/instruct q/k/v + per-stream output projections.
final class DoubleStreamProcessor: Module {
    @ModuleInfo(key: "img_to_q") var imgToQ: Linear
    @ModuleInfo(key: "img_to_k") var imgToK: Linear
    @ModuleInfo(key: "img_to_v") var imgToV: Linear
    @ModuleInfo(key: "instruct_to_q") var instructToQ: Linear
    @ModuleInfo(key: "instruct_to_k") var instructToK: Linear
    @ModuleInfo(key: "instruct_to_v") var instructToV: Linear
    @ModuleInfo(key: "img_out") var imgOut: Linear
    @ModuleInfo(key: "instruct_out") var instructOut: Linear

    init(dim: Int, heads: Int, kvHeads: Int) {
        let headDim = dim / heads
        self._imgToQ.wrappedValue = Linear(dim, heads * headDim, bias: false)
        self._imgToK.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._imgToV.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._instructToQ.wrappedValue = Linear(dim, heads * headDim, bias: false)
        self._instructToK.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._instructToV.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._imgOut.wrappedValue = Linear(heads * headDim, dim, bias: false)
        self._instructOut.wrappedValue = Linear(heads * headDim, dim, bias: false)
        super.init()
    }
}

/// img<->instruct joint attention over concatenated [instruct ; img].
final class DoubleStreamJointAttention: Module {
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    @ModuleInfo var processor: DoubleStreamProcessor
    let heads: Int, kvHeads: Int, headDim: Int, scale: Float

    init(dim: Int, heads: Int, kvHeads: Int, eps: Float = 1e-5) {
        self.heads = heads
        self.kvHeads = kvHeads
        self.headDim = dim / heads
        self.scale = Foundation.pow(Float(headDim), -0.5)
        self._normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._normK.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._toOut.wrappedValue = [Linear(heads * headDim, dim, bias: false)]
        self._processor.wrappedValue = DoubleStreamProcessor(dim: dim, heads: heads, kvHeads: kvHeads)
        super.init()
    }

    /// Returns (joint_out [B,L,dim], L_instruct).
    func callAsFunction(img: MLXArray, instruct: MLXArray, cos: MLXArray, sin: MLXArray)
        -> (MLXArray, Int)
    {
        let p = processor
        let (B, Li) = (instruct.dim(0), instruct.dim(1))
        let Limg = img.dim(1)
        // concat order: instruct first, then img.
        var q = concatenated([p.instructToQ(instruct), p.imgToQ(img)], axis: 1)
        var k = concatenated([p.instructToK(instruct), p.imgToK(img)], axis: 1)
        let v = concatenated([p.instructToV(instruct), p.imgToV(img)], axis: 1)
        let L = Li + Limg
        q = normQ(q.reshaped(B, L, heads, headDim))
        k = normK(k.reshaped(B, L, kvHeads, headDim))
        let vR = v.reshaped(B, L, kvHeads, headDim)
        q = applyRope(q, cos: cos, sin: sin)
        k = applyRope(k, cos: cos, sin: sin)
        let out = sdpa(q, k, vR, heads: heads, kvHeads: kvHeads, scale: scale)  // [B,L,dim]
        let instructOut = p.instructOut(out[0..., ..<Li])
        let imgOut = p.imgOut(out[0..., Li...])
        let merged = concatenated([instructOut, imgOut], axis: 1)
        return (toOut[0](merged), Li)
    }
}

// MARK: - Blocks

/// single_stream / noise_refiner (modulation) and context_refiner (no mod).
final class BasicBlock: Module {
    @ModuleInfo var attn: Attention
    @ModuleInfo(key: "feed_forward") var feedForward: LuminaFeedForward
    @ModuleInfo(key: "norm1") var norm1: Module  // LuminaRMSNormZero (mod) | RMSNorm (no mod)
    @ModuleInfo(key: "ffn_norm1") var ffnNorm1: RMSNorm
    @ModuleInfo var norm2: RMSNorm
    @ModuleInfo(key: "ffn_norm2") var ffnNorm2: RMSNorm
    let modulation: Bool

    init(dim: Int, heads: Int, kvHeads: Int, multipleOf: Int, normEps: Float, modulation: Bool) {
        self.modulation = modulation
        self._attn.wrappedValue = Attention(dim: dim, heads: heads, kvHeads: kvHeads, eps: 1e-5)
        self._feedForward.wrappedValue = LuminaFeedForward(dim: dim, innerDim: 4 * dim, multipleOf: multipleOf)
        self._norm1.wrappedValue = modulation
            ? LuminaRMSNormZero(dim: dim, normEps: normEps)
            : RMSNorm(dimensions: dim, eps: normEps)
        self._ffnNorm1.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
        self._norm2.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
        self._ffnNorm2.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
        super.init()
    }

    func callAsFunction(_ xIn: MLXArray, cos: MLXArray, sin: MLXArray, temb: MLXArray? = nil)
        -> MLXArray
    {
        var x = xIn
        if modulation {
            let (xn, gateMsa, scaleMlp, gateMlp) = (norm1 as! LuminaRMSNormZero)(x, temb!)
            let attnOut = attn(xn, cos: cos, sin: sin)
            x = x + tanh(gateMsa.expandedDimensions(axis: 1)) * norm2(attnOut)
            let mlp = feedForward(ffnNorm1(x) * (1 + scaleMlp.expandedDimensions(axis: 1)))
            x = x + tanh(gateMlp.expandedDimensions(axis: 1)) * ffnNorm2(mlp)
        } else {
            let attnOut = attn((norm1 as! RMSNorm)(x), cos: cos, sin: sin)
            x = x + norm2(attnOut)
            let mlp = feedForward(ffnNorm1(x))
            x = x + ffnNorm2(mlp)
        }
        return x
    }
}

final class DoubleStreamBlock: Module {
    @ModuleInfo(key: "img_instruct_attn") var imgInstructAttn: DoubleStreamJointAttention
    @ModuleInfo(key: "img_self_attn") var imgSelfAttn: Attention
    @ModuleInfo(key: "img_feed_forward") var imgFeedForward: LuminaFeedForward
    @ModuleInfo(key: "img_norm1") var imgNorm1: LuminaRMSNormZero
    @ModuleInfo(key: "img_norm2") var imgNorm2: LuminaRMSNormZero
    @ModuleInfo(key: "img_norm3") var imgNorm3: LuminaRMSNormZero
    @ModuleInfo(key: "img_ffn_norm1") var imgFfnNorm1: RMSNorm
    @ModuleInfo(key: "img_attn_norm") var imgAttnNorm: RMSNorm
    @ModuleInfo(key: "img_self_attn_norm") var imgSelfAttnNorm: RMSNorm
    @ModuleInfo(key: "img_ffn_norm2") var imgFfnNorm2: RMSNorm
    @ModuleInfo(key: "instruct_feed_forward") var instructFeedForward: LuminaFeedForward
    @ModuleInfo(key: "instruct_norm1") var instructNorm1: LuminaRMSNormZero
    @ModuleInfo(key: "instruct_norm2") var instructNorm2: LuminaRMSNormZero
    @ModuleInfo(key: "instruct_ffn_norm1") var instructFfnNorm1: RMSNorm
    @ModuleInfo(key: "instruct_attn_norm") var instructAttnNorm: RMSNorm
    @ModuleInfo(key: "instruct_ffn_norm2") var instructFfnNorm2: RMSNorm

    init(dim: Int, heads: Int, kvHeads: Int, multipleOf: Int, normEps: Float) {
        self._imgInstructAttn.wrappedValue = DoubleStreamJointAttention(dim: dim, heads: heads, kvHeads: kvHeads, eps: 1e-5)
        self._imgSelfAttn.wrappedValue = Attention(dim: dim, heads: heads, kvHeads: kvHeads, eps: 1e-5)
        self._imgFeedForward.wrappedValue = LuminaFeedForward(dim: dim, innerDim: 4 * dim, multipleOf: multipleOf)
        self._imgNorm1.wrappedValue = LuminaRMSNormZero(dim: dim, normEps: normEps)
        self._imgNorm2.wrappedValue = LuminaRMSNormZero(dim: dim, normEps: normEps)
        self._imgNorm3.wrappedValue = LuminaRMSNormZero(dim: dim, normEps: normEps)
        self._imgFfnNorm1.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
        self._imgAttnNorm.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
        self._imgSelfAttnNorm.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
        self._imgFfnNorm2.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
        self._instructFeedForward.wrappedValue = LuminaFeedForward(dim: dim, innerDim: 4 * dim, multipleOf: multipleOf)
        self._instructNorm1.wrappedValue = LuminaRMSNormZero(dim: dim, normEps: normEps)
        self._instructNorm2.wrappedValue = LuminaRMSNormZero(dim: dim, normEps: normEps)
        self._instructFfnNorm1.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
        self._instructAttnNorm.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
        self._instructFfnNorm2.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
        super.init()
    }

    func callAsFunction(img imgIn: MLXArray, instruct instructIn: MLXArray,
                        fullCos: MLXArray, fullSin: MLXArray,
                        imgCos: MLXArray, imgSin: MLXArray, temb: MLXArray)
        -> (MLXArray, MLXArray)
    {
        var img = imgIn
        var instruct = instructIn
        let (imgN1, imgGateMsa, imgScaleMlp, imgGateMlp) = imgNorm1(img, temb)
        let (imgN2, imgShiftMlp, _, _) = imgNorm2(img, temb)
        let (imgN3, imgGateSelf, _, _) = imgNorm3(img, temb)
        let (insN1, insGateMsa, insScaleMlp, insGateMlp) = instructNorm1(instruct, temb)
        let (insN2, insShiftMlp, _, _) = instructNorm2(instruct, temb)

        let (joint, Li) = imgInstructAttn(img: imgN1, instruct: insN1, cos: fullCos, sin: fullSin)
        let insAttn = joint[0..., ..<Li]
        let imgAttn = joint[0..., Li...]
        let imgSelf = imgSelfAttn(imgN3, cos: imgCos, sin: imgSin)

        img = img + tanh(imgGateMsa.expandedDimensions(axis: 1)) * imgAttnNorm(imgAttn)
        img = img + tanh(imgGateSelf.expandedDimensions(axis: 1)) * imgSelfAttnNorm(imgSelf)
        let imgMlpIn = (1 + imgScaleMlp.expandedDimensions(axis: 1)) * imgN2
            + imgShiftMlp.expandedDimensions(axis: 1)
        let imgMlp = imgFeedForward(imgFfnNorm1(imgMlpIn))
        img = img + tanh(imgGateMlp.expandedDimensions(axis: 1)) * imgFfnNorm2(imgMlp)

        instruct = instruct + tanh(insGateMsa.expandedDimensions(axis: 1)) * instructAttnNorm(insAttn)
        let insMlpIn = (1 + insScaleMlp.expandedDimensions(axis: 1)) * insN2
            + insShiftMlp.expandedDimensions(axis: 1)
        let insMlp = instructFeedForward(instructFfnNorm1(insMlpIn))
        instruct = instruct + tanh(insGateMlp.expandedDimensions(axis: 1)) * instructFfnNorm2(insMlp)
        return (img, instruct)
    }
}

// MARK: - Model

public final class BooguImageTransformer2DModel: Module {
    @ModuleInfo(key: "x_embedder") var xEmbedder: Linear
    @ModuleInfo(key: "ref_image_patch_embedder") var refImagePatchEmbedder: Linear
    @ModuleInfo(key: "time_caption_embed") var timeCaptionEmbed: Lumina2CombinedTimestepCaptionEmbedding
    @ModuleInfo(key: "noise_refiner") var noiseRefiner: [BasicBlock]
    @ModuleInfo(key: "ref_image_refiner") var refImageRefiner: [BasicBlock]
    @ModuleInfo(key: "context_refiner") var contextRefiner: [BasicBlock]
    @ModuleInfo(key: "double_stream_layers") var doubleStreamLayers: [DoubleStreamBlock]
    @ModuleInfo(key: "single_stream_layers") var singleStreamLayers: [BasicBlock]
    @ModuleInfo(key: "norm_out") var normOut: LuminaLayerNormContinuous
    @ParameterInfo(key: "image_index_embedding") var imageIndexEmbedding: MLXArray

    let patchSize: Int, inChannels: Int, outChannels: Int
    let axesDimRope: [Int], theta: Int

    /// The DiT's weight dtype (conditioning + latents are cast to this).
    public var dtype: DType { xEmbedder.weight.dtype }

    public init(_ cfg: BooguTransformerConfig) {
        self.patchSize = cfg.patchSize
        self.inChannels = cfg.inChannels
        self.outChannels = cfg.outChannels ?? cfg.inChannels
        self.axesDimRope = cfg.axesDimRope
        self.theta = cfg.theta
        let dim = cfg.hiddenSize
        let h = cfg.numAttentionHeads, kv = cfg.numKVHeads
        let mo = cfg.multipleOf, eps = cfg.normEps
        let patchDim = cfg.patchSize * cfg.patchSize * cfg.inChannels

        self._xEmbedder.wrappedValue = Linear(patchDim, dim, bias: true)
        self._refImagePatchEmbedder.wrappedValue = Linear(patchDim, dim, bias: true)
        self._timeCaptionEmbed.wrappedValue = Lumina2CombinedTimestepCaptionEmbedding(
            hiddenSize: dim, instructionFeatDim: cfg.instructionFeatDim,
            normEps: eps, timestepScale: cfg.timestepScale)
        self._noiseRefiner.wrappedValue = (0..<cfg.numRefinerLayers).map { _ in
            BasicBlock(dim: dim, heads: h, kvHeads: kv, multipleOf: mo, normEps: eps, modulation: true) }
        self._refImageRefiner.wrappedValue = (0..<cfg.numRefinerLayers).map { _ in
            BasicBlock(dim: dim, heads: h, kvHeads: kv, multipleOf: mo, normEps: eps, modulation: true) }
        self._contextRefiner.wrappedValue = (0..<cfg.numRefinerLayers).map { _ in
            BasicBlock(dim: dim, heads: h, kvHeads: kv, multipleOf: mo, normEps: eps, modulation: false) }
        self._doubleStreamLayers.wrappedValue = (0..<cfg.numDoubleStreamLayers).map { _ in
            DoubleStreamBlock(dim: dim, heads: h, kvHeads: kv, multipleOf: mo, normEps: eps) }
        self._singleStreamLayers.wrappedValue = (0..<(cfg.numLayers - cfg.numDoubleStreamLayers)).map { _ in
            BasicBlock(dim: dim, heads: h, kvHeads: kv, multipleOf: mo, normEps: eps, modulation: true) }
        self._normOut.wrappedValue = LuminaLayerNormContinuous(
            dim: dim, condDim: min(dim, 1024),
            outDim: cfg.patchSize * cfg.patchSize * (cfg.outChannels ?? cfg.inChannels), eps: 1e-6)
        self._imageIndexEmbedding.wrappedValue = MLXArray.zeros([5, dim])
        super.init()
    }

    /// latent [B,C,H,W] -> tokens [B, h*w, p*p*C] in (p1 p2 c) order.
    private func patchify(_ latent: MLXArray) -> MLXArray {
        let (B, C, H, W) = (latent.dim(0), latent.dim(1), latent.dim(2), latent.dim(3))
        let p = patchSize
        let (ht, wt) = (H / p, W / p)
        return latent.reshaped(B, C, ht, p, wt, p)
            .transposed(0, 2, 4, 3, 5, 1)  // B h w p1 p2 c
            .reshaped(B, ht * wt, p * p * C)
    }

    private func unpatchify(_ tokens: MLXArray, H: Int, W: Int) -> MLXArray {
        let B = tokens.dim(0)
        let (p, C) = (patchSize, outChannels)
        let (ht, wt) = (H / p, W / p)
        return tokens.reshaped(B, ht, wt, p, p, C)
            .transposed(0, 5, 1, 3, 2, 4)  // B c h p1 w p2
            .reshaped(B, C, ht * p, wt * p)
    }

    /// [cap ; (ref) ; noise] 3-axis positions. ref = (rht, rwt) or nil.
    private func positionIds(LCap: Int, ht: Int, wt: Int, ref: (Int, Int)?) -> [[Int]] {
        var rows: [[Int]] = []
        for i in 0..<LCap { rows.append([i, i, i]) }
        var shift = LCap
        if let (rht, rwt) = ref {
            for r in 0..<rht { for c in 0..<rwt { rows.append([shift, r, c]) } }
            shift += max(rht, rwt)
        }
        for r in 0..<ht { for c in 0..<wt { rows.append([shift, r, c]) } }
        return rows
    }

    /// latent [1,16,H,W]; timestep [1]; instruction [1,L_cap,4096]; ref_latent optional.
    public func callAsFunction(latent: MLXArray, timestep: MLXArray,
                               instructionHiddenStates: MLXArray,
                               refLatent: MLXArray? = nil) -> MLXArray {
        let (H, W) = (latent.dim(2), latent.dim(3))
        let p = patchSize
        let (ht, wt) = (H / p, W / p)
        let LCap = instructionHiddenStates.dim(1)
        let LNoise = ht * wt

        let (temb, caption0) = timeCaptionEmbed(timestep, instructionHiddenStates)
        var x = xEmbedder(patchify(latent))  // noise tokens [1, L_noise, hid]

        var ref: (Int, Int)? = nil
        if let refLatent { ref = (refLatent.dim(2) / p, refLatent.dim(3) / p) }

        let pos = positionIds(LCap: LCap, ht: ht, wt: wt, ref: ref)
        let (cosNp, sinNp) = ropeCosSin(positionIds: pos, axesDim: axesDimRope, theta: theta)
        let fullCos = cosNp[.newAxis, 0..., .newAxis, 0...]
        let fullSin = sinNp[.newAxis, 0..., .newAxis, 0...]
        let capCos = fullCos[0..., ..<LCap], capSin = fullSin[0..., ..<LCap]
        let imgCos = fullCos[0..., LCap...], imgSin = fullSin[0..., LCap...]  // [ref ; noise]

        var caption = caption0
        for layer in contextRefiner { caption = layer(caption, cos: capCos, sin: capSin) }

        var img: MLXArray
        if let ref {
            let rlen = ref.0 * ref.1
            let refCos = imgCos[0..., ..<rlen], refSin = imgSin[0..., ..<rlen]
            let noiseCos = imgCos[0..., rlen...], noiseSin = imgSin[0..., rlen...]
            var refTok = refImagePatchEmbedder(patchify(refLatent!)) + imageIndexEmbedding[0]
            for layer in refImageRefiner { refTok = layer(refTok, cos: refCos, sin: refSin, temb: temb) }
            for layer in noiseRefiner { x = layer(x, cos: noiseCos, sin: noiseSin, temb: temb) }
            img = concatenated([refTok, x], axis: 1)  // [ref ; noise]
        } else {
            for layer in noiseRefiner { x = layer(x, cos: imgCos, sin: imgSin, temb: temb) }
            img = x
        }

        var instruct = caption
        for layer in doubleStreamLayers {
            (img, instruct) = layer(img: img, instruct: instruct, fullCos: fullCos, fullSin: fullSin,
                                    imgCos: imgCos, imgSin: imgSin, temb: temb)
        }

        var hidden = concatenated([instruct, img], axis: 1)  // fuse [instruct ; img]
        for layer in singleStreamLayers { hidden = layer(hidden, cos: fullCos, sin: fullSin, temb: temb) }

        hidden = normOut(hidden, temb)
        let imgTokens = hidden[0..., (hidden.dim(1) - LNoise)...]  // noise tokens are last
        return unpatchify(imgTokens, H: H, W: W)
    }
}
