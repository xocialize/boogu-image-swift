// Weight loading for Boogu-Image-0.1 — Swift mirror of boogu_image_mlx/utils/weights.py.
//
// DiT (`transformer/`): pure Linear + RMSNorm + the bare `image_index_embedding`
// parameter — PT<->MLX layouts identical, NO transposes, keys map 1:1 (942 tensors).
// VAE (`vae/`): diffusers AutoencoderKL — only Conv2d `.weight` (4-D) needs the
// (O,I,H,W) -> (O,H,W,I) channels-last transpose; everything else passes through.

import Foundation
import MLX
import MLXNN

public enum BooguError: Error, CustomStringConvertible {
    case loading(String)
    case invalidInput(String)

    public var description: String {
        switch self {
        case .loading(let m): return "BooguImage loading error: \(m)"
        case .invalidInput(let m): return "BooguImage input error: \(m)"
        }
    }
}

public enum BooguWeights {

    /// Merge every `*.safetensors` shard under `directory` into one dict.
    static func loadAllArrays(directory: URL) throws -> [String: MLXArray] {
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }.sorted {
            $0.lastPathComponent < $1.lastPathComponent
        }
        guard !files.isEmpty else {
            throw BooguError.loading("no .safetensors under \(directory.path)")
        }
        var merged: [String: MLXArray] = [:]
        for f in files {
            merged.merge(try MLX.loadArrays(url: f)) { a, _ in a }
        }
        return merged
    }

    /// Load the DiT from the bf16 `transformer/` shards (no transposes). For the
    /// quantized variants use `loadDiTQuantized` (see Transformer.swift).
    public static func loadDiT(directory: URL, dtype: DType = .bfloat16) throws
        -> BooguImageTransformer2DModel
    {
        let cfg = try BooguTransformerConfig.load(
            directory.appendingPathComponent("config.json"))
        let model = BooguImageTransformer2DModel(cfg)
        var weights: [String: MLXArray] = [:]
        for (k, v) in try loadAllArrays(directory: directory) {
            weights[k] = v.asType(dtype)
        }
        try verifyAndLoad(model: model, weights: weights, label: "DiT")
        return model
    }

    /// quant_config.json shape (matches the published Boogu int4/int8 repos).
    struct QuantConfig: Codable {
        var group_size: Int
        var bits: Int
        var weights_file: String?
    }

    /// Quantization predicate — mirrors pipeline_mlx: attn/feed_forward Linears whose
    /// in-dim is divisible by group_size (everything else stays bf16).
    static func quantPredicate(_ groupSize: Int) -> (String, Module) -> Bool {
        { path, module in
            guard let linear = module as? Linear else { return false }
            guard path.contains("attn") || path.contains("feed_forward") else { return false }
            return linear.weight.dim(1) % groupSize == 0
        }
    }

    /// Load a quantized DiT. `configDir` holds config.json; `quantDir` holds
    /// quant_config.json + the int4/int8 safetensors (same dir for a published snapshot).
    /// Build the model, quantize the attn/ffn scope, then load the packed weights.
    public static func loadDiTQuantized(configDir: URL, quantDir: URL) throws
        -> BooguImageTransformer2DModel
    {
        let cfg = try BooguTransformerConfig.load(configDir.appendingPathComponent("config.json"))
        let model = BooguImageTransformer2DModel(cfg)

        let qcData = try Data(contentsOf: quantDir.appendingPathComponent("quant_config.json"))
        let qc = try JSONDecoder().decode(QuantConfig.self, from: qcData)
        let weightsFile = qc.weights_file ?? "transformer_int\(qc.bits).safetensors"

        quantize(model: model, groupSize: qc.group_size, bits: qc.bits,
                 filter: quantPredicate(qc.group_size))

        let weights = try MLX.loadArrays(url: quantDir.appendingPathComponent(weightsFile))
        try verifyAndLoad(model: model, weights: weights, label: "DiT(int\(qc.bits))")
        return model
    }

    /// Load the DiT from a transformer/ snapshot, auto-selecting quantized (if a
    /// quant_config.json is present) vs bf16/fp32.
    public static func loadDiTAuto(transformerDir: URL, fp32: Bool = false) throws
        -> BooguImageTransformer2DModel
    {
        let qc = transformerDir.appendingPathComponent("quant_config.json")
        if FileManager.default.fileExists(atPath: qc.path) {
            return try loadDiTQuantized(configDir: transformerDir, quantDir: transformerDir)
        }
        return try loadDiT(directory: transformerDir, dtype: fp32 ? .float32 : .bfloat16)
    }

    /// Load the FLUX VAE from the diffusers `vae/` snapshot (Conv2d weights -> NHWC).
    public static func loadVAE(directory: URL, dtype: DType = .float32) throws
        -> AutoencoderKL
    {
        let cfg = try BooguVAEConfig.load(directory.appendingPathComponent("config.json"))
        let vae = AutoencoderKL(cfg)
        var state: [String: MLXArray] = [:]
        for (k, rawValue) in try loadAllArrays(directory: directory) {
            var v = rawValue
            if k.hasSuffix(".weight"), v.ndim == 4 {  // PT (O,I,kH,kW) -> MLX (O,kH,kW,I)
                v = v.transposed(0, 2, 3, 1)
            }
            state[k] = v.asType(dtype)
        }
        try verifyAndLoad(model: vae, weights: state, label: "VAE")
        return vae
    }

    /// Two-way strict load: every module key filled AND every checkpoint key consumed.
    /// A partial load emits garbage with no other symptom (the silent-failure class).
    static func verifyAndLoad(
        model: Module, weights: [String: MLXArray], label: String,
        toleratedExtras: Set<String> = []
    ) throws {
        let moduleKeys = Set(model.parameters().flattened().map(\.0))
        let fileKeys = Set(weights.keys).subtracting(toleratedExtras)
        let missing = moduleKeys.subtracting(fileKeys).sorted()
        guard missing.isEmpty else {
            throw BooguError.loading(
                "\(label): checkpoint missing \(missing.count) module keys, e.g. "
                    + missing.prefix(6).joined(separator: ", "))
        }
        let unused = fileKeys.subtracting(moduleKeys).sorted()
        guard unused.isEmpty else {
            throw BooguError.loading(
                "\(label): \(unused.count) unconsumed checkpoint keys, e.g. "
                    + unused.prefix(6).joined(separator: ", "))
        }
        var filtered = weights
        for k in toleratedExtras { filtered.removeValue(forKey: k) }
        model.update(parameters: ModuleParameters.unflattened(filtered))
        eval(model)
    }
}
