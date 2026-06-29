// Config structs for the three Boogu-Image components. Each decodes the *resolved*
// component `config.json` written by the Python port (extra/underscore keys ignored).

import Foundation

public struct BooguTransformerConfig: Codable, Sendable {
    public var patchSize: Int = 2
    public var inChannels: Int = 16
    public var outChannels: Int?
    public var hiddenSize: Int = 3360
    public var numLayers: Int = 40
    public var numDoubleStreamLayers: Int = 8
    public var numRefinerLayers: Int = 2
    public var numAttentionHeads: Int = 28
    public var numKVHeads: Int = 7
    public var multipleOf: Int = 256
    public var normEps: Float = 1e-5
    public var axesDimRope: [Int] = [40, 40, 40]
    public var axesLens: [Int] = [2048, 1664, 1664]
    public var instructionFeatDim: Int = 4096
    public var timestepScale: Float = 1000.0
    public var theta: Int = 10000

    struct InstructionFeatureConfigs: Codable { var instruction_feat_dim: Int }

    enum CodingKeys: String, CodingKey {
        case patch_size, in_channels, out_channels, hidden_size, num_layers
        case num_double_stream_layers, num_refiner_layers, num_attention_heads
        case num_kv_heads, multiple_of, norm_eps, axes_dim_rope, axes_lens
        case instruction_feature_configs, timestep_scale
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func g<T: Decodable>(_ k: CodingKeys, _ d: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: k) ?? d
        }
        patchSize = try g(.patch_size, 2)
        inChannels = try g(.in_channels, 16)
        outChannels = try c.decodeIfPresent(Int.self, forKey: .out_channels)
        hiddenSize = try g(.hidden_size, 3360)
        numLayers = try g(.num_layers, 40)
        numDoubleStreamLayers = try g(.num_double_stream_layers, 8)
        numRefinerLayers = try g(.num_refiner_layers, 2)
        numAttentionHeads = try g(.num_attention_heads, 28)
        numKVHeads = try g(.num_kv_heads, 7)
        multipleOf = try g(.multiple_of, 256)
        normEps = try g(.norm_eps, 1e-5)
        axesDimRope = try g(.axes_dim_rope, [40, 40, 40])
        axesLens = try g(.axes_lens, [2048, 1664, 1664])
        timestepScale = try g(.timestep_scale, 1000.0)
        let ifc = try c.decode(InstructionFeatureConfigs.self, forKey: .instruction_feature_configs)
        instructionFeatDim = ifc.instruction_feat_dim
    }

    public func encode(to encoder: Encoder) throws {}  // never serialized

    public static func load(_ url: URL) throws -> BooguTransformerConfig {
        try JSONDecoder().decode(BooguTransformerConfig.self, from: Data(contentsOf: url))
    }
}

public struct BooguVAEConfig: Codable, Sendable {
    public var inChannels: Int = 3
    public var outChannels: Int = 3
    public var latentChannels: Int = 16
    public var blockOutChannels: [Int] = [128, 256, 512, 512]
    public var layersPerBlock: Int = 2
    public var normNumGroups: Int = 32
    public var scalingFactor: Float = 0.3611
    public var shiftFactor: Float = 0.1159

    enum CodingKeys: String, CodingKey {
        case in_channels, out_channels, latent_channels, block_out_channels
        case layers_per_block, norm_num_groups, scaling_factor, shift_factor
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func g<T: Decodable>(_ k: CodingKeys, _ d: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: k) ?? d
        }
        inChannels = try g(.in_channels, 3)
        outChannels = try g(.out_channels, 3)
        latentChannels = try g(.latent_channels, 16)
        blockOutChannels = try g(.block_out_channels, [128, 256, 512, 512])
        layersPerBlock = try g(.layers_per_block, 2)
        normNumGroups = try g(.norm_num_groups, 32)
        scalingFactor = try g(.scaling_factor, 0.3611)
        shiftFactor = try g(.shift_factor, 0.1159)
    }

    public func encode(to encoder: Encoder) throws {}

    public static func load(_ url: URL) throws -> BooguVAEConfig {
        try JSONDecoder().decode(BooguVAEConfig.self, from: Data(contentsOf: url))
    }
}

public struct BooguSchedulerConfig: Codable, Sendable {
    public var numTrainTimesteps: Int = 1000
    public var doShift: Bool = true
    public var dynamicTimeShift: Bool = false
    public var timeShiftVersion: String = "v1"
    public var seqLen: Int? = 4096
    public var baseShift: Float = 0.5
    public var maxShift: Float = 1.15
    public var timeShiftV2HalfScalingFactor: Float = 60.0

    enum CodingKeys: String, CodingKey {
        case num_train_timesteps, do_shift, dynamic_time_shift, time_shift_version
        case seq_len, base_shift, max_shift, time_shift_v2_half_scaling_factor
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func g<T: Decodable>(_ k: CodingKeys, _ d: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: k) ?? d
        }
        numTrainTimesteps = try g(.num_train_timesteps, 1000)
        doShift = try g(.do_shift, true)
        dynamicTimeShift = try g(.dynamic_time_shift, false)
        timeShiftVersion = try g(.time_shift_version, "v1")
        seqLen = try c.decodeIfPresent(Int.self, forKey: .seq_len) ?? 4096
        baseShift = try g(.base_shift, 0.5)
        maxShift = try g(.max_shift, 1.15)
        timeShiftV2HalfScalingFactor = try g(.time_shift_v2_half_scaling_factor, 60.0)
    }

    public func encode(to encoder: Encoder) throws {}

    public static func load(_ url: URL) throws -> BooguSchedulerConfig {
        try JSONDecoder().decode(BooguSchedulerConfig.self, from: Data(contentsOf: url))
    }
}
