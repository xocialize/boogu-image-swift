// swift-tools-version: 6.2
// boogu-image-swift — Swift/MLX mirror of Boogu-Image-0.1 (Apache-2.0):
// Qwen3-VL-8B-conditioned OmniGen2-lineage DiT (8 double-stream + 32 single-stream)
// + FLUX.1 AutoencoderKL (16-ch) + FlowMatchEuler static-v1 time-shift scheduler.
// Reference = the parity-locked Python port boogu_image_mlx (github.com/xocialize/
// boogu-image-mlx); weights = mlx-community/Boogu-Image-0.1-{Base,Turbo,Edit}-*.
// The Qwen3-VL conditioner is REUSED from mlx-swift-lm MLXVLM (Path A), not ported.

import PackageDescription

let package = Package(
    name: "BooguImage",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "BooguImage", targets: ["BooguImage"]),
        // MLXEngine wrapper: textToImage (Base/Turbo) + imageEdit (Edit) ModelPackage.
        .library(name: "MLXBoogu", targets: ["MLXBoogu"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.31.4"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "3.31.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
        // Reusable Qwen3-VL backbone exposing last_hidden_state (the conditioner).
        .package(url: "https://github.com/xocialize/qwen3vl-mlx-swift", from: "0.1.0"),
        // MLXEngine contract (MLXToolKit) for the wrapper target only.
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.3.0"),
    ],
    targets: [
        .target(
            name: "BooguImage",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                // Qwen3-VL conditioner reused from the runtime loader (Path A).
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Qwen3VL", package: "qwen3vl-mlx-swift"),
            ],
            path: "Sources/BooguImage"
        ),
        .target(
            name: "MLXBoogu",
            dependencies: [
                "BooguImage",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            ],
            path: "Sources/MLXBoogu"
        ),
        // CLI gate runner (structural + parity gates). The SPM test product's metallib
        // is unreliable, so every Metal-touching gate is a `swift run` mode here.
        .executableTarget(
            name: "BooguGate",
            dependencies: ["BooguImage"],
            path: "Sources/BooguGate"
        ),
        .testTarget(
            name: "BooguImageTests",
            dependencies: ["BooguImage"],
            path: "Tests/BooguImageTests"
        ),
        .testTarget(
            name: "MLXBooguTests",
            dependencies: ["MLXBoogu"],
            path: "Tests/MLXBooguTests"
        ),
    ]
)
