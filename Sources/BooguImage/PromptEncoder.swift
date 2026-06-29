// Boogu-Image conditioning — Qwen3-VL last_hidden_state over the Boogu chat templates.
// Reuses the qwen3vl-mlx-swift backbone (text path parity-locked; vision/Edit WIP).
//
// T2I: system(SYSTEM_T2I) + user(prompt) -> lastHiddenState(inputIds).
// Edit: system(SYSTEM_TI2I) + user(image, prompt) -> lastHiddenState(ids, pixelValues, grid).

import Foundation
import MLX
import Qwen3VL
import Tokenizers

public final class BooguPromptEncoder {
    public let qwen: Qwen3VL
    public let tokenizer: any Tokenizer
    public let dtype: DType
    let processor = Qwen3VLImageProcessor()

    static let systemT2I =
        "You are a helpful assistant that generates high-quality images based on user "
        + "instructions. The instructions are as follows."
    static let systemTI2I =
        "Describe the key features of the input image (color, shape, size, texture, objects, "
        + "background), then explain how the user's text instruction should alter or modify the "
        + "image. Generate a new image that meets the user's requirements while maintaining "
        + "consistency with the original input where appropriate."

    public init(qwen: Qwen3VL, tokenizer: any Tokenizer, dtype: DType) {
        self.qwen = qwen
        self.tokenizer = tokenizer
        self.dtype = dtype
    }

    /// Load the Qwen3-VL conditioner + tokenizer from a stock Qwen3-VL-8B snapshot.
    public static func load(qwenDir: URL, dtype: DType = .bfloat16) async throws -> BooguPromptEncoder {
        var model: Qwen3VL!
        try Device.withDefaultDevice(.cpu) {
            model = try Qwen3VLLoader.load(directory: qwenDir, dtype: dtype)
        }
        let tokenizer = try await AutoTokenizer.from(modelFolder: qwenDir)
        return BooguPromptEncoder(qwen: model, tokenizer: tokenizer, dtype: dtype)
    }

    /// T2I conditioning [1, L, 4096] for a text prompt.
    public func encodeText(_ prompt: String) throws -> MLXArray {
        let text =
            "<|im_start|>system\n\(Self.systemT2I)<|im_end|>\n"
            + "<|im_start|>user\n\(prompt)<|im_end|>\n"
        let ids = tokenizer.encode(text: text, addSpecialTokens: false)
        let inputIds = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
        return try qwen.lastHiddenState(inputIds: inputIds)
    }

    /// TI2I (Edit) conditioning [1, L, 4096] for an image + instruction. Builds the Boogu
    /// edit chat template, expands the single <|image_pad|> to the merged vision-token
    /// count, and runs the vision-merged forward.
    public func encodeImage(rgb: [UInt8], width: Int, height: Int, instruction: String) throws
        -> MLXArray
    {
        let (pixelValues, thw) = processor.preprocess(rgb: rgb, width: width, height: height)
        let merge = processor.mergeSize
        let mergedTokens = thw.t * (thw.h / merge) * (thw.w / merge)

        let text =
            "<|im_start|>system\n\(Self.systemTI2I)<|im_end|>\n"
            + "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\(instruction)<|im_end|>\n"
        var ids = tokenizer.encode(text: text, addSpecialTokens: false)
        guard let padId = tokenizer.convertTokenToId("<|image_pad|>"),
              let idx = ids.firstIndex(of: padId)
        else { throw BooguError.invalidInput("tokenizer lacks <|image_pad|>") }
        ids.replaceSubrange(idx...idx, with: Array(repeating: padId, count: mergedTokens))

        let inputIds = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
        return try qwen.lastHiddenState(
            inputIds: inputIds, pixelValues: pixelValues.asType(dtype), imageGridTHW: [thw])
    }
}
