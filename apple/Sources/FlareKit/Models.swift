import Foundation

/// An opt-in, pinned manifest. Creating a manifest performs no download.
/// Qwen assets are Apache-2.0; applications must retain the upstream notices.
public enum FlareModels {
    public static func qwen3_06B_Q8() throws -> (model: ModelAsset, tokenizer: ModelAsset) {
        let model = try ModelAsset(
            url: URL(string: "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf")!,
            sha256: "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031", bytes: 639446688)
        let tokenizer = try ModelAsset(
            url: URL(string: "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json")!,
            sha256: "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4", bytes: 11422654)
        return (model, tokenizer)
    }
}
