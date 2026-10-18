import Foundation
import XCTest
@testable import FlareKit

/// Explicit opt-in only. Normal CI uses the repository-owned tiny model.
final class RealModelTests: XCTestCase {
    func testPinnedDownloadCacheAndReferenceTokens() async throws {
        guard let directory = ProcessInfo.processInfo.environment["FLARE_REAL_CACHE"] else {
            throw XCTSkip("Set FLARE_REAL_CACHE to opt into a 651 MB public model/tokenizer download")
        }
        let manifest = try FlareModels.qwen3_06B_Q8()
        let store = try ModelStore(directory: URL(fileURLWithPath: directory), maximumBytes: 800_000_000)
        let start = ProcessInfo.processInfo.systemUptime
        let cached = try await store.prepare(model: manifest.model, tokenizer: manifest.tokenizer)
        let prepareMs = (ProcessInfo.processInfo.systemUptime - start) * 1000
        let hitStart = ProcessInfo.processInfo.systemUptime
        let hit = try await store.prepare(model: manifest.model, tokenizer: manifest.tokenizer)
        let hitMs = (ProcessInfo.processInfo.systemUptime - hitStart) * 1000
        XCTAssertEqual(cached.modelURL, hit.modelURL)
        let model = try await FlareModel.load(cached, context: 128)
        let result = try await model.chat(messages: [.init(role: .user, content: "What is the capital of France?")], maxTokens: 32)
        let expectedURL = Bundle.module.url(forResource: "Fixtures", withExtension: nil)!.appendingPathComponent("real-expected.json")
        let expected = try JSONSerialization.jsonObject(with: Data(contentsOf: expectedURL)) as! [String: [UInt32]]
        XCTAssertEqual(result.tokenIDs, expected["tokens"])
        XCTAssertEqual(result.finishReason, "stop")
        let record: [String: Any] = ["prepareMs": prepareMs, "verifiedCacheHitMs": hitMs,
                                    "loadMs": model.loadMs, "firstTokenMs": result.firstTokenMs,
                                    "elapsedMs": result.elapsedMs, "generatedTokens": result.generatedTokens,
                                    "tokenIDs": result.tokenIDs, "text": result.text,
                                    "os": ProcessInfo.processInfo.operatingSystemVersionString,
                                    "modelSHA256": manifest.model.sha256, "tokenizerSHA256": manifest.tokenizer.sha256,
                                    "context": 128, "backend": "native CPU Q8 weights/f32 activations",
                                    "scope": "One host smoke test; not iPhone performance or a controlled speed comparison"]
        let data = try JSONSerialization.data(withJSONObject: record, options: [.prettyPrinted, .sortedKeys])
        print("FLARE_REAL_RESULT " + String(decoding: data, as: UTF8.self))
        if let output = ProcessInfo.processInfo.environment["FLARE_REAL_RESULT"] {
            try data.write(to: URL(fileURLWithPath: output))
        }
        try await model.close()
    }
}
