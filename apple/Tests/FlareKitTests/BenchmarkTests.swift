import Darwin
import Foundation
import XCTest
@testable import FlareKit

/// Opt-in, identical workloads for scalar/optimized builds and physical devices.
final class BenchmarkTests: XCTestCase {
    func testQwenRuntimeBenchmark() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["FLARE_BENCHMARK"] == "1" else {
            throw XCTSkip("Set FLARE_BENCHMARK=1 to run the benchmark (may download 651 MB)")
        }
        let repeats = Int(env["FLARE_BENCH_REPEATS"] ?? "3") ?? 3
        guard (1...20).contains(repeats) else { throw FlareError.invalidAsset("benchmark repeats must be 1...20") }
        let manifest = try FlareModels.qwen3_06B_Q8()
        let store: ModelStore
        if let directory = env["FLARE_BENCH_CACHE"] {
            store = try ModelStore(directory: URL(fileURLWithPath: directory))
        } else {
            store = try ModelStore()
        }
        let cached = try await store.prepare(model: manifest.model, tokenizer: manifest.tokenizer)
        let model = try await FlareModel.load(cached, context: 512)
        let expectedURL = Bundle.module.url(forResource: "Fixtures", withExtension: nil)!.appendingPathComponent("real-expected.json")
        let expected = try JSONSerialization.jsonObject(with: Data(contentsOf: expectedURL)) as! [String: [UInt32]]
        let question = "What is the capital of France?"
        let prompts = [question,
                       String(repeating: "The garden has green trees and red flowers. ", count: 5) + question,
                       String(repeating: "The garden has green trees and red flowers. ", count: 12) + question]
        var records: [[String: Any]] = []
        for (index, prompt) in prompts.enumerated() {
            let warmup = try await model.chat(messages: [.init(role: .user, content: prompt)], maxTokens: 16)
            if index == 0 {
                XCTAssertEqual(warmup.tokenIDs, expected["tokens"])
                XCTAssertEqual(warmup.finishReason, "stop")
            }
            for iteration in 0..<repeats {
                let thermalBefore = ProcessInfo.processInfo.thermalState.rawValue
                let result = try await model.chat(messages: [.init(role: .user, content: prompt)], maxTokens: 16)
                XCTAssertEqual(result.tokenIDs, warmup.tokenIDs)
                XCTAssertEqual(result.finishReason, warmup.finishReason)
                // EOS also requires a decode forward pass, but is excluded from generatedTokens.
                let decodeSteps = result.generatedTokens - 1 + (result.finishReason == "stop" ? 1 : 0)
                let decodeMs = result.elapsedMs - result.firstTokenMs
                records.append(["case": index, "iteration": iteration, "prompt": prompt,
                                "promptTokens": result.promptTokens, "tokenIDs": result.tokenIDs,
                                "finishReason": result.finishReason, "firstTokenMs": result.firstTokenMs,
                                "elapsedMs": result.elapsedMs, "decodeSteps": decodeSteps,
                                "decodeStepsPerSecond": decodeMs > 0 ? Double(decodeSteps) * 1000 / decodeMs : 0,
                                "processPeakResidentBytes": try peakResidentBytes(),
                                "thermalBefore": thermalBefore,
                                "thermalAfter": ProcessInfo.processInfo.thermalState.rawValue])
            }
        }
        try await model.close()
        #if targetEnvironment(simulator)
        let platform = "iOS simulator"
        #elseif os(iOS)
        let platform = "physical iOS device"
        #else
        let platform = "macOS"
        #endif
        let record: [String: Any] = ["runs": records, "loadMs": model.loadMs, "context": 512,
                                    "maxTokens": 16, "warmupsPerCase": 1, "repeats": repeats,
                                    "platform": platform, "hardware": hardwareIdentifier(),
                                    "os": ProcessInfo.processInfo.operatingSystemVersionString,
                                    "sourceRevision": env["FLARE_BENCH_REVISION"] ?? "unspecified",
                                    "buildLabel": env["FLARE_BENCH_LABEL"] ?? "unspecified",
                                    "modelSHA256": manifest.model.sha256, "tokenizerSHA256": manifest.tokenizer.sha256,
                                    "memoryScope": "Process lifetime high-water RSS including XCTest, download/cache and load; Darwin ru_maxrss bytes",
                                    "timingScope": "After tokenization; first model decision, including EOS; excludes cache, load, queue wait. Decode steps include terminal EOS."]
        let data = try JSONSerialization.data(withJSONObject: record, options: [.prettyPrinted, .sortedKeys])
        print("FLARE_BENCH_RESULT " + String(decoding: data, as: UTF8.self))
        if let output = env["FLARE_BENCH_RESULT"] {
            try data.write(to: URL(fileURLWithPath: output))
        }
    }

    private func peakResidentBytes() throws -> Int {
        var usage = rusage()
        guard getrusage(RUSAGE_SELF, &usage) == 0 else { throw FlareError.native("getrusage failed") }
        return usage.ru_maxrss
    }

    private func hardwareIdentifier() -> String {
        var system = utsname()
        uname(&system)
        return withUnsafeBytes(of: &system.machine) { bytes in
            String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
        }
    }
}
