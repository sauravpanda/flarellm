import Foundation
import XCTest
@testable import FlareKit

private final class TextBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var text = ""
    func append(_ value: String) { lock.lock(); text += value; lock.unlock() }
    func read() -> String { lock.lock(); defer { lock.unlock() }; return text }
}

private final class TokenGate: @unchecked Sendable {
    let entered = DispatchSemaphore(value: 0)
    let release = DispatchSemaphore(value: 0)
    var first = true // Accessed only on the native serial queue.
    func waitUntilEntered() -> Bool { entered.wait(timeout: .now() + 10) == .success }
    func token(_ text: String) {
        if first { first = false; entered.signal(); release.wait() }
    }
}

final class RuntimeTests: XCTestCase {
    private var fixture: CachedModel {
        let root = Bundle.module.url(forResource: "Fixtures", withExtension: nil)!
        return CachedModel(modelURL: root.appendingPathComponent("tiny.gguf"), tokenizerURL: root.appendingPathComponent("tokenizer.json"))
    }

    func testLoadStreamReuseAndUnload() async throws {
        let model = try await FlareModel.load(fixture, context: 64)
        let buffer = TextBuffer()
        let first = try await model.generate(prompt: "ace", maxTokens: 16) { buffer.append($0) }
        let expectedURL = Bundle.module.url(forResource: "Fixtures", withExtension: nil)!.appendingPathComponent("expected.json")
        let expected = try JSONSerialization.jsonObject(with: Data(contentsOf: expectedURL)) as! [String: Any]
        XCTAssertEqual(first.tokenIDs, (expected["ids"] as! [NSNumber]).map { $0.uint32Value })
        XCTAssertEqual(first.generatedTokens, 16)
        XCTAssertEqual(first.promptTokens, 3)
        XCTAssertEqual(first.finishReason, "length")
        XCTAssertEqual(first.text, buffer.read())
        XCTAssertGreaterThanOrEqual(first.elapsedMs, first.firstTokenMs)
        let second = try await model.generate(prompt: "ace", maxTokens: 16)
        XCTAssertEqual(second.text, first.text)
        try await model.close()
        do { _ = try await model.generate(prompt: "ace"); XCTFail() }
        catch { XCTAssertEqual(error as? FlareError, .closed) }
    }

    func testCancellationResetsStateAndRejectsConcurrentCalls() async throws {
        let model = try await FlareModel.load(fixture, context: 64)
        let gate = TokenGate()
        let task = Task { try await model.generate(prompt: "ace", maxTokens: 16) { gate.token($0) } }
        let entered = await Task.detached { gate.waitUntilEntered() }.value
        XCTAssertTrue(entered)
        do { _ = try await model.generate(prompt: "ace", maxTokens: 1); XCTFail() }
        catch { XCTAssertEqual(error as? FlareError, .busy) }
        task.cancel()
        gate.release.signal()
        do { _ = try await task.value; XCTFail() } catch is CancellationError {} catch { XCTFail("\(error)") }
        let next = try await model.generate(prompt: "ace", maxTokens: 16)
        XCTAssertEqual(next.generatedTokens, 16)
        try await model.close()
    }

    func testInvalidContextAndOverflowAreErrors() async throws {
        do { _ = try await FlareModel.load(fixture, context: 10000); XCTFail() } catch {}
        let model = try await FlareModel.load(fixture, context: 32)
        do { _ = try await model.generate(prompt: "ace", maxTokens: 32); XCTFail() }
        catch { XCTAssertTrue(String(describing: error).contains("never truncated")) }
        let result = try await model.generate(prompt: "ace", maxTokens: 1)
        XCTAssertEqual(result.generatedTokens, 1)
        try await model.close()
    }
}
