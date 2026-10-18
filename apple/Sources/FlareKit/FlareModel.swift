import CFlare
import Foundation

public struct ChatMessage: Encodable, Sendable {
    public enum Role: String, Encodable, Sendable { case system, user, assistant }
    public let role: Role
    public let content: String
    public init(role: Role, content: String) { self.role = role; self.content = content }
}

public struct GenerationResult: Sendable {
    public let text: String
    public let finishReason: String
    public let generatedTokens: Int
    public let tokenIDs: [UInt32]
    public let promptTokens: Int
    public let firstTokenMs: Double
    public let elapsedMs: Double
}

/// One loaded model with a dedicated serial queue. Calls never run inference on
/// the UI thread. Reuse the instance across requests; close releases model/KV RAM.
public actor FlareModel {
    public typealias TokenHandler = @Sendable (String) -> Void
    private let worker: NativeWorker
    private var generating = false
    private var closed = false
    public nonisolated let loadMs: Double

    private init(worker: NativeWorker, loadMs: Double) { self.worker = worker; self.loadMs = loadMs }

    public static func load(_ cached: CachedModel, context: Int = 512) async throws -> FlareModel {
        guard (32...512).contains(context) else { throw FlareError.native("Context must be 32...512") }
        try Task.checkCancellation()
        let worker = NativeWorker()
        let milliseconds = try await worker.perform {
            try Task.checkCancellation()
            return try worker.open(cached, context: context)
        }
        if Task.isCancelled { try? await worker.perform { worker.close() }; throw CancellationError() }
        return FlareModel(worker: worker, loadMs: milliseconds)
    }

    /// Greedy Qwen3 text generation. The callback runs synchronously on the model
    /// queue and applies backpressure; it must be brief and must not reenter this
    /// model. Dispatch UI work to MainActor. No unbounded output stream is buffered.
    public func generate(prompt: String, maxTokens: Int = 64, onToken: TokenHandler? = nil) async throws -> GenerationResult {
        try await run(Request(prompt: prompt, messages: nil, maxTokens: maxTokens), onToken)
    }

    /// Applies the existing non-thinking Qwen3 template; history belongs to the
    /// application and must fit the context together with reserved output tokens.
    public func chat(messages: [ChatMessage], maxTokens: Int = 64, onToken: TokenHandler? = nil) async throws -> GenerationResult {
        try await run(Request(prompt: nil, messages: messages, maxTokens: maxTokens), onToken)
    }

    private func run(_ request: Request, _ callback: TokenHandler?) async throws -> GenerationResult {
        guard !closed else { throw FlareError.closed }
        guard !generating else { throw FlareError.busy }
        guard request.maxTokens > 0 else { throw FlareError.native("maxTokens must be positive") }
        generating = true
        defer { generating = false }
        let cancellation = CancellationFlag()
        let data = try JSONEncoder().encode(request)
        let json = String(decoding: data, as: UTF8.self)
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await worker.perform { [worker] in
                defer { worker.reset() }
                try cancellation.check()
                try worker.begin(json)
                while true {
                    try cancellation.check()
                    if try worker.prefill() { break }
                }
                var text = ""
                var tokenIDs: [UInt32] = []
                while true {
                    try cancellation.check()
                    let step = try worker.next()
                    try cancellation.check()
                    text += step.text
                    if let token = step.tokenId { tokenIDs.append(token) }
                    if !step.text.isEmpty { callback?(step.text) }
                    try cancellation.check()
                    if let reason = step.finishReason {
                        return GenerationResult(text: text, finishReason: reason,
                                                generatedTokens: step.generatedTokens, tokenIDs: tokenIDs, promptTokens: step.promptTokens,
                                                firstTokenMs: step.firstTokenMs, elapsedMs: step.elapsedMs)
                    }
                }
            }
        } onCancel: { cancellation.cancel() }
    }

    public func close() async throws {
        guard !generating else { throw FlareError.busy }
        if !closed { closed = true; try await worker.perform { [worker] in worker.close() } }
    }
}

private struct Request: Encodable, Sendable {
    let prompt: String?
    let messages: [ChatMessage]?
    let maxTokens: Int
}
private struct Empty: Decodable {}
private struct Loaded: Decodable { let loadMs: Double }
private struct Prefill: Decodable { let ready: Bool }
private struct Step: Decodable {
    let tokenId: UInt32?
    let text: String
    let finishReason: String?
    let generatedTokens: Int
    let promptTokens: Int
    let firstTokenMs: Double
    let elapsedMs: Double
}
private struct Envelope<T: Decodable>: Decodable { let value: T?; let error: String? }

private final class CancellationFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    func cancel() { lock.lock(); cancelled = true; lock.unlock() }
    func check() throws {
        lock.lock(); let value = cancelled; lock.unlock()
        if value { throw CancellationError() }
    }
}

// The queue exclusively owns handle access. Each perform closure retains this
// object until it completes, so deinit cannot free an engine still in use.
private final class NativeWorker: @unchecked Sendable {
    private let queue = DispatchQueue(label: "Flare.native", qos: .userInitiated)
    private var handle: OpaquePointer?
    deinit { if let handle { flare_native_close(handle) } }

    func perform<T: Sendable>(_ work: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            queue.async { continuation.resume(with: Result(catching: work)) }
        }
    }

    func open(_ cached: CachedModel, context: Int) throws -> Double {
        var status: UnsafeMutablePointer<CChar>?
        handle = cached.modelURL.path.withCString { model in
            cached.tokenizerURL.path.withCString { tokenizer in
                flare_native_open(model, tokenizer, UInt32(context), &status)
            }
        }
        let loaded: Loaded = try decode(status)
        guard handle != nil else { throw FlareError.native("Missing model handle") }
        return loaded.loadMs
    }
    func close() { if let handle { flare_native_close(handle); self.handle = nil } }
    func begin(_ request: String) throws {
        guard let handle else { throw FlareError.closed }
        let _: Empty = try request.withCString { try decode(flare_native_begin(handle, $0)) }
    }
    func prefill() throws -> Bool {
        guard let handle else { throw FlareError.closed }
        let result: Prefill = try decode(flare_native_prefill(handle))
        return result.ready
    }
    func next() throws -> Step {
        guard let handle else { throw FlareError.closed }
        return try decode(flare_native_next(handle))
    }
    func reset() { if let handle { flare_native_string_free(flare_native_reset(handle)) } }

    private func decode<T: Decodable>(_ pointer: UnsafeMutablePointer<CChar>?) throws -> T {
        guard let pointer else { throw FlareError.native("Missing native response") }
        defer { flare_native_string_free(pointer) }
        let data = Data(String(cString: pointer).utf8)
        let result = try JSONDecoder().decode(Envelope<T>.self, from: data)
        if let error = result.error { throw FlareError.native(error) }
        guard let value = result.value else { throw FlareError.native("Empty native response") }
        return value
    }
}
