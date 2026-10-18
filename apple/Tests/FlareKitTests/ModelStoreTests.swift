import CryptoKit
import Foundation
import XCTest
@testable import FlareKit

private actor Counter {
    var value = 0
    func increment() { value += 1 }
}

final class ModelStoreTests: XCTestCase {
    private func directory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: root) }
        return root
    }
    private func asset(_ data: Data, hash: String? = nil) throws -> ModelAsset {
        try ModelAsset(url: URL(string: "https://example.com/pinned/model")!,
                       sha256: hash ?? SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(),
                       bytes: Int64(data.count))
    }
    private func transport(_ data: Data, count: Counter, status: Int = 200) -> ModelStore.Download {
        { url, _ in
            await count.increment()
            let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try data.write(to: temporary)
            return (temporary, HTTPURLResponse(url: url, statusCode: status, httpVersion: nil, headerFields: nil)!)
        }
    }

    func testVerifiedCacheHitWorksOfflineAndCorruptionIsReplaced() async throws {
        let bytes = Data("verified model bytes".utf8), count = Counter()
        let root = try directory(), item = try asset(bytes)
        let store = try ModelStore(directory: root, maximumBytes: 100, download: transport(bytes, count: count))
        let first = try await store.ensure(item)
        let second = try await store.ensure(item)
        XCTAssertEqual(first, second)
        let calls = await count.value
        XCTAssertEqual(calls, 1)
        try Data(repeating: 0, count: bytes.count).write(to: first)
        let corrupt = try await store.cachedURL(for: item)
        XCTAssertNil(corrupt)
        let restored = try await store.ensure(item)
        XCTAssertEqual(try Data(contentsOf: restored), bytes)
        let callsAfterRepair = await count.value
        XCTAssertEqual(callsAfterRepair, 2)
        let usage = try await store.cachedBytes()
        XCTAssertEqual(usage, Int64(bytes.count))
        try await store.remove(item)
        let missing = try await store.cachedURL(for: item)
        XCTAssertNil(missing)
    }

    func testWrongHashLengthAndHTTPNeverPublish() async throws {
        let bytes = Data("abcd".utf8)
        for failure in ["hash", "length", "http"] {
            let root = try directory(), count = Counter()
            let item = try asset(bytes, hash: failure == "hash" ? String(repeating: "0", count: 64) : nil)
            let store = try ModelStore(directory: root, maximumBytes: 100,
                                      download: transport(failure == "length" ? Data("abc".utf8) : bytes,
                                                          count: count, status: failure == "http" ? 404 : 200))
            do { _ = try await store.ensure(item); XCTFail("Published \(failure)") } catch {}
            let size = try await store.cachedBytes()
            XCTAssertEqual(size, 0)
            XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.path), [])
        }
    }

    func testQuotaAndHTTPSFailBeforeDownload() async throws {
        let count = Counter(), bytes = Data("abcd".utf8)
        let store = try ModelStore(directory: directory(), maximumBytes: 3, download: transport(bytes, count: count))
        do { _ = try await store.ensure(asset(bytes)); XCTFail() }
        catch { XCTAssertEqual(error as? FlareError, .storageLimit) }
        let good = try asset(bytes)
        let bad = try ModelAsset(url: URL(string: "http://example.com/model")!, sha256: good.sha256, bytes: good.bytes)
        do { _ = try await store.ensure(bad); XCTFail() }
        catch { XCTAssertEqual(error as? FlareError, .invalidAsset("Model download requires HTTPS")) }
        let calls = await count.value
        XCTAssertEqual(calls, 0)
    }

    func testInFlightAssetsReserveQuotaAndCannotBeRemoved() async throws {
        let root = try directory(), first = try asset(Data("abcd".utf8)), second = try asset(Data("efgh".utf8))
        let entered = expectation(description: "reserved download")
        let store = try ModelStore(directory: root, maximumBytes: 6) { _, _ in
            entered.fulfill()
            try await Task.sleep(nanoseconds: 60_000_000_000)
            throw FlareError.native("Should have been cancelled")
        }
        let task = Task { try await store.ensure(first) }
        await fulfillment(of: [entered], timeout: 5)
        do { _ = try await store.ensure(first); XCTFail() }
        catch { XCTAssertEqual(error as? FlareError, .busy) }
        do { try await store.remove(first); XCTFail() }
        catch { XCTAssertEqual(error as? FlareError, .busy) }
        do { _ = try await store.ensure(second); XCTFail() }
        catch { XCTAssertEqual(error as? FlareError, .storageLimit) }
        task.cancel()
        _ = try? await task.value
        try await store.remove(first) // Reservation was released after cancellation.
    }

    func testCancellationLeavesNoCacheEntryAndCanRetry() async throws {
        let root = try directory(), item = try asset(Data("abcd".utf8))
        let entered = expectation(description: "download started")
        let store = try ModelStore(directory: root, maximumBytes: 100) { _, _ in
            entered.fulfill()
            try await Task.sleep(nanoseconds: 60_000_000_000)
            throw FlareError.native("Should have been cancelled")
        }
        let task = Task { try await store.ensure(item) }
        await fulfillment(of: [entered], timeout: 5)
        task.cancel()
        do { _ = try await task.value; XCTFail() } catch is CancellationError {} catch { XCTFail("\(error)") }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.path), [])
        let repaired = try ModelStore(directory: root, maximumBytes: 100, download: transport(Data("abcd".utf8), count: Counter()))
        _ = try await repaired.ensure(item)
    }
}
