import CryptoKit
import Foundation

public enum FlareError: Error, Equatable, Sendable {
    case invalidAsset(String)
    case integrity(String)
    case httpStatus(Int)
    case storageLimit
    case busy
    case closed
    case native(String)
}

/// Pin both the exact content hash and byte length, not a mutable model name.
public struct ModelAsset: Sendable {
    public let url: URL
    public let sha256: String
    public let bytes: Int64

    public init(url: URL, sha256: String, bytes: Int64) throws {
        guard sha256.count == 64, sha256.allSatisfy({ "0123456789abcdef".contains($0) }), bytes > 0 else {
            throw FlareError.invalidAsset("Expected a lowercase SHA-256 and positive byte length")
        }
        self.url = url
        self.sha256 = sha256
        self.bytes = bytes
    }
}

public struct CachedModel: Sendable {
    public let modelURL: URL
    public let tokenizerURL: URL
}

/// One store per root directory. Immutable, verified files persist offline until
/// explicitly removed. Use an app-private directory, not a shared app-group root.
public actor ModelStore {
    public typealias Progress = @Sendable (Int64, Int64) -> Void
    typealias Download = @Sendable (URL, Progress?) async throws -> (URL, URLResponse)
    private let root: URL
    private let maximumBytes: Int64
    private let download: Download
    private let allowLocalHTTP: Bool
    private var reservations: [String: Int64] = [:]

    public init(directory: URL? = nil, maximumBytes: Int64 = 2_000_000_000) throws {
        try self.init(directory: directory ?? Self.defaultDirectory(), maximumBytes: maximumBytes,
                      allowLocalHTTP: false, download: { url, progress in try await Self.fetch(url, progress) })
    }

    // Injectable transport keeps integrity/cancellation tests offline. Production
    // clients always use URLSession's disk download and require HTTPS.
    init(directory: URL, maximumBytes: Int64, allowLocalHTTP: Bool = false,
         download: @escaping Download) throws {
        guard maximumBytes > 0, directory.isFileURL else { throw FlareError.invalidAsset("Invalid cache configuration") }
        self.root = directory
        self.maximumBytes = maximumBytes
        self.allowLocalHTTP = allowLocalHTTP
        self.download = download
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var location = directory
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try location.setResourceValues(values)
    }

    private static func defaultDirectory() throws -> URL {
        try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                    appropriateFor: nil, create: true).appendingPathComponent("Flare/Models", isDirectory: true)
    }

    private static func fetch(_ url: URL, _ progress: Progress?) async throws -> (URL, URLResponse) {
        let delegate = DownloadProgress(progress)
        let request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData)
        return try await URLSession.shared.download(for: request, delegate: delegate)
    }

    /// A hit verifies disk contents and performs no network request. Corrupt files
    /// are removed; callers can then explicitly download them again.
    public func cachedURL(for asset: ModelAsset) throws -> URL? {
        try Task.checkCancellation()
        let path = destination(asset)
        guard FileManager.default.fileExists(atPath: path.path) else { return nil }
        do {
            try verify(path, asset)
            return path
        } catch is CancellationError { throw CancellationError() }
        catch {
            try FileManager.default.removeItem(at: path)
            return nil
        }
    }

    /// Foreground download. Cancellation/failure never publishes partial files.
    /// Concurrent requests for the same asset return busy; different assets can
    /// download concurrently within this store's reserved storage budget.
    public func ensure(_ asset: ModelAsset, progress: Progress? = nil) async throws -> URL {
        guard reservations[asset.sha256] == nil else { throw FlareError.busy }
        if let path = try cachedURL(for: asset) { return path }
        guard permitted(asset.url) else { throw FlareError.invalidAsset("Model download requires HTTPS") }
        let used = try cachedBytes()
        let reserved = reservations.values.reduce(Int64(0), +)
        guard asset.bytes <= maximumBytes, used <= maximumBytes - asset.bytes,
              reserved <= maximumBytes - asset.bytes - used else { throw FlareError.storageLimit }
        reservations[asset.sha256] = asset.bytes
        defer { reservations.removeValue(forKey: asset.sha256) }
        try Task.checkCancellation()
        let (temporary, response) = try await download(asset.url, progress)
        defer { try? FileManager.default.removeItem(at: temporary) }
        try Task.checkCancellation()
        guard let http = response as? HTTPURLResponse else { throw FlareError.httpStatus(-1) }
        guard http.statusCode == 200 else { throw FlareError.httpStatus(http.statusCode) }
        guard let finalURL = http.url, permitted(finalURL) else { throw FlareError.invalidAsset("Insecure download redirect") }
        try verify(temporary, asset)
        let staging = root.appendingPathComponent(".\(UUID().uuidString).partial")
        defer { try? FileManager.default.removeItem(at: staging) }
        // Move into the destination filesystem before publishing with an atomic
        // rename. Verification is streaming and never reads a full model into RAM.
        try FileManager.default.moveItem(at: temporary, to: staging)
        try Task.checkCancellation()
        let path = destination(asset)
        try FileManager.default.moveItem(at: staging, to: path)
        progress?(asset.bytes, asset.bytes)
        return path
    }

    public func prepare(model: ModelAsset, tokenizer: ModelAsset, progress: Progress? = nil) async throws -> CachedModel {
        let modelURL = try await ensure(model, progress: progress)
        let tokenizerURL = try await ensure(tokenizer)
        return CachedModel(modelURL: modelURL, tokenizerURL: tokenizerURL)
    }

    public func remove(_ asset: ModelAsset) throws {
        guard reservations[asset.sha256] == nil else { throw FlareError.busy }
        let path = destination(asset)
        if FileManager.default.fileExists(atPath: path.path) { try FileManager.default.removeItem(at: path) }
    }

    public func cachedBytes() throws -> Int64 {
        try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles])
            .filter { $0.pathExtension == "bin" }
            .reduce(0) { total, url in total + Int64(try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) }
    }

    private func destination(_ asset: ModelAsset) -> URL { root.appendingPathComponent(asset.sha256 + ".bin") }
    private func permitted(_ url: URL) -> Bool {
        url.scheme == "https" || (allowLocalHTTP && url.scheme == "http" && ["127.0.0.1", "localhost", "::1"].contains(url.host ?? ""))
    }

    private func verify(_ path: URL, _ asset: ModelAsset) throws {
        let size = try path.resourceValues(forKeys: [.fileSizeKey]).fileSize
        guard size.map(Int64.init) == asset.bytes else { throw FlareError.integrity("Wrong file length") }
        let file = try FileHandle(forReadingFrom: path)
        defer { try? file.close() }
        var hash = SHA256()
        while true {
            try Task.checkCancellation()
            let chunk = try file.read(upToCount: 256 * 1024) ?? Data()
            if chunk.isEmpty { break }
            hash.update(data: chunk)
        }
        let actual = hash.finalize().map { String(format: "%02x", $0) }.joined()
        guard actual == asset.sha256 else { throw FlareError.integrity("SHA-256 mismatch") }
    }
}

private final class DownloadProgress: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let progress: ModelStore.Progress?
    init(_ progress: ModelStore.Progress?) { self.progress = progress }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {}
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        progress?(totalBytesWritten, totalBytesExpectedToWrite)
    }
}
