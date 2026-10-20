// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "FlareKit",
    platforms: [.iOS("18.0"), .macOS(.v14)],
    products: [.library(name: "FlareKit", targets: ["FlareKit"])],
    targets: [
        .binaryTarget(name: "CFlare", path: "Frameworks/FlareNative.xcframework"),
        .target(name: "FlareKit", dependencies: ["CFlare"]),
        .testTarget(name: "FlareKitTests", dependencies: ["FlareKit"], resources: [.copy("Fixtures")])
    ]
)
