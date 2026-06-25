// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let llamaVersion = "b9786"
let llamaChecksum = "f4f97927ac6f352d06cc3039e51086b4e38594fab6272ebc59e65069e5603df8"

let package = Package(
    name: "swift-llama-cpp",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    products: [
        .library(
            name: "SwiftLlama",
            targets: ["SwiftLlama"]),
    ],
    dependencies: [
    ],
    targets: [
        .target(
            name: "SwiftLlama",
            dependencies: [
                "llama"
            ]
        ),
        .binaryTarget(
            name: "llama",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/\(llamaVersion)/llama-\(llamaVersion)-xcframework.zip",
            checksum: llamaChecksum
        ),
        .testTarget(
            name: "SwiftLlamaTests",
            dependencies: ["SwiftLlama"],
            resources: [.copy("Resources")]
        ),
    ]
)
