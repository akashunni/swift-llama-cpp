import Foundation
import Testing
@testable import SwiftLlama

enum TestModelSupport {
    private static let gpuOverrideKey = "SWIFT_LLAMA_TEST_USE_GPU"
    private static let environmentKeys = [
        "SWIFT_LLAMA_TEST_MODEL_PATH",
        "LLAMA_MODEL_PATH"
    ]

    static func modelURL() -> URL? {
        let fileManager = FileManager.default
        let environment = ProcessInfo.processInfo.environment

        for key in environmentKeys {
            if let rawValue = environment[key], !rawValue.isEmpty {
                let candidate = URL(fileURLWithPath: rawValue)
                if fileManager.fileExists(atPath: candidate.path(percentEncoded: false)) {
                    return candidate
                }
            }
        }

        if let bundled = Bundle.module.url(
            forResource: "Llama-3.2-1B-Instruct-Q4_K_M",
            withExtension: "gguf",
            subdirectory: "Resources"
        ) {
            return bundled
        }

        if let bundled = Bundle.module.url(
            forResource: "Resources/Llama-3.2-1B-Instruct-Q4_K_M",
            withExtension: "gguf"
        ) {
            return bundled
        }

        return nil
    }

    static func makeModel() throws -> LlamaModel? {
        guard let url = modelURL() else { return nil }
        return LlamaModel(path: url.path(percentEncoded: false))
    }

    static func makeService(
        batchSize: UInt32 = 128,
        maxTokenCount: UInt32 = 256,
        useGPU: Bool? = nil
    ) -> LlamaService? {
        guard let modelUrl = modelURL() else { return nil }
        let resolvedUseGPU = useGPU ?? shouldUseGPU()
        return LlamaService(
            modelUrl: modelUrl,
            config: .init(
                batchSize: batchSize,
                maxTokenCount: maxTokenCount,
                useGPU: resolvedUseGPU
            )
        )
    }

    static func makeLlama(
        batchSize: UInt32 = 128,
        maxTokenCount: UInt32 = 256,
        useGPU: Bool? = nil
    ) throws -> Llama? {
        guard let modelURL = modelURL() else { return nil }
        let resolvedUseGPU = useGPU ?? shouldUseGPU()
        return try Llama(
            modelPath: modelURL.path(percentEncoded: false),
            config: .init(
                batchSize: batchSize,
                maxTokenCount: maxTokenCount,
                useGPU: resolvedUseGPU
            )
        )
    }

    static func shouldUseGPU() -> Bool {
        guard let raw = ProcessInfo.processInfo.environment[gpuOverrideKey]?.lowercased() else {
            return true
        }
        switch raw {
        case "0", "false", "no", "off":
            return false
        default:
            return true
        }
    }

    static func simpleMessages() -> [LlamaChatMessage] {
        [
            .init(role: .system, content: "You are a concise assistant."),
            .init(role: .user, content: "Write one short sentence about local language models.")
        ]
    }
}
