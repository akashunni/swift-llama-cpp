import Foundation
import Testing
@testable import SwiftLlama

@Suite(.serialized)
struct LlamaServiceTests {

    private struct Person: Codable {
        let name: String
        let age: Int
        let city: String?
    }

    private func makeService(maxTokenCount: UInt32 = 160) -> LlamaService? {
        TestModelSupport.makeService(batchSize: 128, maxTokenCount: maxTokenCount)
    }

    private func loadGrammar(named name: String) throws -> String {
        let url = try #require(Bundle.module.url(forResource: name, withExtension: "gbnf", subdirectory: "Resources"))
        return try String(contentsOf: url, encoding: .utf8)
    }

    private func makeGrammarSamplingConfig(
        grammarName: String,
        temperature: Float = 0.1,
        seed: UInt32 = 42
    ) throws -> LlamaSamplingConfig {
        .init(
            temperature: temperature,
            seed: seed,
            grammarConfig: .init(
                grammar: try loadGrammar(named: grammarName),
                grammarRoot: "root"
            )
        )
    }

    private func parseJSONObject(_ text: String) throws -> [String: Any] {
        let data = try #require(text.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func parseJSONArray(_ text: String) throws -> [Any] {
        let data = try #require(text.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [Any])
    }

    private func parseStringArray(_ text: String) throws -> [String] {
        let data = try #require(text.data(using: .utf8))
        return try JSONDecoder().decode([String].self, from: data)
    }

    @Test("streamCompletion rejects an empty message list")
    func streamCompletionRejectsEmptyMessages() async throws {
        try await TestProgress.run("LlamaServiceTests.streamCompletionRejectsEmptyMessages") {
            guard let service = makeService() else {
                TestProgress.skipped("LlamaServiceTests.streamCompletionRejectsEmptyMessages", reason: "No GGUF test model available")
                return
            }

            do {
                _ = try await service.streamCompletion(of: [], samplingConfig: .init(temperature: 0.1, seed: 1))
                Issue.record("Expected streamCompletion(of:samplingConfig:) to reject an empty message array.")
            } catch let error as LlamaError {
                #expect(error == .emptyMessageArray)
            }
        }
    }

    @Test("Service exposes model architecture and family")
    func serviceExposesModelArchitectureAndFamily() async throws {
        try await TestProgress.run("LlamaServiceTests.serviceExposesModelArchitectureAndFamily") {
            guard let service = makeService() else {
                TestProgress.skipped("LlamaServiceTests.serviceExposesModelArchitectureAndFamily", reason: "No GGUF test model available")
                return
            }

            let architecture = try await service.modelArchitecture()
            let family = try await service.modelFamily()

            switch architecture {
            case "llama":
                #expect(family == .llama)
            case let value? where value.hasPrefix("gemma"):
                #expect(family == .gemma)
            default:
                #expect(family == .unknown || family == .gemma)
            }
        }
    }

    @Test("Plain text respond returns non-empty output")
    func respondReturnsText() async throws {
        try await TestProgress.run("LlamaServiceTests.respondReturnsText") {
            guard let service = makeService() else {
                TestProgress.skipped("LlamaServiceTests.respondReturnsText", reason: "No GGUF test model available")
                return
            }
            let text = try await service.respond(
                to: TestModelSupport.simpleMessages(),
                samplingConfig: .init(temperature: 0.2, seed: 11)
            )

            #expect(!text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
    }

    @Test("Plain text respond is deterministic with the same seed")
    func respondIsDeterministic() async throws {
        try await TestProgress.run("LlamaServiceTests.respondIsDeterministic") {
            guard let serviceA = makeService() else {
                TestProgress.skipped("LlamaServiceTests.respondIsDeterministic", reason: "No GGUF test model available")
                return
            }
            guard let serviceB = makeService() else {
                TestProgress.skipped("LlamaServiceTests.respondIsDeterministic", reason: "No GGUF test model available")
                return
            }
            let sampling = LlamaSamplingConfig(temperature: 0.1, seed: 12345)

            let first = try await serviceA.respond(to: TestModelSupport.simpleMessages(), samplingConfig: sampling)
            let second = try await serviceB.respond(to: TestModelSupport.simpleMessages(), samplingConfig: sampling)

            #expect(first == second)
        }
    }

    @Test("Top-K constrained sampling still produces output")
    func topKSamplingProducesOutput() async throws {
        try await TestProgress.run("LlamaServiceTests.topKSamplingProducesOutput") {
            guard let service = makeService() else {
                TestProgress.skipped("LlamaServiceTests.topKSamplingProducesOutput", reason: "No GGUF test model available")
                return
            }
            let text = try await service.respond(
                to: TestModelSupport.simpleMessages(),
                samplingConfig: .init(temperature: 0.8, seed: 9, topK: 5)
            )

            #expect(!text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
    }

    @Test("Repetition penalty configuration still produces output")
    func repetitionPenaltyProducesOutput() async throws {
        try await TestProgress.run("LlamaServiceTests.repetitionPenaltyProducesOutput") {
            let isGemmaFamily = try TestModelSupport.isGemmaFamily()
            guard let service = makeService(maxTokenCount: isGemmaFamily ? 768 : 192) else {
                TestProgress.skipped("LlamaServiceTests.repetitionPenaltyProducesOutput", reason: "No GGUF test model available")
                return
            }
            let sampling = LlamaSamplingConfig(
                temperature: isGemmaFamily ? 0.0 : 0.2,
                seed: 21,
                grammarConfig: .init(
                    grammar: try loadGrammar(named: "json_array_strings"),
                    grammarRoot: "root"
                ),
                repetitionPenaltyConfig: .init(
                    lastN: isGemmaFamily ? 8 : 32,
                    repeatPenalty: isGemmaFamily ? 1.02 : 1.2,
                    freqPenalty: isGemmaFamily ? 0.0 : 0.1,
                    presentPenalty: isGemmaFamily ? 0.0 : 0.1
                )
            )

            let text = try await service.respond(
                to: [
                    LlamaChatMessage(role: .system, content: "Respond only with a JSON array of strings."),
                    LlamaChatMessage(role: .user, content: "Return exactly three short words as a compact JSON array.")
                ],
                samplingConfig: sampling
            )
            let strings = try parseStringArray(text)
            if isGemmaFamily {
                #expect(!strings.isEmpty)
            } else {
                #expect(strings.count == 3)
            }
            #expect(strings.allSatisfy { !$0.isEmpty })
        }
    }

    @Test("GBNF object grammar produces valid JSON")
    func jsonGrammarProducesJSONObject() async throws {
        try await TestProgress.run("LlamaServiceTests.jsonGrammarProducesJSONObject") {
            guard let service = makeService(maxTokenCount: 96) else {
                TestProgress.skipped("LlamaServiceTests.jsonGrammarProducesJSONObject", reason: "No GGUF test model available")
                return
            }
            let sampling = try makeGrammarSamplingConfig(grammarName: "json")
            let messages = [
                LlamaChatMessage(role: .system, content: "Respond only with valid JSON."),
                LlamaChatMessage(role: .user, content: "Return a simple object with a few fields.")
            ]

            let text = try await service.respond(to: messages, samplingConfig: sampling)
            let object = try parseJSONObject(text)

            #expect(!object.isEmpty)
        }
    }

    @Test("GBNF array grammar produces valid JSON arrays")
    func jsonArrayGrammarProducesJSONArray() async throws {
        try await TestProgress.run("LlamaServiceTests.jsonArrayGrammarProducesJSONArray") {
            guard let service = makeService(maxTokenCount: 256) else {
                TestProgress.skipped("LlamaServiceTests.jsonArrayGrammarProducesJSONArray", reason: "No GGUF test model available")
                return
            }
            let sampling = try makeGrammarSamplingConfig(grammarName: "json_array")
            let messages = [
                LlamaChatMessage(role: .system, content: "Respond only with a valid JSON array."),
                LlamaChatMessage(role: .user, content: "Return exactly one short JSON array on a single line, such as [1,2,3].")
            ]

            let text = try await service.respond(to: messages, samplingConfig: sampling)
            _ = try parseJSONArray(text)
        }
    }

    @Test("GBNF string-array grammar decodes to string arrays")
    func jsonStringArrayGrammarProducesStrings() async throws {
        try await TestProgress.run("LlamaServiceTests.jsonStringArrayGrammarProducesStrings") {
            guard let service = makeService(maxTokenCount: 96) else {
                TestProgress.skipped("LlamaServiceTests.jsonStringArrayGrammarProducesStrings", reason: "No GGUF test model available")
                return
            }
            let sampling = try makeGrammarSamplingConfig(grammarName: "json_array_strings")
            let messages = [
                LlamaChatMessage(role: .system, content: "Respond only with a JSON array of strings."),
                LlamaChatMessage(role: .user, content: "Return a few programming languages.")
            ]

            let text = try await service.respond(to: messages, samplingConfig: sampling)
            let strings = try parseStringArray(text)

            #expect(!strings.isEmpty)
            #expect(strings.allSatisfy { !$0.isEmpty })
        }
    }

    @Test("Typed respond decodes a generated object")
    func typedRespondDecodesObject() async throws {
        try await TestProgress.run("LlamaServiceTests.typedRespondDecodesObject") {
            guard let service = makeService(maxTokenCount: 128) else {
                TestProgress.skipped("LlamaServiceTests.typedRespondDecodesObject", reason: "No GGUF test model available")
                return
            }
            let person = try await service.respond(
                to: [
                    LlamaChatMessage(role: .system, content: "Respond only with JSON matching the schema. Do not include explanations or markdown."),
                    LlamaChatMessage(role: .user, content: "Return exactly one JSON object with name \"Ada\", age 36, and city \"London\".")
                ],
                generating: Person.self
            )

            #expect(person.name == "Ada")
            #expect(person.age == 36)
            #expect(person.city == "London")
        }
    }

    @Test("Typed respond decodes arrays of strings")
    func typedRespondDecodesArray() async throws {
        try await TestProgress.run("LlamaServiceTests.typedRespondDecodesArray") {
            guard let service = makeService(maxTokenCount: 256) else {
                TestProgress.skipped("LlamaServiceTests.typedRespondDecodesArray", reason: "No GGUF test model available")
                return
            }
            let strings = try await service.respond(
                to: [
                    LlamaChatMessage(role: .system, content: "Respond only with a JSON array of strings."),
                    LlamaChatMessage(role: .user, content: "Return exactly this JSON array of strings: [\"apple\",\"banana\",\"mango\"].")
                ],
                generating: [String].self
            )

            #expect(!strings.isEmpty)
            #expect(strings.allSatisfy { !$0.isEmpty })
        }
    }

    @Test("Streaming completion can be consumed incrementally")
    func streamCompletionYieldsTokens() async throws {
        try await TestProgress.run("LlamaServiceTests.streamCompletionYieldsTokens") {
            guard let service = makeService(maxTokenCount: 96) else {
                TestProgress.skipped("LlamaServiceTests.streamCompletionYieldsTokens", reason: "No GGUF test model available")
                return
            }
            let stream = try await service.streamCompletion(
                of: TestModelSupport.simpleMessages(),
                samplingConfig: .init(temperature: 0.2, seed: 5)
            )

            var output = ""
            var tokenCount = 0

            for try await token in stream {
                output += token
                tokenCount += 1
                if tokenCount == 8 {
                    await service.stopCompletion()
                    break
                }
            }

            #expect(tokenCount > 0)
            #expect(!output.isEmpty)
        }
    }
}
