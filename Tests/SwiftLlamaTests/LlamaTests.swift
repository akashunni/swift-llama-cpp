import Foundation
import Testing
@testable import SwiftLlama

@Suite(.serialized)
struct LlamaTests {

    @Test("Completion initialization and token generation produce output")
    @MainActor
    func completionFlowProducesTokens() async throws {
        try await TestProgress.runOnMainActor("LlamaTests.completionFlowProducesTokens") {
            guard let sut = try TestModelSupport.makeLlama(maxTokenCount: 128) else {
                TestProgress.skipped("LlamaTests.completionFlowProducesTokens", reason: "No GGUF test model available")
                return
            }

            await sut.updateSamplingConfig(.init(temperature: 0.3, seed: 7))
            try await sut.initializeCompletion(messages: TestModelSupport.simpleMessages())

            var emittedText = ""
            for _ in 0..<16 {
                let nextToken = try await sut.generateNextToken()
                switch nextToken {
                case .token(let token):
                    emittedText += token
                case .endOfString:
                    break
                }
            }

            #expect(!emittedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
    }

    @Test("Tokenize and detokenize keep a stable round trip")
    func tokenizeDetokenizeRoundtrip() throws {
        try TestProgress.run("LlamaTests.tokenizeDetokenizeRoundtrip") {
            guard let model = try TestModelSupport.makeModel() else {
                TestProgress.skipped("LlamaTests.tokenizeDetokenizeRoundtrip", reason: "No GGUF test model available")
                return
            }
            let text = "Hello, 世界! Emojis: 🚀🔥"
            let tokens = model.tokenize(text: text, addBos: false, special: false)
            let detokenized = model.detokenize(tokens: tokens, removeSpecial: true, unparseSpecial: false)
            let reparsed = model.tokenize(text: detokenized, addBos: false, special: false)

            #expect(!tokens.isEmpty)
            #expect(reparsed == tokens)
        }
    }
}
