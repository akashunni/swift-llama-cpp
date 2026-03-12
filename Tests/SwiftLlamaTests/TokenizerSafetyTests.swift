import Testing
import Foundation
@testable import SwiftLlama

struct TokenizerSafetyTests {
    @Test("Tokenize very long input")
    func testLongInputTokenization() throws {
        let model = try #require(LlamaModel(path: URL.llama1B.path))
        // Create a long string with many characters that might trigger byte fallback
        // to ensure we have a high token-to-char ratio if possible,
        // or just a very long string to test the context truncation.
        let longText = String(repeating: "Testing tokenizer safety with long input strings. ", count: 100)
        
        let tokens = model.tokenize(text: longText, addBos: true, special: true)
        #expect(!tokens.isEmpty)
        
        // Ensure it doesn't exceed trained context size
        let maxTokens = model.trainedContextSize()
        #expect(tokens.count <= Int(maxTokens))
    }

    @Test("Tokenize empty input")
    func testEmptyInput() throws {
        let model = try #require(LlamaModel(path: URL.llama1B.path))
        let tokens = model.tokenize(text: "", addBos: true, special: true)
        #expect(tokens.isEmpty)
    }

    @Test("Normal input tokenization")
    func testNormalInput() throws {
        let model = try #require(LlamaModel(path: URL.llama1B.path))
        let text = "Hello, world!"
        let tokens = model.tokenize(text: text, addBos: true, special: true)
        #expect(!tokens.isEmpty)
        
        let detokenized = model.detokenize(tokens: tokens)
        #expect(detokenized.contains("Hello"))
    }

    @Test("piece(from:) with various tokens")
    func testPieceSafety() throws {
        let model = try #require(LlamaModel(path: URL.llama1B.path))
        
        // Test common token
        let piece1 = model.piece(from: model.bosToken())
        #expect(piece1.count >= 0)
        
        let piece2 = model.piece(from: 100)
        #expect(!piece2.isEmpty)
        
        // Test invalid token
        let pieceInvalid = model.piece(from: -1)
        #expect(pieceInvalid == "")
    }
}
