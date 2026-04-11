//
//  LlamaModel.swift
//  PrivateAI
//
//  Created by Piotr Gorzelany on 12/02/2024.
//

import Foundation
import llama

enum LlamaModelError: Error {
    case initializationError
}

public enum LlamaModelFamily: String, Sendable {
    case llama
    case gemma
    case unknown
}

public final class LlamaModel {

    // MARK: - Properties

    let modelPointer: OpaquePointer
    let vocabPointer: OpaquePointer

    // MARK: - Lifecycle

    public init?(path: String, parameters: llama_model_params = llama_model_default_params()) {
        guard let modelPointer = llama_model_load_from_file(path, parameters), let vocabPointer = llama_model_get_vocab(modelPointer) else {
            return nil
        }
        self.modelPointer = modelPointer
        self.vocabPointer = vocabPointer
    }

    /// Initializes a model from multiple GGUF split files.
    /// The `paths` must be ordered correctly.
    public init?(paths: [String], parameters: llama_model_params = llama_model_default_params()) {
        var cStrings: [UnsafeMutablePointer<CChar>?] = paths.map { strdup($0) }
        defer { cStrings.forEach { if let p = $0 { free(UnsafeMutablePointer(mutating: p)) } } }
        let count = cStrings.count
        let result = cStrings.withUnsafeMutableBufferPointer { buf in
            buf.baseAddress!.withMemoryRebound(to: UnsafePointer<CChar>?.self, capacity: count) { reboundPtr in
                llama_model_load_from_splits(reboundPtr, size_t(count), parameters)
            }
        }
        guard let modelPointer = result, let vocabPointer = llama_model_get_vocab(modelPointer) else {
            return nil
        }
        self.modelPointer = modelPointer
        self.vocabPointer = vocabPointer
    }

    deinit {
        llama_model_free(modelPointer)
    }

    // MARK: - Methods

    // Helper to convert a null-terminated CChar buffer into Swift String without deprecation warnings
    private static func stringFromNullTerminated(_ buffer: [CChar]) -> String {
        let units: [UInt8] = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return String(decoding: units, as: UTF8.self)
    }

    /// Text context size used during training.
    public func trainedContextSize() -> Int32 {
        llama_model_n_ctx_train(modelPointer)
    }

    /// A string describing the model type.
    public func description() -> String {
        let bufferSize = 1024
        var buffer = [CChar](repeating: 0, count: bufferSize)
        let descriptionBufferSize = llama_model_desc(modelPointer, &buffer, bufferSize)
        guard descriptionBufferSize > 0 else {
            fatalError("Something went wrong")
        }
        return Self.stringFromNullTerminated(buffer)
    }

    /// Render token text for a token id.
    public func string(from token: llama_token) -> String {
        guard token >= 0 && token < vocabularySize() else {
            return ""
        }
        guard let results = llama_vocab_get_text(vocabPointer, token) else {
            return ""
        }
        return String(cString: results, encoding: .utf8) ?? ""
    }

    /// Convert a token id to its piece (optionally rendering special tokens).
    public func piece(from token: llama_token, renderSpecial: Bool = false, lstrip: Int32 = 0) -> String {
        guard token >= 0 && token < vocabularySize() else {
            return ""
        }
        var bufferSize: Int32 = 64
        var buffer = [CChar](repeating: 0, count: Int(bufferSize))
        var charCount = llama_token_to_piece(vocabPointer, token, &buffer, bufferSize, lstrip, renderSpecial)
        if charCount < 0 {
            // Buffer too small, resize and retry
            bufferSize = -charCount
            buffer = [CChar](repeating: 0, count: Int(bufferSize))
            charCount = llama_token_to_piece(vocabPointer, token, &buffer, bufferSize, lstrip, renderSpecial)
        }
        guard charCount > 0 else {
            return ""
        }
        let chars = Array(buffer.prefix(upTo: Int(charCount))) + [0]
        return String(cString: chars, encoding: .utf8) ?? ""
    }

    /// Beginning-of-sentence token id.
    public func bosToken() -> llama_token {
        llama_vocab_bos(vocabPointer)
    }

    /// Whether a BOS token should be added automatically.
    public func shouldAddBos() -> Bool {
        let addBos = llama_vocab_get_add_bos(vocabPointer)
        if addBos {
            return llama_vocab_type(vocabPointer) == LLAMA_VOCAB_TYPE_SPM
        }
        return addBos
    }

    /// End-of-sentence token id.
    public func eosToken() -> llama_token {
        llama_vocab_eos(vocabPointer)
    }

    /// Whether the token is an end-of-generation token (e.g. EOS/EOT).
    public func isEogToken(_ token: llama_token) -> Bool {
        llama_vocab_is_eog(vocabPointer, token)
    }

    /// Convert the provided text into tokens.
    /// - Parameters:
    ///   - addBos: Allow to add BOS/EOS if model is configured so.
    ///   - special: Allow tokenizing special/control tokens.
    public func tokenize(text: String, addBos: Bool, special: Bool) -> [llama_token] {
        guard !text.isEmpty else {
            return []
        }
        let utf8Count = Int32(text.utf8.count)
        let maxTokens = trainedContextSize()
        // Heuristic: token buffer = utf8Count + 2 (BOS/EOS and potential extra)
        var tokenBufferSize = utf8Count + (addBos ? 1 : 0) + 1
        var tokensBuffer = [llama_token](repeating: llama_token(), count: Int(tokenBufferSize))
        var tokenCount = llama_tokenize(vocabPointer, text, utf8Count, &tokensBuffer, tokenBufferSize, addBos, special)
        
        if tokenCount < 0 {
            // Buffer too small, resize and retry
            tokenBufferSize = -tokenCount
            tokensBuffer = [llama_token](repeating: llama_token(), count: Int(tokenBufferSize))
            tokenCount = llama_tokenize(vocabPointer, text, utf8Count, &tokensBuffer, tokenBufferSize, addBos, special)
        }
        
        guard tokenCount > 0 else {
            return []
        }
        
        // Final sanity check: ensure we don't return more than what the context can handle if that was intent
        // though llama_tokenize already respects context size if passed as n_tokens_max
        // but currently we pass tokenBufferSize as n_tokens_max to get the full tokenization first.
        let result = Array(tokensBuffer.prefix(upTo: Int(tokenCount)))
        if result.count > maxTokens {
            return Array(result.prefix(Int(maxTokens)))
        }
        return result
    }

    /// Convert tokens back to text (inverse of tokenize)
    /// Convert tokens back to text (inverse of tokenize()).
    public func detokenize(tokens: [llama_token], removeSpecial: Bool = true, unparseSpecial: Bool = false) -> String {
        guard !tokens.isEmpty else { return "" }
        // Heuristic buffer: tokens * avg 4 bytes + 16
        var bufSize = Int32(tokens.count * 4 + 16)
        var buffer = [CChar](repeating: 0, count: Int(bufSize))
        var written: Int32 = -1
        repeat {
            written = tokens.withUnsafeBufferPointer { ptr in
                llama_detokenize(vocabPointer, ptr.baseAddress, Int32(tokens.count), &buffer, bufSize, removeSpecial, unparseSpecial)
            }
            if written < 0 { // need bigger buffer
                bufSize = -written + 1
                buffer = [CChar](repeating: 0, count: Int(bufSize))
            }
        } while written < 0
        return Self.stringFromNullTerminated(buffer)
    }

    /// Number of tokens in the vocabulary.
    public func vocabularySize() -> Int32 {
        llama_vocab_n_tokens(vocabPointer)
    }

    /// Apply chat template using the default model template (or custom by name).
    public func applyChatTemplate(to messages: [LlamaChatMessage], addAssistant: Bool? = nil) -> String {
        guard !messages.isEmpty else { return "" }
        let cTemplatePointer = llama_model_chat_template(modelPointer, nil)
        guard cTemplatePointer != nil else {
            return fallbackPromptForModelFamily(from: messages, addAssistant: addAssistant)
        }

        // Convert Swift messages to C messages
        var cMessages = messages.map { message -> llama_chat_message in
           let roleCString = strdup(message.role.rawValue)
           let contentCString = strdup(message.content)
           return llama_chat_message(role: roleCString, content: contentCString)
        }

        // Initial buffer size
        let bufferSizeMultiplier = 3
        var bufferSize = max(256, bufferSizeMultiplier * messages.reduce(0) { $0 + $1.content.count })
        var buffer = [CChar](repeating: 0, count: bufferSize)

        var resultSize: Int32 = 0
        repeat {
           // If the buffer was too small, increase the buffer size
           if resultSize >= Int32(bufferSize) {
               bufferSize = Int(resultSize + 1) // the buffer has to be null (0) terminated
               buffer = [CChar](repeating: 0, count: bufferSize)
           }

           resultSize = llama_chat_apply_template(
               cTemplatePointer,
               &cMessages,
               messages.count,
               addAssistant ?? (messages.last?.role != .assistant),
               &buffer,
               Int32(bufferSize)
           )
        } while resultSize >= Int32(bufferSize)

        // Free the allocated C strings
        for message in cMessages {
           free(UnsafeMutablePointer(mutating: message.role))
           free(UnsafeMutablePointer(mutating: message.content))
        }

        // Negative or zero means template application failed or produced nothing.
        guard resultSize > 0 else {
            return fallbackPromptForModelFamily(from: messages, addAssistant: addAssistant)
        }

        let rendered = Self.stringFromNullTerminated(buffer)
        return rendered.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? fallbackPromptForModelFamily(from: messages, addAssistant: addAssistant)
            : rendered
    }

    /// Apply chat template by template name found in the model.
    public func applyChatTemplate(name: String, to messages: [LlamaChatMessage], addAssistant: Bool? = nil) -> String {
        guard !messages.isEmpty else { return "" }
        let cTemplatePointer = name.withCString { cname in
            llama_model_chat_template(modelPointer, cname)
        }
        guard cTemplatePointer != nil else {
            return fallbackPromptForModelFamily(from: messages, addAssistant: addAssistant)
        }
        // Convert Swift messages to C messages
        var cMessages = messages.map { message -> llama_chat_message in
           let roleCString = strdup(message.role.rawValue)
           let contentCString = strdup(message.content)
           return llama_chat_message(role: roleCString, content: contentCString)
        }
        let bufferSizeMultiplier = 3
        var bufferSize = max(256, bufferSizeMultiplier * messages.reduce(0) { $0 + $1.content.count })
        var buffer = [CChar](repeating: 0, count: bufferSize)
        var resultSize: Int32 = 0
        repeat {
            if resultSize >= Int32(bufferSize) {
                bufferSize = Int(resultSize + 1)
                buffer = [CChar](repeating: 0, count: bufferSize)
            }
            resultSize = llama_chat_apply_template(
                cTemplatePointer,
                &cMessages,
                messages.count,
                addAssistant ?? (messages.last?.role != .assistant),
                &buffer,
                Int32(bufferSize)
            )
        } while resultSize >= Int32(bufferSize)
        for message in cMessages {
            free(UnsafeMutablePointer(mutating: message.role))
            free(UnsafeMutablePointer(mutating: message.content))
        }
        guard resultSize > 0 else {
            return fallbackPromptForModelFamily(from: messages, addAssistant: addAssistant)
        }
        let rendered = Self.stringFromNullTerminated(buffer)
        return rendered.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? fallbackPromptForModelFamily(from: messages, addAssistant: addAssistant)
            : rendered
    }

    public func architecture() -> String? {
        metaValue(forKey: "general.architecture")?.lowercased()
    }

    public func modelFamily() -> LlamaModelFamily {
        guard let architecture = architecture() else {
            return fallbackModelFamily()
        }

        switch architecture {
        case "llama":
            return .llama
        case let value where value.hasPrefix("gemma"):
            return .gemma
        default:
            return fallbackModelFamily()
        }
    }

    private func fallbackPrompt(from messages: [LlamaChatMessage], addAssistant: Bool?) -> String {
        var parts = messages.map { "\($0.role.rawValue.capitalized): \($0.content)" }
        if addAssistant ?? (messages.last?.role != .assistant) {
            parts.append("Assistant:")
        }
        return parts.joined(separator: "\n\n")
    }

    private func gemmaPrompt(from messages: [LlamaChatMessage], addAssistant: Bool?) -> String {
        let normalizedMessages = normalizeMessagesForGemma(messages)
        var prompt = normalizedMessages.map { message in
            let role = message.role == .assistant ? "model" : "user"
            return "<start_of_turn>\(role)\n\(message.content)<end_of_turn>"
        }.joined(separator: "\n")

        if addAssistant ?? (normalizedMessages.last?.role != .assistant) {
            if !prompt.isEmpty {
                prompt += "\n"
            }
            prompt += "<start_of_turn>model\n"
        }

        return prompt
    }

    private func normalizeMessagesForGemma(_ messages: [LlamaChatMessage]) -> [LlamaChatMessage] {
        var normalized: [LlamaChatMessage] = []
        var pendingInstruction: [String] = []

        for message in messages {
            switch message.role {
            case .system:
                pendingInstruction.append(message.content.trimmingCharacters(in: .whitespacesAndNewlines))
            case .user:
                let content = ([pendingInstruction.joined(separator: "\n\n"), message.content]
                    .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty })
                    .joined(separator: "\n\n")
                normalized.append(.init(role: .user, content: content))
                pendingInstruction.removeAll(keepingCapacity: true)
            case .assistant:
                if !pendingInstruction.isEmpty {
                    let merged = pendingInstruction.joined(separator: "\n\n")
                    normalized.append(.init(role: .user, content: merged))
                    pendingInstruction.removeAll(keepingCapacity: true)
                }
                normalized.append(message)
            }
        }

        if !pendingInstruction.isEmpty {
            normalized.append(.init(role: .user, content: pendingInstruction.joined(separator: "\n\n")))
        }

        return normalized
    }

    private func fallbackPromptForModelFamily(from messages: [LlamaChatMessage], addAssistant: Bool?) -> String {
        switch modelFamily() {
        case .gemma:
            return gemmaPrompt(from: messages, addAssistant: addAssistant)
        case .llama, .unknown:
            return fallbackPrompt(from: messages, addAssistant: addAssistant)
        }
    }

    private func fallbackModelFamily() -> LlamaModelFamily {
        if let tokenizerModel = metaValue(forKey: "tokenizer.ggml.model")?.lowercased(),
           tokenizerModel.contains("gemma") {
            return .gemma
        }

        if let modelName = metaValue(forKey: "general.name")?.lowercased(),
           modelName.contains("gemma") {
            return .gemma
        }

        return .unknown
    }

    /// Total number of parameters in the model.
    public func numberOfParameters() -> UInt64 {
        return llama_model_n_params(modelPointer)
    }

    // MARK: - Model & Vocab Introspection

    /// Model and vocab introspection helpers.
    public func ropeType() -> llama_rope_type { llama_model_rope_type(modelPointer) }
    public func nEmbed() -> Int32 { llama_model_n_embd(modelPointer) }
    public func nLayer() -> Int32 { llama_model_n_layer(modelPointer) }
    public func nHead() -> Int32 { llama_model_n_head(modelPointer) }
    public func nHeadKV() -> Int32 { llama_model_n_head_kv(modelPointer) }
    public func nSWA() -> Int32 { llama_model_n_swa(modelPointer) }
    public func ropeFreqScaleTrain() -> Float { llama_model_rope_freq_scale_train(modelPointer) }
    public func nClassifierOutputs() -> UInt32 { llama_model_n_cls_out(modelPointer) }
    public func classifierLabel(at index: UInt32) -> String? {
        guard let cstr = llama_model_cls_label(modelPointer, index) else { return nil }
        return String(cString: cstr)
    }
    public func modelSizeBytes() -> UInt64 { llama_model_size(modelPointer) }
    public func hasEncoder() -> Bool { llama_model_has_encoder(modelPointer) }
    public func hasDecoder() -> Bool { llama_model_has_decoder(modelPointer) }
    public func decoderStartToken() -> llama_token { llama_model_decoder_start_token(modelPointer) }
    public func isRecurrent() -> Bool { llama_model_is_recurrent(modelPointer) }
    public func isDiffusion() -> Bool { llama_model_is_diffusion(modelPointer) }

    // Vocab helpers
    public func vocabType() -> llama_vocab_type { llama_vocab_type(vocabPointer) }
    public func vocabScore(for token: llama_token) -> Float { llama_vocab_get_score(vocabPointer, token) }
    public func vocabAttr(for token: llama_token) -> llama_token_attr { llama_vocab_get_attr(vocabPointer, token) }
    public func isControl(token: llama_token) -> Bool { llama_vocab_is_control(vocabPointer, token) }
    public func sepToken() -> llama_token { llama_vocab_sep(vocabPointer) }
    public func nlToken() -> llama_token { llama_vocab_nl(vocabPointer) }
    public func padToken() -> llama_token { llama_vocab_pad(vocabPointer) }
    public func maskToken() -> llama_token { llama_vocab_mask(vocabPointer) }
    public func addEos() -> Bool { llama_vocab_get_add_eos(vocabPointer) }
    public func addSep() -> Bool { llama_vocab_get_add_sep(vocabPointer) }
    public func fimPre() -> llama_token { llama_vocab_fim_pre(vocabPointer) }
    public func fimSuf() -> llama_token { llama_vocab_fim_suf(vocabPointer) }
    public func fimMid() -> llama_token { llama_vocab_fim_mid(vocabPointer) }
    public func fimPad() -> llama_token { llama_vocab_fim_pad(vocabPointer) }
    public func fimRep() -> llama_token { llama_vocab_fim_rep(vocabPointer) }
    public func fimSep() -> llama_token { llama_vocab_fim_sep(vocabPointer) }

    // Metadata
    /// Read model GGUF metadata value by key as string.
    public func metaValue(forKey key: String) -> String? {
        let bufSize = 512
        var buffer = [CChar](repeating: 0, count: bufSize)
        let res = llama_model_meta_val_str(modelPointer, key, &buffer, bufSize)
        guard res >= 0 else { return nil }
        return Self.stringFromNullTerminated(buffer)
    }
    /// Number of model GGUF metadata key/value pairs.
    public func metaCount() -> Int32 { llama_model_meta_count(modelPointer) }
    /// Read metadata key name by index.
    public func metaKey(at index: Int32) -> String? {
        let bufSize = 512
        var buffer = [CChar](repeating: 0, count: bufSize)
        let res = llama_model_meta_key_by_index(modelPointer, index, &buffer, bufSize)
        guard res >= 0 else { return nil }
        return Self.stringFromNullTerminated(buffer)
    }
    /// Read metadata value as a string by index.
    public func metaValue(at index: Int32) -> String? {
        let bufSize = 512
        var buffer = [CChar](repeating: 0, count: bufSize)
        let res = llama_model_meta_val_str_by_index(modelPointer, index, &buffer, bufSize)
        guard res >= 0 else { return nil }
        return Self.stringFromNullTerminated(buffer)
    }

    // Save model
    /// Save the model to a file.
    public func save(to path: String) {
        llama_model_save_to_file(modelPointer, path)
    }

    // Built-in chat templates
    /// Get list of built-in chat templates.
    public func builtinChatTemplates(maxCount: Int = 64) -> [String] {
        var result: [String] = []
        var ptrs = Array<UnsafePointer<CChar>?>(repeating: nil, count: maxCount)
        let n = ptrs.withUnsafeMutableBufferPointer { buf in
            llama_chat_builtin_templates(buf.baseAddress, size_t(maxCount))
        }
        if n > 0 {
            for i in 0..<min(Int(n), maxCount) {
                if let p = ptrs[i] { result.append(String(cString: p)) }
            }
        }
        return result
    }

    // Quantize helper (wraps the C quantize function)
    @discardableResult
    /// Quantize a model file.
    public static func quantizeModel(inputPath: String, outputPath: String, params: inout llama_model_quantize_params) -> UInt32 {
        llama_model_quantize(inputPath, outputPath, &params)
    }

    /// Default quantization parameters.
    public static func defaultQuantizeParams() -> llama_model_quantize_params {
        llama_model_quantize_default_params()
    }

    // MARK: - Split utilities

    /// Build a split GGUF final path for this chunk.
    public static func splitPath(pathPrefix: String, splitNo: Int32, splitCount: Int32) -> String {
        var buf = [CChar](repeating: 0, count: 1024)
        _ = pathPrefix.withCString { c in
            llama_split_path(&buf, buf.count, c, splitNo, splitCount)
        }
        return Self.stringFromNullTerminated(buf)
    }

    /// Extract the path prefix from a split path if and only if the split_no and split_count match.
    public static func splitPrefix(splitPath: String, splitNo: Int32, splitCount: Int32) -> String? {
        var buf = [CChar](repeating: 0, count: 1024)
        let n = splitPath.withCString { c in
            llama_split_prefix(&buf, buf.count, c, splitNo, splitCount)
        }
        if n <= 0 { return nil }
        return Self.stringFromNullTerminated(buf)
    }
}
