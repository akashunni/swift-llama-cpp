import Foundation
import llama

enum NextToken {
    case token(String)
    case endOfString
}

final actor Llama {
    private let model: LlamaModel
    let context: LlamaContext
    private var batch: LlamaBatch
    private var sampler: LlamaSampler!

    // Configuration

    private let config: LlamaConfig
    let maxTokenCount: UInt32
    /// Tracks the current position in the token sequence during decoding.
    var currentTokenPosition: Int32 = 0
    var processedTokens: [llama_token] = []

    init(modelPath: String, config: LlamaConfig) throws {
        self.config = config
        LlamaBackend.initialize()
        var model_params = llama_model_default_params()

        // useGPU = false: keep GPU disabled (legacy flag, overrides gpuLayerCount)
        if !config.useGPU {
            model_params.n_gpu_layers = 0
        } else {
            // n_gpu_layers controls how many transformer layers are offloaded to
            // the GPU (Metal on Apple Silicon / iOS device).
            //   0   = CPU only
            //   999 = attempt to offload all layers
            // Casting to Int32 is safe: llama.cpp uses int32_t for this field.
            model_params.n_gpu_layers = Int32(config.gpuLayerCount)
        }

        #if targetEnvironment(simulator)
        // Metal is not available in the iOS simulator; force CPU-only.
        model_params.n_gpu_layers = 0
        print("Running on simulator, force use n_gpu_layers = 0")
        #endif

        let model = LlamaModel(path: modelPath, parameters: model_params)
        guard let model else {
            print("Could not load model at \(modelPath)")
            throw LlamaError.couldNotInitializeContext
        }

        let n_threads = ProcessInfo.processInfo.processorCount - 1
        print("Using \(n_threads) threads")

        var contextParam = llama_context_default_params()
        // n_ctx: the context window size (tokens). Sourced from config.contextSize.
        contextParam.n_ctx = UInt32(config.contextSize)
        // n_threads / n_threads_batch: CPU thread count for inference and batch processing.
        contextParam.n_threads       = Int32(config.threadCount)
        contextParam.n_threads_batch = Int32(config.threadCount)
        contextParam.n_batch = config.batchSize
        contextParam.n_ubatch = config.batchSize
        contextParam.offload_kqv = true

        let context = LlamaContext(model: model, parameters: contextParam)
        guard let context else {
            print("Could not load context!")
            throw LlamaError.couldNotInitializeContext
        }


        self.maxTokenCount = min(UInt32(model.trainedContextSize()), config.maxTokenCount)
        self.model = context.model
        self.context = context
        self.batch = .init(initialSize: Int32(config.batchSize))
    }

    deinit {
        // llama_backend_free() is not called here because it's global and 
        // destructive. LlamaBackend manages the global state.
    }

    // Expose some backend/system utilities for convenience
    /// Return system info string from the backend.
    static func printSystemInfo() -> String {
        guard let c = llama_print_system_info() else { return "" }
        return String(cString: c)
    }

    /// Expose the underlying context to trusted callers (tests / advanced users).
    /// Access is actor-isolated; callers must `await`.
    func contextHandle() -> LlamaContext { context }

    // MARK: - Testing & Introspection helpers (actor-safe)

    func getLastLogits() -> [Float]? { context.lastLogits() }
    func getEmbeddings() -> [Float]? { context.embeddings(at: -1) }
    func enableEmbeddingsOutput(_ enabled: Bool) { context.setEmbeddingsOutput(enabled) }
    func saveStateData() -> Data { context.saveState() }
    func loadStateData(_ data: Data) -> Bool { context.loadState(data) }
    func setThreads(nThreads: Int32, nThreadsBatch: Int32) { context.setThreads(nThreads: nThreads, nThreadsBatch: nThreadsBatch) }
    func getThreads() -> (Int32, Int32) { (context.nThreads(), context.nThreadsBatch()) }
    func kvMinPosition() -> Int32 { context.memory.minPosition(for: 0) }
    func kvMaxPosition() -> Int32 { context.memory.maxPosition(for: 0) }
    func clearKV() { context.clearKVCache() }

    /// Return the full processed token id sequence (prompt + generated).
    func getProcessedTokenIds() -> [llama_token] { processedTokens }

    func initializeCompletion(messages: [LlamaChatMessage], addAssistant: Bool? = nil) throws {
        let formattedPrompt = model.applyChatTemplate(to: messages, addAssistant: addAssistant)
        try initializeCompletion(text: formattedPrompt)
    }

    private func initializeCompletion(text: String) throws {
        print("attempting to complete \"\(text)\"")

        let tokenList = model.tokenize(text: text, addBos: model.shouldAddBos(), special: true)
        guard tokenList.count < maxTokenCount - 4 else {
            throw LlamaError.contextSizeLimitExeeded
        }

        if tokenList.starts(with: processedTokens) {
            print("### Using cached processing")
            try processPrompt(tokens: Array(tokenList[processedTokens.count...]), startIndex: processedTokens.count)
        } else {
            // Check if we can optimize by only clearing from the divergence point
            let divergenceIndex = findDivergenceIndex(newTokenList: tokenList, processedTokens: processedTokens)
            
            if divergenceIndex > 0 && shouldUsePartialOptimization(divergenceIndex: divergenceIndex, totalProcessed: processedTokens.count) {
                print("### Using partial optimization from position \(divergenceIndex)")
                do {
                    try optimizedReprocessing(newTokenList: tokenList, divergenceIndex: divergenceIndex)
                } catch {
                    print("Partial optimization failed, falling back to full reprocessing")
                    clear()
                    try processPrompt(tokens: tokenList, startIndex: 0)
                }
            } else {
                print("### Full reprocessing required")
                clear()
                try processPrompt(tokens: tokenList, startIndex: 0)
            }
        }
    }

    /// Find the index where the two token lists diverge
    private func findDivergenceIndex(newTokenList: [llama_token], processedTokens: [llama_token]) -> Int {
        let minLength = min(newTokenList.count, processedTokens.count)
        for i in 0..<minLength {
            if newTokenList[i] != processedTokens[i] {
                return i
            }
        }
        return minLength
    }
    
    /// Decide whether to use partial optimization based on the divergence point
    private func shouldUsePartialOptimization(divergenceIndex: Int, totalProcessed: Int) -> Bool {
        // Only use partial optimization if:
        // 1. We have a significant amount of processed tokens (at least 10)
        // 2. The divergence is not too early (at least 50% of tokens match)
        // 3. The divergence is not at the very beginning
        
        guard divergenceIndex > 0 && totalProcessed >= 10 else { return false }
        
        let matchPercentage = Double(divergenceIndex) / Double(totalProcessed)
        return matchPercentage >= 0.5 // At least 50% of tokens match
    }
    
    /// Optimized reprocessing that only clears cache from the divergence point
    private func optimizedReprocessing(newTokenList: [llama_token], divergenceIndex: Int) throws {
        // Clear KV cache from the divergence point onward
        context.clearKVCacheFromPosition(Int32(divergenceIndex))
        
        // Update our internal state
        processedTokens = Array(processedTokens[0..<divergenceIndex])
        currentTokenPosition = Int32(divergenceIndex)
        
        // Process only the tokens from the divergence point onward
        let tokensToProcess = Array(newTokenList[divergenceIndex...])
        try processPrompt(tokens: tokensToProcess, startIndex: divergenceIndex)
    }

    func generateNextToken() throws -> NextToken {
        // Stop before sampling if we've reached the context limit to avoid mutating sampler state
        if currentTokenPosition >= Int32(maxTokenCount) {
            return .endOfString
        }
        let newTokenId = sampler.sample(context: context)

        if model.isEogToken(newTokenId) || currentTokenPosition >= Int32(maxTokenCount) {
            return .endOfString
        }

        batch.reset()
        batch.addToken(newTokenId, at: currentTokenPosition, logits: true)
        processedTokens.append(newTokenId)

        currentTokenPosition += 1
        try context.decode(batch: batch)

        return .token(model.piece(from: newTokenId))
    }

    func updateSamplingConfig(_ config: LlamaSamplingConfig) {
        self.sampler = .init(config: config, model: model)
    }

    private func clear() {
        context.clearKVCache()
        processedTokens = []
        batch = .init(initialSize: Int32(config.batchSize))
    }

    private func processBatch() throws {
        do {
            try context.decode(batch: batch)
        } catch {
            print("llama_decode() failed")
            throw LlamaError.decodingError
        }
    }

    private func processPrompt(tokens: [llama_token], startIndex: Int) throws {
        guard !tokens.isEmpty else {
            // No new tokens to process, but the sampler needs valid logits.
            // Re-decode the last cached token with logits enabled.
            guard !processedTokens.isEmpty else { return }
            let lastPos = Int32(processedTokens.count - 1)
            let lastToken = processedTokens[processedTokens.count - 1]
            batch.reset()
            batch.addToken(lastToken, at: lastPos, logits: true)
            try processBatch()
            return
        }
        batch.reset()

        for i in 0..<tokens.count {
            let tokenPosition = startIndex + i
            let tokenId = tokens[i]
            // Request logits only for the very last token in the full sequence
            let isLast = (i == tokens.count - 1)
            batch.addToken(tokenId, at: Int32(tokenPosition), logits: isLast)
            processedTokens.append(tokenId)
            if batch.size == config.batchSize {
                // If the last token happens to land exactly at the batch boundary,
                // we already marked it with logits=true above.
                try processBatch()
                if !isLast {
                    batch.reset()
                }
            }
        }

        // Decode any remaining tokens in the batch that weren't flushed in the loop
        if batch.size > 0 {
            try processBatch()
        }

        currentTokenPosition = Int32(processedTokens.count)
    }
}
