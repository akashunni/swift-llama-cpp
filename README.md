# swift-llama-cpp

Run any LLM locally on iOS or MacOS. Powered by [llama.cpp](https://github.com/ggml-org/llama.cpp)

To browse upstream C/C++ source at the same revision as the pinned xcframework, see [Reference/README.md](Reference/README.md).

## Coverage

This wrapper covers:
- Model loading (single file and splits), save, metadata, size, params, encoder/decoder flags
- Vocab API (token text, score, attrs, special tokens), tokenize/detokenize
- Context creation/free, threads, embeddings/attention/warmup toggles
- Memory API (sequence remove/copy/keep/add/div/min/max/canShift)
- Encode/decode, logits/embeddings getters, synchronize
- State/session save-load, per-sequence state
- Chat templates (apply and list built-ins)
- Sampler chain (grammar, top-k, top-p, temp, penalties, dist), sample/accept/reset/clone
- LoRA adapter load/apply/remove/clear, control vectors
- Backend init/free, capability queries, system info, logging hook

## Basic usage

Here is a quick example of how to use `SwiftLlama` to generate text from a model.

First, make sure you have a GGUF model file accessible in your project. You can download models from sources like [Hugging Face](https://huggingface.co/models?search=gguf).

```swift
import SwiftLlama
import Foundation

// 1. Get the model URL
// Make sure to add a GGUF model to your project and get its URL.
guard let modelUrl = Bundle.main.url(forResource: "your-model-name", withExtension: "gguf") else {
    print("Model file not found")
    return
}

// 2. Initialize the LlamaService
// This service manages the model and context.
let llamaService = LlamaService(modelUrl: modelUrl, config: .init(batchSize: 256, maxTokenCount: 4096, useGPU: true))

// 3. Prepare your messages
// The conversation history can be provided as an array of messages.
let messages = [
    LlamaChatMessage(role: .system, content: "You are a helpful assistant."),
    LlamaChatMessage(role: .user, content: "Tell me a short story."),
]

// 4. Generate text
// The `streamCompletion` method returns an `AsyncThrowingStream` of tokens.
do {
    let stream = try await llamaService.streamCompletion(of: messages, samplingConfig: .init(temperature: 0.8, seed: 42))
    var generatedText = ""
    for try await token in stream {
        generatedText += token
        print("Generated token: \(token)")
    }
    print("Generated text: \(generatedText)")
} catch {
    print("Error generating text: \(error.localizedDescription)")
}
``` 

## Running Tests

The test suite supports the bundled small test model as well as external GGUF models.

Run the full suite with the bundled test model:

```bash
swift test
```

Run the suite with a specific GGUF:

```bash
SWIFT_LLAMA_TEST_MODEL_PATH="/absolute/path/to/model.gguf" swift test
```

The test helper checks model paths in this order:

1. `SWIFT_LLAMA_TEST_MODEL_PATH`
2. `LLAMA_MODEL_PATH`
3. the bundled test model in `Tests/SwiftLlamaTests/Resources`

If a large model is unstable on GPU, force CPU for the run:

```bash
SWIFT_LLAMA_TEST_MODEL_PATH="/absolute/path/to/model.gguf" SWIFT_LLAMA_TEST_USE_GPU=0 swift test
```

### Common runs

Run only service and structured-output tests against a Llama-family model:

```bash
SWIFT_LLAMA_TEST_MODEL_PATH="/absolute/path/to/llama.gguf" swift test --filter 'LlamaServiceTests|LlamaTypedGrammarTests'
```

Run Gemma-family structured-output tests:

```bash
SWIFT_LLAMA_TEST_MODEL_PATH="/absolute/path/to/gemma.gguf" swift test --filter 'LlamaServiceTests|LlamaTypedGrammarTests'
```

Run low-level context tests separately:

```bash
SWIFT_LLAMA_TEST_MODEL_PATH="/absolute/path/to/model.gguf" swift test --filter LlamaContextTests
```

Run a single suite while debugging a failure:

```bash
SWIFT_LLAMA_TEST_MODEL_PATH="/absolute/path/to/model.gguf" swift test --filter LlamaServiceTests
SWIFT_LLAMA_TEST_MODEL_PATH="/absolute/path/to/model.gguf" swift test --filter LlamaTypedGrammarTests
SWIFT_LLAMA_TEST_MODEL_PATH="/absolute/path/to/model.gguf" swift test --filter LlamaTests
```

Long-running model-backed tests print progress in the terminal:

- `[TEST START] ...`
- `[TEST PASS] ...`
- `[TEST SKIP] ...`
- `[TEST FAIL] ...`

### Notes

- Model-backed tests are designed to be functional and portable, not tuned to one exact model family.
- Some tests are serialized to avoid loading multiple large models at once.
- Context-level tests use a smaller CPU-only setup so larger GGUFs can run on laptops more reliably.
