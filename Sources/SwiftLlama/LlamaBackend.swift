import Foundation
import llama

public enum LlamaBackend {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var isInitialized = false

    /// Initialize the llama + ggml backend. Safe to call multiple times.
    public static func initialize() {
        lock.lock()
        defer { lock.unlock() }
        if !isInitialized {
            llama_backend_init()
            isInitialized = true
        }
    }
    
    // shutdown() is removed because llama_backend_free() is global and 
    // destructive to all active contexts/models. In a long-lived process 
    // like a test runner, it's safer to let the OS clean up at exit.

    /// Whether mmap/mlock/gpu offload/rpc are supported by the compiled library.
    public static var supportsMmap: Bool { llama_supports_mmap() }
    public static var supportsMlock: Bool { llama_supports_mlock() }
    public static var supportsGpuOffload: Bool { llama_supports_gpu_offload() }
    public static var supportsRpc: Bool { llama_supports_rpc() }
    /// Maximum devices and parallel sequences
    public static var maxDevices: Int { Int(llama_max_devices()) }
    public static var maxParallelSequences: Int { Int(llama_max_parallel_sequences()) }

    /// Initialize NUMA with a given strategy.
    public static func numaInit(_ strategy: ggml_numa_strategy) { llama_numa_init(strategy) }

    /// Microsecond timer from llama.cpp
    public static func timeMicros() -> Int64 { llama_time_us() }

    /// Return system info string provided by llama.cpp
    public static func systemInfo() -> String {
        guard let c = llama_print_system_info() else { return "" }
        return String(cString: c)
    }

    /// Attach the library-managed auto threadpool to a context.
    public static func attachAutoThreadpool(to context: LlamaContext) {
        llama_attach_threadpool(context.contextPointer, nil, nil)
    }

    /// Detach any threadpools from the context.
    public static func detachThreadpool(from context: LlamaContext) {
        llama_detach_threadpool(context.contextPointer)
    }
}

