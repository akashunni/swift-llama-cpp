//
//  LlamaConfig.swift
//  LlamaSwift
//
//  Created by Piotr Gorzelany on 05/11/2024.
//

import Foundation

public struct LlamaConfig: Equatable, Sendable {
    public let batchSize: UInt32
    public let maxTokenCount: UInt32
    public let useGPU: Bool

    // MARK: - GPU offload (n_gpu_layers)

    /// Number of transformer layers to offload to GPU (Metal).
    /// - 0   = CPU only (default, backward-compatible).
    /// - 999 = attempt to offload all layers to GPU.
    /// Has no effect on the simulator (Metal is unavailable there).
    public let gpuLayerCount: Int

    // MARK: - Threading (n_threads)

    /// Number of CPU threads to use for inference.
    /// Defaults to the number of logical processors on the device.
    /// Reducing this can lower power draw; increasing beyond core count rarely helps.
    public let threadCount: Int

    // MARK: - Context window (n_ctx)

    /// Maximum number of tokens in the context window (prompt + generation).
    /// Larger values consume proportionally more memory.
    /// Defaults to 2048 for backward compatibility.
    public let contextSize: Int

    public init(
        batchSize: UInt32,
        maxTokenCount: UInt32,
        useGPU: Bool = true,
        gpuLayerCount: Int = 999,                                        // default 999 → offload all layers to GPU
        threadCount: Int = ProcessInfo.processInfo.processorCount,        // use all logical CPUs
        contextSize: Int = 2048                                           // classic llama.cpp default
    ) {
        self.batchSize = batchSize
        self.maxTokenCount = maxTokenCount
        self.useGPU = useGPU
        self.gpuLayerCount = gpuLayerCount
        self.threadCount = threadCount
        self.contextSize = contextSize
    }
}
