import Foundation

enum TestProgress {
    static func run(_ name: String, operation: () throws -> Void) throws {
        let start = CFAbsoluteTimeGetCurrent()
        print("[TEST START] \(name)")
        do {
            try operation()
            print("[TEST PASS] \(name) (\(formatDuration(since: start)))")
        } catch {
            print("[TEST FAIL] \(name) (\(formatDuration(since: start)))")
            throw error
        }
    }

    static func run(_ name: String, operation: () async throws -> Void) async throws {
        let start = CFAbsoluteTimeGetCurrent()
        print("[TEST START] \(name)")
        do {
            try await operation()
            print("[TEST PASS] \(name) (\(formatDuration(since: start)))")
        } catch {
            print("[TEST FAIL] \(name) (\(formatDuration(since: start)))")
            throw error
        }
    }

    @MainActor
    static func runOnMainActor(_ name: String, operation: @MainActor () async throws -> Void) async throws {
        let start = CFAbsoluteTimeGetCurrent()
        print("[TEST START] \(name)")
        do {
            try await operation()
            print("[TEST PASS] \(name) (\(formatDuration(since: start)))")
        } catch {
            print("[TEST FAIL] \(name) (\(formatDuration(since: start)))")
            throw error
        }
    }

    static func skipped(_ name: String, reason: String) {
        print("[TEST SKIP] \(name) - \(reason)")
    }

    private static func formatDuration(since start: CFAbsoluteTime) -> String {
        String(format: "%.2fs", CFAbsoluteTimeGetCurrent() - start)
    }
}
