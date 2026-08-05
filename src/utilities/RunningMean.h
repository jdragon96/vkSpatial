#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

// Thread-safe running average for pipeline liveness/timing: ONE writer (a worker thread) records a
// sample per item; MANY readers (the render thread's HUD) read the mean + count. Lock-free — the
// single writer means the read-modify-write of the mean is never contended, and the render thread
// only ever reads. Count doubling as a liveness signal: a stage whose count keeps rising is alive.
namespace util {

    class RunningMean {
    public:
        void Add(double x) {
            const std::uint64_t n = m_count.load(std::memory_order_relaxed) + 1;
            const double mean = m_mean.load(std::memory_order_relaxed);
            m_mean.store(mean + (x - mean) / double(n), std::memory_order_relaxed);
            m_count.store(n, std::memory_order_release);
        }

        double Mean() const { return m_mean.load(std::memory_order_relaxed); }
        std::uint64_t Count() const { return m_count.load(std::memory_order_acquire); }

    private:
        std::atomic<double> m_mean{0.0};
        std::atomic<std::uint64_t> m_count{0};
    };

    // RAII: records the wall-clock time of its own scope into `mean` when it goes out of scope.
    class ScopedMean {
    public:
        explicit ScopedMean(RunningMean &mean) : m_mean(mean), m_start(Clock::now()) {}
        ~ScopedMean() {
            const double ms =
                    std::chrono::duration<double, std::milli>(Clock::now() - m_start).count();
            m_mean.Add(ms);
        }
        ScopedMean(const ScopedMean &) = delete;
        ScopedMean &operator=(const ScopedMean &) = delete;

    private:
        using Clock = std::chrono::steady_clock;
        RunningMean &m_mean;
        Clock::time_point m_start;
    };

} // namespace util
