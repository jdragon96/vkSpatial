#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Lightweight wall-clock timing for pipeline profiling: a monotonic Stopwatch, a StageProfiler that
// aggregates per-named-stage samples (count / total / avg / min / max / %), and a ScopedStageTimer
// that records one sample for the block it lives in. Header-only, single-threaded (per-frame
// instrumentation), no dependency on the render/compute stack.
namespace util {

    // Monotonic stopwatch (steady_clock). ElapsedMs() is the time since construction/Reset.
    class Stopwatch {
    public:
        Stopwatch() : m_start(Clock::now()) {}
        void Reset() { m_start = Clock::now(); }
        double ElapsedMs() const {
            return std::chrono::duration<double, std::milli>(Clock::now() - m_start).count();
        }

    private:
        using Clock = std::chrono::steady_clock;
        Clock::time_point m_start;
    };

    // Accumulates timing samples keyed by stage name (first-seen order preserved for the report).
    class StageProfiler {
    public:
        void Add(std::string_view stage, double ms) {
            const std::string key(stage);
            const auto it = m_stats.find(key);
            Stat &s = (it == m_stats.end()) ? newStat(key) : it->second;
            ++s.count;
            s.totalMs += ms;
            s.lastMs = ms;
            s.minMs = std::min(s.minMs, ms);
            s.maxMs = std::max(s.maxMs, ms);
        }

        std::size_t Count(std::string_view stage) const {
            const Stat *s = find(stage);
            return s ? s->count : 0;
        }
        double LastMs(std::string_view stage) const {
            const Stat *s = find(stage);
            return s ? s->lastMs : 0.0;
        }
        double TotalMs(std::string_view stage) const {
            const Stat *s = find(stage);
            return s ? s->totalMs : 0.0;
        }
        double AvgMs(std::string_view stage) const {
            const Stat *s = find(stage);
            return (s && s->count) ? s->totalMs / double(s->count) : 0.0;
        }

        bool Empty() const { return m_order.empty(); }
        void Reset() {
            m_stats.clear();
            m_order.clear();
        }

        // Formatted table: stage, count, total, avg, min, max, and % of the summed total.
        std::string Report(std::string_view title = "timing") const {
            double grand = 0.0;
            for (const std::string &n : m_order) grand += m_stats.at(n).totalMs;
            std::ostringstream os;
            char line[256];
            std::snprintf(line, sizeof line, "== %.*s (%zu stages, total %.2f ms) ==\n",
                          int(title.size()), title.data(), m_order.size(), grand);
            os << line;
            std::snprintf(line, sizeof line, "  %-14s %7s %11s %10s %10s %10s %7s\n", "stage",
                          "count", "total(ms)", "avg(ms)", "min(ms)", "max(ms)", "%");
            os << line;
            for (const std::string &n : m_order) {
                const Stat &s = m_stats.at(n);
                const double avg = s.count ? s.totalMs / double(s.count) : 0.0;
                const double pct = grand > 0.0 ? 100.0 * s.totalMs / grand : 0.0;
                std::snprintf(line, sizeof line, "  %-14s %7zu %11.2f %10.3f %10.3f %10.3f %6.1f%%\n",
                              n.c_str(), s.count, s.totalMs, avg, s.minMs, s.maxMs, pct);
                os << line;
            }
            return os.str();
        }

    private:
        struct Stat {
            std::size_t count = 0;
            double totalMs = 0.0;
            double lastMs = 0.0;
            double minMs = std::numeric_limits<double>::infinity();
            double maxMs = 0.0;
        };
        Stat &newStat(const std::string &key) {
            m_order.push_back(key);
            return m_stats[key];
        }
        const Stat *find(std::string_view stage) const {
            const auto it = m_stats.find(std::string(stage));
            return it == m_stats.end() ? nullptr : &it->second;
        }

        std::unordered_map<std::string, Stat> m_stats;
        std::vector<std::string> m_order;
    };

    // RAII: times its own lifetime and records one sample under `stage` when it goes out of scope.
    class ScopedStageTimer {
    public:
        ScopedStageTimer(StageProfiler &profiler, std::string_view stage)
            : m_profiler(profiler), m_stage(stage) {}
        ~ScopedStageTimer() { m_profiler.Add(m_stage, m_sw.ElapsedMs()); }
        ScopedStageTimer(const ScopedStageTimer &) = delete;
        ScopedStageTimer &operator=(const ScopedStageTimer &) = delete;

    private:
        StageProfiler &m_profiler;
        std::string m_stage;
        Stopwatch m_sw;
    };

} // namespace util
