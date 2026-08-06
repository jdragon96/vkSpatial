#pragma once

#include "Engine/Pipeline/Reconstruction/FrameLoader.h" // EstimateCameraHint
#include "Engine/Pipeline/Reconstruction/ReconstructionSource.h"

#include "utilities/PointCloudIO.h" // util::LoadPly

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace Engine::Pipeline {

    struct FileSourceConfig {
        std::vector<std::string> files;
        double intervalMs = 0.0;
        bool loop = false;
    };

    class FileFrameSource : public IFrameSource {
    public:
        explicit FileFrameSource(FileSourceConfig config) : m_cfg(std::move(config)) {}

        EAcquisitionType Type() const override { return EAcquisitionType::File; }
        const char *Name() const override { return "file"; }

        void Open() override {
            m_cursor = 0;
            m_hasLast = false;
            std::lock_guard<std::mutex> lock(m_mutex);
            m_closed = false;
        }

        bool Next(Frame &out) override {
            for (;;) {
                if (m_cursor >= m_cfg.files.size()) {
                    if (!m_cfg.loop || m_cfg.files.empty()) return false;
                    m_cursor = 0; // loop back to the start
                }
                if (!waitForNextSlot()) return false;

                Frame fr;
                std::string filePath = m_cfg.files[m_cursor++];
                if (!util::LoadPly(filePath, fr.pts, fr.nrm) || fr.nrm.size() != fr.pts.size())
                    continue;
                fr.cam = EstimateCameraHint(fr.pts, fr.nrm);
                m_lastEmit = Clock::now();
                m_hasLast = true;
                out = std::move(fr);
                return true;
            }
        }

        void Close() override {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_closed = true;
            }
            m_cv.notify_all();
        }

    private:
        using Clock = std::chrono::steady_clock;

        bool waitForNextSlot() {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (m_closed) return false;
            if (m_cfg.intervalMs <= 0.0 || !m_hasLast) return true;
            const auto due = m_lastEmit + std::chrono::duration_cast<Clock::duration>(
                                                  std::chrono::duration<double, std::milli>(m_cfg.intervalMs));
            m_cv.wait_until(lock, due, [&] { return m_closed; }); // early wake on Close()
            return !m_closed;
        }

        FileSourceConfig m_cfg;
        std::size_t m_cursor = 0;

        std::mutex m_mutex;
        std::condition_variable m_cv;
        bool m_closed = false;
        bool m_hasLast = false;
        Clock::time_point m_lastEmit{};
    };

} // namespace Engine::Pipeline
