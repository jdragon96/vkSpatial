#include "Realsense/RealSenseD435Recorder.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <utility>

namespace Realsense {
    namespace {
        namespace fs = std::filesystem;

        std::string FormatDepthFrameFilename(int frameIndex) {
            char name[32];
            std::snprintf(name, sizeof(name), "depth_%04d.bin", frameIndex);
            return std::string(name);
        }

        void WriteIntrinsics(const fs::path &path, const Realsense::CameraIntrinsics &intrinsics) {
            std::ofstream file(path);
            if (!file.is_open())
                throw std::runtime_error("Realsense::RealSenseD435Recorder: cannot write " + path.string());
            file << std::setprecision(std::numeric_limits<float>::max_digits10) << intrinsics.fx
                 << ' ' << intrinsics.fy << ' ' << intrinsics.cx << ' ' << intrinsics.cy << ' '
                 << intrinsics.width << ' ' << intrinsics.height << ' ' << intrinsics.depthScale
                 << ' ' << intrinsics.stereoBaselineMeters << '\n';
        }

        Realsense::CameraIntrinsics ReadIntrinsics(const fs::path &path) {
            std::ifstream file(path);
            if (!file.is_open())
                throw std::runtime_error("Realsense::RealSenseD435Recorder: cannot open " +
                                         path.string());

            Realsense::CameraIntrinsics intrinsics;
            file >> intrinsics.fx >> intrinsics.fy >> intrinsics.cx >> intrinsics.cy >>
                    intrinsics.width >> intrinsics.height;
            if (!file)
                throw std::runtime_error("Realsense::RealSenseD435Recorder: malformed " +
                                         path.string());
            if (intrinsics.width <= 0 || intrinsics.height <= 0)
                throw std::runtime_error(
                        "Realsense::RealSenseD435Recorder: non-positive width/height in " +
                        path.string());

            if (!(file >> intrinsics.depthScale) || !(intrinsics.depthScale > 0.0f))
                intrinsics.depthScale = 0.001f;
            if (!(file >> intrinsics.stereoBaselineMeters) ||
                !(intrinsics.stereoBaselineMeters > 0.0f))
                intrinsics.stereoBaselineMeters = 0.05f;

            return intrinsics;
        }

        std::vector<std::string> ListDepthFrameFiles(const fs::path &directory) {
            std::vector<std::string> paths;
            for (const fs::directory_entry &entry: fs::directory_iterator(directory)) {
                if (!entry.is_regular_file()) continue;
                const fs::path &entryPath = entry.path();
                if (entryPath.filename().string().rfind("depth_", 0) == 0 &&
                    entryPath.extension() == ".bin")
                    paths.push_back(entryPath.string());
            }
            std::sort(paths.begin(), paths.end());
            return paths;
        }
    }

    RealSenseD435Recorder::RealSenseD435Recorder(std::string directory)
        : m_intrinsics(ReadIntrinsics(fs::path(directory) / "intrinsics.txt")),
          m_framePaths(ListDepthFrameFiles(directory)) {
        const std::uintmax_t expectedBytes = std::uintmax_t(m_intrinsics.width) *
                                             std::uintmax_t(m_intrinsics.height) *
                                             sizeof(std::uint16_t);
        for (const std::string &framePath: m_framePaths) {
            std::error_code errorCode;
            const std::uintmax_t actualBytes = fs::file_size(framePath, errorCode);
            if (errorCode || actualBytes != expectedBytes)
                throw std::runtime_error(
                        "Realsense::RealSenseD435Recorder: truncated or malformed recording: " +
                        framePath);
        }
    }

    bool RealSenseD435Recorder::Grab(Realsense::DepthFrame &out) {
        if (m_device) return record(out);

        out = Realsense::DepthFrame{};
        if (m_nextFrame >= int(m_framePaths.size())) return false;

        const std::string &framePath = m_framePaths[std::size_t(m_nextFrame)];
        std::ifstream frameFile(framePath, std::ios::binary);
        if (!frameFile.is_open())
            throw std::runtime_error("Realsense::RealSenseD435Recorder: cannot open " + framePath);

        const std::size_t sampleCount =
                std::size_t(m_intrinsics.width) * std::size_t(m_intrinsics.height);
        m_frame.assign(sampleCount, 0);
        frameFile.read(reinterpret_cast<char *>(m_frame.data()),
                       std::streamsize(sampleCount * sizeof(std::uint16_t)));
        if (!frameFile)
            throw std::runtime_error("Realsense::RealSenseD435Recorder: short read: " + framePath);

        out.rawZ16 = m_frame.data();
        ++m_nextFrame;
        return true;
    }

    RealSenseD435Recorder::RealSenseD435Recorder(std::unique_ptr<Realsense::IDepthProvider> device,
                                                 std::string directory)
        : m_device(std::move(device)), m_directory(std::move(directory)) {
        if (!m_device)
            throw std::invalid_argument(
                    "Realsense::RealSenseD435Recorder: recording needs a device to record from");
        if (fs::exists(m_directory) && !ListDepthFrameFiles(m_directory).empty())
            throw std::runtime_error("Realsense::RealSenseD435Recorder: " + m_directory +
                                     " already holds a recording. Point --record at a different "
                                     "directory to keep it, or move it aside first.");
    }

    bool RealSenseD435Recorder::record(Realsense::DepthFrame &out) {
        if (!m_device->Grab(out)) return false;
        if (!out.rawZ16)
            throw std::runtime_error("Realsense::RealSenseD435Recorder: the wrapped provider "
                                     "returned a frame with no Z16 image");

        const Realsense::CameraIntrinsics &intrinsics = m_device->Intrinsics();

        if (m_writtenFrameCount == 0) {
            fs::create_directories(m_directory);
            WriteIntrinsics(fs::path(m_directory) / "intrinsics.txt", intrinsics);
        }

        const fs::path framePath =
                fs::path(m_directory) / FormatDepthFrameFilename(m_writtenFrameCount);
        std::ofstream frameFile(framePath, std::ios::binary);
        if (!frameFile.is_open())
            throw std::runtime_error("Realsense::RealSenseD435Recorder: cannot write " +
                                     framePath.string());

        const std::size_t sampleCount = std::size_t(intrinsics.width) * std::size_t(intrinsics.height);
        frameFile.write(reinterpret_cast<const char *>(out.rawZ16),
                        std::streamsize(sampleCount * sizeof(std::uint16_t)));
        if (!frameFile)
            throw std::runtime_error("Realsense::RealSenseD435Recorder: write failed for " +
                                     framePath.string());

        ++m_writtenFrameCount;
        return true;
    }
}
