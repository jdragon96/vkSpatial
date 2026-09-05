#include "Pipeline/Acquisition/DepthRecording.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace Pipeline {

    namespace {

        namespace fs = std::filesystem;

        std::string FormatDepthFrameFilename(int frameIndex) {
            char name[32];
            std::snprintf(name, sizeof(name), "depth_%04d.bin", frameIndex);
            return std::string(name);
        }

        // Every float printed with enough digits (max_digits10) to read back bit-for-bit.
        void WriteIntrinsics(const fs::path &path, const CameraIntrinsics &intrinsics) {
            std::ofstream file(path);
            if (!file.is_open())
                throw std::runtime_error("DepthRecorder: cannot write " + path.string());
            file << std::setprecision(std::numeric_limits<float>::max_digits10) << intrinsics.fx
                 << ' ' << intrinsics.fy << ' ' << intrinsics.cx << ' ' << intrinsics.cy << ' '
                 << intrinsics.width << ' ' << intrinsics.height << '\n';
        }

        CameraIntrinsics ReadIntrinsics(const fs::path &path) {
            std::ifstream file(path);
            if (!file.is_open())
                throw std::runtime_error("RecordedDepthProvider: cannot open " + path.string());

            CameraIntrinsics intrinsics;
            file >> intrinsics.fx >> intrinsics.fy >> intrinsics.cx >> intrinsics.cy >>
                    intrinsics.width >> intrinsics.height;
            if (!file) throw std::runtime_error("RecordedDepthProvider: malformed " + path.string());
            if (intrinsics.width <= 0 || intrinsics.height <= 0)
                throw std::runtime_error("RecordedDepthProvider: non-positive width/height in " +
                                          path.string());
            return intrinsics;
        }

        // Every depth_*.bin directly inside `directory`, sorted -- the zero-padded %04d name makes
        // lexical order match recording order.
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

    } // namespace

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // DepthRecorder
    ///////////////////////////////////////////////////////////////////////////////////////////////

    DepthRecorder::DepthRecorder(std::unique_ptr<IDepthProvider> device, std::string directory)
        : m_device(std::move(device)), m_directory(std::move(directory)) {
        // Refuse a directory that already holds a capture. Overwriting from index 0 would leave
        // the previous take's higher-numbered frames in place, and RecordedDepthProvider globs
        // every depth_*.bin -- so a 5-frame re-record over a 30-frame take replays as one
        // 30-frame capture with a teleport at frame 5. Same intrinsics, same file sizes, so the
        // size check cannot see it.
        //
        // Checked here rather than on the first frame because a caller that opens a window before
        // streaming -- depth_live_viewer does -- would otherwise build its whole render stack and
        // then throw. Nothing is created yet: the directory and its intrinsics.txt still wait for
        // a frame to actually arrive.
        if (fs::exists(m_directory) && !ListDepthFrameFiles(m_directory).empty())
            // Names the alternative, not the deletion. A capture is minutes of someone's time in
            // front of a sensor and cannot be regenerated from anything on disk; an error message
            // that opens with "remove it" invites destroying exactly that.
            throw std::runtime_error("DepthRecorder: " + m_directory +
                                     " already holds a recording. Point --record at a different "
                                     "directory to keep it, or move it aside first.");
    }

    const CameraIntrinsics &DepthRecorder::Intrinsics() const { return m_device->Intrinsics(); }

    bool DepthRecorder::Grab(DepthFrame &out) {
        if (!m_device->Grab(out)) return false;

        // The file format is float32 metres, so this is the one place that pays for the unpack when
        // the device handed over Z16. It stays float32 on disk: 16-bit would need the scale stored
        // alongside it, and a scale mistake corrupts every reconstruction made from the recording
        // without ever looking wrong.
        EnsureMetres(out, m_device->Intrinsics());

        // On the first frame only: create the directory and write the one intrinsics.txt that
        // describes every frame in it.
        if (m_recordedFrameCount == 0) {
            fs::create_directories(m_directory);
            WriteIntrinsics(fs::path(m_directory) / "intrinsics.txt", m_device->Intrinsics());
        }

        const fs::path framePath =
                fs::path(m_directory) / FormatDepthFrameFilename(m_recordedFrameCount);
        std::ofstream frameFile(framePath, std::ios::binary);
        if (!frameFile.is_open())
            throw std::runtime_error("DepthRecorder: cannot write " + framePath.string());
        frameFile.write(reinterpret_cast<const char *>(out.depth.data()),
                         std::streamsize(out.depth.size() * sizeof(float)));
        if (!frameFile)
            throw std::runtime_error("DepthRecorder: write failed for " + framePath.string());

        ++m_recordedFrameCount;
        return true;
    }

    int DepthRecorder::RecordedFrameCount() const { return m_recordedFrameCount; }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // RecordedDepthProvider
    ///////////////////////////////////////////////////////////////////////////////////////////////

    RecordedDepthProvider::RecordedDepthProvider(std::string directory)
        : m_intrinsics(ReadIntrinsics(fs::path(directory) / "intrinsics.txt")),
          m_framePaths(ListDepthFrameFiles(directory)) {
        // A short read that quietly yielded a partial depth image would corrupt a reconstruction
        // with no symptom, so every recorded frame's size is checked up front, not on first use.
        const std::uintmax_t expectedBytes = std::uintmax_t(m_intrinsics.width) *
                                              std::uintmax_t(m_intrinsics.height) * sizeof(float);
        for (const std::string &framePath: m_framePaths) {
            std::error_code errorCode;
            const std::uintmax_t actualBytes = fs::file_size(framePath, errorCode);
            if (errorCode || actualBytes != expectedBytes)
                throw std::runtime_error(
                        "RecordedDepthProvider: truncated or malformed recording: " + framePath);
        }
    }

    const CameraIntrinsics &RecordedDepthProvider::Intrinsics() const { return m_intrinsics; }

    bool RecordedDepthProvider::Grab(DepthFrame &out) {
        if (m_nextFrame >= int(m_framePaths.size())) return false;

        const std::string &framePath = m_framePaths[std::size_t(m_nextFrame)];
        std::ifstream frameFile(framePath, std::ios::binary);
        if (!frameFile.is_open())
            throw std::runtime_error("RecordedDepthProvider: cannot open " + framePath);

        const std::size_t sampleCount =
                std::size_t(m_intrinsics.width) * std::size_t(m_intrinsics.height);
        out.depth.assign(sampleCount, 0.0f);
        out.rawZ16 = nullptr; // a recording holds metres; there is no device buffer behind it
        out.depthScale = 0.0f;
        frameFile.read(reinterpret_cast<char *>(out.depth.data()),
                        std::streamsize(sampleCount * sizeof(float)));
        if (!frameFile) throw std::runtime_error("RecordedDepthProvider: short read: " + framePath);

        ++m_nextFrame;
        return true;
    }

    int RecordedDepthProvider::FrameCount() const { return int(m_framePaths.size()); }

} // namespace Pipeline
