#include "Engine/Pipeline/Registration/Tracker.h" // Engine::Pipeline::TrackerRegistry
#include "Engine/Pipeline/Pipeline.h"  // Engine::Pipeline::Pipeline / Config / EAcquisitionType

#include "utilities/PointCloudIO.h"

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
namespace ep = Engine::Pipeline;
using Eigen::Vector3f;

namespace {

    // Write a +Z plane patch as an ASCII PLY (with normals) so the File source can load it.
    void writePlanePly(const std::string &path, float z) {
        std::vector<Vector3f> pts, nrm;
        for (int i = -20; i <= 20; ++i)
            for (int j = -20; j <= 20; ++j) {
                pts.emplace_back(i * 0.02f, j * 0.02f, z);
                nrm.emplace_back(0.0f, 0.0f, 1.0f);
            }
        ASSERT_TRUE(util::SavePly(path, pts, nrm));
    }

    // A unique temp dir holding N frame_%04d.ply files (removed by the fixture dtor). `files` is the
    // created path list, in order — exactly what a config-driven File source replays.
    struct FrameDir {
        fs::path dir;
        std::vector<std::string> files;
        explicit FrameDir(int n) {
            dir = fs::temp_directory_path() /
                  ("pipe_test_" + std::string(::testing::UnitTest::GetInstance()
                                                      ->current_test_info()
                                                      ->name()));
            fs::create_directories(dir);
            for (int k = 0; k < n; ++k) {
                char name[32];
                std::snprintf(name, sizeof name, "frame_%04d.ply", k);
                const std::string path = (dir / name).string();
                writePlanePly(path, 0.0f);
                files.push_back(path);
            }
        }
        ~FrameDir() {
            std::error_code ec;
            fs::remove_all(dir, ec);
        }
    };

    // Config-driven File source over `files`, paced at `intervalMs` (0 = as fast as consumed).
    ep::Pipeline::Config makeConfig(const std::vector<std::string> &files, double intervalMs) {
        ep::Pipeline::Config cfg;
        cfg.map.baseVoxel = 0.05f;
        cfg.map.truncation = 0.15f;
        cfg.acquisition.type = ep::EAcquisitionType::File;
        cfg.acquisition.framePaths = files;
        cfg.acquisition.intervalMs = intervalMs;
        return cfg; // no densityFrames -> base-only integration (still yields entries)
    }

    std::unique_ptr<ep::Tracker> identity() {
        return ep::TrackerRegistry::Default().Create("identity");
    }

    bool waitProcessed(const ep::Pipeline &p, int target, int seconds = 60) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
        while (p.ProcessedFrame() < target && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return p.ProcessedFrame() >= target;
    }

} // namespace

// End-to-end: a config-driven File source streams PLYs through Reconstruction -> ICP(identity) ->
// Integration, and the pipeline publishes a non-empty model. Proves the module works off the render
// thread and that Config::source builds the File strategy.
TEST(Pipeline, FileSourceProducesModel) {
    const FrameDir frames(3);

    ep::Pipeline pipe(makeConfig(frames.files, 0.0), identity());
    pipe.Start();

    ASSERT_TRUE(waitProcessed(pipe, 2)) << "pipeline did not integrate all frames";
    pipe.CheckErrors(); // rethrow any worker exception
    const auto snap = pipe.LatestModel();
    ASSERT_NE(snap, nullptr);
    EXPECT_GE(snap->processedFrame, 2);
    EXPECT_GT(snap->entries.size(), 100u);
    // First-seen frame is carried per voxel (AdvancedEntry::firstFrame, GPU-stamped), not in parallel
    // isNew/firstFrame arrays. Every entry must carry a valid frame within the processed range.
    for (const auto &e: snap->entries) {
        EXPECT_GE(e.firstFrame, 0);
        EXPECT_LE(e.firstFrame, snap->processedFrame);
    }
    pipe.Stop();
}

// The paced File source (intervalMs) still delivers every frame; Type() reports File.
TEST(Pipeline, PacedFileSourceDeliversAllFrames) {
    const FrameDir frames(3);

    ep::Pipeline pipe(makeConfig(frames.files, 15.0), identity()); // ~66 fps pacing
    EXPECT_EQ(pipe.Type(), ep::EAcquisitionType::File);
    pipe.Start();

    ASSERT_TRUE(waitProcessed(pipe, 2));
    pipe.CheckErrors();
    EXPECT_NE(pipe.LatestModel(), nullptr);
    pipe.Stop();
}

// Reconfigure (a runtime option toggle in the viewer) rebuilds the worker stages IN PLACE and keeps
// producing models. Regression: each stage's destructor runs Stop() -> Interrupt() -> m_comm.*.Close(),
// so the old stages MUST be destroyed before the comm they reference -- rebuilding comm-first was a
// use-after-free that crashed on the first toggle. Repeated to exercise back-to-back rebuilds.
TEST(Pipeline, ReconfigureRebuildsCleanly) {
    const FrameDir frames(5);
    ep::Pipeline pipe(makeConfig(frames.files, 0.0), identity());
    pipe.Start();
    ASSERT_TRUE(waitProcessed(pipe, 1)) << "pipeline did not start";

    for (int i = 0; i < 3; ++i) {
        ep::Pipeline::Config cfg = makeConfig(frames.files, 0.0);
        cfg.map.submap = (i % 2 == 0); // flip an option, as a UI toggle would
        pipe.Reconfigure(std::move(cfg), identity());
        ASSERT_TRUE(waitProcessed(pipe, 1)) << "no model after Reconfigure #" << i;
        pipe.CheckErrors();                 // rethrow any worker exception from the rebuild
        EXPECT_NE(pipe.LatestModel(), nullptr);
    }
    pipe.Stop();
}

// Stop() before the source is exhausted must not hang or crash (interruptible shutdown).
TEST(Pipeline, StopIsCleanMidStream) {
    const FrameDir frames(50);

    ep::Pipeline pipe(makeConfig(frames.files, 50.0), identity()); // slow pacing -> Stop mid-stream
    pipe.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(80)); // let a couple frames through
    pipe.Stop();                                                // must return promptly (no hang)
    pipe.CheckErrors();
    SUCCEED();
}
