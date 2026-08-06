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

    // A 3-plane corner (constrains all 6 DoF for point-to-plane) -- same fixture shape as
    // GpuIcp.SolveMatchesCpuOnCorner (test_gpuIcp.cpp) and Icp.RecoversKnownTransform (test_icp.cpp),
    // used here to build a hand-made ModelSnapshot + Frame that exercise GpuIcpTracker::Track directly,
    // deterministically -- no async Pipeline/thread-race involved.
    struct Corner {
        std::vector<Vector3f> pts, nrm;
    };

    Corner makeCorner() {
        Corner c;
        auto addPlane = [&](const Vector3f &o, const Vector3f &u, const Vector3f &v, const Vector3f &n) {
            for (int i = -10; i <= 10; ++i)
                for (int j = -10; j <= 10; ++j) {
                    c.pts.push_back(o + u * (i * 0.03f) + v * (j * 0.03f));
                    c.nrm.push_back(n);
                }
        };
        addPlane({0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1});
        addPlane({0, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0});
        addPlane({0, 0, 0}, {1, 0, 0}, {0, 0, 1}, {0, 1, 0});
        return c;
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

// End-to-end: the GPU point-to-plane ICP tracker (registered as "icp") runs on its own lazily-created
// Context inside the ICP thread, alongside the Integration thread's Context -- two live GPU contexts at
// once. Also proves "icp-cpu" (the pre-existing CPU tracker) is still registered under its new name.
TEST(Pipeline, GpuIcpTrackerRuns) {
    const FrameDir frames(3);
    ep::Pipeline::Config cfg = makeConfig(frames.files, 0.0);
    ep::Pipeline pipe(std::move(cfg), ep::TrackerRegistry::Default().Create("icp"));
    ASSERT_NE(ep::TrackerRegistry::Default().Create("icp"), nullptr);
    ASSERT_NE(ep::TrackerRegistry::Default().Create("icp-cpu"), nullptr);
    pipe.Start();
    ASSERT_TRUE(waitProcessed(pipe, 2)) << "gpu-icp pipeline did not integrate frames";
    pipe.CheckErrors();
    EXPECT_NE(pipe.LatestModel(), nullptr);
    pipe.Stop();
}

// Direct, deterministic exercise of GpuIcpTracker::Track -- no async Pipeline, so no race with
// IntegrationThread's first Publish (Pipeline.GpuIcpTrackerRuns above can be racily satisfied by the
// tracker's model==nullptr early-return alone, with 3 frames draining before any model exists; this
// test instead hand-builds a ModelSnapshot + Frame and calls Track() straight through the base Tracker
// interface, so the crop, the lazy Context/GpuPointToPlaneIcp construction, and Solve() are all
// actually exercised). Model = an unperturbed 3-plane corner (constrains all 6 DoF); frame = that same
// corner moved by a small known SE(3) perturbation (so frame plays the role of "src" in
// GpuIcp.SolveMatchesCpuOnCorner). Track should recover ~perturb^-1.
TEST(Pipeline, GpuIcpTrackerRecoversPerturbation) {
    const Corner corner = makeCorner();

    ep::ModelSnapshot model;
    model.entries.reserve(corner.pts.size());
    for (std::size_t i = 0; i < corner.pts.size(); ++i) {
        ep::AdvancedEntry e{};
        e.center = corner.pts[i];
        e.normal = corner.nrm[i];
        model.entries.push_back(e);
    }

    Eigen::Isometry3f perturb = Eigen::Isometry3f::Identity();
    perturb.translate(Vector3f(0.02f, -0.015f, 0.01f));
    perturb.rotate(Eigen::AngleAxisf(0.03f, Vector3f::UnitZ()));

    ep::Frame frame;
    frame.pts.reserve(corner.pts.size());
    frame.nrm.reserve(corner.pts.size());
    for (std::size_t i = 0; i < corner.pts.size(); ++i) {
        frame.pts.push_back(perturb * corner.pts[i]);
        frame.nrm.push_back(perturb.rotation() * corner.nrm[i]);
    }

    const std::unique_ptr<ep::Tracker> tracker = ep::TrackerRegistry::Default().Create("icp");
    ASSERT_NE(tracker, nullptr);

    const ep::TrackingResult r = tracker->Track(frame, &model, Eigen::Isometry3f::Identity());

    ASSERT_TRUE(r.valid) << "GPU ICP tracker did not converge on a well-constrained corner fixture";
    EXPECT_GT(r.inliers, 0u);

    // r.pose should recover ~perturb^-1 (aligns the perturbed frame back onto the model). Same
    // T*known - I convergence check as Icp.RecoversKnownTransform (test_icp.cpp), same tolerance.
    const Eigen::Matrix4f err = r.pose.matrix() * perturb.matrix() - Eigen::Matrix4f::Identity();
    EXPECT_LT(err.norm(), 5e-3f) << "pose:\n" << r.pose.matrix() << "\nperturb:\n" << perturb.matrix();
}

// The model-crop's exclusion branch: entries all sit far outside the frame's AABB + maxCorrDist
// margin, so fewer than 3 survive the crop and Track must return the prior pose UNCHANGED (nothing
// local to align to), never touching Solve.
TEST(Pipeline, GpuIcpTrackerCropExcludesFarModel) {
    const Corner corner = makeCorner(); // near-origin frame -> a small AABB

    ep::Frame frame;
    frame.pts = corner.pts;
    frame.nrm = corner.nrm;

    ep::ModelSnapshot farModel;
    for (int i = 0; i < 5; ++i) {
        ep::AdvancedEntry e{};
        e.center = Vector3f(100.0f + i * 0.1f, 100.0f, 100.0f); // far outside AABB+0.1 margin
        e.normal = Vector3f(0, 0, 1);
        farModel.entries.push_back(e);
    }

    const std::unique_ptr<ep::Tracker> tracker = ep::TrackerRegistry::Default().Create("icp");
    ASSERT_NE(tracker, nullptr);

    Eigen::Isometry3f prior = Eigen::Isometry3f::Identity();
    prior.translate(Vector3f(0.5f, -0.3f, 0.2f));
    prior.rotate(Eigen::AngleAxisf(0.1f, Vector3f::UnitY()));

    const ep::TrackingResult r = tracker->Track(frame, &farModel, prior);

    EXPECT_FALSE(r.valid);
    EXPECT_TRUE(r.pose.matrix().isApprox(prior.matrix(), 1e-6f))
            << "pose:\n" << r.pose.matrix() << "\nprior:\n" << prior.matrix();
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
