#include "Pipeline/Registration/Tracker.h" // Pipeline::TrackerRegistry
#include "Pipeline/Pipeline.h"  // Pipeline::Pipeline / Config / EAcquisitionType
#include "Pipeline/CommunicationModule.h"       // Pipeline::CommunicationModule
#include "Pipeline/Registration/RegistrationThread.h"
#include "Pipeline/Reconstruction/ReconstructionThread.h" // MakeAcquisitionSource
#include "Pipeline/Reconstruction/DepthCameraFrameSource.h"
#include "Pipeline/Reconstruction/DepthRecording.h"

#include "utilities/PointCloudIO.h"

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
namespace ep = Pipeline;
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
    // First-seen frame is carried per voxel (TSDFVoxel::firstFrame, GPU-stamped), not in parallel
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
        TSDFVoxel e{};
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
        TSDFVoxel e{};
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

// Moving-camera regression: unlike GpuIcpTrackerRecoversPerturbation (which keeps `priorPose` at
// Identity and moves the FRAME instead, so frame.pts and model.center happen to sit in the same
// numeric region regardless of any world-vs-sensor-frame mixup), this test gives Track() a `priorPose`
// with a REAL translation away from the origin -- as a moving camera actually would have. The model is
// a world-frame corner surface sitting at that same non-origin location; frame.pts is the SENSOR-LOCAL
// expression of that surface (i.e. what the camera would actually capture, near its own origin). If the
// crop AABB is computed from raw (sensor-frame) frame.pts instead of priorPose-transformed (world-frame)
// points, it lands nowhere near the world model -> <3 survivors -> Track returns the prior unchanged
// (invalid). GpuIcpTracker.cpp:Track must transform by priorPose before cropping.
TEST(Pipeline, GpuIcpTrackerRecoversMovingCameraPose) {
    const Corner corner = makeCorner();

    // The world model sits far from the origin -- a real map region a moving camera has driven to.
    const Vector3f worldOffset(4.0f, -3.0f, 2.5f);
    ep::ModelSnapshot model;
    model.entries.reserve(corner.pts.size());
    for (std::size_t i = 0; i < corner.pts.size(); ++i) {
        TSDFVoxel e{};
        e.center = corner.pts[i] + worldOffset;
        e.normal = corner.nrm[i];
        model.entries.push_back(e);
    }

    // The TRUE camera pose (sensor -> world): a real translation to the model's location + a modest
    // rotation -- this is what Track's crop must transform frame.pts by before comparing to the
    // world-frame model.
    Eigen::Isometry3f truePose = Eigen::Isometry3f::Identity();
    truePose.translate(worldOffset);
    truePose.rotate(Eigen::AngleAxisf(0.05f, Vector3f::UnitY()));

    // A small extra perturbation on top of truePose so ICP has an actual residual to solve, not just a
    // trivial zero-correction (same magnitude as GpuIcpTrackerRecoversPerturbation's `perturb`).
    Eigen::Isometry3f extraPerturb = Eigen::Isometry3f::Identity();
    extraPerturb.translate(Vector3f(0.02f, -0.015f, 0.01f));
    extraPerturb.rotate(Eigen::AngleAxisf(0.03f, Vector3f::UnitZ()));
    const Eigen::Isometry3f priorPose = truePose * extraPerturb;

    // Sensor-local frame.pts: the world model surface expressed in the (true) sensor frame, so that
    // truePose * frame.pts[i] == model.entries[i].center exactly.
    ep::Frame frame;
    frame.pts.reserve(corner.pts.size());
    frame.nrm.reserve(corner.pts.size());
    for (std::size_t i = 0; i < corner.pts.size(); ++i) {
        frame.pts.push_back(truePose.inverse() * model.entries[i].center);
        frame.nrm.push_back(truePose.inverse().rotation() * corner.nrm[i]);
    }

    const std::unique_ptr<ep::Tracker> tracker = ep::TrackerRegistry::Default().Create("icp");
    ASSERT_NE(tracker, nullptr);

    const ep::TrackingResult r = tracker->Track(frame, &model, priorPose);

    ASSERT_TRUE(r.valid) << "GPU ICP tracker did not converge for a moving camera (world-frame model, "
                             "non-identity priorPose) -- likely the sensor-frame-vs-world-frame crop bug";
    EXPECT_GT(r.inliers, 0u);

    // r.pose should recover ~truePose (the transform that exactly aligns frame.pts back onto the
    // world-frame model).
    const Eigen::Matrix4f err = r.pose.matrix() - truePose.matrix();
    EXPECT_LT(err.norm(), 5e-3f) << "pose:\n" << r.pose.matrix() << "\ntruePose:\n" << truePose.matrix();
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

// ---------------------------------------------------------------------------------------------
// Registration-quality plan, Task 5 (Tier 3): constant-velocity motion model in
// RegistrationThread::Run.
namespace {

    // A Tracker that ALWAYS reports success on a known, pre-scripted pose sequence -- ignoring
    // whatever prior it's given. This test is about what prior RegistrationThread::Run COMPUTES and
    // hands to Track (the motion model), not about whether ICP itself converges, so the tracker just
    // needs to behave like a perfect one: it records every priorPose it receives, in call order, into
    // a shared vector the test inspects after the thread has been Stop()'d (joined).
    class RecordingStraightLineTracker : public ep::Tracker {
    public:
        RecordingStraightLineTracker(std::vector<Eigen::Isometry3f> truePoses,
                                     std::shared_ptr<std::vector<Eigen::Isometry3f>> recordedPriors)
            : m_truePoses(std::move(truePoses)), m_recordedPriors(std::move(recordedPriors)) {}

        const char *Name() const override { return "recording-straight-line"; }

        ep::TrackingResult Track(const ep::Frame &, const ep::ModelSnapshot *,
                                 const Eigen::Isometry3f &priorPose) override {
            m_recordedPriors->push_back(priorPose);
            ep::TrackingResult r;
            r.valid = m_callIndex < m_truePoses.size();
            r.pose = r.valid ? m_truePoses[m_callIndex] : priorPose;
            ++m_callIndex;
            return r;
        }

    private:
        std::vector<Eigen::Isometry3f> m_truePoses;
        std::shared_ptr<std::vector<Eigen::Isometry3f>> m_recordedPriors;
        std::size_t m_callIndex = 0;
    };

} // namespace

// On a straight-line pose sequence, the constant-velocity prior (previousPose advanced by the SAME
// delta that got from previousPreviousPose to previousPose) should predict the next true pose far
// better than just re-handing the tracker the previous pose unchanged -- that better starting point is
// exactly the value the motion model is meant to add (matters most when per-frame motion is large
// relative to the correspondence gate). Frame 3's prior is the first one built from TWO real (tracked,
// non-default-Identity) previous poses -- frames 1 and 2's actual results -- so it's the first
// genuine constant-velocity prediction (frame 2's prior also happens to land correctly, but only
// because previousPreviousPose was still its Identity initial value -- a degenerate case, not the
// motion model actually extrapolating two tracked poses).
TEST(RegistrationThread, ConstantVelocityPriorBeatsPreviousPoseOnStraightLine) {
    ep::CommunicationModule comm;

    // Straight-line translation along +X, 0.1m/frame -- true poses for frames 1, 2, 3.
    const float step = 0.1f;
    auto truePoseAt = [&](int k) {
        Eigen::Isometry3f p = Eigen::Isometry3f::Identity();
        p.translate(Vector3f(step * float(k), 0.0f, 0.0f));
        return p;
    };
    const std::vector<Eigen::Isometry3f> truePoses = {truePoseAt(1), truePoseAt(2), truePoseAt(3)};

    auto recordedPriors = std::make_shared<std::vector<Eigen::Isometry3f>>();
    auto tracker = std::make_unique<RecordingStraightLineTracker>(truePoses, recordedPriors);

    ep::RegistrationThread rt(comm, std::move(tracker));
    rt.Start();

    for (int i = 0; i < 3; ++i) comm.capturedFrames.Push(ep::Frame{});

    // Drain exactly 3 tracked frames before Stop() -- Stop() sets the cooperative stop flag, which
    // RegistrationThread::Run only re-checks at the top of its loop, so calling Stop() without first
    // waiting could abandon already-queued-but-unprocessed frames. Waiting for all 3 to come out the
    // other end guarantees the tracker's 3rd Track() call (and hence its 3rd recorded prior) happened.
    ep::TrackedFrame tf;
    for (int i = 0; i < 3; ++i) ASSERT_TRUE(comm.trackedFrames.Pop(tf)) << "tracked frame " << i;

    rt.Stop();
    ASSERT_EQ(rt.Error(), nullptr);

    ASSERT_EQ(recordedPriors->size(), 3u);
    const Eigen::Isometry3f constantVelocityPrior = (*recordedPriors)[2];
    const Eigen::Isometry3f previousPoseOnlyPrior = truePoses[1]; // frame 3's prior WITHOUT the motion model

    const float constantVelocityError =
            (constantVelocityPrior.translation() - truePoses[2].translation()).norm();
    const float previousPoseOnlyError =
            (previousPoseOnlyPrior.translation() - truePoses[2].translation()).norm();

    std::printf("[motion-model] constant-velocity prior error %.5f | previous-pose-only prior error %.5f\n",
                constantVelocityError, previousPoseOnlyError);

    EXPECT_LT(constantVelocityError, previousPoseOnlyError)
            << "constant-velocity prior should predict the next pose on a straight line better than just "
               "reusing the previous pose";
    EXPECT_NEAR(constantVelocityError, 0.0f, 1e-5f)
            << "on an exact straight line, the constant-velocity prediction should be near-exact";
}

///////////////////////////////////////////////////////////////////////////////////////////////
// Injected source factory -- how a device (a camera) reaches the pipeline. A camera is a handle,
// not a path list, so AcquisitionConfig cannot describe one by value.
///////////////////////////////////////////////////////////////////////////////////////////////

namespace {

    // Hands out a fixed number of identical plane frames, and counts how many times it was built.
    class CountingFrameSource : public ep::IFrameSource {
    public:
        CountingFrameSource(int frames, int *buildCount) : m_left(frames) { ++(*buildCount); }

        ep::EAcquisitionType Type() const override { return ep::EAcquisitionType::DepthCamera; }
        const char *Name() const override { return "counting"; }

        bool Next(ep::Frame &out) override {
            if (m_left-- <= 0) return false;
            out.pts.clear();
            out.nrm.clear();
            for (int i = -6; i <= 6; ++i)
                for (int j = -6; j <= 6; ++j) {
                    out.pts.emplace_back(float(i) * 0.02f, float(j) * 0.02f, 1.0f);
                    out.nrm.emplace_back(0.0f, 0.0f, -1.0f);
                }
            out.cam = Eigen::Vector3f::Zero();
            return true;
        }

    private:
        int m_left;
    };

} // namespace

TEST(PipelineSource, InjectedFactoryBuildsTheSource) {
    int buildCount = 0;
    ep::AcquisitionConfig acquisition;
    acquisition.type = ep::EAcquisitionType::DepthCamera;
    acquisition.makeSource = [&] { return std::make_unique<CountingFrameSource>(3, &buildCount); };

    std::unique_ptr<ep::IFrameSource> source = ep::MakeAcquisitionSource(acquisition);
    ASSERT_NE(source, nullptr);
    EXPECT_EQ(buildCount, 1);
    EXPECT_EQ(source->Type(), ep::EAcquisitionType::DepthCamera);

    int delivered = 0;
    ep::Frame frame;
    while (source->Next(frame)) ++delivered;
    EXPECT_EQ(delivered, 3);
}

// Without a factory a device type has nothing to build from, and saying so beats a null source
// that fails later inside the acquisition thread.
TEST(PipelineSource, DeviceTypeWithoutAFactoryIsRejected) {
    ep::AcquisitionConfig acquisition;
    acquisition.type = ep::EAcquisitionType::DepthCamera;
    EXPECT_THROW(ep::MakeAcquisitionSource(acquisition), std::invalid_argument);
}

// A factory that returns nothing must be caught where it is called. Handing a null IFrameSource
// to ReconstructionThread makes Run() exit immediately and the pipeline look merely empty.
TEST(PipelineSource, FactoryReturningNullIsRejected) {
    ep::AcquisitionConfig acquisition;
    acquisition.type = ep::EAcquisitionType::DepthCamera;
    acquisition.makeSource = [] { return std::unique_ptr<ep::IFrameSource>(); };
    EXPECT_THROW(ep::MakeAcquisitionSource(acquisition), std::invalid_argument);
}

// The factory takes precedence on File too, so a caller can substitute a decorated or synthetic
// source without inventing a config field for it.
TEST(PipelineSource, FactoryOverridesTheFileDescription) {
    int buildCount = 0;
    ep::AcquisitionConfig acquisition;
    acquisition.type = ep::EAcquisitionType::File;
    acquisition.framePaths = {"/no/such/frame.ply"};
    acquisition.makeSource = [&] { return std::make_unique<CountingFrameSource>(1, &buildCount); };

    std::unique_ptr<ep::IFrameSource> source = ep::MakeAcquisitionSource(acquisition);
    EXPECT_EQ(buildCount, 1);
    EXPECT_STREQ(source->Name(), "counting");
}

// End-to-end: a depth device reaches the map through the injected factory. This is the shape
// realsense_scan uses -- IDepthProvider -> DepthCameraFrameSource -> Pipeline -- with a synthetic
// provider standing in for the camera so it runs without hardware.
namespace {

    // A slanted wall with a nearer box in it: a plane alone leaves point-to-plane ICP with three
    // unconstrained DoF, and the box is what pins them.
    class SyntheticDepthProvider : public ep::IDepthProvider {
    public:
        explicit SyntheticDepthProvider(int frames) : m_left(frames) {
            m_intrinsics.width = 64;
            m_intrinsics.height = 48;
            m_intrinsics.fx = m_intrinsics.fy = 40.0f;
            m_intrinsics.cx = 32.0f;
            m_intrinsics.cy = 24.0f;
        }

        const ep::CameraIntrinsics &Intrinsics() const override { return m_intrinsics; }

        bool Grab(ep::DepthFrame &out) override {
            if (m_left-- <= 0) return false;
            out.depth.assign(std::size_t(m_intrinsics.width) * m_intrinsics.height, 0.0f);
            for (int v = 0; v < m_intrinsics.height; ++v)
                for (int u = 0; u < m_intrinsics.width; ++u) {
                    const float x = (float(u) - m_intrinsics.cx) / m_intrinsics.fx;
                    const float y = (float(v) - m_intrinsics.cy) / m_intrinsics.fy;
                    float z = 1.20f + 0.25f * x;
                    if (std::abs(x) < 0.20f && std::abs(y) < 0.20f) z = 0.85f;
                    out.depth[std::size_t(v) * m_intrinsics.width + u] = z;
                }
            return true;
        }

    private:
        int m_left;
        ep::CameraIntrinsics m_intrinsics;
    };

} // namespace

TEST(Pipeline, DepthDeviceAccumulatesAMap) {
    ep::Pipeline::Config cfg;
    cfg.map.baseVoxel = 0.01f;
    cfg.map.truncation = 0.03f;
    cfg.map.submap = false;
    cfg.acquisition.type = ep::EAcquisitionType::DepthCamera;
    cfg.acquisition.makeSource = [] {
        return std::make_unique<ep::DepthCameraFrameSource>(
                std::make_unique<SyntheticDepthProvider>(6));
    };

    ep::Pipeline pipe(cfg, ep::TrackerRegistry::Default().Create("identity"));
    pipe.Start();
    ASSERT_TRUE(waitProcessed(pipe, 3)) << "the depth source never reached the map";
    pipe.Stop();
    pipe.CheckErrors();

    const auto snapshot = pipe.LatestModel();
    ASSERT_NE(snapshot, nullptr);
    EXPECT_GT(snapshot->entries.size(), 0u) << "no voxels were integrated from the depth frames";
    EXPECT_FLOAT_EQ(snapshot->voxel, cfg.map.baseVoxel);

    // The surface must land where the geometry is. Back-projection puts the wall near z = 1.2 and
    // the box at 0.85, so a map centred anywhere else means the frontend or the routing is wrong.
    float nearestZ = 1e9f, farthestZ = -1e9f;
    for (const TSDFVoxel &voxel: snapshot->entries) {
        nearestZ = std::min(nearestZ, voxel.center.z());
        farthestZ = std::max(farthestZ, voxel.center.z());
    }
    EXPECT_GT(nearestZ, 0.5f);
    EXPECT_LT(farthestZ, 2.0f);
}

// Same depth source, but tracked by ICP instead of identity. The provider hands out an IDENTICAL
// frame every time, so the true pose is identity throughout and the map must stay the size of one
// frame. A tracker that diverges instead smears each frame to its own wrong pose and the map grows
// without bound.
//
// Waits on integratedFrames, NOT on ProcessedFrame reaching a chosen number: trackedFrames is a
// lossy channel (capacity 4, drops when full), so with integration slower than tracking a finite
// source can never reach a fixed count. Demanding one just burns the deadline -- measured at
// acq=9 align=8 integ=4 drop=4 on this fixture.
TEST(Pipeline, IcpOnAStaticSceneKeepsTheMapTheSizeOfOneFrame) {
    auto runWith = [](const char *trackerName) {
        ep::Pipeline::Config cfg;
        cfg.map.baseVoxel = 0.01f;
        cfg.map.truncation = 0.03f;
        cfg.map.submap = false;
        cfg.acquisition.type = ep::EAcquisitionType::DepthCamera;
        cfg.acquisition.makeSource = [] {
            return std::make_unique<ep::DepthCameraFrameSource>(
                    std::make_unique<SyntheticDepthProvider>(8));
        };

        ep::Pipeline pipe(cfg, ep::TrackerRegistry::Default().Create(trackerName));
        pipe.Start();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (pipe.GetStats().integratedFrames < 3 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        const std::uint64_t integrated = pipe.GetStats().integratedFrames;
        pipe.Stop();
        pipe.CheckErrors();

        Vector3f low = Vector3f::Constant(1e9f), high = Vector3f::Constant(-1e9f);
        const auto snapshot = pipe.LatestModel();
        if (snapshot)
            for (const TSDFVoxel &voxel: snapshot->entries) {
                low = low.cwiseMin(voxel.center);
                high = high.cwiseMax(voxel.center);
            }
        EXPECT_GE(integrated, 3u) << trackerName << " never integrated enough frames to judge";
        return (high - low).eval();
    };

    const Vector3f identityExtent = runWith("identity");
    const Vector3f icpExtent = runWith("icp");

    // Identity is the reference: the true poses ARE identity here, so ICP cannot legitimately do
    // better, and much worse means divergence.
    ASSERT_GT(identityExtent.maxCoeff(), 0.1f) << "the reference run produced no map";
    EXPECT_LT(icpExtent.maxCoeff(), identityExtent.maxCoeff() * 1.5f + 0.05f)
            << "ICP smeared a static scene: identity extent " << identityExtent.transpose()
            << ", icp extent " << icpExtent.transpose();
}

// Closing the device is what keeps the NEXT run able to open it: a RealSense left streaming holds
// its USB interface and every later open fails until the cable is re-seated. So the pipeline must
// release the device when acquisition ENDS -- not merely when the objects are destroyed.
namespace {

    class CloseCountingDepthProvider : public ep::IDepthProvider {
    public:
        CloseCountingDepthProvider(int frames, int *closeCount)
            : m_left(frames), m_closeCount(closeCount) {
            m_intrinsics.width = 8;
            m_intrinsics.height = 6;
            m_intrinsics.fx = m_intrinsics.fy = 10.0f;
            m_intrinsics.cx = 4.0f;
            m_intrinsics.cy = 3.0f;
        }

        const ep::CameraIntrinsics &Intrinsics() const override { return m_intrinsics; }

        bool Grab(ep::DepthFrame &out) override {
            if (m_left-- <= 0) return false;
            out.depth.assign(std::size_t(m_intrinsics.width) * m_intrinsics.height, 1.0f);
            return true;
        }

        void Close() override { ++(*m_closeCount); }

    private:
        int m_left;
        int *m_closeCount;
        ep::CameraIntrinsics m_intrinsics;
    };

} // namespace

TEST(DepthFrontend, FrameSourceForwardsCloseToTheDevice) {
    int closeCount = 0;
    {
        ep::DepthCameraFrameSource source(
                std::make_unique<CloseCountingDepthProvider>(1, &closeCount));
        source.Close();
        EXPECT_EQ(closeCount, 1) << "the frame source swallowed Close(), so the device kept running";
    }
}

// DepthRecorder decorates a provider. A decorator that swallows Close() leaves the wrapped camera
// streaming while looking perfectly correct at the call site.
TEST(DepthFrontend, RecorderForwardsCloseToTheWrappedDevice) {
    int closeCount = 0;
    const std::filesystem::path dir =
            std::filesystem::temp_directory_path() / "vkbvh_depth_close_forward";
    std::filesystem::remove_all(dir);

    ep::DepthRecorder recorder(std::make_unique<CloseCountingDepthProvider>(1, &closeCount),
                               dir.string());
    recorder.Close();
    EXPECT_EQ(closeCount, 1);
    std::filesystem::remove_all(dir);
}

// The end that matters: running the pipeline to the end of its source must release the device.
TEST(Pipeline, ReleasesTheDeviceWhenAcquisitionEnds) {
    int closeCount = 0;
    ep::Pipeline::Config cfg;
    cfg.map.baseVoxel = 0.05f;
    cfg.map.truncation = 0.15f;
    cfg.map.submap = false;
    cfg.acquisition.type = ep::EAcquisitionType::DepthCamera;
    cfg.acquisition.makeSource = [&] {
        return std::make_unique<ep::DepthCameraFrameSource>(
                std::make_unique<CloseCountingDepthProvider>(2, &closeCount));
    };

    {
        ep::Pipeline pipe(cfg, ep::TrackerRegistry::Default().Create("identity"));
        pipe.Start();
        pipe.Stop();
        pipe.CheckErrors();
        EXPECT_GT(closeCount, 0) << "the pipeline stopped without releasing the device";
    }
}

// Acquisition-stage downsampling. Two points inside one map voxel are indistinguishable to the
// map, but the finer one is still paid for by ICP, by the frame queues, and by every copy between
// -- so it is dropped where it is produced, not at integration.
TEST(Pipeline, AcquisitionReducesFramesToTheMapsFinestVoxel) {
    // 200x200 points across 1 m: 5 mm apart, far denser than the 0.05 m map voxel below.
    auto makeDenseSource = [] {
        class DenseSource : public ep::IFrameSource {
        public:
            ep::EAcquisitionType Type() const override { return ep::EAcquisitionType::DepthCamera; }
            const char *Name() const override { return "dense"; }
            bool Next(ep::Frame &out) override {
                if (m_left-- <= 0) return false;
                out.pts.clear();
                out.nrm.clear();
                for (int i = 0; i < 200; ++i)
                    for (int j = 0; j < 200; ++j) {
                        out.pts.emplace_back(float(i) * 0.005f, float(j) * 0.005f, 1.0f);
                        out.nrm.emplace_back(0.0f, 0.0f, -1.0f);
                    }
                return true;
            }

        private:
            int m_left = 1;
        };
        return std::make_unique<DenseSource>();
    };

    ep::AcquisitionConfig acquisition;
    acquisition.type = ep::EAcquisitionType::DepthCamera;
    acquisition.makeSource = makeDenseSource;

    // Off: the frame arrives whole.
    acquisition.downsampleVoxel = -1.0f; // negative disables
    {
        ep::CommunicationModule comm;
        ep::ReconstructionThread stage(comm, acquisition);
        stage.Start();
        ep::Frame frame;
        ASSERT_TRUE(comm.capturedFrames.Pop(frame));
        EXPECT_EQ(frame.pts.size(), 200u * 200u);
        stage.Stop();
    }

    // On at 0.05 m: 1 m of surface can hold at most 21x21 occupied cells, so the frame must come
    // out two orders of magnitude smaller -- and still cover the same surface.
    acquisition.downsampleVoxel = 0.05f;
    {
        ep::CommunicationModule comm;
        ep::ReconstructionThread stage(comm, acquisition);
        stage.Start();
        ep::Frame frame;
        ASSERT_TRUE(comm.capturedFrames.Pop(frame));
        EXPECT_LE(frame.pts.size(), 21u * 21u);
        EXPECT_GE(frame.pts.size(), 20u * 20u) << "the surface itself must survive, not just shrink";
        EXPECT_EQ(frame.pts.size(), frame.nrm.size()) << "Frame's contract is pts.size() == nrm.size()";
        for (const Eigen::Vector3f &n: frame.nrm) EXPECT_NEAR(n.z(), -1.0f, 1e-3f);
        stage.Stop();
    }
}

// Left at 0, the pipeline fills the knob from the map's finest level -- baseVoxel, or half of it
// when the submap's detail level exists.
TEST(Pipeline, DownsampleVoxelDefaultsToTheMapsFinestLevel) {
    ep::Pipeline::Config cfg;
    cfg.map.baseVoxel = 0.04f;
    cfg.acquisition.type = ep::EAcquisitionType::DepthCamera;
    cfg.acquisition.makeSource = [] {
        return std::make_unique<ep::DepthCameraFrameSource>(
                std::make_unique<SyntheticDepthProvider>(1));
    };

    cfg.map.submap = false;
    {
        ep::Pipeline pipe(cfg, ep::TrackerRegistry::Default().Create("identity"));
        EXPECT_FLOAT_EQ(pipe.DownsampleVoxel(), 0.04f);
    }
    cfg.map.submap = true;
    {
        ep::Pipeline pipe(cfg, ep::TrackerRegistry::Default().Create("identity"));
        EXPECT_FLOAT_EQ(pipe.DownsampleVoxel(), 0.02f) << "the detail level is half the base voxel";
    }
}

// A recording must be processed losslessly. With dropping on, a slower configuration silently
// processes fewer frames, so any measurement taken across configurations compares drop rates
// rather than the thing under test.
TEST(Pipeline, OfflineSourceProcessesEveryFrame) {
    constexpr int kFrames = 40;
    auto config = [kFrames](bool realTime) {
        ep::Pipeline::Config cfg;
        cfg.map.baseVoxel = 0.02f;
        cfg.map.truncation = 0.06f;
        cfg.map.submap = false;
        cfg.acquisition.type = ep::EAcquisitionType::DepthCamera;
        cfg.acquisition.realTime = realTime;
        cfg.acquisition.makeSource = [=] {
            return std::make_unique<ep::DepthCameraFrameSource>(
                    std::make_unique<SyntheticDepthProvider>(kFrames));
        };
        return cfg;
    };

    auto run = [](ep::Pipeline::Config cfg) {
        ep::Pipeline pipe(cfg, ep::TrackerRegistry::Default().Create("identity"));
        pipe.Start();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        std::uint64_t last = 0;
        int stalled = 0;
        while (std::chrono::steady_clock::now() < deadline) {
            const ep::PipelineStats s = pipe.GetStats();
            if (s.integratedFrames == last) {
                if (last > 0 && ++stalled > 100) break; // 2 s with no progress
            } else {
                stalled = 0;
                last = s.integratedFrames;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        const ep::PipelineStats s = pipe.GetStats();
        pipe.Stop();
        pipe.CheckErrors();
        return s;
    };

    const ep::PipelineStats offline = run(config(false));
    EXPECT_EQ(offline.integratedFrames, std::uint64_t(kFrames))
            << "an offline source must lose nothing; dropped " << offline.trackDropped;
    EXPECT_EQ(offline.trackDropped, 0u);

    // Real-time keeps the drop policy: bounded latency is the point of it, and losing frames is
    // the price. Asserted only as "still runs", since whether it drops depends on the machine.
    const ep::PipelineStats live = run(config(true));
    EXPECT_GT(live.integratedFrames, 0u);
}
