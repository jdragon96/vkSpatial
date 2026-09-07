#include "Pipeline/CommunicationModule.h" // Pipeline::CommunicationModule
#include "Pipeline/Pipeline.h"            // Pipeline::Pipeline / Config / EAcquisitionSource
#include "Pipeline/Acquisition/AcquisitionThread.h" // MakeDepthProvider, IDepthProvider
#include "Realsense/RealSenseD435Recorder.h"
#include "Pipeline/Registration/GpuIcpTracker.h"
#include "Registration/Frontend/GpuPointToPlaneIcp.h"
#include "Pipeline/Registration/PointToPlaneIcpTracker.h"
#include "Pipeline/Registration/RegistrationThread.h"
#include "Pipeline/Registration/RelocalizingIcpTracker.h"
#include "Pipeline/Registration/Tracker.h" // Pipeline::TrackerRegistry

#include "utilities/PointCloudIO.h"
#include "Common/PointCloud.h"
#include "Registration/RegistrationConfig.h"
#include "Registration/RegistrationParam.h"
#include "Registration/RegistrationResult.h"

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
namespace ep = Pipeline;
using Eigen::Vector3f;

namespace {

    // A depth image of a plane 1 m away, in Z16 -- the only pixel format the acquisition axis
    // carries, so a test source hands out exactly what a D400 does. 48x48 at fx 60 back-projects
    // to a 0.8 m patch, the same size the retired PLY fixture wrote.
    class PlaneDepthProvider : public ep::IDepthProvider {
    public:
        explicit PlaneDepthProvider(int frames, double frameIntervalMs = 0.0)
            : m_left(frames), m_frameIntervalMs(frameIntervalMs) {
            m_intrinsics.width = 48;
            m_intrinsics.height = 48;
            m_intrinsics.fx = m_intrinsics.fy = 60.0f;
            m_intrinsics.cx = m_intrinsics.cy = 24.0f;
            m_intrinsics.depthScale = 0.001f;
            m_intrinsics.stereoBaselineMeters = 0.05f; // ValidationMask refuses a zero baseline
            m_frame.assign(std::size_t(m_intrinsics.width) * m_intrinsics.height,
                           std::uint16_t(1.0f / m_intrinsics.depthScale));
        }

        const ep::CameraIntrinsics &Intrinsics() const override { return m_intrinsics; }

        bool Grab(ep::DepthFrame &out) override {
            if (m_left-- <= 0) return false;
            // Paced in the device, not in the config: a camera is what has a frame rate, and the
            // acquisition stage no longer owns a clock.
            if (m_frameIntervalMs > 0.0)
                std::this_thread::sleep_for(
                        std::chrono::microseconds(std::int64_t(m_frameIntervalMs * 1000.0)));
            out.rawZ16 = m_frame.data();
            return true;
        }

    private:
        int m_left;
        double m_frameIntervalMs;
        ep::CameraIntrinsics m_intrinsics;
        std::vector<std::uint16_t> m_frame;
    };

    // Config-driven depth source: `frames` identical plane images through the real GPU front end.
    ep::Pipeline::Config makeConfig(int frames, double frameIntervalMs = 0.0) {
        ep::Pipeline::Config cfg;
        cfg.map.baseVoxel = 0.05f;
        cfg.map.truncation = 0.15f;
        cfg.acquisition.source = ep::EAcquisitionSource::Realsense;
        cfg.acquisition.makeProvider = [frames, frameIntervalMs] {
            return std::make_unique<PlaneDepthProvider>(frames, frameIntervalMs);
        };
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

// End-to-end: a config-driven depth source streams frames through Acquisition -> ICP(identity) ->
// Integration, and the pipeline publishes a non-empty model. Proves the module works off the render
// thread, and that the whole GPU front end runs on whatever the provider hands it.
TEST(Pipeline, DepthSourceProducesModel) {
    ep::Pipeline pipe(makeConfig(3), identity());
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

// A device that paces itself still delivers every frame; Source() reports what was configured.
TEST(Pipeline, PacedDeviceDeliversAllFrames) {
    ep::Pipeline pipe(makeConfig(3, 15.0), identity()); // ~66 fps pacing
    EXPECT_EQ(pipe.Source(), ep::EAcquisitionSource::Realsense);
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
    ep::Pipeline pipe(makeConfig(5), identity());
    pipe.Start();
    ASSERT_TRUE(waitProcessed(pipe, 1)) << "pipeline did not start";

    for (int i = 0; i < 3; ++i) {
        ep::Pipeline::Config cfg = makeConfig(5);
        cfg.map.submap = (i % 2 == 0); // flip an option, as a UI toggle would
        // An ACQUISITION option too, not just a map one: realsense_scan's option panel edits both,
        // and rebuilding the map stage while reusing a front end configured for the old settings
        // would produce frames that silently disagree with the config the caller just applied.
        cfg.acquisition.scoreThreshold = (i % 2 == 0) ? 0.5f : 0.95f;
        cfg.acquisition.normal.planeFitRadius = 2 + i;
        pipe.Reconfigure(std::move(cfg), identity());
        ASSERT_TRUE(waitProcessed(pipe, 1)) << "no model after Reconfigure #" << i;
        pipe.CheckErrors(); // rethrow any worker exception from the rebuild
        EXPECT_NE(pipe.LatestModel(), nullptr);
    }
    pipe.Stop();
}

// End-to-end: the GPU point-to-plane ICP tracker (registered as "icp") runs on its own lazily-created
// Context inside the ICP thread, alongside the Integration thread's Context -- two live GPU contexts at
// once. Also proves "icp-cpu" (the pre-existing CPU tracker) is still registered under its new name.
TEST(Pipeline, GpuIcpTrackerRuns) {
    ep::Pipeline::Config cfg = makeConfig(3);
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
    EXPECT_LT(err.norm(), 5e-3f) << "pose:\n"
                                 << r.pose.matrix() << "\nperturb:\n"
                                 << perturb.matrix();
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
            << "pose:\n"
            << r.pose.matrix() << "\nprior:\n"
            << prior.matrix();
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
    EXPECT_LT(err.norm(), 5e-3f) << "pose:\n"
                                 << r.pose.matrix() << "\ntruePose:\n"
                                 << truePose.matrix();
}

// Stop() before the source is exhausted must not hang or crash (interruptible shutdown).
TEST(Pipeline, StopIsCleanMidStream) {
    ep::Pipeline pipe(makeConfig(50, 50.0), identity()); // slow pacing -> Stop mid-stream
    pipe.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(80)); // let a couple frames through
    pipe.Stop();                                                // must return promptly (no hang)
    pipe.CheckErrors();
    SUCCEED();
}

namespace {

    class RecordingStraightLineTracker : public ep::Tracker {
    public:
        RecordingStraightLineTracker(std::vector<Eigen::Isometry3f> truePoses,
                                     std::shared_ptr<std::vector<Eigen::Isometry3f>> recordedPriors)
            : m_truePoses(std::move(truePoses)), m_recordedPriors(std::move(recordedPriors)) {}

        const char *Name() const override { return "recording-straight-line"; }

        ep::TrackingResult Track(const ep::Frame &,
                                 const ep::ModelSnapshot *,
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
// Injected provider factory -- how a device reaches the pipeline. A camera is a handle, not a path
// list, so AcquisitionConfig cannot describe one by value.
///////////////////////////////////////////////////////////////////////////////////////////////

namespace {

    // Hands out a fixed number of identical depth images, and counts how many times it was built.
    class CountingDepthProvider : public ep::IDepthProvider {
    public:
        CountingDepthProvider(int frames, int *buildCount) : m_left(frames) {
            ++(*buildCount);
            m_intrinsics.width = 32;
            m_intrinsics.height = 32;
            m_intrinsics.fx = m_intrinsics.fy = 40.0f;
            m_intrinsics.cx = m_intrinsics.cy = 16.0f;
            m_intrinsics.depthScale = 0.001f;
            m_intrinsics.stereoBaselineMeters = 0.05f;
            m_frame.assign(std::size_t(m_intrinsics.width) * m_intrinsics.height, 1000);
        }

        const ep::CameraIntrinsics &Intrinsics() const override { return m_intrinsics; }

        bool Grab(ep::DepthFrame &out) override {
            if (m_left-- <= 0) return false;
            out.rawZ16 = m_frame.data();
            return true;
        }

    private:
        int m_left;
        ep::CameraIntrinsics m_intrinsics;
        std::vector<std::uint16_t> m_frame;
    };

} // namespace

TEST(PipelineSource, InjectedFactoryBuildsTheProvider) {
    int buildCount = 0;
    ep::AcquisitionConfig acquisition;
    acquisition.source = ep::EAcquisitionSource::Realsense;
    acquisition.makeProvider = [&] { return std::make_unique<CountingDepthProvider>(3, &buildCount); };

    std::unique_ptr<ep::IDepthProvider> provider = ep::MakeDepthProvider(acquisition);
    ASSERT_NE(provider, nullptr);
    EXPECT_EQ(buildCount, 1) << "the factory must be called exactly once";

    int delivered = 0;
    ep::DepthFrame frame;
    while (provider->Grab(frame)) ++delivered;
    EXPECT_EQ(delivered, 3);
}

// A device no longer needs an injected factory -- describing it in the config is the whole point
// of the source enum -- but it still has to be described COMPLETELY. RealsenseFile without a
// directory has nothing to replay, and saying so beats a device that opens and delivers nothing.
TEST(PipelineSource, RealsenseFileWithoutARecordingIsRejected) {
    ep::AcquisitionConfig acquisition;
    acquisition.source = ep::EAcquisitionSource::RealsenseFile;
    EXPECT_THROW(ep::MakeDepthProvider(acquisition), std::invalid_argument);
}

// A factory that returns nothing must be caught where it is called: AcquisitionThread would
// dereference the null provider on its first Grab, inside the worker thread, where the failure is
// a crash rather than a rejected configuration.
TEST(PipelineSource, FactoryReturningNullIsRejected) {
    ep::AcquisitionConfig acquisition;
    acquisition.makeProvider = [] { return std::unique_ptr<ep::IDepthProvider>(); };
    EXPECT_THROW(ep::MakeDepthProvider(acquisition), std::invalid_argument);
}

// An injected provider outranks the source enum, and must do so even where the enum names a
// source that would itself have succeeded -- otherwise a tool that wraps a device (realsense_scan
// wrapping it in a DepthRecorder) silently gets the bare device instead of its chain.
TEST(PipelineSource, FactoryOverridesTheConfiguredSource) {
    int buildCount = 0;
    ep::AcquisitionConfig acquisition;
    acquisition.source = ep::EAcquisitionSource::RealsenseFile;
    acquisition.recordingDirectory = "/no/such/recording";
    acquisition.makeProvider = [&] { return std::make_unique<CountingDepthProvider>(1, &buildCount); };

    std::unique_ptr<ep::IDepthProvider> provider = ep::MakeDepthProvider(acquisition);
    ASSERT_NE(provider, nullptr) << "the configured source ran instead of the injected device";
    EXPECT_EQ(buildCount, 1);
}

// End-to-end: a depth device reaches the map through the injected factory. This is the shape
// realsense_scan uses -- IDepthProvider -> the GPU front end -> Pipeline -- with a synthetic
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
            m_intrinsics.depthScale = 0.001f;
            m_intrinsics.stereoBaselineMeters = 0.05f;

            m_frame.assign(std::size_t(m_intrinsics.width) * m_intrinsics.height, 0);
            for (int v = 0; v < m_intrinsics.height; ++v)
                for (int u = 0; u < m_intrinsics.width; ++u) {
                    const float x = (float(u) - m_intrinsics.cx) / m_intrinsics.fx;
                    const float y = (float(v) - m_intrinsics.cy) / m_intrinsics.fy;
                    float z = 1.20f + 0.25f * x;
                    if (std::abs(x) < 0.20f && std::abs(y) < 0.20f) z = 0.85f;
                    m_frame[std::size_t(v) * m_intrinsics.width + u] =
                            std::uint16_t(std::lround(z / m_intrinsics.depthScale));
                }
        }

        const ep::CameraIntrinsics &Intrinsics() const override { return m_intrinsics; }

        bool Grab(ep::DepthFrame &out) override {
            if (m_left-- <= 0) return false;
            out.rawZ16 = m_frame.data();
            return true;
        }

    private:
        int m_left;
        ep::CameraIntrinsics m_intrinsics;
        std::vector<std::uint16_t> m_frame;
    };

} // namespace

TEST(Pipeline, DepthDeviceAccumulatesAMap) {
    ep::Pipeline::Config cfg;
    cfg.map.baseVoxel = 0.01f;
    cfg.map.truncation = 0.03f;
    cfg.map.submap = false;
    cfg.acquisition.source = ep::EAcquisitionSource::Realsense;
    cfg.acquisition.makeProvider = [] {
        return std::make_unique<SyntheticDepthProvider>(6);
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
        cfg.acquisition.source = ep::EAcquisitionSource::Realsense;
        cfg.acquisition.makeProvider = [] {
            return std::make_unique<SyntheticDepthProvider>(8);
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
            m_intrinsics.depthScale = 0.001f;
            m_intrinsics.stereoBaselineMeters = 0.05f;
            m_frame.assign(std::size_t(m_intrinsics.width) * m_intrinsics.height, 1000);
        }

        const ep::CameraIntrinsics &Intrinsics() const override { return m_intrinsics; }

        bool Grab(ep::DepthFrame &out) override {
            if (m_left-- <= 0) return false;
            out.rawZ16 = m_frame.data();
            return true;
        }

        void Close() override { ++(*m_closeCount); }

    private:
        int m_left;
        int *m_closeCount;
        ep::CameraIntrinsics m_intrinsics;
        std::vector<std::uint16_t> m_frame;
    };

} // namespace

// Recording decorates a provider. A decorator that swallows Close() leaves the wrapped camera
// streaming while looking perfectly correct at the call site.
TEST(DepthFrontend, RecorderForwardsCloseToTheWrappedDevice) {
    int closeCount = 0;
    const std::filesystem::path dir =
            std::filesystem::temp_directory_path() / "vkbvh_depth_close_forward";
    std::filesystem::remove_all(dir);

    Realsense::RealSenseD435Recorder recorder(
            std::make_unique<CloseCountingDepthProvider>(1, &closeCount), dir.string());
    recorder.Close();
    EXPECT_EQ(closeCount, 1);
    std::filesystem::remove_all(dir);
}

namespace {

    // A ramp, so a frame written at the wrong stride or byte width reads back visibly wrong rather
    // than accidentally matching a constant image.
    class RampDepthProvider : public ep::IDepthProvider {
    public:
        explicit RampDepthProvider(int frames) : m_left(frames) {
            m_intrinsics.width = 7; // deliberately not a power of two: a stride bug shows up
            m_intrinsics.height = 5;
            m_intrinsics.fx = m_intrinsics.fy = 9.0f;
            m_intrinsics.cx = 3.5f;
            m_intrinsics.cy = 2.5f;
            m_intrinsics.depthScale = 0.00025f; // NOT the 0.001 default: proves it is stored
            m_intrinsics.stereoBaselineMeters = 0.0499f;
        }

        const ep::CameraIntrinsics &Intrinsics() const override { return m_intrinsics; }

        bool Grab(ep::DepthFrame &out) override {
            if (m_left-- <= 0) return false;
            const std::size_t pixels = std::size_t(m_intrinsics.width) * m_intrinsics.height;
            m_frame.resize(pixels);
            for (std::size_t i = 0; i < pixels; ++i)
                m_frame[i] = std::uint16_t(1000 + m_emitted * 100 + i);
            ++m_emitted;
            out.rawZ16 = m_frame.data();
            return true;
        }

        int Emitted() const { return m_emitted; }

    private:
        int m_left;
        int m_emitted = 0;
        ep::CameraIntrinsics m_intrinsics;
        std::vector<std::uint16_t> m_frame;
    };

} // namespace

// The recording format's whole job: what the device produced is what comes back. Asserted per
// pixel, not per frame count -- a wrong element width or a wrong stride still writes the right
// NUMBER of frames, and the tools downstream would report a plausible-looking depth range from
// them. depthScale and the baseline ride along because a replay that loses them scores every
// frame with a silently different sigma_z.
TEST(DepthFrontend, ARecordingReplaysTheExactZ16TheDeviceProduced) {
    constexpr int kFrames = 3;
    const std::filesystem::path dir =
            std::filesystem::temp_directory_path() / "vkbvh_depth_roundtrip";
    std::filesystem::remove_all(dir);

    RampDepthProvider reference(kFrames); // an independent copy of what was recorded
    std::vector<std::vector<std::uint16_t>> written;
    {
        auto device = std::make_unique<RampDepthProvider>(kFrames);
        Realsense::RealSenseD435Recorder recorder(std::move(device), dir.string());
        const std::size_t pixels = std::size_t(reference.Intrinsics().width) *
                                   reference.Intrinsics().height;
        ep::DepthFrame frame;
        while (recorder.Grab(frame)) written.emplace_back(frame.rawZ16, frame.rawZ16 + pixels);
        EXPECT_EQ(recorder.FrameCount(), kFrames);
    }
    ASSERT_EQ(int(written.size()), kFrames);

    Realsense::RealSenseD435Recorder replay(dir.string());
    const ep::CameraIntrinsics &back = replay.Intrinsics();
    const ep::CameraIntrinsics &original = reference.Intrinsics();
    EXPECT_EQ(replay.FrameCount(), kFrames);
    EXPECT_EQ(back.width, original.width);
    EXPECT_EQ(back.height, original.height);
    EXPECT_FLOAT_EQ(back.fx, original.fx);
    EXPECT_FLOAT_EQ(back.cy, original.cy);
    EXPECT_FLOAT_EQ(back.depthScale, original.depthScale);
    EXPECT_FLOAT_EQ(back.stereoBaselineMeters, original.stereoBaselineMeters);

    const std::size_t pixels = std::size_t(back.width) * back.height;
    ep::DepthFrame frame;
    for (int k = 0; k < kFrames; ++k) {
        ASSERT_TRUE(replay.Grab(frame)) << "recording ended at frame " << k;
        ASSERT_NE(frame.rawZ16, nullptr);
        for (std::size_t i = 0; i < pixels; ++i)
            ASSERT_EQ(frame.rawZ16[i], written[std::size_t(k)][i])
                    << "frame " << k << " pixel " << i;
    }
    EXPECT_FALSE(replay.Grab(frame)) << "replay ran past the end of the recording";

    std::filesystem::remove_all(dir);
}

// A truncated depth_*.bin must be refused at construction, not handed back as a partial image:
// short frames corrupt a reconstruction with no symptom at all. This is also the check that
// catches a recording left in the retired float32-metres format.
TEST(DepthFrontend, ATruncatedRecordingIsRefused) {
    const std::filesystem::path dir =
            std::filesystem::temp_directory_path() / "vkbvh_depth_truncated";
    std::filesystem::remove_all(dir);
    {
        Realsense::RealSenseD435Recorder recorder(std::make_unique<RampDepthProvider>(1),
                                                  dir.string());
        ep::DepthFrame frame;
        ASSERT_TRUE(recorder.Grab(frame));
    }
    ASSERT_NO_THROW(Realsense::RealSenseD435Recorder{dir.string()});

    std::filesystem::resize_file(dir / "depth_0000.bin", 4);
    EXPECT_THROW(Realsense::RealSenseD435Recorder{dir.string()}, std::runtime_error);

    std::filesystem::remove_all(dir);
}

// Recording over a directory that already holds one would replay as a single capture with a
// teleport where the shorter take ended, and no size check downstream can see that.
TEST(DepthFrontend, RecordingIntoADirectoryThatAlreadyHoldsOneIsRefused) {
    const std::filesystem::path dir =
            std::filesystem::temp_directory_path() / "vkbvh_depth_overwrite";
    std::filesystem::remove_all(dir);
    {
        Realsense::RealSenseD435Recorder recorder(std::make_unique<RampDepthProvider>(1),
                                                  dir.string());
        ep::DepthFrame frame;
        ASSERT_TRUE(recorder.Grab(frame));
    }

    EXPECT_THROW(Realsense::RealSenseD435Recorder(std::make_unique<RampDepthProvider>(1),
                                                  dir.string()),
                 std::runtime_error);

    std::filesystem::remove_all(dir);
}

// The end that matters: running the pipeline to the end of its source must release the device.
TEST(Pipeline, ReleasesTheDeviceWhenAcquisitionEnds) {
    int closeCount = 0;
    ep::Pipeline::Config cfg;
    cfg.map.baseVoxel = 0.05f;
    cfg.map.truncation = 0.15f;
    cfg.map.submap = false;
    cfg.acquisition.source = ep::EAcquisitionSource::Realsense;
    cfg.acquisition.makeProvider = [&] {
        return std::make_unique<CloseCountingDepthProvider>(2, &closeCount);
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
    // A 200x200 image of a wall 1 m away at fx 200 back-projects to points z/fx = 5 mm apart, far
    // denser than the 0.05 m map voxel below.
    auto makeDenseProvider = [] {
        class DenseDepthProvider : public ep::IDepthProvider {
        public:
            DenseDepthProvider() {
                m_intrinsics.width = m_intrinsics.height = 200;
                m_intrinsics.fx = m_intrinsics.fy = 200.0f;
                m_intrinsics.cx = m_intrinsics.cy = 100.0f;
                m_intrinsics.depthScale = 0.001f;
                m_intrinsics.stereoBaselineMeters = 0.05f;
                m_frame.assign(std::size_t(m_intrinsics.width) * m_intrinsics.height, 1000);
            }

            const ep::CameraIntrinsics &Intrinsics() const override { return m_intrinsics; }

            bool Grab(ep::DepthFrame &out) override {
                if (m_left-- <= 0) return false;
                out.rawZ16 = m_frame.data();
                return true;
            }

        private:
            int m_left = 1;
            ep::CameraIntrinsics m_intrinsics;
            std::vector<std::uint16_t> m_frame;
        };
        return std::make_unique<DenseDepthProvider>();
    };

    ep::AcquisitionConfig acquisition;
    acquisition.source = ep::EAcquisitionSource::Realsense;
    acquisition.makeProvider = makeDenseProvider;

    // Off: the frame arrives whole.
    acquisition.downsampleVoxel = -1.0f; // negative disables
    {
        ep::CommunicationModule comm;
        ep::AcquisitionThread stage(comm, acquisition);
        stage.Start();
        ep::Frame frame;
        ASSERT_TRUE(comm.capturedFrames.Pop(frame));
        // Not exactly 200x200: the normal stencil cannot reach the border, so a ring of pixels is
        // outside the estimator's domain. What matters is that the frame arrives essentially whole.
        EXPECT_GT(frame.pts.size(), 30000u);
        stage.Stop();
    }

    // On at 0.05 m: 1 m of surface can hold at most 21x21 occupied cells, so the frame must come
    // out two orders of magnitude smaller -- and still cover the same surface.
    acquisition.downsampleVoxel = 0.05f;
    {
        ep::CommunicationModule comm;
        ep::AcquisitionThread stage(comm, acquisition);
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

// Acquisition-stage reduction is a throughput tool, not a free optimisation, so it stays off
// unless asked for. Deriving it from the map voxel looks principled and is not: the submap assigns
// its detail level by measuring point DENSITY, so reducing a frame to the map's own resolution
// guarantees no region is ever dense enough to earn detail. Measured on a 477-frame D435 capture
// at map voxel 0.05, that default cost 84% of the reconstructed surface (312,933 -> 48,784).
TEST(Pipeline, DownsamplingIsOffUnlessAskedFor) {
    ep::Pipeline::Config cfg;
    cfg.map.baseVoxel = 0.04f;
    cfg.map.submap = true;
    cfg.acquisition.source = ep::EAcquisitionSource::Realsense;
    cfg.acquisition.makeProvider = [] {
        return std::make_unique<SyntheticDepthProvider>(1);
    };

    {
        ep::Pipeline pipe(cfg, ep::TrackerRegistry::Default().Create("identity"));
        EXPECT_FLOAT_EQ(pipe.DownsampleVoxel(), 0.0f) << "nothing may enable this implicitly";
    }

    cfg.acquisition.downsampleVoxel = 0.01f;
    {
        ep::Pipeline pipe(cfg, ep::TrackerRegistry::Default().Create("identity"));
        EXPECT_FLOAT_EQ(pipe.DownsampleVoxel(), 0.01f) << "an explicit request must survive";
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
        cfg.acquisition.source = ep::EAcquisitionSource::Realsense;
        cfg.acquisition.realTime = realTime;
        cfg.acquisition.makeProvider = [=] {
            return std::make_unique<SyntheticDepthProvider>(kFrames);
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

// A solve that satisfies ten correspondences out of tens of thousands determines six degrees of
// freedom about as well as noise does, yet minInliers alone accepts it. On a real capture that let
// a frame which had drifted off the map latch onto a handful of strays, report the resulting pose
// as good, and -- through the constant-velocity prior, which doubles whatever the last frame did --
// run away: 10.78 m of claimed motion in a single frame at 30 fps.
TEST(Registration, FitnessGateRejectsASolveBackedByAlmostNoOverlap) {
    Common::PointCloud target;
    for (int i = -20; i <= 20; ++i)
        for (int j = -20; j <= 20; ++j) {
            target.points.emplace_back(float(i) * 0.01f, float(j) * 0.01f, 0.0f);
            target.normals.emplace_back(0.0f, 0.0f, 1.0f);
        }

    // Source: a small patch overlapping the target, plus far more points nowhere near it. Only the
    // patch can find correspondences, so the solve is backed by a small share of the source.
    std::vector<Eigen::Vector3f> source, sourceNormals;
    for (int i = -3; i <= 3; ++i)
        for (int j = -3; j <= 3; ++j) {
            source.emplace_back(float(i) * 0.01f, float(j) * 0.01f, 0.0f);
            sourceNormals.emplace_back(0.0f, 0.0f, 1.0f);
        }
    const std::size_t overlapping = source.size();
    for (int i = 0; i < 400; ++i) {
        source.emplace_back(50.0f + float(i) * 0.01f, 50.0f, 50.0f);
        sourceNormals.emplace_back(0.0f, 0.0f, 1.0f);
    }
    ASSERT_LT(double(overlapping) / double(source.size()), 0.2);

    Engine::Core::Context context;
    ep::GpuPointToPlaneIcp icp(context);
    Registration::RegistrationParam params;
    params.maxCorrDist = 0.1f;
    params.huberScale = 0.05f;

    // The historical gate: an absolute inlier floor the overlapping patch clears on its own.
    params.minFitness = 0.0f;
    const Registration::RegistrationResult ungated =
            icp.Solve(source, sourceNormals, target, Eigen::Matrix4f::Identity(), params);
    EXPECT_TRUE(ungated.valid) << "fixture is wrong: the patch must produce enough inliers to pass "
                                  "the bare minInliers floor, or this proves nothing";
    EXPECT_LT(ungated.fitness, 0.2f);

    params.minFitness = 0.4f;
    const Registration::RegistrationResult gated =
            icp.Solve(source, sourceNormals, target, Eigen::Matrix4f::Identity(), params);
    EXPECT_FALSE(gated.valid) << "a solve backed by " << gated.fitness * 100.0f
                              << "% of the source must not be reported as a good track";
}

namespace {

    // A Tracker with a scripted verdict, so the integration stage's fusion policy can be exercised
    // without depending on whether real ICP happens to converge. Also records, per call, the frame
    // index of the map it was handed -- which is what the lock-step test below inspects.
    class ScriptedVerdictTracker : public ep::Tracker {
    public:
        ScriptedVerdictTracker(bool firstFrameSucceeds, ep::ETrackFailure failure, float strayMetres,
                               std::shared_ptr<std::vector<int>> observedModelFrames = nullptr)
            : m_firstFrameSucceeds(firstFrameSucceeds), m_failure(failure), m_stray(strayMetres),
              m_observedModelFrames(std::move(observedModelFrames)) {}

        const char *Name() const override { return "scripted-verdict"; }

        ep::TrackingResult Track(const ep::Frame &, const ep::ModelSnapshot *model,
                                 const Eigen::Isometry3f &) override {
            if (m_observedModelFrames)
                m_observedModelFrames->push_back(model ? model->processedFrame : -1);
            const bool firstCall = m_callIndex++ == 0;
            // failure == None scripts a tracker that never fails, so every frame is fused.
            const bool succeeds = m_failure == ep::ETrackFailure::None || (firstCall && m_firstFrameSucceeds);
            ep::TrackingResult r;
            if (succeeds) {
                r.valid = true;
                r.failure = ep::ETrackFailure::None;
                r.pose = Eigen::Isometry3f::Identity();
                return r;
            }
            r.valid = false;
            r.failure = m_failure;
            r.pose = Eigen::Isometry3f(Eigen::Translation3f(m_stray, 0.0f, 0.0f));
            return r;
        }

    private:
        bool m_firstFrameSucceeds;
        ep::ETrackFailure m_failure;
        float m_stray;
        std::shared_ptr<std::vector<int>> m_observedModelFrames;
        std::size_t m_callIndex = 0;
    };

} // namespace

// A track that failed its overlap gate carries a knowingly-wrong pose, and the map it corrupts is
// the next frame's alignment target. Measured on the 477-frame capture/ recording: roughly half the
// frames take that path, so half the map was being built from poses the tracker itself rejected.
//
// The assertion is on the map's EXTENT, not its entry count: fusing at the previous pose (the old
// behaviour) piles the frame on top of the existing surface, which barely moves a count but smears
// the extent by the stray displacement. A count-based assertion passes the bug.
TEST(Pipeline, AFrameWhoseOverlapGateFailedIsNotFusedIntoTheMap) {
    constexpr int kFrames = 6;
    constexpr float kStrayMetres = 5.0f;
    ep::Pipeline::Config cfg = makeConfig(kFrames);
    cfg.acquisition.realTime = false;

    auto tracker = std::make_unique<ScriptedVerdictTracker>(
            /*firstFrameSucceeds=*/true, ep::ETrackFailure::LowOverlap, kStrayMetres);
    ep::Pipeline pipe(cfg, std::move(tracker));
    pipe.Start();
    pipe.SetPaused(false);
    ASSERT_TRUE(waitProcessed(pipe, kFrames - 1));
    pipe.CheckErrors();

    const std::shared_ptr<const ep::ModelSnapshot> model = pipe.LatestModel();
    ASSERT_NE(model, nullptr);
    ASSERT_FALSE(model->entries.empty()) << "frame 0 was adopted, so the map must exist";
    ASSERT_TRUE(model->hasAlloc);

    // One 0.8 m plane patch plus a truncation band on each side -- nowhere near the 5 m stray.
    const float extentX = model->allocMax.x() - model->allocMin.x();
    EXPECT_LT(extentX, 2.0f) << "map spans " << extentX
                             << " m in x: a frame rejected for low overlap was fused anyway";

    const ep::PipelineStats stats = pipe.GetStats();
    EXPECT_EQ(stats.skippedFusions, std::uint64_t(kFrames - 1));
    EXPECT_EQ(stats.processedFrame, kFrames - 1) << "a skipped fusion must still advance the frame "
                                                    "index -- callers wait on it";
    pipe.Stop();
}

// The first frames legitimately have no map to align against: GpuIcpTracker reports NoModel while
// model->entries is empty. Refusing to fuse those is not a safety measure but a deadlock -- the map
// never bootstraps, so every later frame is NoModel too and the reconstruction stays empty. This is
// the guard on ShouldFuse()'s NoModel exemption; it fails if the policy becomes "skip everything
// the tracker rejected".
TEST(Pipeline, AFrameWithNoMapYetIsStillFused) {
    constexpr int kFrames = 4;
    ep::Pipeline::Config cfg = makeConfig(kFrames);
    cfg.acquisition.realTime = false;

    auto tracker = std::make_unique<ScriptedVerdictTracker>(
            /*firstFrameSucceeds=*/false, ep::ETrackFailure::NoModel, /*strayMetres=*/0.0f);
    ep::Pipeline pipe(cfg, std::move(tracker));
    pipe.Start();
    pipe.SetPaused(false);
    ASSERT_TRUE(waitProcessed(pipe, kFrames - 1));
    pipe.CheckErrors();

    const std::shared_ptr<const ep::ModelSnapshot> model = pipe.LatestModel();
    ASSERT_NE(model, nullptr);
    EXPECT_FALSE(model->entries.empty()) << "nothing was fused, so the map can never bootstrap";
    EXPECT_EQ(pipe.GetStats().skippedFusions, std::uint64_t(0));
    pipe.Stop();
}

// Blocking channels make a replay lossless; they do NOT make it reproducible. The map reaches the
// tracker through a latest-wins Mailbox, and registration may run ahead of integration by the whole
// trackedFrames capacity, so WHICH map version frame N aligns against depends on thread scheduling.
// That map is the alignment target, so the pose changes, so the next map changes: the run diverges.
// Measured on capture/ before the handshake -- four runs of ONE command reported trajectory lengths
// of 1.47, 6.45, 7.84 and 136.76 metres.
//
// Asserted as the exact lock-step sequence rather than "two runs agree", because a flaky-timing bug
// can make two runs agree by luck; -1, 0, 1, 2, ... can only hold if every frame really did wait.
TEST(Pipeline, ALosslessReplayAlignsEachFrameAgainstEveryEarlierFrame) {
    constexpr int kFrames = 8;
    ep::Pipeline::Config cfg = makeConfig(kFrames);
    cfg.acquisition.realTime = false; // a recording: lossless AND lock-stepped

    auto observed = std::make_shared<std::vector<int>>();
    auto tracker = std::make_unique<ScriptedVerdictTracker>(
            /*firstFrameSucceeds=*/true, ep::ETrackFailure::None, /*strayMetres=*/0.0f, observed);
    ep::Pipeline pipe(cfg, std::move(tracker));
    pipe.Start();
    pipe.SetPaused(false);
    ASSERT_TRUE(waitProcessed(pipe, kFrames - 1));
    pipe.CheckErrors();
    pipe.Stop(); // joins the registration thread, so `observed` is safe to read

    ASSERT_GE(observed->size(), std::size_t(kFrames));
    for (int k = 0; k < kFrames; ++k)
        EXPECT_EQ((*observed)[std::size_t(k)], k - 1)
                << "frame " << k << " aligned against a map holding frames 0.." << (*observed)[k]
                << " instead of 0.." << (k - 1) << ": integration was not waited for";
}

namespace {

    // Perfect tracker on a scripted pose sequence, with scripted rejections -- for testing what
    // prior RegistrationThread hands to Track around a rejection, not whether ICP converges.
    class RecordingTrackerWithScriptedRejections : public ep::Tracker {
    public:
        RecordingTrackerWithScriptedRejections(std::vector<Eigen::Isometry3f> truePoses,
                                               std::set<std::size_t> rejectedCalls,
                                               std::shared_ptr<std::vector<Eigen::Isometry3f>> priors)
            : m_truePoses(std::move(truePoses)), m_rejectedCalls(std::move(rejectedCalls)),
              m_recordedPriors(std::move(priors)) {}

        const char *Name() const override { return "recording-with-rejections"; }

        ep::TrackingResult Track(const ep::Frame &, const ep::ModelSnapshot *,
                                 const Eigen::Isometry3f &priorPose) override {
            m_recordedPriors->push_back(priorPose);
            const std::size_t call = m_callIndex++;
            ep::TrackingResult r;
            if (m_rejectedCalls.count(call) || call >= m_truePoses.size()) {
                r.valid = false;
                r.failure = ep::ETrackFailure::LowOverlap;
                r.pose = priorPose;
                return r;
            }
            r.valid = true;
            r.failure = ep::ETrackFailure::None;
            r.pose = m_truePoses[call];
            return r;
        }

    private:
        std::vector<Eigen::Isometry3f> m_truePoses;
        std::set<std::size_t> m_rejectedCalls;
        std::shared_ptr<std::vector<Eigen::Isometry3f>> m_recordedPriors;
        std::size_t m_callIndex = 0;
    };

} // namespace

// After a rejection, the first re-adopted pose and the pose from BEFORE the rejection are two frame
// intervals apart. Re-arming the constant-velocity prior from that pair extrapolates a two-interval
// delta as if it were one -- the prior overshoots by a full frame of motion, which on the 477-frame
// capture/ recording pushed every second solve out of its convergence basin: a self-sustaining
// period-2 oscillation that rejected 229 of 477 frames (adopt fit ~0.9 / reject fit ~0.12,
// perfectly alternating). The prior may extrapolate only from two CONSECUTIVELY adopted poses;
// until it has them, the safe prior is the previous pose unchanged.
TEST(RegistrationThread, VelocityPriorNeedsTwoConsecutiveAdoptionsAfterARejection) {
    ep::CommunicationModule comm;

    const float v = 0.1f; // metres per frame along +X
    auto poseAt = [&](int k) {
        Eigen::Isometry3f p = Eigen::Isometry3f::Identity();
        p.translate(Vector3f(v * float(k), 0.0f, 0.0f));
        return p;
    };
    // Call k adopts pose p_{k+1}; call 3 is rejected.
    std::vector<Eigen::Isometry3f> truePoses;
    for (int k = 1; k <= 7; ++k) truePoses.push_back(poseAt(k));

    auto priors = std::make_shared<std::vector<Eigen::Isometry3f>>();
    auto tracker = std::make_unique<RecordingTrackerWithScriptedRejections>(
            truePoses, std::set<std::size_t>{3}, priors);

    ep::RegistrationThread rt(comm, std::move(tracker));
    rt.Start();
    // Interleaved push/pop: trackedFrames is a capacity-4 DROPPING channel here (default comm), so
    // pushing all 7 up front can drop tracked frames and leave the later Pops blocked forever.
    ep::TrackedFrame tf;
    for (int i = 0; i < 7; ++i) {
        comm.capturedFrames.Push(ep::Frame{});
        ASSERT_TRUE(comm.trackedFrames.Pop(tf)) << "tracked frame " << i;
    }
    rt.Stop();
    ASSERT_EQ(rt.Error(), nullptr);
    ASSERT_EQ(priors->size(), 7u);

    // Call 4 (first after the rejection): previous pose unchanged -- already the behaviour.
    EXPECT_NEAR(((*priors)[4].translation() - poseAt(3).translation()).norm(), 0.0f, 1e-6f)
            << "the frame right after a rejection must start from the previous pose";

    // Call 5 (second after the rejection): the last two adoptions (p3 at call 2, p5 at call 4) are
    // NOT consecutive frames, so no velocity may be extrapolated from them. The buggy re-arm
    // produces p5 + (p5 - p3) = p7 here -- a double-length extrapolation.
    EXPECT_NEAR(((*priors)[5].translation() - poseAt(5).translation()).norm(), 0.0f, 1e-6f)
            << "prior two frames after a rejection extrapolated a two-interval delta: got x="
            << (*priors)[5].translation().x() << ", want the previous pose x=" << poseAt(5).translation().x();

    // Call 6: calls 4 and 5 were consecutive adoptions (p5, p6) -- a true one-interval delta, so
    // the velocity prior is legitimately re-armed and predicts p7 exactly.
    EXPECT_NEAR(((*priors)[6].translation() - poseAt(7).translation()).norm(), 0.0f, 1e-6f)
            << "after two consecutive adoptions the velocity prior must be re-armed";
}

// The fitness gate cannot catch every mis-convergence: on the 477-frame capture/ recording the
// solve that first corrupted the map claimed a 0.225 m single-frame step at fitness 0.571 -- past
// the 0.4 gate, physically impossible for a 30 fps hand-held camera (bound ~0.05 m/frame; docs and
// PipelineStats both state it). The step gate is the second line: a solve whose translation from
// the prior exceeds maxStepMeters is a mis-convergence BY PHYSICS, whatever its fitness says.
TEST(Pipeline, GpuIcpTrackerRejectsAPhysicallyImplausibleStep) {
    const Corner corner = makeCorner();

    ep::ModelSnapshot model;
    model.entries.reserve(corner.pts.size());
    for (std::size_t i = 0; i < corner.pts.size(); ++i) {
        TSDFVoxel e{};
        e.center = corner.pts[i];
        e.normal = corner.nrm[i];
        model.entries.push_back(e);
    }

    // Same well-behaved ~0.027 m recovery as GpuIcpTrackerRecoversPerturbation...
    Eigen::Isometry3f perturb = Eigen::Isometry3f::Identity();
    perturb.translate(Vector3f(0.02f, -0.015f, 0.01f));
    ep::Frame frame;
    for (std::size_t i = 0; i < corner.pts.size(); ++i) {
        frame.pts.push_back(perturb * corner.pts[i]);
        frame.nrm.push_back(perturb.rotation() * corner.nrm[i]);
    }

    // ...which a tightened gate must reject as implausible, returning the prior untouched.
    {
        auto tracker = std::make_unique<ep::GpuIcpTracker>();
        tracker->SetMaxStepMeters(0.005f);
        const ep::TrackingResult r = tracker->Track(frame, &model, Eigen::Isometry3f::Identity());
        EXPECT_FALSE(r.valid) << "a 0.027 m step passed a 0.005 m physical bound";
        EXPECT_EQ(r.failure, ep::ETrackFailure::ImplausibleMotion);
        EXPECT_TRUE(r.pose.matrix().isApprox(Eigen::Matrix4f::Identity()))
                << "a gated solve must hand back the prior, not the implausible pose";
    }

    // The DEFAULT bound (0.08 m -- above any real hand-held per-frame motion) must not touch the
    // same well-behaved solve; this is the guard that the gate does not strangle normal tracking.
    {
        auto tracker = std::make_unique<ep::GpuIcpTracker>();
        const ep::TrackingResult r = tracker->Track(frame, &model, Eigen::Isometry3f::Identity());
        EXPECT_TRUE(r.valid) << "the default step bound rejected an ordinary 0.027 m recovery";
        EXPECT_EQ(r.failure, ep::ETrackFailure::None);
    }

    // Fusion policy: an implausible-motion frame has a real local map its wrong pose would corrupt.
    EXPECT_FALSE(ep::ShouldFuse(false, ep::ETrackFailure::ImplausibleMotion));
}

// The velocity prior earns its keep only when per-frame motion is large: its prediction error is
// the pose-estimate noise in the last delta, re-applied. At 0.1 m/frame (the straight-line test)
// that noise is a rounding error on the prediction; at the ~5 mm/frame of a slow hand-held sweep
// the noise IS the prediction, and measured on capture/ it pushed every third solve out of its
// convergence basin (153 of 477 frames gated as implausible, period-3) while the plain
// previous-pose prior tracked all 476. Below the motion threshold the previous pose is already
// inside the solve's basin, so extrapolation adds noise and nothing else.
TEST(RegistrationThread, SlowMotionUsesThePreviousPosePrior) {
    ep::CommunicationModule comm;

    const float v = 0.005f; // 5 mm per frame -- capture/'s measured scale, far below the threshold
    auto poseAt = [&](int k) {
        Eigen::Isometry3f p = Eigen::Isometry3f::Identity();
        p.translate(Vector3f(v * float(k), 0.0f, 0.0f));
        return p;
    };
    std::vector<Eigen::Isometry3f> truePoses;
    for (int k = 1; k <= 5; ++k) truePoses.push_back(poseAt(k));

    auto priors = std::make_shared<std::vector<Eigen::Isometry3f>>();
    auto tracker = std::make_unique<RecordingTrackerWithScriptedRejections>(
            truePoses, std::set<std::size_t>{}, priors);

    ep::RegistrationThread rt(comm, std::move(tracker));
    rt.Start();
    ep::TrackedFrame tf;
    for (int i = 0; i < 5; ++i) {
        comm.capturedFrames.Push(ep::Frame{});
        ASSERT_TRUE(comm.trackedFrames.Pop(tf)) << "tracked frame " << i;
    }
    rt.Stop();
    ASSERT_EQ(rt.Error(), nullptr);
    ASSERT_EQ(priors->size(), 5u);

    // Every prior from call 2 on must be the PREVIOUS pose, not a velocity extrapolation: at this
    // motion scale extrapolating gains nothing and doubles the noise.
    for (int k = 2; k < 5; ++k)
        EXPECT_NEAR(((*priors)[std::size_t(k)].translation() - poseAt(k).translation()).norm(), 0.0f,
                    1e-6f)
                << "call " << k << " got a velocity-extrapolated prior at 5 mm/frame motion (x="
                << (*priors)[std::size_t(k)].translation().x() << ", previous pose x="
                << poseAt(k).translation().x() << ")";
}

namespace {

    // Meter-scale bumpy sphere — the same shape test_registration.cpp proves FPFH+RANSAC on (mm
    // scale there). The radius perturbation breaks both the FPFH degeneracy of a smooth surface and
    // the sphere's rotational symmetry, so global registration and point-to-plane ICP are each
    // well-posed on it. Model entries carry tsdf=0 and the snapshot truncationDistance=0, so the
    // sub-voxel surface point IS the entry center — the fixture needs no TSDF band model.
    struct SphereFixture {
        ep::ModelSnapshot model;
        ep::Frame frame; // the model surface expressed in an identity sensor pose
    };

    SphereFixture makeBumpySphereFixture(int n = 600, float radius = 0.2f, float voxel = 0.02f) {
        SphereFixture f;
        f.model.voxel = voxel;
        f.model.entries.reserve(std::size_t(n));
        f.frame.pts.reserve(std::size_t(n));
        f.frame.nrm.reserve(std::size_t(n));
        for (int i = 0; i < n; ++i) {
            const float a = 2.399963f * float(i), z = 1.0f - 2.0f * (float(i) + 0.5f) / float(n);
            const float rr = std::sqrt(std::max(0.0f, 1.0f - z * z));
            const Vector3f direction(rr * std::cos(a), rr * std::sin(a), z);
            const Vector3f point = radius * (1.0f + 0.15f * std::sin(0.7f * float(i))) * direction;
            TSDFVoxel e{};
            e.center = point;
            e.normal = direction;
            f.model.entries.push_back(e);
            f.frame.pts.push_back(point);
            f.frame.nrm.push_back(direction);
        }
        return f;
    }

    // A prior far enough from the truth (identity) that local ICP cannot adopt a solve — the sphere
    // surfaces barely graze, so the solve either finds too few correspondences, keeps too small a
    // share, or would have to claim a step no hand-held camera makes. All three are the
    // "healthy map, failed solve" causes a relocalizer must react to (never NoLocalTarget, which
    // would mean the fixture parked the frame entirely off the map and armed nothing).
    Eigen::Isometry3f lostPrior() {
        Eigen::Isometry3f prior = Eigen::Isometry3f::Identity();
        prior.translate(Vector3f(0.25f, 0.25f, 0.25f));
        return prior;
    }

} // namespace

// The composite relocalizing tracker is a registered strategy like every other tracker.
TEST(Pipeline, TrackerRegistryHasIcpPlusGlobal) {
    const std::unique_ptr<ep::Tracker> tracker = ep::TrackerRegistry::Default().Create("icp+global");
    ASSERT_NE(tracker, nullptr);
    EXPECT_STREQ(tracker->Name(), "icp+global");
}

// The standalone "global" tracker must scale its pipeline to the MAP resolution (model->voxel),
// exactly as GpuIcpTracker scales maxCorrDist. The default Registration::RegistrationConfig is mm-scale
// (voxelSize 5.0); on a metre-scale scene that collapses each cloud to a handful of octant blobs,
// which cannot express a real sensor transform. The frame is therefore the model surface expressed
// in a NON-trivial sensor pose (an identical frame would let even blob-level matching recover
// identity, hiding the mis-scale), and the prior is far off: global registration is prior-free and
// must recover the true pose regardless.
TEST(Pipeline, GlobalTrackerScalesToModelVoxelAndRecovers) {
    const SphereFixture f = makeBumpySphereFixture();
    Eigen::Isometry3f truePose = Eigen::Isometry3f::Identity();
    truePose.translate(Vector3f(0.15f, -0.1f, 0.1f));
    truePose.rotate(Eigen::AngleAxisf(0.5f, Vector3f(0.2f, 0.7f, 0.6f).normalized()));
    ep::Frame frame;
    frame.pts.reserve(f.frame.pts.size());
    frame.nrm.reserve(f.frame.nrm.size());
    const Eigen::Isometry3f sensorFromWorld = truePose.inverse();
    for (std::size_t i = 0; i < f.frame.pts.size(); ++i) {
        frame.pts.push_back(sensorFromWorld * f.frame.pts[i]);
        frame.nrm.push_back(sensorFromWorld.rotation() * f.frame.nrm[i]);
    }
    const std::unique_ptr<ep::Tracker> tracker = ep::TrackerRegistry::Default().Create("global");
    ASSERT_NE(tracker, nullptr);

    const ep::TrackingResult r = tracker->Track(frame, &f.model, lostPrior());

    ASSERT_TRUE(r.valid) << "global registration failed on a metre-scale scene (voxelSize not "
                            "scaled to model->voxel?)";
    const Eigen::Isometry3f err = r.pose * truePose.inverse();
    EXPECT_LT(err.translation().norm(), 0.03f);
    EXPECT_LT(Eigen::AngleAxisf(err.rotation()).angle(), 0.05f);
}

// When global registration fails against an EXISTING map, the failure must be classified as a
// solve failure (TooFewInliers/LowOverlap — not fusible), never left at the default NoModel:
// ShouldFuse(NoModel) is true, so the misclassification would fuse the failed solve's garbage pose
// into the map. Fixture: a frame whose points all collapse into one downsample cell, so feature
// matching cannot produce the 3 correspondences RANSAC needs — guaranteed invalid.
TEST(Pipeline, GlobalTrackerFailureIsNotFusible) {
    const SphereFixture f = makeBumpySphereFixture();
    ep::Frame degenerate;
    for (int i = 0; i < 5; ++i) {
        degenerate.pts.emplace_back(0.1f + 0.002f * float(i), 0.1f, 0.1f);
        degenerate.nrm.emplace_back(0.0f, 0.0f, 1.0f);
    }
    const std::unique_ptr<ep::Tracker> tracker = ep::TrackerRegistry::Default().Create("global");
    ASSERT_NE(tracker, nullptr);

    const ep::TrackingResult r = tracker->Track(degenerate, &f.model, Eigen::Isometry3f::Identity());

    ASSERT_FALSE(r.valid);
    EXPECT_EQ(r.failure, ep::ETrackFailure::TooFewInliers);
    EXPECT_FALSE(ep::ShouldFuse(r.valid, r.failure))
            << "a failed global solve against an existing map must never be fused";
}

// The composite: local ICP fails N consecutive times against a healthy map -> ONE global
// relocalization attempt fires, recovers the identity truth from a prior 0.43 m off, and the
// result passes the local refine gates (the refine is seeded AT the global pose, so the
// maxStepMeters gate never sees the recovery jump). Afterwards plain local tracking resumes
// without further attempts.
TEST(Pipeline, RelocalizingTrackerRecoversAfterConsecutiveFailures) {
    const SphereFixture f = makeBumpySphereFixture();
    ep::RelocalizingIcpTracker tracker;

    for (int i = 0; i < ep::RelocalizingIcpTracker::kDefaultFailuresBeforeGlobal - 1; ++i) {
        const ep::TrackingResult r = tracker.Track(f.frame, &f.model, lostPrior());
        EXPECT_FALSE(r.valid) << "call " << i << " should fail against the displaced prior";
        EXPECT_NE(r.failure, ep::ETrackFailure::NoModel) << "call " << i;
        EXPECT_NE(r.failure, ep::ETrackFailure::NoLocalTarget)
                << "call " << i << ": fixture parked the frame off the map — arms nothing";
        EXPECT_EQ(tracker.Stats().relocalizationAttempts, 0u) << "fired before N failures";
    }

    const ep::TrackingResult r = tracker.Track(f.frame, &f.model, lostPrior());
    ASSERT_TRUE(r.valid) << "relocalization did not recover a trackable pose";
    EXPECT_LT(r.pose.translation().norm(), 0.01f);
    EXPECT_LT(Eigen::AngleAxisf(r.pose.rotation()).angle(), 0.02f);
    EXPECT_EQ(tracker.Stats().relocalizationAttempts, 1u);
    EXPECT_EQ(tracker.Stats().relocalizationSuccesses, 1u);

    // Recovered -> plain local tracking, no further global attempts.
    const ep::TrackingResult next = tracker.Track(f.frame, &f.model, r.pose);
    EXPECT_TRUE(next.valid);
    EXPECT_EQ(tracker.Stats().relocalizationAttempts, 1u);
}

// A failed relocalization must (1) return the LOCAL failure cause — never a fusible one — and
// (2) reset the failure counter, so the expensive global pipeline is throttled to once per N
// failures instead of running every frame while lost. Fixture: a flat plane frame against the
// sphere map — local ICP always fails (plane normals are ~orthogonal to the equator-band normals,
// so the compatibility gate starves the solve), and no rigid pose can put a plane on a sphere, so
// any coarse global hit is rejected by the local refine gates.
TEST(Pipeline, RelocalizingTrackerThrottlesFailedGlobalAttempts) {
    const SphereFixture f = makeBumpySphereFixture();
    ep::Frame plane;
    for (int i = -20; i <= 20; ++i)
        for (int j = -20; j <= 20; ++j) {
            plane.pts.emplace_back(float(i) * 0.02f, float(j) * 0.02f, 0.0f);
            plane.nrm.emplace_back(0.0f, 0.0f, 1.0f);
        }
    ep::RelocalizingIcpTracker tracker;
    constexpr int kN = ep::RelocalizingIcpTracker::kDefaultFailuresBeforeGlobal;

    for (int call = 1; call <= 2 * kN; ++call) {
        const ep::TrackingResult r = tracker.Track(plane, &f.model, Eigen::Isometry3f::Identity());
        EXPECT_FALSE(r.valid) << "call " << call;
        EXPECT_FALSE(ep::ShouldFuse(r.valid, r.failure))
                << "call " << call << ": a failed relocalization leaked a fusible failure cause";
        // One attempt per N consecutive failures — the counter must reset after a failed attempt.
        EXPECT_EQ(tracker.Stats().relocalizationAttempts, std::uint64_t(call / kN)) << "call " << call;
    }
    EXPECT_EQ(tracker.Stats().relocalizationSuccesses, 0u);
}

// Bootstrap guard: NoModel failures (no map yet) must never arm the global fallback — there is
// nothing to relocalize against, and the first frames of every run pass through this state.
TEST(Pipeline, RelocalizingTrackerDoesNotFireGlobalWhileBootstrapping) {
    ep::RelocalizingIcpTracker tracker;
    for (int i = 0; i <= ep::RelocalizingIcpTracker::kDefaultFailuresBeforeGlobal; ++i) {
        const ep::TrackingResult r = tracker.Track(ep::Frame{}, nullptr, Eigen::Isometry3f::Identity());
        EXPECT_FALSE(r.valid);
        EXPECT_EQ(r.failure, ep::ETrackFailure::NoModel);
    }
    EXPECT_EQ(tracker.Stats().relocalizationAttempts, 0u);
}

// The DEFAULT step gate must scale with the map resolution, like maxCorrDist/huberScale already
// do. Registration::kDefaultTrackerMaxStepMeters (0.08) encodes "hand-held 30 fps, metres" — on a map whose
// units make the voxel comparable to or larger than 0.08 (scanData ~mm units: voxel 5.7;
// scan_out: voxel 0.5), solve jitter alone exceeds it and the gate rejects nearly every frame
// (measured: scanData 59/60, scan_out 76/90 ImplausibleMotion). The tuned capture/ behaviour is
// the ratio 0.08/0.05 = 1.6 voxels, so the default becomes max(0.08, 1.6*voxel) — bit-identical
// at voxel 0.05. An EXPLICIT SetMaxStepMeters stays absolute: the caller knows their units.
TEST(Pipeline, GpuIcpTrackerScalesDefaultStepGateToMapVoxel) {
    const Corner corner = makeCorner();
    ep::ModelSnapshot model;
    model.voxel = 0.5f; // coarse map: 1.6*voxel = 0.8 world units
    model.entries.reserve(corner.pts.size());
    for (std::size_t i = 0; i < corner.pts.size(); ++i) {
        TSDFVoxel e{};
        e.center = corner.pts[i];
        e.normal = corner.nrm[i];
        model.entries.push_back(e);
    }
    Eigen::Isometry3f perturb = Eigen::Isometry3f::Identity();
    perturb.translate(Vector3f(0.2f, -0.15f, 0.1f)); // recovery step ~0.27: > 0.08, < 1.6*voxel
    perturb.rotate(Eigen::AngleAxisf(0.03f, Vector3f::UnitZ()));
    ep::Frame frame;
    frame.pts.reserve(corner.pts.size());
    frame.nrm.reserve(corner.pts.size());
    for (std::size_t i = 0; i < corner.pts.size(); ++i) {
        frame.pts.push_back(perturb * corner.pts[i]);
        frame.nrm.push_back(perturb.rotation() * corner.nrm[i]);
    }

    // Default gate: the ~0.27-unit recovery is well under 1.6 voxels — must be adopted.
    {
        auto tracker = std::make_unique<ep::GpuIcpTracker>();
        const ep::TrackingResult r = tracker->Track(frame, &model, Eigen::Isometry3f::Identity());
        ASSERT_TRUE(r.valid) << "default step gate did not scale with model->voxel (failure "
                             << int(r.failure) << ")";
        const Eigen::Matrix4f err = r.pose.matrix() * perturb.matrix() - Eigen::Matrix4f::Identity();
        EXPECT_LT(err.norm(), 5e-3f);
    }

    // Explicit bound: absolute, never voxel-scaled — the same solve must be rejected.
    {
        auto tracker = std::make_unique<ep::GpuIcpTracker>();
        tracker->SetMaxStepMeters(0.05f);
        const ep::TrackingResult r = tracker->Track(frame, &model, Eigen::Isometry3f::Identity());
        EXPECT_FALSE(r.valid);
        EXPECT_EQ(r.failure, ep::ETrackFailure::ImplausibleMotion);
    }
}

// The CPU tracker ("icp-cpu") must classify a failed solve exactly as GpuIcpTracker would — the two
// trackers must judge a solve alike, and the fusion policy depends on the cause: leaving the failure
// at the default NoModel makes ShouldFuse() true, so the failed solve's garbage pose would be fused
// into the map that is the next frame's alignment target. Fixture: a flat +Z plane frame against the
// sphere map — the normal-compatibility gate starves the solve (equator-band normals are orthogonal
// to the plane normal), so it ends with fewer correspondences than minInliers.
TEST(Pipeline, CpuIcpTrackerClassifiesAStarvedSolveAsTooFewInliers) {
    const SphereFixture f = makeBumpySphereFixture();
    ep::Frame plane;
    for (int i = -20; i <= 20; ++i)
        for (int j = -20; j <= 20; ++j) {
            plane.pts.emplace_back(float(i) * 0.02f, float(j) * 0.02f, 0.0f);
            plane.nrm.emplace_back(0.0f, 0.0f, 1.0f);
        }
    ep::PointToPlaneIcpTracker tracker;

    const ep::TrackingResult r = tracker.Track(plane, &f.model, Eigen::Isometry3f::Identity());

    ASSERT_FALSE(r.valid);
    EXPECT_EQ(r.failure, ep::ETrackFailure::TooFewInliers);
    EXPECT_FALSE(ep::ShouldFuse(r.valid, r.failure))
            << "a failed CPU solve against an existing map must never be fused";
}

// Same-judgement parity, fitness side: GpuIcpTracker defaults minFitness to 0.4 when the caller set
// none, and the CPU align itself has no fitness gate at all — so without the tracker enforcing the
// same default, a solve keeping only a sliver of the frame (here 1/3: the rest of the points are far
// off the map and find no correspondence) is adopted by icp-cpu and rejected by icp. Enough inliers,
// too small a share -> LowOverlap, not fusible.
TEST(Pipeline, CpuIcpTrackerDefaultsTheFitnessGateLikeTheGpuTracker) {
    const SphereFixture f = makeBumpySphereFixture();
    ep::Frame frame = f.frame; // 600 on-map points...
    for (int i = 0; i < 1200; ++i) { // ...plus twice as many far off the map: fitness ~0.33 < 0.4
        frame.pts.emplace_back(10.0f + 0.001f * float(i), 10.0f, 10.0f);
        frame.nrm.emplace_back(0.0f, 0.0f, 1.0f);
    }
    ep::PointToPlaneIcpTracker tracker;

    const ep::TrackingResult r = tracker.Track(frame, &f.model, Eigen::Isometry3f::Identity());

    ASSERT_FALSE(r.valid) << "a fitness-0.33 solve passed: the CPU tracker is not defaulting the "
                             "0.4 fitness gate the GPU tracker enforces";
    EXPECT_EQ(r.failure, ep::ETrackFailure::LowOverlap);
    EXPECT_FALSE(ep::ShouldFuse(r.valid, r.failure));
}

// Same-judgement parity, step-gate side: the CPU tracker's DEFAULT maxStepMeters must scale with
// the map resolution exactly like GpuIcpTracker's (max(0.08, 1.6*voxel) — see
// Pipeline.GpuIcpTrackerScalesDefaultStepGateToMapVoxel for the measured rationale). An explicit
// caller-set bound stays absolute.
TEST(Pipeline, CpuIcpTrackerScalesDefaultStepGateToMapVoxel) {
    const Corner corner = makeCorner();
    ep::ModelSnapshot model;
    model.voxel = 0.5f; // coarse map: 1.6*voxel = 0.8 world units
    model.entries.reserve(corner.pts.size());
    for (std::size_t i = 0; i < corner.pts.size(); ++i) {
        TSDFVoxel e{};
        e.center = corner.pts[i];
        e.normal = corner.nrm[i];
        model.entries.push_back(e);
    }
    Eigen::Isometry3f perturb = Eigen::Isometry3f::Identity();
    perturb.translate(Vector3f(0.2f, -0.15f, 0.1f)); // recovery step ~0.27: > 0.08, < 1.6*voxel
    perturb.rotate(Eigen::AngleAxisf(0.03f, Vector3f::UnitZ()));
    ep::Frame frame;
    frame.pts.reserve(corner.pts.size());
    frame.nrm.reserve(corner.pts.size());
    for (std::size_t i = 0; i < corner.pts.size(); ++i) {
        frame.pts.push_back(perturb * corner.pts[i]);
        frame.nrm.push_back(perturb.rotation() * corner.nrm[i]);
    }

    // Default gate: the ~0.27-unit recovery is well under 1.6 voxels — must be adopted.
    {
        ep::PointToPlaneIcpTracker tracker;
        const ep::TrackingResult r = tracker.Track(frame, &model, Eigen::Isometry3f::Identity());
        ASSERT_TRUE(r.valid) << "default step gate did not scale with model->voxel (failure "
                             << int(r.failure) << ")";
        const Eigen::Matrix4f err = r.pose.matrix() * perturb.matrix() - Eigen::Matrix4f::Identity();
        EXPECT_LT(err.norm(), 5e-3f);
    }

    // Explicit bound: absolute, never voxel-scaled — the same solve must be rejected.
    {
        Registration::RegistrationParam params;
        params.maxStepMeters = 0.05f;
        ep::PointToPlaneIcpTracker tracker(params);
        const ep::TrackingResult r = tracker.Track(frame, &model, Eigen::Isometry3f::Identity());
        EXPECT_FALSE(r.valid);
        EXPECT_EQ(r.failure, ep::ETrackFailure::ImplausibleMotion);
    }
}

namespace {

    // A tracker whose only job is to report fixed relocalization counters -- proves the
    // Tracker::Stats() -> RegistrationThread -> Pipeline::GetStats() plumbing without needing an
    // actual lost-and-recovered run.
    struct FixedStatsTracker final : ep::Tracker {
        const char *Name() const override { return "fixed-stats"; }
        ep::TrackingResult Track(const ep::Frame &, const ep::ModelSnapshot *, const Eigen::Isometry3f &) override {
            return {};
        }
        ep::TrackerStats Stats() const override {
            ep::TrackerStats stats;
            stats.relocalizationAttempts = 7;
            stats.relocalizationSuccesses = 3;
            return stats;
        }
    };

} // namespace

// The relocalization fallback is expensive and fires inside the registration thread; if its
// counters never reach PipelineStats, a live run cannot tell "tracking is fine" from "the global
// fallback is silently firing every N frames". The pipeline must surface whatever the tracker's
// Stats() reports.
TEST(Pipeline, StatsSurfaceTheTrackersRelocalizationCounters) {
    ep::Pipeline pipe(makeConfig(1), std::make_unique<FixedStatsTracker>());

    const ep::PipelineStats stats = pipe.GetStats();

    EXPECT_EQ(stats.relocalizationAttempts, 7u);
    EXPECT_EQ(stats.relocalizationSuccesses, 3u);
}


namespace {

    // Collect the centers ForEachEntryInBox visits, sorted for comparison.
    std::vector<Vector3f> collectEntriesInBox(const ep::ModelSnapshot &snap,
                                              const Vector3f &minimum,
                                              const Vector3f &maximum) {
        std::vector<Vector3f> centers;
        ep::ForEachEntryInBox(snap, minimum, maximum,
                              [&](const TSDFVoxel &entry) { centers.push_back(entry.center); });
        std::sort(centers.begin(), centers.end(), [](const Vector3f &a, const Vector3f &b) {
            if (a.x() != b.x()) return a.x() < b.x();
            if (a.y() != b.y()) return a.y() < b.y();
            return a.z() < b.z();
        });
        return centers;
    }

} // namespace

// The snapshot entry index must return EXACTLY what the linear AABB filter returns -- including
// entries sitting on bucket boundaries, and entries outside the box that share a bucket with ones
// inside (the per-entry re-check).
TEST(Pipeline, SnapshotEntryIndexQueryMatchesTheLinearFilter) {
    ep::ModelSnapshot indexed;
    indexed.voxel = 0.02f;
    for (int x = 0; x < 20; ++x)
        for (int y = 0; y < 20; ++y) {
            TSDFVoxel e{};
            e.center = Vector3f(float(x) * 0.1f, float(y) * 0.1f, 0.5f);
            e.normal = Vector3f(0, 0, 1);
            indexed.entries.push_back(e);
        }
    ep::BuildSnapshotEntryIndex(indexed);
    ASSERT_GT(indexed.entryBucketSize, 0.0f);

    ep::ModelSnapshot linear = indexed; // same entries, then drop the index -> linear fallback
    linear.entryBucketSize = 0.0f;
    linear.entryBuckets.clear();

    const Vector3f minimum(0.35f, 0.35f, 0.0f), maximum(1.25f, 1.25f, 1.0f);
    const std::vector<Vector3f> viaIndex = collectEntriesInBox(indexed, minimum, maximum);
    const std::vector<Vector3f> viaLinear = collectEntriesInBox(linear, minimum, maximum);

    ASSERT_FALSE(viaLinear.empty());
    ASSERT_EQ(viaIndex.size(), viaLinear.size());
    for (std::size_t i = 0; i < viaIndex.size(); ++i)
        EXPECT_EQ(viaIndex[i], viaLinear[i]) << "entry " << i;

    // A box past the map must stay empty through the index too.
    EXPECT_TRUE(collectEntriesInBox(indexed, Vector3f(50, 50, 50), Vector3f(60, 60, 60)).empty());
}

// A hand-built snapshot (every direct-tracker test, and any caller predating the index) has no
// buckets; queries must fall back to the linear scan instead of silently returning nothing.
TEST(Pipeline, SnapshotWithoutAnIndexFallsBackToTheLinearScan) {
    ep::ModelSnapshot snap;
    snap.voxel = 0.02f; // voxel set but index never built
    TSDFVoxel e{};
    e.center = Vector3f(0.1f, 0.2f, 0.3f);
    snap.entries.push_back(e);

    const std::vector<Vector3f> hits =
            collectEntriesInBox(snap, Vector3f(0, 0, 0), Vector3f(1, 1, 1));
    ASSERT_EQ(hits.size(), 1u);
    EXPECT_EQ(hits[0], e.center);
}

// The GPU tracker must CONSULT the index when one exists: a lying index (bucketSize set, buckets
// empty) must starve the crop into NoLocalTarget even though a linear scan of `entries` would have
// found the whole map. This is the seam proof -- without it the tracker could keep its linear scan
// and every other test would still pass.
TEST(Pipeline, GpuIcpTrackerConsultsTheSnapshotEntryIndex) {
    SphereFixture f = makeBumpySphereFixture();
    ep::BuildSnapshotEntryIndex(f.model);
    ASSERT_GT(f.model.entryBucketSize, 0.0f);
    f.model.entryBuckets.clear(); // the lie: an index that says "nothing anywhere"

    ep::GpuIcpTracker tracker;
    const ep::TrackingResult r = tracker.Track(f.frame, &f.model, Eigen::Isometry3f::Identity());

    EXPECT_FALSE(r.valid);
    EXPECT_EQ(r.failure, ep::ETrackFailure::NoLocalTarget)
            << "the tracker ignored the entry index and scanned entries linearly";
}

// The pipeline's published snapshots must carry the index (built on the integration thread), and
// it must cover every entry exactly once -- a partial index would silently shrink every crop.
TEST(Pipeline, PublishedSnapshotsCarryAFullEntryIndex) {
    ep::Pipeline pipe(makeConfig(3), identity());
    pipe.Start();
    ASSERT_TRUE(waitProcessed(pipe, 2));
    pipe.CheckErrors();
    const auto snap = pipe.LatestModel();
    ASSERT_NE(snap, nullptr);
    ASSERT_FALSE(snap->entries.empty());

    EXPECT_GT(snap->entryBucketSize, 0.0f);
    std::size_t indexedEntries = 0;
    for (const auto &bucket: snap->entryBuckets) indexedEntries += bucket.second.size();
    EXPECT_EQ(indexedEntries, snap->entries.size());
    pipe.Stop();
}

// Reconfigure destroys the old stages BEFORE building the new ones -- it has to, because the old
// acquisition stage still holds the camera and the new one cannot open it until that is released.
// So a constructor that throws (a D435 reopened too soon after Close is the real case) leaves every
// stage pointer null, and the next GetStats() dereferences one.
//
// realsense_scan's "Apply & restart" hit exactly this: the throw escaped the ImGui callback and
// killed the process. The pipeline does not have to survive as a working pipeline -- the device is
// genuinely gone -- but it must stay ANSWERABLE so the viewer can report the failure.
TEST(Pipeline, AFailedReconfigureLeavesThePipelineAnswerable) {
    int providerBuilds = 0;
    ep::Pipeline::Config cfg = makeConfig(3);
    cfg.acquisition.makeProvider = [&providerBuilds]() -> std::unique_ptr<ep::IDepthProvider> {
        if (++providerBuilds > 1) throw std::runtime_error("device busy");
        return std::make_unique<PlaneDepthProvider>(3);
    };

    ep::Pipeline pipe(cfg, identity());
    pipe.Start();
    ASSERT_EQ(providerBuilds, 1);

    EXPECT_THROW(pipe.Reconfigure(cfg, identity()), std::runtime_error);

    // Every one of these dereferenced a stage pointer that the failed rebuild left null.
    EXPECT_NO_THROW((void) pipe.GetStats());
    EXPECT_NO_THROW((void) pipe.LatestModel());
    EXPECT_NO_THROW((void) pipe.ProcessedFrame());
    EXPECT_NO_THROW(pipe.SetPaused(true));
    EXPECT_NO_THROW(pipe.CheckErrors());
    EXPECT_NO_THROW(pipe.Stop());
}

// What realsense_scan's "Apply & restart" actually does, over a real recording: rebuild every
// stage while the caller holds the Pipeline. This tears down and recreates the acquisition
// thread's Engine::Core::Context and the integration thread's TSDF each time, which the synthetic
// Reconfigure test above does not -- it injects a provider and never touches the depth front end.
TEST(Pipeline, ReconfigureOverARecordingRebuildsRepeatedly) {
    const std::filesystem::path capture = std::filesystem::path(VKBVH_SOURCE_DIR) / "capture";
    if (!std::filesystem::exists(capture / "intrinsics.txt")) GTEST_SKIP() << "no capture/";

    ep::Pipeline::Config cfg;
    cfg.map.baseVoxel = 0.03f;
    cfg.map.truncation = 0.09f;
    cfg.map.submap = false;
    cfg.acquisition.source = ep::EAcquisitionSource::RealsenseFile;
    cfg.acquisition.recordingDirectory = capture.string();
    cfg.acquisition.realTime = false;

    ep::Pipeline pipe(cfg, identity());
    pipe.Start();
    ASSERT_TRUE(waitProcessed(pipe, 1)) << "the recording never started";

    for (int i = 0; i < 3; ++i) {
        cfg.acquisition.scoreThreshold = 0.5f + 0.1f * float(i);
        pipe.Reconfigure(cfg, identity());
        ASSERT_TRUE(waitProcessed(pipe, 1)) << "no frames after Reconfigure #" << i;
        pipe.CheckErrors();
        EXPECT_NE(pipe.LatestModel(), nullptr) << "no model after Reconfigure #" << i;
    }
    pipe.Stop();
}

// Tracker::Configure has to reach whichever tracker actually solves, including through a
// composite. Callers used to configure via dynamic_cast<GpuIcpTracker*>, which MISSES on
// "icp+global" -- RelocalizingIcpTracker CONTAINS a GpuIcpTracker rather than deriving from one --
// so every gate was silently dropped for exactly the tracker whose fallback makes tuning matter.
//
// Asserted through behaviour, not a getter: maxStepMeters is set far below the fixture's known
// displacement, so the solve must be refused as ImplausibleMotion. Left unconfigured, the default
// gate is max(0.08, 1.6*voxel) and the same solve is accepted -- which is what made the dropped
// gate invisible.
TEST(Pipeline, ConfigureReachesTheSolverThroughEveryTracker) {
    const Corner corner = makeCorner();

    ep::ModelSnapshot model;
    model.voxel = 0.02f; // default gate = max(0.08, 1.6*0.02) = 0.08 m
    model.entries.reserve(corner.pts.size());
    for (std::size_t i = 0; i < corner.pts.size(); ++i) {
        TSDFVoxel e{};
        e.center = corner.pts[i];
        e.normal = corner.nrm[i];
        model.entries.push_back(e);
    }

    // ~0.027 m of translation: comfortably under the 0.08 m default, far over the 0.001 m gate set
    // below. That gap is what makes the assertion discriminate.
    Eigen::Isometry3f perturb = Eigen::Isometry3f::Identity();
    perturb.translate(Vector3f(0.02f, -0.015f, 0.01f));
    perturb.rotate(Eigen::AngleAxisf(0.03f, Vector3f::UnitZ()));

    ep::Frame frame;
    for (std::size_t i = 0; i < corner.pts.size(); ++i) {
        frame.pts.push_back(perturb * corner.pts[i]);
        frame.nrm.push_back(perturb.rotation() * corner.nrm[i]);
    }

    for (const char *name: {"icp", "icp-cpu", "icp+global"}) {
        const std::unique_ptr<ep::Tracker> unconfigured = ep::TrackerRegistry::Default().Create(name);
        ASSERT_NE(unconfigured, nullptr) << name;
        const ep::TrackingResult loose =
                unconfigured->Track(frame, &model, Eigen::Isometry3f::Identity());
        ASSERT_TRUE(loose.valid) << name << ": the fixture must converge with the default gate, or "
                                          "the tight-gate assertion below proves nothing";

        const std::unique_ptr<ep::Tracker> configured = ep::TrackerRegistry::Default().Create(name);
        Registration::RegistrationParam gates;
        gates.maxStepMeters = 0.001f; // far below the fixture's displacement
        configured->Configure(gates);
        const ep::TrackingResult tight =
                configured->Track(frame, &model, Eigen::Isometry3f::Identity());

        EXPECT_FALSE(tight.valid) << name << ": Configure did not reach the solver";
        EXPECT_EQ(tight.failure, ep::ETrackFailure::ImplausibleMotion) << name;
    }
}

// A tracker with no gates must ignore Configure rather than reject it -- callers configure
// whatever the registry handed them without knowing which one it is.
TEST(Pipeline, ConfigureIsHarmlessOnGatelessTrackers) {
    const std::unique_ptr<ep::Tracker> tracker = ep::TrackerRegistry::Default().Create("identity");
    ASSERT_NE(tracker, nullptr);
    Registration::RegistrationParam gates;
    gates.maxStepMeters = 0.001f;
    gates.minFitness = 0.99f;
    EXPECT_NO_THROW(tracker->Configure(gates));

    ep::Frame frame;
    frame.pts.emplace_back(0.0f, 0.0f, 1.0f);
    frame.nrm.emplace_back(0.0f, 0.0f, 1.0f);
    const ep::TrackingResult r = tracker->Track(frame, nullptr, Eigen::Isometry3f::Identity());
    EXPECT_TRUE(r.valid) << "identity stopped accepting frames after being configured";
}
