#include "ReconstructionPipeline.h"

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

using pipeline::Frame;
using pipeline::IdentityAlignment;
using pipeline::ReconstructionPipeline;
using Eigen::Vector3f;

namespace {
    Frame makePlaneFrame() {
        Frame fr;
        for (int i = -20; i <= 20; ++i)
            for (int j = -20; j <= 20; ++j) {
                fr.pts.emplace_back(i * 0.02f, j * 0.02f, 0.0f);
                fr.nrm.emplace_back(0.0f, 0.0f, 1.0f);
            }
        fr.cam = Vector3f(0, 0, 1);
        return fr;
    }

    bool waitProcessed(const ReconstructionPipeline &p, int target, int seconds = 60) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
        while (p.ProcessedFrame() < target && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return p.ProcessedFrame() >= target;
    }

    ReconstructionPipeline::Config cornerCfg(std::shared_ptr<const std::vector<Frame>> frames) {
        ReconstructionPipeline::Config cfg;
        cfg.map.baseVoxel = 0.05f;
        cfg.map.truncation = 0.15f;
        cfg.captureQueue = 16; // > frame count -> no drops in the test
        (void) frames;         // density is learned online now (no pre-scan frame set)
        return cfg;
    }
} // namespace

TEST(ReconstructionPipeline, IntegratesPushedFramesOffThread) {
    auto frames = std::make_shared<std::vector<Frame>>();
    for (int k = 0; k < 3; ++k) frames->push_back(makePlaneFrame());

    ReconstructionPipeline pipe;
    pipe.Start(cornerCfg(frames), std::make_unique<IdentityAlignment>());
    for (int k = 0; k < 3; ++k) pipe.PushFrame((*frames)[k]);

    ASSERT_TRUE(waitProcessed(pipe, 2));
    const auto snap = pipe.LatestModel();
    ASSERT_NE(snap, nullptr);
    EXPECT_EQ(snap->processedFrame, 2);
    EXPECT_GT(snap->entries.size(), 100u);
    EXPECT_EQ(snap->entries.size(), snap->isNew.size());
    EXPECT_EQ(snap->entries.size(), snap->firstFrame.size());
    EXPECT_EQ(pipe.RequestedFrame(), 2);
    pipe.Stop();
}

TEST(ReconstructionPipeline, ResetRestartsProcessedCounter) {
    auto frames = std::make_shared<std::vector<Frame>>();
    for (int k = 0; k < 3; ++k) frames->push_back(makePlaneFrame());

    ReconstructionPipeline pipe;
    pipe.Start(cornerCfg(frames), std::make_unique<IdentityAlignment>());
    for (int k = 0; k < 3; ++k) pipe.PushFrame((*frames)[k]);
    ASSERT_TRUE(waitProcessed(pipe, 2));

    pipe.Reset();
    EXPECT_EQ(pipe.RequestedFrame(), -1);
    pipe.PushFrame((*frames)[0]);
    ASSERT_TRUE(waitProcessed(pipe, 0));
    const auto snap = pipe.LatestModel();
    ASSERT_NE(snap, nullptr);
    EXPECT_EQ(snap->processedFrame, 0); // counter restarted after reset
    EXPECT_GT(snap->entries.size(), 100u);
    pipe.Stop();
}
