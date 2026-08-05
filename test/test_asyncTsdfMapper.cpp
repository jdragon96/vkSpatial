#include "AsyncTsdfMapper.h"

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

using asyncmap::AsyncTsdfMapper;
using asyncmap::Config;
using asyncmap::MapperFrame;
using asyncmap::planStep;
using asyncmap::StepKind;
using Eigen::Vector3f;

// ---- pure step-planning logic (no Vulkan) ----

TEST(AsyncMapperStep, ForwardOneFrame) {
    const auto s = planStep(/*shown=*/2, /*target=*/5);
    EXPECT_EQ(s.kind, StepKind::Forward);
    EXPECT_EQ(s.frame, 3); // one step toward target
}

TEST(AsyncMapperStep, CaughtUpWaits) {
    EXPECT_EQ(planStep(5, 5).kind, StepKind::Wait);
    EXPECT_EQ(planStep(-1, -1).kind, StepKind::Wait); // nothing requested yet
}

TEST(AsyncMapperStep, BackwardResets) {
    EXPECT_EQ(planStep(/*shown=*/9, /*target=*/3).kind, StepKind::Reset);
}

TEST(AsyncMapperStep, FirstForwardFromEmpty) {
    const auto s = planStep(/*shown=*/-1, /*target=*/0);
    EXPECT_EQ(s.kind, StepKind::Forward);
    EXPECT_EQ(s.frame, 0);
}

// ---- headless smoke: worker integrates to the requested frame off-thread ----

namespace {
    MapperFrame makePlaneFrame(float z) {
        MapperFrame fr;
        for (int i = -20; i <= 20; ++i)
            for (int j = -20; j <= 20; ++j) {
                fr.pts.emplace_back(i * 0.02f, j * 0.02f, z);
                fr.nrm.emplace_back(0.0f, 0.0f, 1.0f);
            }
        fr.cam = Vector3f(0, 0, z + 1.0f);
        return fr;
    }
} // namespace

TEST(AsyncMapper, IntegratesToRequestedFrameOffThread) {
    auto frames = std::make_shared<std::vector<MapperFrame>>();
    frames->push_back(makePlaneFrame(0.0f));
    frames->push_back(makePlaneFrame(0.0f));
    frames->push_back(makePlaneFrame(0.0f));

    Config cfg;
    cfg.baseVoxel = 0.05f;
    cfg.truncation = 0.15f;

    AsyncTsdfMapper mapper;
    mapper.Start(cfg, frames);
    mapper.RequestFrame(2);

    // Poll for the worker to catch up (bounded wait).
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (mapper.ProcessedFrame() < 2 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

    ASSERT_EQ(mapper.ProcessedFrame(), 2);
    const auto snap = mapper.Latest();
    ASSERT_NE(snap, nullptr);
    EXPECT_EQ(snap->processedFrame, 2);
    EXPECT_GT(snap->entries.size(), 100u);
    EXPECT_EQ(snap->entries.size(), snap->isNew.size());
    // Some level must hold the model. This plane is one uniformly-dense block, so detail carries it
    // and base is (correctly) empty -- base only covers non-dense + dense/sparse seam blocks -- so the
    // coverage check is base + detail, not base alone.
    EXPECT_GE(snap->baseTiles + snap->detailTiles, 1u);
    mapper.Stop();
}
