#include "Pipeline/Integration/FusionGate.h" // Pipeline::FusionGate
#include "Pipeline/Types.h"                  // Pipeline::FusionGateConfig, ETrackFailure

#include <gtest/gtest.h>

namespace {

    using Pipeline::ETrackFailure;
    using Pipeline::FusionGate;
    using Pipeline::FusionGateConfig;

    // A frame the tracker solved successfully, at the given quality.
    bool AdmitSolved(FusionGate &gate, float fitness, float rmse = 0.0f) {
        return gate.Admit(true, ETrackFailure::None, fitness, rmse);
    }

    // The first Admit of a gate's life is the map's seed and passes unconditionally. A test that
    // means to exercise a later gate has to spend it first, or its assertions are vacuous.
    void ConsumeSeedFrame(FusionGate &gate) {
        ASSERT_TRUE(gate.Admit(false, ETrackFailure::NoModel, 0.0f, 0.0f));
    }

    FusionGateConfig BootstrapConfig(int frames, float fitness) {
        FusionGateConfig config;
        config.bootstrapConsecutiveFrames = frames;
        config.bootstrapMinFitness = fitness;
        return config;
    }

    /// A default-constructed gate is OFF: it must not change what ShouldFuse already decided,
    /// or every existing caller (identity tracker, folder eval) silently stops fusing.
    TEST(FusionGate, DefaultConfigAdmitsEveryFrame) {
        FusionGate gate;
        EXPECT_TRUE(gate.IsArmed());
        for (int i = 0; i < 10; ++i) {
            EXPECT_TRUE(AdmitSolved(gate, 0.0f, 10.0f)) << "frame " << i;
        }
        EXPECT_EQ(gate.BootstrapHeldFrames(), 0u);
    }

    /// The identity tracker reports fitness 0 always. A gate that judged it would starve the map.
    TEST(FusionGate, DefaultConfigAdmitsZeroFitnessTracker) {
        FusionGate gate;
        for (int i = 0; i < 5; ++i) EXPECT_TRUE(AdmitSolved(gate, 0.0f));
    }

    /// The map must be seeded or the tracker reports NoModel forever and no fitness ever exists
    /// (see ShouldFuse in Types.h). So the first frame passes regardless of its quality.
    TEST(FusionGate, SeedFrameIsAdmittedDespiteFailingBootstrapFitness) {
        FusionGate gate(BootstrapConfig(5, 0.80f));
        EXPECT_FALSE(gate.IsArmed());
        EXPECT_TRUE(gate.Admit(false, ETrackFailure::NoModel, 0.0f, 0.0f));
    }

    /// After the seed, fusion is held until the run of good frames completes.
    TEST(FusionGate, HoldsFusionUntilConsecutiveRunCompletes) {
        FusionGate gate(BootstrapConfig(5, 0.80f));
        ASSERT_TRUE(gate.Admit(false, ETrackFailure::NoModel, 0.0f, 0.0f)); // seed

        for (int i = 0; i < 5; ++i) {
            EXPECT_FALSE(AdmitSolved(gate, 0.85f)) << "arming frame " << i << " must not fuse";
        }
        EXPECT_TRUE(gate.IsArmed());
        EXPECT_EQ(gate.BootstrapHeldFrames(), 5u);
        EXPECT_TRUE(AdmitSolved(gate, 0.85f)); // fusion resumes on the frame after arming
    }

    /// A single sub-threshold frame restarts the run -- "5 consecutive", not "5 total".
    TEST(FusionGate, ConsecutiveCounterResetsOnASubThresholdFrame) {
        FusionGate gate(BootstrapConfig(5, 0.80f));
        ASSERT_TRUE(gate.Admit(false, ETrackFailure::NoModel, 0.0f, 0.0f)); // seed

        for (int i = 0; i < 4; ++i) ASSERT_FALSE(AdmitSolved(gate, 0.85f));
        ASSERT_FALSE(AdmitSolved(gate, 0.79f)); // miss -> back to zero
        EXPECT_FALSE(gate.IsArmed());

        for (int i = 0; i < 4; ++i) EXPECT_FALSE(AdmitSolved(gate, 0.85f));
        EXPECT_FALSE(gate.IsArmed()) << "four good frames after the reset must not arm";
        EXPECT_FALSE(AdmitSolved(gate, 0.85f)); // fifth completes the run
        EXPECT_TRUE(gate.IsArmed());
    }

    /// Arming is one-way: a later bad frame must not send the map back to bootstrap.
    TEST(FusionGate, StaysArmedAfterALaterLowFitnessFrame) {
        FusionGate gate(BootstrapConfig(2, 0.80f));
        ASSERT_TRUE(gate.Admit(false, ETrackFailure::NoModel, 0.0f, 0.0f));
        ASSERT_FALSE(AdmitSolved(gate, 0.85f));
        ASSERT_FALSE(AdmitSolved(gate, 0.85f));
        ASSERT_TRUE(gate.IsArmed());

        EXPECT_TRUE(AdmitSolved(gate, 0.10f));
        EXPECT_TRUE(gate.IsArmed());
    }

    TEST(FusionGate, MinimumFusionFitnessRejectsAWeakSolve) {
        FusionGateConfig config;
        config.minimumFusionFitness = 0.50f;
        FusionGate gate(config);
        ConsumeSeedFrame(gate);

        EXPECT_TRUE(AdmitSolved(gate, 0.50f)) << "the threshold itself must pass";
        EXPECT_FALSE(AdmitSolved(gate, 0.49f));
        EXPECT_EQ(gate.FusionRejectedByFitness(), 1u);
    }

    TEST(FusionGate, MaximumFusionRmseRejectsANoisySolve) {
        FusionGateConfig config;
        config.maximumFusionRmse = 0.010f;
        FusionGate gate(config);
        ConsumeSeedFrame(gate);

        EXPECT_TRUE(AdmitSolved(gate, 0.9f, 0.010f)) << "the threshold itself must pass";
        EXPECT_FALSE(AdmitSolved(gate, 0.9f, 0.011f));
        EXPECT_EQ(gate.FusionRejectedByRmse(), 1u);
    }

    /// NoModel / NoLocalTarget frames are the ones ShouldFuse deliberately lets through so the map
    /// can bootstrap and grow into new space. They carry no meaningful fitness, so judging them on
    /// it would freeze the map exactly where it is supposed to expand.
    TEST(FusionGate, FramesWithoutASolveBypassTheSteadyGates) {
        FusionGateConfig config;
        config.minimumFusionFitness = 0.50f;
        config.maximumFusionRmse = 0.010f;
        FusionGate gate(config);
        ConsumeSeedFrame(gate);
        ASSERT_TRUE(AdmitSolved(gate, 0.9f, 0.001f));

        EXPECT_TRUE(gate.Admit(false, ETrackFailure::NoLocalTarget, 0.0f, 0.0f));
        EXPECT_TRUE(gate.Admit(false, ETrackFailure::NoModel, 0.0f, 0.0f));
        EXPECT_EQ(gate.FusionRejectedByFitness(), 0u);
        EXPECT_EQ(gate.FusionRejectedByRmse(), 0u);
    }

    /// Every refusal has to be countable -- a silently dropped frame is indistinguishable from a
    /// frame that was never captured.
    TEST(FusionGate, RefusalsAreCounted) {
        FusionGateConfig config = BootstrapConfig(3, 0.80f);
        config.minimumFusionFitness = 0.50f;
        FusionGate gate(config);

        ASSERT_TRUE(gate.Admit(false, ETrackFailure::NoModel, 0.0f, 0.0f));
        for (int i = 0; i < 3; ++i) ASSERT_FALSE(AdmitSolved(gate, 0.85f));
        ASSERT_TRUE(gate.IsArmed());
        ASSERT_FALSE(AdmitSolved(gate, 0.20f));

        EXPECT_EQ(gate.BootstrapHeldFrames(), 3u);
        EXPECT_EQ(gate.FusionRejectedByFitness(), 1u);
    }

} // namespace
