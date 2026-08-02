#include "utilities/StageProfiler.h"

#include <gtest/gtest.h>

#include <string>

TEST(Stopwatch, MonotonicNonNegative) {
    util::Stopwatch sw;
    EXPECT_GE(sw.ElapsedMs(), 0.0);
    sw.Reset();
    EXPECT_GE(sw.ElapsedMs(), 0.0);
}

TEST(StageProfiler, AccumulatesPerStage) {
    util::StageProfiler p;
    p.Add("a", 10.0);
    p.Add("a", 20.0);
    p.Add("b", 5.0);
    EXPECT_EQ(p.Count("a"), 2u);
    EXPECT_EQ(p.Count("b"), 1u);
    EXPECT_DOUBLE_EQ(p.TotalMs("a"), 30.0);
    EXPECT_DOUBLE_EQ(p.AvgMs("a"), 15.0);
    EXPECT_DOUBLE_EQ(p.LastMs("a"), 20.0);
    EXPECT_DOUBLE_EQ(p.AvgMs("b"), 5.0);
}

TEST(StageProfiler, UnknownStageIsZero) {
    util::StageProfiler p;
    EXPECT_EQ(p.Count("nope"), 0u);
    EXPECT_DOUBLE_EQ(p.AvgMs("nope"), 0.0);
    EXPECT_DOUBLE_EQ(p.LastMs("nope"), 0.0);
}

TEST(StageProfiler, ReportListsStagesAndPercentages) {
    util::StageProfiler p;
    p.Add("integrate", 30.0);
    p.Add("download", 70.0); // 70% of 100
    const std::string r = p.Report("test");
    EXPECT_NE(r.find("test"), std::string::npos);
    EXPECT_NE(r.find("integrate"), std::string::npos);
    EXPECT_NE(r.find("download"), std::string::npos);
    EXPECT_NE(r.find("70.0%"), std::string::npos); // download share of the summed total
}

TEST(StageProfiler, ResetClears) {
    util::StageProfiler p;
    p.Add("a", 1.0);
    EXPECT_FALSE(p.Empty());
    p.Reset();
    EXPECT_TRUE(p.Empty());
    EXPECT_EQ(p.Count("a"), 0u);
}

TEST(StageProfiler, ScopedTimerRecordsOneSample) {
    util::StageProfiler p;
    {
        util::ScopedStageTimer t(p, "x");
    }
    EXPECT_EQ(p.Count("x"), 1u);
    EXPECT_GE(p.LastMs("x"), 0.0);
}
