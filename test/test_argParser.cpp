#include "utilities/ArgParser.h"

#include <gtest/gtest.h>

#include <initializer_list>
#include <string>
#include <vector>

namespace {

    // Build a mutable (argc, argv) from string literals, keeping the backing strings alive.
    struct Argv {
        std::vector<std::string> store;
        std::vector<char *> ptrs;
        Argv(std::initializer_list<std::string> a) : store(a) {
            for (std::string &s : store) ptrs.push_back(s.data());
        }
        int argc() const { return int(ptrs.size()); }
        char **argv() { return ptrs.data(); }
    };

} // namespace

TEST(ArgParser, ValuePresentAndAbsent) {
    Argv a{"prog", "--dir", "scans/bunny", "--voxel", "0.5"};
    EXPECT_TRUE(util::ArgValue(a.argc(), a.argv(), "--dir").has_value());
    EXPECT_EQ(util::ArgValue(a.argc(), a.argv(), "--dir").value(), "scans/bunny");
    EXPECT_FALSE(util::ArgValue(a.argc(), a.argv(), "--missing").has_value());
}

TEST(ArgParser, StringFallback) {
    Argv a{"prog", "--out", "recon.ply"};
    EXPECT_EQ(util::ArgString(a.argc(), a.argv(), "--out", "def"), "recon.ply");
    EXPECT_EQ(util::ArgString(a.argc(), a.argv(), "--nope", "def"), "def");
}

TEST(ArgParser, IntAndFloatParse) {
    Argv a{"prog", "--frames", "90", "--voxel", "0.25"};
    EXPECT_EQ(util::ArgInt(a.argc(), a.argv(), "--frames", 1), 90);
    EXPECT_EQ(util::ArgInt(a.argc(), a.argv(), "--nope", 7), 7);
    EXPECT_FLOAT_EQ(util::ArgFloat(a.argc(), a.argv(), "--voxel", 1.0f), 0.25f);
    EXPECT_FLOAT_EQ(util::ArgFloat(a.argc(), a.argv(), "--nope", 1.5f), 1.5f);
}

TEST(ArgParser, FlagPresence) {
    Argv a{"prog", "--no-view", "--dir", "x"};
    EXPECT_TRUE(util::HasFlag(a.argc(), a.argv(), "--no-view"));
    EXPECT_FALSE(util::HasFlag(a.argc(), a.argv(), "--hermite"));
    // "--dir" is an option key, but as a bare token it is still "present" for HasFlag.
    EXPECT_TRUE(util::HasFlag(a.argc(), a.argv(), "--dir"));
}

TEST(ArgParser, FirstMatchWins) {
    Argv a{"prog", "--voxel", "0.5", "--voxel", "0.1"};
    EXPECT_FLOAT_EQ(util::ArgFloat(a.argc(), a.argv(), "--voxel", 9.0f), 0.5f);
}

// A key in the final position has no following value: ArgValue is empty (fallback used), yet the
// bare key is still detectable as a flag.
TEST(ArgParser, TrailingKeyHasNoValueButIsAFlag) {
    Argv a{"prog", "--dir"};
    EXPECT_FALSE(util::ArgValue(a.argc(), a.argv(), "--dir").has_value());
    EXPECT_EQ(util::ArgString(a.argc(), a.argv(), "--dir", "def"), "def");
    EXPECT_TRUE(util::HasFlag(a.argc(), a.argv(), "--dir"));
}

TEST(ArgParser, EmptyArgs) {
    Argv a{"prog"};
    EXPECT_FALSE(util::ArgValue(a.argc(), a.argv(), "--dir").has_value());
    EXPECT_FALSE(util::HasFlag(a.argc(), a.argv(), "--dir"));
    EXPECT_EQ(util::ArgInt(a.argc(), a.argv(), "--n", 42), 42);
}

// ---- ArgParser (fluent class) ----

TEST(ArgParserClass, MustSatisfiedIsOk) {
    Argv a{"prog", "--dir", "scans/x", "--voxel", "0.5"};
    util::ArgParser arg =
            util::BuildArgParser(a.argc(), a.argv()).Must("--dir", "need --dir").Option("--voxel");
    EXPECT_TRUE(arg.Ok());
    EXPECT_TRUE(bool(arg));
    EXPECT_EQ(arg.Value("--dir"), "scans/x");
    EXPECT_FLOAT_EQ(arg.ValueFloat("--voxel", 1.0f), 0.5f);
}

TEST(ArgParserClass, MustMissingPrintsMessageAndFails) {
    Argv a{"prog", "--voxel", "0.5"};
    testing::internal::CaptureStderr();
    util::ArgParser arg = util::BuildArgParser(a.argc(), a.argv()).Must("--dir", "need --dir");
    const std::string err = testing::internal::GetCapturedStderr();
    EXPECT_NE(err.find("need --dir"), std::string::npos);
    EXPECT_FALSE(arg.Ok());
    EXPECT_FALSE(bool(arg));
}

// A Must key present with no following value is treated as unsatisfied (and prints).
TEST(ArgParserClass, TrailingMustKeyHasNoValueSoFails) {
    Argv a{"prog", "--dir"};
    testing::internal::CaptureStderr();
    util::ArgParser arg = util::BuildArgParser(a.argc(), a.argv()).Must("--dir", "need --dir value");
    const std::string err = testing::internal::GetCapturedStderr();
    EXPECT_NE(err.find("need --dir value"), std::string::npos);
    EXPECT_FALSE(arg.Ok());
}

TEST(ArgParserClass, TypedReadsAndFlag) {
    Argv a{"prog", "--frames", "12", "--fov", "55", "--no-view"};
    util::ArgParser arg = util::BuildArgParser(a.argc(), a.argv());
    EXPECT_EQ(arg.ValueInt("--frames", 1), 12);
    EXPECT_FLOAT_EQ(arg.ValueFloat("--fov", 1.0f), 55.0f);
    EXPECT_TRUE(arg.Has("--no-view"));
    EXPECT_FALSE(arg.Has("--headless"));
    EXPECT_EQ(arg.Value("--missing", "def"), "def");
}

TEST(ArgParserClass, DeclaredListsMustAndOptionInOrder) {
    Argv a{"prog", "--dir", "x"};
    util::ArgParser arg = util::BuildArgParser(a.argc(), a.argv())
                                  .Must("--dir", "e")
                                  .Option("--voxel")
                                  .Option("--trunc");
    ASSERT_EQ(arg.Declared().size(), 3u);
    EXPECT_EQ(arg.Declared()[0], "--dir");
    EXPECT_EQ(arg.Declared()[1], "--voxel");
    EXPECT_EQ(arg.Declared()[2], "--trunc");
}

// Each unsatisfied Must prints its own message; satisfied ones stay silent.
TEST(ArgParserClass, MultipleMustAccumulateFailures) {
    Argv a{"prog", "--dir", "x"};
    testing::internal::CaptureStderr();
    util::ArgParser arg = util::BuildArgParser(a.argc(), a.argv())
                                  .Must("--dir", "need dir")
                                  .Must("--mesh", "need mesh");
    const std::string err = testing::internal::GetCapturedStderr();
    EXPECT_EQ(err.find("need dir"), std::string::npos); // --dir satisfied -> not printed
    EXPECT_NE(err.find("need mesh"), std::string::npos); // --mesh missing -> printed
    EXPECT_FALSE(arg.Ok());
}
