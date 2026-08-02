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
