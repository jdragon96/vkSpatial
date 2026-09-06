#include "Common/PointCloud.h"
#include "utilities/PointCloudIO.h" // the pre-existing ASCII reader, as a cross-check

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

namespace {

    namespace fs = std::filesystem;

    struct TempPly {
        fs::path path;
        explicit TempPly(const std::string &name)
            : path(fs::temp_directory_path() / ("vkbvh_pointcloud_" + name + ".ply")) {
            fs::remove(path);
        }
        ~TempPly() { fs::remove(path); }
        std::string String() const { return path.string(); }
    };

    Common::PointCloud makeCloud(bool withNormals) {
        Common::PointCloud cloud;
        for (int i = 0; i < 9; ++i) {
            cloud.points.emplace_back(float(i) * 0.25f, float(i) * -0.5f, float(i) + 0.125f);
            if (withNormals) cloud.normals.emplace_back(0.0f, float(i) * 0.125f, 1.0f);
        }
        return cloud;
    }

    void ExpectSameCloud(const Common::PointCloud &read, const Common::PointCloud &written) {
        ASSERT_EQ(read.points.size(), written.points.size());
        ASSERT_EQ(read.normals.size(), written.normals.size());
        for (std::size_t i = 0; i < written.points.size(); ++i) {
            EXPECT_FLOAT_EQ(read.points[i].x(), written.points[i].x()) << "point " << i;
            EXPECT_FLOAT_EQ(read.points[i].y(), written.points[i].y()) << "point " << i;
            EXPECT_FLOAT_EQ(read.points[i].z(), written.points[i].z()) << "point " << i;
        }
        for (std::size_t i = 0; i < written.normals.size(); ++i) {
            EXPECT_FLOAT_EQ(read.normals[i].x(), written.normals[i].x()) << "normal " << i;
            EXPECT_FLOAT_EQ(read.normals[i].y(), written.normals[i].y()) << "normal " << i;
            EXPECT_FLOAT_EQ(read.normals[i].z(), written.normals[i].z()) << "normal " << i;
        }
    }

} // namespace

TEST(PointCloudIo, RoundTripsPointsAndNormalsInBothFormats) {
    for (const Common::EPointCloudFormat format:
         {Common::EPointCloudFormat::Binary, Common::EPointCloudFormat::Ascii}) {
        const bool binary = format == Common::EPointCloudFormat::Binary;
        TempPly file(binary ? "roundtrip_binary" : "roundtrip_ascii");

        const Common::PointCloud written = makeCloud(/*withNormals=*/true);
        ASSERT_TRUE(written.Save(file.String(), format));

        Common::PointCloud read;
        ASSERT_TRUE(read.Load(file.String()));
        ExpectSameCloud(read, written);
    }
}

// A cloud with no normals must come back with no normals -- not with a zero-filled array the same
// length as the points, which every consumer would then treat as real surface orientation.
TEST(PointCloudIo, ACloudWithoutNormalsStaysWithoutNormals) {
    TempPly file("no_normals");
    const Common::PointCloud written = makeCloud(/*withNormals=*/false);
    ASSERT_TRUE(written.Save(file.String()));

    Common::PointCloud read;
    ASSERT_TRUE(read.Load(file.String()));
    EXPECT_EQ(read.points.size(), written.points.size());
    EXPECT_TRUE(read.normals.empty());
}

// Normals that do not pair one-to-one with points are not normals. Writing the short array would
// mis-pair every point past the gap, and the file would look entirely valid.
TEST(PointCloudIo, RaggedNormalsAreNotWritten) {
    TempPly file("ragged_normals");
    Common::PointCloud written = makeCloud(/*withNormals=*/true);
    written.normals.pop_back();
    ASSERT_TRUE(written.Save(file.String()));

    Common::PointCloud read;
    ASSERT_TRUE(read.Load(file.String()));
    EXPECT_EQ(read.points.size(), written.points.size());
    EXPECT_TRUE(read.normals.empty()) << "a short normal array was written as if it paired up";
}

// Load REPLACES, unlike util::LoadPly which appends. A Load onto a non-empty cloud otherwise
// concatenates two scans into one that reads as plausible and is wrong everywhere.
TEST(PointCloudIo, LoadReplacesRatherThanAppends) {
    TempPly file("replace");
    const Common::PointCloud written = makeCloud(/*withNormals=*/true);
    ASSERT_TRUE(written.Save(file.String()));

    Common::PointCloud read = makeCloud(/*withNormals=*/true); // already populated
    ASSERT_TRUE(read.Load(file.String()));
    ExpectSameCloud(read, written);
}

TEST(PointCloudIo, LoadFailsOnAMissingFileAndLeavesTheCloudAlone) {
    Common::PointCloud cloud = makeCloud(/*withNormals=*/true);
    const std::size_t before = cloud.points.size();
    EXPECT_FALSE(cloud.Load((fs::temp_directory_path() / "vkbvh_no_such_cloud.ply").string()));
    EXPECT_EQ(cloud.points.size(), before) << "a failed Load emptied the cloud";
}

// ASCII output has to stay readable by the loader four example tools already use, or the two
// halves of this repo's PLY handling drift apart with nothing to notice.
TEST(PointCloudIo, AsciiOutputIsReadableByTheExistingLoader) {
    TempPly file("interop");
    const Common::PointCloud written = makeCloud(/*withNormals=*/true);
    ASSERT_TRUE(written.Save(file.String(), Common::EPointCloudFormat::Ascii));

    std::vector<Eigen::Vector3f> points, normals;
    ASSERT_TRUE(util::LoadPly(file.String(), points, normals));
    ASSERT_EQ(points.size(), written.points.size());
    ASSERT_EQ(normals.size(), written.normals.size());
    for (std::size_t i = 0; i < points.size(); ++i)
        EXPECT_FLOAT_EQ(points[i].z(), written.points[i].z()) << "point " << i;
}
