#include "Common/PointCloud.h"
#include "Common/PointCloud.h" // PointCloud::SortTargetIntoCanonicalOrder

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <algorithm>
#include <random>
#include <vector>

namespace {

    using Eigen::Vector3f;
    using Common::PointCloud;

    // Normals are derived from the point so a mis-paired reorder is detectable by inspection alone.
    Vector3f NormalFor(const Vector3f &point) { return Vector3f(point.z(), point.x(), point.y()); }

    Common::PointCloud MakeCloud(std::vector<Vector3f> points) {
        Common::PointCloud cloud;
        cloud.normals.reserve(points.size());
        for (const Vector3f &p: points) cloud.normals.push_back(NormalFor(p));
        cloud.points = std::move(points);
        return cloud;
    }

    std::vector<Vector3f> GridPoints(int side) {
        std::vector<Vector3f> points;
        for (int x = 0; x < side; ++x)
            for (int y = 0; y < side; ++y)
                for (int z = 0; z < side; ++z)
                    points.push_back(Vector3f(float(x) * 0.5f, float(y) * 0.25f, float(z) * 0.125f));
        return points;
    }

} // namespace

TEST(SortTargetIntoCanonicalOrder, OrdersLexicographicallyByPosition) {
    Common::PointCloud cloud = MakeCloud({{1.0f, 0.0f, 0.0f}, {0.0f, 2.0f, 0.0f}, {0.0f, 1.0f, 5.0f}, {0.0f, 1.0f, 2.0f}});
    cloud.SortTargetIntoCanonicalOrder();

    ASSERT_EQ(cloud.points.size(), 4u);
    EXPECT_EQ(cloud.points[0], Vector3f(0.0f, 1.0f, 2.0f));
    EXPECT_EQ(cloud.points[1], Vector3f(0.0f, 1.0f, 5.0f));
    EXPECT_EQ(cloud.points[2], Vector3f(0.0f, 2.0f, 0.0f));
    EXPECT_EQ(cloud.points[3], Vector3f(1.0f, 0.0f, 0.0f));
}

// The whole reason the sort exists: the map hands its crop over in whatever order the hash happened
// to iterate, and the GPU reduction sums floats in array order. Two runs that saw the same points in
// different orders must reduce identically, or an A/B measurement compares scheduling, not settings.
TEST(SortTargetIntoCanonicalOrder, IsIndependentOfInputPermutation) {
    const std::vector<Vector3f> points = GridPoints(12);
    Common::PointCloud reference = MakeCloud(points);
    reference.SortTargetIntoCanonicalOrder();

    std::mt19937 rng(12345);
    for (int trial = 0; trial < 4; ++trial) {
        std::vector<Vector3f> shuffled = points;
        std::shuffle(shuffled.begin(), shuffled.end(), rng);
        Common::PointCloud cloud = MakeCloud(shuffled);
        cloud.SortTargetIntoCanonicalOrder();

        ASSERT_EQ(cloud.points.size(), reference.points.size());
        for (std::size_t i = 0; i < cloud.points.size(); ++i)
            ASSERT_EQ(cloud.points[i], reference.points[i]) << "trial " << trial << " index " << i;
    }
}

// A reorder that moved points without their normals would leave the point cloud looking correct
// while every point-to-plane residual used the wrong plane.
TEST(SortTargetIntoCanonicalOrder, KeepsEachNormalWithItsPoint) {
    std::vector<Vector3f> points = GridPoints(10);
    std::mt19937 rng(999);
    std::shuffle(points.begin(), points.end(), rng);
    Common::PointCloud cloud = MakeCloud(points);

    cloud.SortTargetIntoCanonicalOrder();

    ASSERT_EQ(cloud.normals.size(), cloud.points.size());
    for (std::size_t i = 0; i < cloud.points.size(); ++i)
        ASSERT_EQ(cloud.normals[i], NormalFor(cloud.points[i])) << "index " << i;
}

// Malformed input must be left alone rather than half-reordered -- Solve rejects it downstream.
TEST(SortTargetIntoCanonicalOrder, LeavesAMismatchedCloudUntouched) {
    Common::PointCloud cloud;
    cloud.points = {{1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
    cloud.normals = {{0.0f, 0.0f, 1.0f}};

    cloud.SortTargetIntoCanonicalOrder();

    EXPECT_EQ(cloud.points[0], Vector3f(1.0f, 0.0f, 0.0f));
    EXPECT_EQ(cloud.points[1], Vector3f(0.0f, 0.0f, 0.0f));
}
