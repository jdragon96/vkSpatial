#include "example2/shape_fixtures.h"

#include <gtest/gtest.h>

using fixtures::Shape;
using Eigen::Vector3f;

namespace {

    TEST(ShapeFixturesNormal, CubePlusXFace) {
        const Vector3f n = fixtures::NearestNormal(Shape::Cube, {fixtures::kCubeHalf, 0.3f, -0.2f});
        EXPECT_NEAR(n.x(), 1.0f, 1e-4f);
        EXPECT_NEAR(n.y(), 0.0f, 1e-4f);
        EXPECT_NEAR(n.z(), 0.0f, 1e-4f);
    }

    TEST(ShapeFixturesNormal, CubeMinusZFace) {
        const Vector3f n = fixtures::NearestNormal(Shape::Cube, {0.1f, -0.4f, -fixtures::kCubeHalf});
        EXPECT_NEAR(n.z(), -1.0f, 1e-4f);
    }

    TEST(ShapeFixturesNormal, CubeEdgeBisector) {
        const Vector3f n = fixtures::NearestNormal(
                Shape::Cube, {fixtures::kCubeHalf + 0.05f, fixtures::kCubeHalf + 0.05f, 0.0f});
        EXPECT_NEAR(n.x(), n.y(), 1e-4f);
        EXPECT_GT(n.x(), 0.0f);
        EXPECT_NEAR(n.z(), 0.0f, 1e-4f);
    }

    TEST(ShapeFixturesNormal, CylinderSideRadial) {
        const Vector3f n = fixtures::NearestNormal(Shape::Cylinder, {fixtures::kCylRadius, 0.0f, 0.5f});
        EXPECT_NEAR(n.x(), 1.0f, 1e-4f);
        EXPECT_NEAR(n.z(), 0.0f, 1e-4f);
    }

    TEST(ShapeFixturesNormal, CylinderCapAxial) {
        const Vector3f n = fixtures::NearestNormal(Shape::Cylinder, {0.3f, -0.2f, fixtures::kCylHalfZ});
        EXPECT_NEAR(n.z(), 1.0f, 1e-4f);
    }

    TEST(ShapeFixturesNormal, AngleDeg) {
        EXPECT_NEAR(fixtures::NormalAngleDeg(Vector3f::UnitX(), Vector3f::UnitY()), 90.0f, 1e-3f);
        EXPECT_NEAR(fixtures::NormalAngleDeg(Vector3f::UnitX(), Vector3f::UnitX()), 0.0f, 1e-3f);
    }

} // namespace
