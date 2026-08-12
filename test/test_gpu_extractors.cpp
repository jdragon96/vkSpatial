#include "Engine/Spatial/Extraction/GpuExtractorRegistry.h"
#include "Engine/Spatial/Extraction/ExtractorRegistry.h"
#include "Engine/Spatial/Extraction/MeshConnectivity.h"
#include "Engine/Core/Context.h"
#include "Engine/Eval/RmseMetrics.h"
#include "isosurface_test_util.h"
#include <gtest/gtest.h>
using namespace Engine::Spatial::Extraction;

TEST(GpuExtractors, McGpuMatchesCpuOnSphere) {
    Engine::Core::Context ctx;
    auto gpuReg = GpuExtractorRegistry::Default();
    ASSERT_TRUE(gpuReg.Has("mc-gpu"));
    ASSERT_NE(gpuReg.Create("mc-gpu", ctx), nullptr);
    EXPECT_EQ(gpuReg.Create("nonexistent", ctx), nullptr);

    VoxelField field = isotest::SphereField(0.7f, 0.1f, 12);
    SurfaceMesh cpu = ExtractorRegistry::Default().Create("mc")->Extract(field, ExtractParams{});
    SurfaceMesh gpu = gpuReg.Create("mc-gpu", ctx)->Extract(field, ExtractParams{});

    ASSERT_GT(gpu.triangles.size(), 100u);
    // Same tables + same weld -> vertex sets coincide to fp tolerance, both directions.
    EXPECT_LT(Engine::Eval::NearestNeighbourRMSE(gpu.vertices, cpu.vertices), 1e-4f);
    EXPECT_LT(Engine::Eval::NearestNeighbourRMSE(cpu.vertices, gpu.vertices), 1e-4f);
    // And the GPU mesh is itself closed+manifold like the CPU one.
    ConnectivityReport r = AnalyzeConnectivity(gpu);
    EXPECT_TRUE(r.IsEdgeManifold());
    EXPECT_TRUE(r.IsClosed());
}
