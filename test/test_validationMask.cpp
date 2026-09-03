#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"
#include "Pipeline/Reconstruction/Algorithm/ValidationMask.h"
#include "Pipeline/Reconstruction/DepthCameraFrameSource.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {

    // A depth buffer the GPU can read, filled from `depth`.
    std::unique_ptr<Engine::Core::Buffer> UploadDepth(Engine::Core::Context &context,
                                                      const std::vector<float> &depth) {
        auto buffer = std::make_unique<Engine::Core::Buffer>(context);
        const uint32_t bytes = uint32_t(depth.size() * sizeof(float));
        buffer->AllocateHostVisible(bytes);
        std::memcpy(buffer->MappedPtr(), depth.data(), bytes);
        buffer->MakeVisibleToGPU(bytes);
        return buffer;
    }

    std::vector<float> ReadBack(Engine::Core::Buffer &buffer, std::size_t count) {
        const uint32_t bytes = uint32_t(count * sizeof(float));
        buffer.MakeVisibleToCPU(bytes);
        std::vector<float> out(count);
        std::memcpy(out.data(), buffer.MappedPtr(), bytes);
        return out;
    }

} // namespace

// The C++ mirror and the GLSL struct must agree on the stride, and nothing in either file makes
// that true by itself -- the static_asserts in ValidationMask.h only constrain the C++ side, so a
// vec3 added to the shader alone would go unnoticed until the map came out wrong.
//
// The mask is allocated for MORE pixels than the dispatch covers and pre-filled with a sentinel.
// If the GPU stride were the 32 bytes an `{ int; vec3; }` struct produces, clearing 4 pixels would
// zero bytes 0, 32, 64 and 96 and leave the sentinel in between -- so "the first
// pixelCount * sizeof bytes are zero AND the rest still hold the sentinel" pins the stride from
// both sides at once.
TEST(ValidationMask, ClearWritesExactlyOneEntryPerPixelAtTheCppStride) {
    Engine::Core::Context context;
    const int width = 4, height = 1;
    const std::size_t covered = std::size_t(width) * height;
    const std::size_t allocated = covered * 8; // room for a wrong, larger stride to show up in

    ValidationMask mask(context, width, height);
    Engine::Core::Buffer probe(context);
    const uint32_t probeBytes = uint32_t(allocated * sizeof(ValidationMaskProperty));
    probe.AllocateHostVisibleReadback(probeBytes);

    {
        Engine::Compute::CommandBatch batch(context);
        batch.FillBuffer(probe.Handle(), 0, probeBytes, 0xFFFFFFFFu);
        batch.Barrier();
        mask.RecordClear(batch, probe, width, height);
        batch.Submit();
    }

    probe.MakeVisibleToCPU(probeBytes);
    const auto *words = static_cast<const std::uint32_t *>(probe.MappedPtr());
    const std::size_t clearedWords = covered * sizeof(ValidationMaskProperty) / sizeof(std::uint32_t);

    for (std::size_t i = 0; i < clearedWords; ++i)
        EXPECT_EQ(words[i], 0u) << "word " << i << " inside the dispatched region was not cleared: "
                                   "the GPU stride is larger than sizeof(ValidationMaskProperty)";
    for (std::size_t i = clearedWords; i < allocated * sizeof(ValidationMaskProperty) / 4; ++i)
        EXPECT_EQ(words[i], 0xFFFFFFFFu)
                << "word " << i << " past the dispatched region was overwritten";
}

// Every pixel, not just the interior: a dispatch rounded up to the workgroup size must not leave
// the tail of a row untouched, and must not wrap onto the next row either.
TEST(ValidationMask, ClearCoversAnImageThatIsNotAWorkgroupMultiple) {
    Engine::Core::Context context;
    const int width = 37, height = 19; // coprime with the 16x16 workgroup on both axes
    const std::size_t count = std::size_t(width) * height;

    ValidationMask mask(context, width, height);
    Engine::Core::Buffer probe(context);
    const uint32_t bytes = uint32_t(count * sizeof(ValidationMaskProperty));
    probe.AllocateHostVisibleReadback(bytes);

    {
        Engine::Compute::CommandBatch batch(context);
        batch.FillBuffer(probe.Handle(), 0, bytes, 0xFFFFFFFFu);
        batch.Barrier();
        mask.RecordClear(batch, probe, width, height);
        batch.Submit();
    }

    probe.MakeVisibleToCPU(bytes);
    const auto *entries = static_cast<const ValidationMaskProperty *>(probe.MappedPtr());
    for (std::size_t i = 0; i < count; ++i)
        EXPECT_EQ(entries[i].valid, 0u) << "pixel " << i << " of " << count << " was not cleared";
}

// The GPU [H1] must reproduce Pipeline::PrefilterDepth exactly. A port of this filter has four
// places to go quietly wrong -- the tolerance taken from the neighbour instead of the centre, the
// centre excluded from its own window, invalid pixels interpolated rather than left invalid, and
// the border cropped rather than clipped -- and none of them changes the output on a fixture that
// is uniform, valid and interior. So the fixture is noisy, holed, and stepped.
TEST(ValidationMask, PrefilterMatchesTheCpuPathBitExactly) {
    Engine::Core::Context context;
    const int width = 61, height = 43;
    const std::size_t count = std::size_t(width) * height;

    std::mt19937 rng(1234u);
    std::uniform_real_distribution<float> noise(-0.004f, 0.004f);
    std::vector<float> depth(count, 0.0f);
    for (int v = 0; v < height; ++v)
        for (int u = 0; u < width; ++u) {
            // Two surfaces with a step between them, plus scattered dropouts.
            float z = (u < width / 2) ? 1.0f : 1.6f;
            z += noise(rng);
            if ((u * 7 + v * 13) % 23 == 0) z = 0.0f; // invalid speckle
            depth[std::size_t(v) * width + u] = z;
        }

    Pipeline::DepthFilterOptions filter;
    filter.prefilterWindow = 5;
    const std::vector<float> expected =
            Pipeline::PrefilterDepth(depth, width, height, filter.prefilterWindow, filter);

    ValidationMask mask(context, width, height);
    auto source = UploadDepth(context, depth);
    Engine::Core::Buffer filtered(context);
    filtered.AllocateHostVisibleReadback(uint32_t(count * sizeof(float)));

    {
        Engine::Compute::CommandBatch batch(context);
        mask.RecordWindowAveraging(batch, *source, filtered, width, height, filter.prefilterWindow,
                                  filter.relativeDepthJump, filter.minimumDepthJump);
        batch.Submit();
    }

    const std::vector<float> actual = ReadBack(filtered, count);
    ASSERT_EQ(actual.size(), expected.size());
    int mismatches = 0;
    for (std::size_t i = 0; i < count; ++i) {
        // Bit-exact is the goal, but the two paths sum in different orders, so a float mean of the
        // same set can land one ulp apart. A tolerance far below the depth noise still catches
        // every structural difference -- a different neighbour set moves the mean by millimetres.
        if (std::abs(actual[i] - expected[i]) > 1e-6f && ++mismatches <= 5)
            ADD_FAILURE() << "pixel " << i << " (u=" << (i % width) << ", v=" << (i / width)
                          << "): gpu " << actual[i] << " cpu " << expected[i];
    }
    EXPECT_EQ(mismatches, 0);
}

// window <= 1 is the identity, and the destination is a separate buffer that nothing else fills,
// so the kernel still has to write it.
TEST(ValidationMask, PrefilterWithAWindowOfOneCopiesTheSource) {
    Engine::Core::Context context;
    const int width = 8, height = 8;
    const std::size_t count = std::size_t(width) * height;

    std::vector<float> depth(count);
    for (std::size_t i = 0; i < count; ++i) depth[i] = 0.5f + 0.01f * float(i);

    ValidationMask mask(context, width, height);
    auto source = UploadDepth(context, depth);
    Engine::Core::Buffer filtered(context);
    filtered.AllocateHostVisibleReadback(uint32_t(count * sizeof(float)));

    {
        Engine::Compute::CommandBatch batch(context);
        batch.FillBuffer(filtered.Handle(), 0, uint32_t(count * sizeof(float)), 0xDEADBEEFu);
        batch.Barrier();
        mask.RecordWindowAveraging(batch, *source, filtered, width, height, 1, 0.02f, 0.005f);
        batch.Submit();
    }

    EXPECT_EQ(ReadBack(filtered, count), depth);
}

// Pins WHOSE depth the tolerance comes from. tau = max(0.005, 0.02 z) grows with range, so a
// geometric ramp z(u) = z0 * r^u with r = 1.0202 puts every horizontal neighbour gap exactly
// between the two candidate tolerances:
//
//   gap to the right neighbour = 0.0202 z   >  tau(centre)    = 0.0200 z      -> reject
//                                           <  tau(neighbour) = 0.0204 z      -> accept
//
// and the left neighbour flips the other way. Reading tau from the neighbour instead of the
// centre therefore changes the admitted set in every column, in both directions. A uniform or
// stepped fixture cannot see this: there the gaps are far inside or far outside both tolerances.
TEST(ValidationMask, PrefilterTakesTheToleranceFromTheCentrePixel) {
    Engine::Core::Context context;
    const int width = 61, height = 9;
    const std::size_t count = std::size_t(width) * height;

    std::vector<float> depth(count);
    for (int v = 0; v < height; ++v) {
        float z = 0.6f;
        for (int u = 0; u < width; ++u) {
            depth[std::size_t(v) * width + u] = z;
            z *= 1.0202f;
        }
    }

    Pipeline::DepthFilterOptions filter;
    filter.prefilterWindow = 3;
    const std::vector<float> expected =
            Pipeline::PrefilterDepth(depth, width, height, filter.prefilterWindow, filter);

    ValidationMask mask(context, width, height);
    auto source = UploadDepth(context, depth);
    Engine::Core::Buffer filtered(context);
    filtered.AllocateHostVisibleReadback(uint32_t(count * sizeof(float)));

    {
        Engine::Compute::CommandBatch batch(context);
        mask.RecordWindowAveraging(batch, *source, filtered, width, height, filter.prefilterWindow,
                                  filter.relativeDepthJump, filter.minimumDepthJump);
        batch.Submit();
    }

    const std::vector<float> actual = ReadBack(filtered, count);
    // The fixture is only worth anything if the filter actually rejected something here.
    int changed = 0;
    for (std::size_t i = 0; i < count; ++i)
        if (std::abs(expected[i] - depth[i]) > 1e-7f) ++changed;
    ASSERT_GT(changed, 0) << "the ramp passed through untouched: no neighbour was ever rejected";

    int mismatches = 0;
    for (std::size_t i = 0; i < count; ++i)
        if (std::abs(actual[i] - expected[i]) > 1e-6f && ++mismatches <= 5)
            ADD_FAILURE() << "pixel " << i << " (u=" << (i % width) << "): gpu " << actual[i]
                          << " cpu " << expected[i];
    EXPECT_EQ(mismatches, 0);
}

///////////////////////////////////////////////////////////////////////////////////////////////
// [H2] BuildVertexGrid + RangeGate
///////////////////////////////////////////////////////////////////////////////////////////////

namespace {

    struct VertexGridResult {
        std::vector<Eigen::Vector4f> vertices;
        std::vector<ValidationMaskProperty> mask;
        ValidationMaskCounters counters;
    };

    VertexGridResult RunVertexGrid(Engine::Core::Context &context, const std::vector<float> &depth,
                                   const Pipeline::CameraIntrinsics &intrinsics,
                                   const Pipeline::DepthFilterOptions &filter) {
        const std::size_t count = depth.size();
        ValidationMask mask(context, intrinsics.width, intrinsics.height);
        auto source = UploadDepth(context, depth);

        Engine::Core::Buffer vertices(context);
        Engine::Core::Buffer maskBuffer(context);
        Engine::Core::Buffer counter(context);
        vertices.AllocateHostVisibleReadback(uint32_t(count * sizeof(Eigen::Vector4f)));
        maskBuffer.AllocateHostVisibleReadback(uint32_t(count * sizeof(ValidationMaskProperty)));
        counter.AllocateHostVisibleReadback(uint32_t(sizeof(ValidationMaskCounters)));

        {
            Engine::Compute::CommandBatch batch(context);
            batch.FillBuffer(counter.Handle(), 0, sizeof(ValidationMaskCounters), 0u);
            // Sentinel, so "the kernel zeroed this vertex" is distinguishable from "the allocation
            // came back zero". Without it the rejected-pixel assertion passes on an empty kernel.
            batch.FillBuffer(vertices.Handle(), 0, uint32_t(count * sizeof(Eigen::Vector4f)),
                             0x7F7FFFFFu);
            batch.Barrier();
            mask.RecordBuildVertexGrid(batch, *source, vertices, maskBuffer, counter, intrinsics,
                                       filter.minimumDepthMeters, filter.maximumDepthMeters);
            batch.Submit();
        }

        VertexGridResult result;
        vertices.MakeVisibleToCPU(uint32_t(count * sizeof(Eigen::Vector4f)));
        maskBuffer.MakeVisibleToCPU(uint32_t(count * sizeof(ValidationMaskProperty)));
        counter.MakeVisibleToCPU(uint32_t(sizeof(ValidationMaskCounters)));
        result.vertices.resize(count);
        result.mask.resize(count);
        std::memcpy(result.vertices.data(), vertices.MappedPtr(), count * sizeof(Eigen::Vector4f));
        std::memcpy(result.mask.data(), maskBuffer.MappedPtr(),
                    count * sizeof(ValidationMaskProperty));
        std::memcpy(&result.counters, counter.MappedPtr(), sizeof(ValidationMaskCounters));
        return result;
    }

    Pipeline::CameraIntrinsics TestIntrinsics(int width, int height) {
        Pipeline::CameraIntrinsics intrinsics;
        intrinsics.width = width;
        intrinsics.height = height;
        intrinsics.fx = 383.18f;
        intrinsics.fy = 381.44f; // deliberately != fx: a kernel using fx for both still passes
                                 // every square fixture, so the two must differ here
        intrinsics.cx = 313.94f;
        intrinsics.cy = 237.30f;
        return intrinsics;
    }

} // namespace

// The back-projection, checked against the formula rather than against a second copy of the same
// code. cx/cy are off-centre and fx != fy, so a kernel that swapped an axis or dropped a principal
// point offset cannot pass by symmetry.
TEST(ValidationMask, VertexGridBackProjectsEveryValidPixel) {
    Engine::Core::Context context;
    const Pipeline::CameraIntrinsics intrinsics = TestIntrinsics(23, 17);
    const std::size_t count = std::size_t(intrinsics.width) * intrinsics.height;

    std::vector<float> depth(count);
    for (std::size_t i = 0; i < count; ++i) depth[i] = 0.4f + 0.017f * float(i % 61);

    const VertexGridResult result = RunVertexGrid(context, depth, intrinsics, {});

    for (int v = 0; v < intrinsics.height; ++v)
        for (int u = 0; u < intrinsics.width; ++u) {
            const std::size_t i = std::size_t(v) * intrinsics.width + u;
            const float z = depth[i];
            EXPECT_EQ(result.mask[i].valid, 1u) << "pixel " << i;
            EXPECT_NEAR(result.vertices[i].x(), (float(u) - intrinsics.cx) / intrinsics.fx * z, 1e-5f)
                    << "pixel (" << u << "," << v << ")";
            EXPECT_NEAR(result.vertices[i].y(), (float(v) - intrinsics.cy) / intrinsics.fy * z, 1e-5f)
                    << "pixel (" << u << "," << v << ")";
            EXPECT_NEAR(result.vertices[i].z(), z, 1e-6f) << "pixel (" << u << "," << v << ")";
        }
    EXPECT_EQ(result.counters.rejectedByRange, 0u);
}

// A pixel the matcher produced no depth for is invalid, and it is NOT a range rejection: the two
// have opposite corrections (one is the sensor's own dropout, the other is a knob the caller set),
// so charging a dropout to the range counter would make the counter useless for tuning it.
TEST(ValidationMask, VertexGridMarksZeroDepthInvalidWithoutChargingTheRangeCounter) {
    Engine::Core::Context context;
    const Pipeline::CameraIntrinsics intrinsics = TestIntrinsics(16, 4);
    const std::size_t count = std::size_t(intrinsics.width) * intrinsics.height;

    std::vector<float> depth(count, 1.0f);
    depth[3] = 0.0f;
    depth[20] = -1.0f; // a negative reading is as absent as a zero one

    const VertexGridResult result = RunVertexGrid(context, depth, intrinsics, {});

    EXPECT_EQ(result.mask[3].valid, 0u);
    EXPECT_EQ(result.mask[20].valid, 0u);
    EXPECT_EQ(result.mask[4].valid, 1u);
    EXPECT_EQ(result.counters.rejectedByRange, 0u) << "a dropout is not a range rejection";
}

// Each end of the gate is independent and 0 disables it, matching DepthFilterOptions.
TEST(ValidationMask, VertexGridAppliesEachEndOfTheRangeGateIndependently) {
    Engine::Core::Context context;
    const Pipeline::CameraIntrinsics intrinsics = TestIntrinsics(8, 2);
    const std::size_t count = std::size_t(intrinsics.width) * intrinsics.height;

    std::vector<float> depth(count, 1.0f);
    depth[0] = 0.15f; // below a near gate
    depth[1] = 6.0f;  // above a far gate

    Pipeline::DepthFilterOptions off;
    const VertexGridResult none = RunVertexGrid(context, depth, intrinsics, off);
    EXPECT_EQ(none.mask[0].valid, 1u) << "0 must disable the near end";
    EXPECT_EQ(none.mask[1].valid, 1u) << "0 must disable the far end";
    EXPECT_EQ(none.counters.rejectedByRange, 0u);

    Pipeline::DepthFilterOptions nearOnly;
    nearOnly.minimumDepthMeters = 0.3f;
    const VertexGridResult near = RunVertexGrid(context, depth, intrinsics, nearOnly);
    EXPECT_EQ(near.mask[0].valid, 0u);
    EXPECT_EQ(near.mask[1].valid, 1u) << "the near gate must not reject a far pixel";
    EXPECT_EQ(near.counters.rejectedByRange, 1u);

    Pipeline::DepthFilterOptions farOnly;
    farOnly.maximumDepthMeters = 4.0f;
    const VertexGridResult far = RunVertexGrid(context, depth, intrinsics, farOnly);
    EXPECT_EQ(far.mask[0].valid, 1u) << "the far gate must not reject a near pixel";
    EXPECT_EQ(far.mask[1].valid, 0u);
    EXPECT_EQ(far.counters.rejectedByRange, 1u);
}

// An out-of-range pixel must be INVALID, not merely unemitted: the later passes read this mask to
// decide whether a neighbour exists, and a reading the sensor could not have made must not be
// allowed to set the normal of the pixel next to it. The vertex it leaves behind must therefore
// be zeroed too, so nothing can pick it up by reading the grid without the mask.
TEST(ValidationMask, VertexGridLeavesNoVertexBehindForARejectedPixel) {
    Engine::Core::Context context;
    const Pipeline::CameraIntrinsics intrinsics = TestIntrinsics(8, 2);
    const std::size_t count = std::size_t(intrinsics.width) * intrinsics.height;

    std::vector<float> depth(count, 1.0f);
    depth[5] = 9.0f;

    Pipeline::DepthFilterOptions filter;
    filter.maximumDepthMeters = 4.0f;
    const VertexGridResult result = RunVertexGrid(context, depth, intrinsics, filter);

    EXPECT_EQ(result.mask[5].valid, 0u);
    EXPECT_EQ(result.vertices[5].x(), 0.0f);
    EXPECT_EQ(result.vertices[5].y(), 0.0f);
    EXPECT_EQ(result.vertices[5].z(), 0.0f);
}

///////////////////////////////////////////////////////////////////////////////////////////////
// [H3] ForwardJumpGuard + EstimateNormal
///////////////////////////////////////////////////////////////////////////////////////////////

namespace {

    struct NormalGridResult {
        ValidationMaskCounters counters;
        std::vector<Eigen::Vector4f> vertices;
        std::vector<Eigen::Vector4f> normals;
        std::vector<ValidationMaskProperty> mask;
    };

    NormalGridResult RunNormalGrid(Engine::Core::Context &context, const std::vector<float> &depth,
                                   const Pipeline::CameraIntrinsics &intrinsics,
                                   const Pipeline::DepthFilterOptions &filter) {
        const std::size_t count = depth.size();
        ValidationMask mask(context, intrinsics.width, intrinsics.height);
        auto source = UploadDepth(context, depth);

        Engine::Core::Buffer vertices(context), normals(context), maskBuffer(context),
                counter(context);
        vertices.AllocateHostVisibleReadback(uint32_t(count * sizeof(Eigen::Vector4f)));
        normals.AllocateHostVisibleReadback(uint32_t(count * sizeof(Eigen::Vector4f)));
        maskBuffer.AllocateHostVisibleReadback(uint32_t(count * sizeof(ValidationMaskProperty)));
        counter.AllocateHostVisibleReadback(uint32_t(sizeof(ValidationMaskCounters)));

        {
            Engine::Compute::CommandBatch batch(context);
            batch.FillBuffer(counter.Handle(), 0, sizeof(ValidationMaskCounters), 0u);
            batch.FillBuffer(normals.Handle(), 0, uint32_t(count * sizeof(Eigen::Vector4f)),
                             0x7F7FFFFFu);
            batch.Barrier();
            mask.RecordBuildVertexGrid(batch, *source, vertices, maskBuffer, counter, intrinsics,
                                       filter.minimumDepthMeters, filter.maximumDepthMeters);
            batch.Barrier();
            mask.RecordEstimateNormal(batch, vertices, maskBuffer, normals, counter,
                                      intrinsics.width, intrinsics.height, filter);
            batch.Submit();
        }

        NormalGridResult result;
        counter.MakeVisibleToCPU(uint32_t(sizeof(ValidationMaskCounters)));
        std::memcpy(&result.counters, counter.MappedPtr(), sizeof(ValidationMaskCounters));
        const uint32_t vectorBytes = uint32_t(count * sizeof(Eigen::Vector4f));
        vertices.MakeVisibleToCPU(vectorBytes);
        normals.MakeVisibleToCPU(vectorBytes);
        maskBuffer.MakeVisibleToCPU(uint32_t(count * sizeof(ValidationMaskProperty)));
        result.vertices.resize(count);
        result.normals.resize(count);
        result.mask.resize(count);
        std::memcpy(result.vertices.data(), vertices.MappedPtr(), vectorBytes);
        std::memcpy(result.normals.data(), normals.MappedPtr(), vectorBytes);
        std::memcpy(result.mask.data(), maskBuffer.MappedPtr(),
                    count * sizeof(ValidationMaskProperty));
        return result;
    }

} // namespace

// The whole reason `emitted` exists as a field of its own. A pixel the forward guard refuses still
// HAS a depth measurement, and [H4]/[H5] count neighbours by `valid` -- folding the guard's verdict
// into `valid` would silently change every neighbour count downstream.
TEST(ValidationMask, TheForwardGuardRejectsWithoutClearingValid) {
    Engine::Core::Context context;
    const Pipeline::CameraIntrinsics intrinsics = TestIntrinsics(32, 8);
    const std::size_t count = std::size_t(intrinsics.width) * intrinsics.height;

    std::vector<float> depth(count);
    for (int v = 0; v < intrinsics.height; ++v)
        for (int u = 0; u < intrinsics.width; ++u)
            depth[std::size_t(v) * intrinsics.width + u] = (u < 16) ? 1.0f : 2.0f;

    const NormalGridResult result = RunNormalGrid(context, depth, intrinsics, {});

    // Column 15 straddles the step through its u+1 neighbour, so the guard refuses it.
    const std::size_t straddling = std::size_t(2) * intrinsics.width + 15;
    EXPECT_EQ(result.mask[straddling].emitted, 0u);
    EXPECT_EQ(result.mask[straddling].valid, 1u)
            << "the guard cleared valid; the neighbour counts in [H4]/[H5] would shift";
}

// A plane facing the camera keeps every pixel the CPU loop reaches, with the normal pointing back
// along the view direction.
TEST(ValidationMask, EstimateNormalOnAPlaneEmitsEveryInteriorPixel) {
    Engine::Core::Context context;
    const Pipeline::CameraIntrinsics intrinsics = TestIntrinsics(20, 12);
    const std::size_t count = std::size_t(intrinsics.width) * intrinsics.height;
    const std::vector<float> depth(count, 1.5f);

    const NormalGridResult result = RunNormalGrid(context, depth, intrinsics, {});

    int emitted = 0;
    for (int v = 0; v < intrinsics.height; ++v)
        for (int u = 0; u < intrinsics.width; ++u) {
            const std::size_t i = std::size_t(v) * intrinsics.width + u;
            // BackprojectDepth walks [0,W-1) x [0,H-1): it needs the u+1 and v+1 neighbours.
            const bool reachable = (u + 1 < intrinsics.width) && (v + 1 < intrinsics.height);
            ASSERT_EQ(result.mask[i].emitted, reachable ? 1u : 0u)
                    << "pixel (" << u << "," << v << ")";
            if (!reachable) continue;
            ++emitted;
            EXPECT_NEAR(result.normals[i].z(), -1.0f, 1e-3f) << "pixel (" << u << "," << v << ")";
        }
    EXPECT_EQ(emitted, (intrinsics.width - 1) * (intrinsics.height - 1));
}

// Mirrors DepthFrontend.NormalsFaceTheCameraAcrossAWideFieldOfView: the orientation flip is against
// the pixel's own ray, so a kernel testing n.z() alone leaves the image edges inverted, and an
// inverted normal flips the sign of every TSDF update and point-to-plane residual.
TEST(ValidationMask, EstimatedNormalsFaceTheCameraAcrossAWideFieldOfView) {
    Engine::Core::Context context;
    Pipeline::CameraIntrinsics intrinsics = TestIntrinsics(96, 96);
    intrinsics.fx = intrinsics.fy = 48.0f; // 45 degrees of ray angle at the corner
    intrinsics.cx = intrinsics.cy = 48.0f;
    const std::size_t count = std::size_t(intrinsics.width) * intrinsics.height;
    const std::vector<float> depth(count, 1.5f);

    const NormalGridResult result = RunNormalGrid(context, depth, intrinsics, {});

    for (std::size_t i = 0; i < count; ++i) {
        if (result.mask[i].emitted == 0u) continue;
        const Eigen::Vector3f normal = result.normals[i].head<3>();
        const Eigen::Vector3f point = result.vertices[i].head<3>();
        EXPECT_LT(normal.dot(point), 0.0f) << "pixel " << i << " has a normal facing away";
    }
}

// The strongest check available: the CPU emits its points in the same row-major order this pass
// marks them in, so the k-th emitted pixel must equal the k-th CPU point exactly.
TEST(ValidationMask, EstimateNormalMatchesTheCpuEmitOrderAndValues) {
    Engine::Core::Context context;
    const Pipeline::CameraIntrinsics intrinsics = TestIntrinsics(47, 31);
    const std::size_t count = std::size_t(intrinsics.width) * intrinsics.height;

    std::mt19937 rng(99u);
    std::uniform_real_distribution<float> noise(-0.003f, 0.003f);
    std::vector<float> depth(count);
    for (int v = 0; v < intrinsics.height; ++v)
        for (int u = 0; u < intrinsics.width; ++u) {
            float z = (u < 20) ? 1.0f : 1.9f;            // a step the guard must refuse to span
            z += 0.004f * float(v) + noise(rng);          // a slant it must keep
            if ((u * 5 + v * 11) % 29 == 0) z = 0.0f;     // dropouts
            depth[std::size_t(v) * intrinsics.width + u] = z;
        }

    Pipeline::DepthFrame frame;
    frame.depth = depth;
    const Pipeline::Frame expected = Pipeline::BackprojectDepth(frame, intrinsics, {});

    const NormalGridResult result = RunNormalGrid(context, depth, intrinsics, {});

    std::size_t k = 0;
    for (int v = 0; v + 1 < intrinsics.height; ++v)
        for (int u = 0; u + 1 < intrinsics.width; ++u) {
            const std::size_t i = std::size_t(v) * intrinsics.width + u;
            if (result.mask[i].emitted == 0u) continue;
            ASSERT_LT(k, expected.pts.size())
                    << "the GPU emitted more pixels than the CPU did, first extra at (" << u << ","
                    << v << ")";
            EXPECT_NEAR((result.vertices[i].head<3>() - expected.pts[k]).norm(), 0.0f, 1e-5f)
                    << "point " << k << " at (" << u << "," << v << ")";
            EXPECT_NEAR((result.normals[i].head<3>() - expected.nrm[k]).norm(), 0.0f, 1e-4f)
                    << "normal " << k << " at (" << u << "," << v << ")";
            ++k;
        }
    EXPECT_EQ(k, expected.pts.size()) << "the GPU emitted fewer pixels than the CPU did";
    EXPECT_GT(k, 0u);
}

// The neighbour-valid check is not redundant with the jump guard, though it very nearly is. [H2]
// zeroes a rejected pixel's vertex, so an invalid neighbour reads as z = 0 and the guard refuses
// it on |0 - z| > tau for any ordinary depth. The two part company below the absolute floor of
// tau: at z = 4 mm, |0 - z| = 0.004 is INSIDE tau = max(0.005, 0.02 z), and only the valid check
// still rejects. Unphysical for a D435, but it is what keeps the check from being dead code.
TEST(ValidationMask, TheNeighbourValidCheckStillRejectsBelowTheJumpFloor) {
    Engine::Core::Context context;
    const Pipeline::CameraIntrinsics intrinsics = TestIntrinsics(8, 4);
    const std::size_t count = std::size_t(intrinsics.width) * intrinsics.height;

    std::vector<float> depth(count, 0.004f);
    const std::size_t centre = std::size_t(1) * intrinsics.width + 1;
    depth[centre + 1] = 0.0f; // the u+1 neighbour is a dropout

    const NormalGridResult result = RunNormalGrid(context, depth, intrinsics, {});

    EXPECT_EQ(result.mask[centre].valid, 1u) << "the centre itself still has a measurement";
    EXPECT_EQ(result.mask[centre].emitted, 0u)
            << "an invalid neighbour was accepted because its zeroed vertex sits inside the "
               "jump tolerance at this depth";
}

///////////////////////////////////////////////////////////////////////////////////////////////
// [H4] StraddlesADepthStep / [H5] CountSameSurfaceNeighbours / [H6] IncidenceGate
//
// All three live in the normal kernel because the CPU order interleaves them with the normal:
// H3 -> H4 -> H5 -> EstimateNormal -> H6. Any other arrangement stops matching the CPU.
///////////////////////////////////////////////////////////////////////////////////////////////

// One fixture, every gate: the CPU emit list and the GPU emitted pixels must agree index for
// index with the gates on, exactly as they do with the gates off.
TEST(ValidationMask, EveryGateMatchesTheCpuEmitOrderAndValues) {
    Engine::Core::Context context;
    const Pipeline::CameraIntrinsics intrinsics = TestIntrinsics(53, 37);
    const std::size_t count = std::size_t(intrinsics.width) * intrinsics.height;

    std::mt19937 rng(4242u);
    std::uniform_real_distribution<float> noise(-0.003f, 0.003f);
    std::vector<float> depth(count);
    for (int v = 0; v < intrinsics.height; ++v)
        for (int u = 0; u < intrinsics.width; ++u) {
            float z = (u < 22) ? 1.0f : 1.8f;                     // step -> H3/H4
            z += 0.02f * float(v) + noise(rng);                   // slant -> H6 has something to cut
            if ((u * 3 + v * 7) % 17 == 0) z = 0.0f;              // dropouts -> H5 has islands
            if (u >= 40 && u < 43 && v >= 10 && v < 13) z = 0.6f; // a near blob on the far half
            depth[std::size_t(v) * intrinsics.width + u] = z;
        }

    for (int minimumValidNeighbours : {0, 6, 8})
        for (bool symmetric : {false, true})
            for (float maximumIncidenceDegrees : {0.0f, 75.0f}) {
                Pipeline::DepthFilterOptions filter;
                filter.symmetricDepthJumpGuard = symmetric;
                filter.minimumValidNeighbours = minimumValidNeighbours;
                filter.maximumIncidenceDegrees = maximumIncidenceDegrees;

                Pipeline::DepthFrame frame;
                frame.depth = depth;
                Pipeline::DepthFilterStats cpuStats;
                const Pipeline::Frame expected =
                        Pipeline::BackprojectDepth(frame, intrinsics, filter, &cpuStats);

                const NormalGridResult result = RunNormalGrid(context, depth, intrinsics, filter);

                SCOPED_TRACE("neighbours=" + std::to_string(minimumValidNeighbours) +
                             " symmetric=" + std::to_string(int(symmetric)) +
                             " incidence=" + std::to_string(maximumIncidenceDegrees));

                std::size_t k = 0;
                for (int v = 0; v + 1 < intrinsics.height; ++v)
                    for (int u = 0; u + 1 < intrinsics.width; ++u) {
                        const std::size_t i = std::size_t(v) * intrinsics.width + u;
                        if (result.mask[i].emitted == 0u) continue;
                        ASSERT_LT(k, expected.pts.size())
                                << "GPU emitted more than the CPU, first extra at (" << u << ","
                                << v << ")";
                        EXPECT_NEAR((result.vertices[i].head<3>() - expected.pts[k]).norm(), 0.0f,
                                    1e-5f)
                                << "point " << k;
                        EXPECT_NEAR((result.normals[i].head<3>() - expected.nrm[k]).norm(), 0.0f,
                                    1e-4f)
                                << "normal " << k;
                        ++k;
                    }
                EXPECT_EQ(k, expected.pts.size()) << "GPU emitted fewer than the CPU";
                EXPECT_GT(k, 0u);

                // Counters must agree too, per cause. The CPU counts range rejections over PIXELS
                // and the other two over candidate POINTS -- summing them would hide a rejection
                // charged to the wrong gate.
                EXPECT_EQ(result.counters.emittedPoints, cpuStats.emittedPoints.load());
                EXPECT_EQ(result.counters.rejectedByNeighbourSupport,
                          cpuStats.rejectedByNeighbourSupport.load());
                EXPECT_EQ(result.counters.rejectedByIncidence,
                          cpuStats.rejectedByIncidence.load());
            }
}

// [H4] in isolation: the trailing edge of a step is what the forward guard cannot see, and it is
// where the 400 mm streaks come from.
TEST(ValidationMask, TheSymmetricGuardRejectsTheTrailingEdgeOfAStep) {
    Engine::Core::Context context;
    const Pipeline::CameraIntrinsics intrinsics = TestIntrinsics(32, 8);
    const std::size_t count = std::size_t(intrinsics.width) * intrinsics.height;

    std::vector<float> depth(count);
    for (int v = 0; v < intrinsics.height; ++v)
        for (int u = 0; u < intrinsics.width; ++u)
            depth[std::size_t(v) * intrinsics.width + u] = (u < 16) ? 2.0f : 1.0f;

    // Column 16 leads the near half: its right and down neighbours are its own surface, so the
    // forward guard admits it, while its left neighbour is a metre behind.
    const std::size_t trailing = std::size_t(3) * intrinsics.width + 16;

    Pipeline::DepthFilterOptions off;
    EXPECT_EQ(RunNormalGrid(context, depth, intrinsics, off).mask[trailing].emitted, 1u)
            << "the forward-only guard admits this pixel -- that is the point of the fixture";

    Pipeline::DepthFilterOptions symmetric;
    symmetric.symmetricDepthJumpGuard = true;
    EXPECT_EQ(RunNormalGrid(context, depth, intrinsics, symmetric).mask[trailing].emitted, 0u);
}

// [H6] in isolation, and against the pixel's own ray rather than the optical axis: on a
// fronto-parallel wall every normal is (0,0,-1), so a gate reading normal.z sees zero incidence
// everywhere and cuts nothing, yet at a 45 degree ray angle that wall really is 54.7 degrees off
// the ray.
TEST(ValidationMask, TheIncidenceGateMeasuresAgainstTheViewRay) {
    Engine::Core::Context context;
    Pipeline::CameraIntrinsics intrinsics = TestIntrinsics(96, 96);
    intrinsics.fx = intrinsics.fy = 48.0f;
    intrinsics.cx = intrinsics.cy = 48.0f;
    const std::size_t count = std::size_t(intrinsics.width) * intrinsics.height;
    const std::vector<float> depth(count, 1.5f);

    Pipeline::DepthFilterOptions gate;
    gate.maximumIncidenceDegrees = 40.0f;
    const NormalGridResult result = RunNormalGrid(context, depth, intrinsics, gate);

    std::size_t emitted = 0;
    for (std::size_t i = 0; i < count; ++i) emitted += result.mask[i].emitted;
    EXPECT_GT(emitted, 0u) << "the centre of the wall is square to its ray and must survive";
    EXPECT_LT(emitted, std::size_t(intrinsics.width - 1) * (intrinsics.height - 1))
            << "a gate reading normal.z alone keeps the whole wall";
    EXPECT_GT(result.counters.rejectedByIncidence, 0u);
}

///////////////////////////////////////////////////////////////////////////////////////////////
// Compaction
///////////////////////////////////////////////////////////////////////////////////////////////

namespace {

    struct CompactResult {
        std::vector<Eigen::Vector4f> points;
        std::vector<Eigen::Vector4f> normals;
        std::uint32_t count = 0;
    };

    CompactResult RunFullChain(Engine::Core::Context &context, const std::vector<float> &depth,
                              const Pipeline::CameraIntrinsics &intrinsics,
                              const Pipeline::DepthFilterOptions &filter) {
        const std::size_t count = depth.size();
        const int height = intrinsics.height;
        ValidationMask mask(context, intrinsics.width, height);
        auto source = UploadDepth(context, depth);

        Engine::Core::Buffer vertices(context), normals(context), maskBuffer(context),
                counter(context), rowOffset(context), points(context), compactNormals(context);
        const uint32_t vectorBytes = uint32_t(count * sizeof(Eigen::Vector4f));
        vertices.AllocateHostVisibleReadback(vectorBytes);
        normals.AllocateHostVisibleReadback(vectorBytes);
        maskBuffer.AllocateHostVisibleReadback(uint32_t(count * sizeof(ValidationMaskProperty)));
        counter.AllocateHostVisibleReadback(uint32_t(sizeof(ValidationMaskCounters)));
        rowOffset.AllocateHostVisibleReadback(uint32_t((height + 1) * sizeof(std::uint32_t)));
        points.AllocateHostVisibleReadback(vectorBytes);
        compactNormals.AllocateHostVisibleReadback(vectorBytes);

        {
            Engine::Compute::CommandBatch batch(context);
            batch.FillBuffer(counter.Handle(), 0, sizeof(ValidationMaskCounters), 0u);
            batch.FillBuffer(points.Handle(), 0, vectorBytes, 0u);
            batch.FillBuffer(compactNormals.Handle(), 0, vectorBytes, 0u);
            batch.Barrier();
            mask.RecordBuildVertexGrid(batch, *source, vertices, maskBuffer, counter, intrinsics,
                                       filter.minimumDepthMeters, filter.maximumDepthMeters);
            batch.Barrier();
            mask.RecordEstimateNormal(batch, vertices, maskBuffer, normals, counter,
                                      intrinsics.width, height, filter);
            batch.Barrier();
            mask.RecordCompactPoints(batch, maskBuffer, vertices, normals, rowOffset, points,
                                     compactNormals, intrinsics.width, height);
            batch.Submit();
        }

        const uint32_t offsetBytes = uint32_t((height + 1) * sizeof(std::uint32_t));
        rowOffset.MakeVisibleToCPU(offsetBytes);
        CompactResult result;
        std::memcpy(&result.count,
                    static_cast<const std::uint32_t *>(rowOffset.MappedPtr()) + height,
                    sizeof(std::uint32_t));

        points.MakeVisibleToCPU(vectorBytes);
        compactNormals.MakeVisibleToCPU(vectorBytes);
        result.points.resize(result.count);
        result.normals.resize(result.count);
        if (result.count > 0) {
            std::memcpy(result.points.data(), points.MappedPtr(),
                        result.count * sizeof(Eigen::Vector4f));
            std::memcpy(result.normals.data(), compactNormals.MappedPtr(),
                        result.count * sizeof(Eigen::Vector4f));
        }
        return result;
    }

} // namespace

// The end of the chain: the compacted arrays must equal BackprojectDepth's output element for
// element. Not as sets -- in ORDER. An atomicAdd append would pass a set comparison and still
// make every replay diverge, because this cloud is the ICP source and its centroid is a float sum.
TEST(ValidationMask, CompactionReproducesTheCpuFrameInOrder) {
    Engine::Core::Context context;
    const Pipeline::CameraIntrinsics intrinsics = TestIntrinsics(59, 41);
    const std::size_t pixels = std::size_t(intrinsics.width) * intrinsics.height;

    std::mt19937 rng(7u);
    std::uniform_real_distribution<float> noise(-0.003f, 0.003f);
    std::vector<float> depth(pixels);
    for (int v = 0; v < intrinsics.height; ++v)
        for (int u = 0; u < intrinsics.width; ++u) {
            float z = (u < 25) ? 1.1f : 1.9f;
            z += 0.015f * float(v) + noise(rng);
            if ((u * 3 + v * 5) % 19 == 0) z = 0.0f;
            depth[std::size_t(v) * intrinsics.width + u] = z;
        }

    for (int minimumValidNeighbours : {0, 6})
        for (bool symmetric : {false, true}) {
            Pipeline::DepthFilterOptions filter;
            filter.minimumValidNeighbours = minimumValidNeighbours;
            filter.symmetricDepthJumpGuard = symmetric;
            SCOPED_TRACE("neighbours=" + std::to_string(minimumValidNeighbours) +
                         " symmetric=" + std::to_string(int(symmetric)));

            Pipeline::DepthFrame frame;
            frame.depth = depth;
            const Pipeline::Frame expected = Pipeline::BackprojectDepth(frame, intrinsics, filter);

            const CompactResult actual = RunFullChain(context, depth, intrinsics, filter);

            ASSERT_EQ(actual.count, expected.pts.size());
            ASSERT_GT(actual.count, 0u);
            for (std::size_t i = 0; i < actual.count; ++i) {
                EXPECT_NEAR((actual.points[i].head<3>() - expected.pts[i]).norm(), 0.0f, 1e-5f)
                        << "point " << i;
                EXPECT_NEAR((actual.normals[i].head<3>() - expected.nrm[i]).norm(), 0.0f, 1e-4f)
                        << "normal " << i;
            }
        }
}

// Same input, same bytes out. The order leaks this repo closed were only visible as divergence
// across runs, so the property is worth asserting directly rather than inferred from the pass above.
TEST(ValidationMask, CompactionIsByteIdenticalAcrossRuns) {
    Engine::Core::Context context;
    const Pipeline::CameraIntrinsics intrinsics = TestIntrinsics(48, 32);
    const std::size_t pixels = std::size_t(intrinsics.width) * intrinsics.height;

    std::mt19937 rng(11u);
    std::uniform_real_distribution<float> noise(-0.004f, 0.004f);
    std::vector<float> depth(pixels);
    for (std::size_t i = 0; i < pixels; ++i)
        depth[i] = (i % 7 == 0) ? 0.0f : 1.3f + noise(rng);

    Pipeline::DepthFilterOptions filter;
    filter.minimumValidNeighbours = 6;
    const CompactResult first = RunFullChain(context, depth, intrinsics, filter);
    const CompactResult second = RunFullChain(context, depth, intrinsics, filter);

    ASSERT_GT(first.count, 0u);
    EXPECT_EQ(first.count, second.count);
    EXPECT_EQ(first.points, second.points);
    EXPECT_EQ(first.normals, second.normals);
}

// A frame where nothing survives must produce a count of zero rather than a stale one.
TEST(ValidationMask, CompactionOfAnEmptyFrameCountsZero) {
    Engine::Core::Context context;
    const Pipeline::CameraIntrinsics intrinsics = TestIntrinsics(16, 8);
    const std::vector<float> depth(std::size_t(intrinsics.width) * intrinsics.height, 0.0f);

    EXPECT_EQ(RunFullChain(context, depth, intrinsics, {}).count, 0u);
}

///////////////////////////////////////////////////////////////////////////////////////////////
// Execute -- the whole chain in one call
///////////////////////////////////////////////////////////////////////////////////////////////

// The capstone. Execute() must reproduce BackprojectDepth exactly, over every gate combination,
// including the prefilter -- which the per-pass tests exercise separately but never together with
// the gates, and it changes the depth every later pass reads.
TEST(ValidationMask, ExecuteReproducesTheCpuFrontEnd) {
    Engine::Core::Context context;
    const Pipeline::CameraIntrinsics intrinsics = TestIntrinsics(57, 39);
    const std::size_t pixels = std::size_t(intrinsics.width) * intrinsics.height;

    std::mt19937 rng(31337u);
    std::uniform_real_distribution<float> noise(-0.004f, 0.004f);
    std::vector<float> depth(pixels);
    for (int v = 0; v < intrinsics.height; ++v)
        for (int u = 0; u < intrinsics.width; ++u) {
            float z = (u < 24) ? 1.05f : 1.85f;
            z += 0.018f * float(v) + noise(rng);
            if ((u * 3 + v * 7) % 17 == 0) z = 0.0f;
            if (u >= 44 && u < 47 && v >= 8 && v < 11) z = 0.55f;
            depth[std::size_t(v) * intrinsics.width + u] = z;
        }

    for (int prefilterWindow : {0, 3, 5})
        for (int minimumValidNeighbours : {0, 6})
            for (bool symmetric : {false, true}) {
                Pipeline::DepthFilterOptions filter;
                filter.prefilterWindow = prefilterWindow;
                filter.minimumValidNeighbours = minimumValidNeighbours;
                filter.symmetricDepthJumpGuard = symmetric;
                filter.maximumDepthMeters = 4.0f;
                SCOPED_TRACE("window=" + std::to_string(prefilterWindow) +
                             " neighbours=" + std::to_string(minimumValidNeighbours) +
                             " symmetric=" + std::to_string(int(symmetric)));

                Pipeline::DepthFrame frame;
                frame.depth = depth;
                Pipeline::DepthFilterStats cpuStats;
                const Pipeline::Frame expected =
                        Pipeline::BackprojectDepth(frame, intrinsics, filter, &cpuStats);

                ValidationMask mask(context, intrinsics.width, intrinsics.height);
                auto source = UploadDepth(context, depth);
                {
                    Engine::Compute::CommandBatch batch(context);
                    mask.Execute(batch, *source, intrinsics, filter);
                    batch.Submit();
                }

                const std::uint32_t count = mask.PointCount();
                ASSERT_EQ(count, expected.pts.size());
                ASSERT_GT(count, 0u);

                const auto points = mask.DownloadPoints();
                const auto normals = mask.DownloadNormals();
                for (std::size_t i = 0; i < count; ++i) {
                    EXPECT_NEAR((points[i] - expected.pts[i]).norm(), 0.0f, 1e-5f) << "point " << i;
                    EXPECT_NEAR((normals[i] - expected.nrm[i]).norm(), 0.0f, 1e-4f)
                            << "normal " << i;
                }

                const ValidationMaskCounters counters = mask.DownloadCounters();
                EXPECT_EQ(counters.emittedPoints, cpuStats.emittedPoints.load());
                EXPECT_EQ(counters.rejectedByRange, cpuStats.rejectedByRange.load());
                EXPECT_EQ(counters.rejectedByNeighbourSupport,
                          cpuStats.rejectedByNeighbourSupport.load());
            }
}

// Counters are per frame, not cumulative: Execute zeroes them, so running twice must not double
// them. The CPU stats object is owned by the caller and accumulates across frames -- this one is
// scoped to the pass, and mixing the two conventions would silently inflate every report.
TEST(ValidationMask, ExecuteResetsItsCountersEachRun) {
    Engine::Core::Context context;
    const Pipeline::CameraIntrinsics intrinsics = TestIntrinsics(24, 16);
    const std::size_t pixels = std::size_t(intrinsics.width) * intrinsics.height;
    std::vector<float> depth(pixels, 1.2f);
    depth[10] = 9.0f; // one range rejection

    Pipeline::DepthFilterOptions filter;
    filter.maximumDepthMeters = 4.0f;

    ValidationMask mask(context, intrinsics.width, intrinsics.height);
    auto source = UploadDepth(context, depth);

    std::uint32_t firstCount = 0;
    ValidationMaskCounters firstCounters;
    for (int run = 0; run < 2; ++run) {
        {
            Engine::Compute::CommandBatch batch(context);
            mask.Execute(batch, *source, intrinsics, filter);
            batch.Submit();
        }
        const ValidationMaskCounters counters = mask.DownloadCounters();
        if (run == 0) {
            firstCount = mask.PointCount();
            firstCounters = counters;
            EXPECT_EQ(counters.rejectedByRange, 1u);
        } else {
            EXPECT_EQ(mask.PointCount(), firstCount);
            EXPECT_EQ(counters.rejectedByRange, firstCounters.rejectedByRange);
            EXPECT_EQ(counters.emittedPoints, firstCounters.emittedPoints);
        }
    }
}

// The buffers are sized from the constructor's dimensions, so a mismatched intrinsics would write
// past them. Fail loudly instead.
TEST(ValidationMask, ExecuteRejectsIntrinsicsThatDoNotMatchTheAllocation) {
    Engine::Core::Context context;
    ValidationMask mask(context, 16, 16);
    Pipeline::CameraIntrinsics wrong = TestIntrinsics(32, 16);
    std::vector<float> depth(16 * 16, 1.0f);
    auto source = UploadDepth(context, depth);

    Engine::Compute::CommandBatch batch(context);
    EXPECT_THROW(mask.Execute(batch, *source, wrong, {}), std::runtime_error);
}
