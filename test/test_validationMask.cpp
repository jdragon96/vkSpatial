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
        mask.RecordPrefilterDepth(batch, *source, filtered, width, height, filter.prefilterWindow,
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
        mask.RecordPrefilterDepth(batch, *source, filtered, width, height, 1, 0.02f, 0.005f);
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
        mask.RecordPrefilterDepth(batch, *source, filtered, width, height, filter.prefilterWindow,
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
