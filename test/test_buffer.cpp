#include "Engine/Core/Buffer.h"
#include "Engine/Core/Context.h"

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

// A host-visible mapped buffer: writes through MappedPtr are visible to a GPU-side Download.
TEST(Buffer, HostVisibleMappedRoundTrip) {
    Engine::Core::Context ctx;
    Engine::Core::Buffer buf(ctx);
    buf.AllocateHostVisible(16 * sizeof(float));
    ASSERT_NE(buf.MappedPtr(), nullptr);

    std::vector<float> src(16);
    for (int i = 0; i < 16; ++i) src[i] = float(i) * 1.5f;
    std::memcpy(buf.MappedPtr(), src.data(), src.size() * sizeof(float));
    buf.MakeVisibleToGPU(uint32_t(src.size() * sizeof(float)));

    std::vector<float> dst(16, 0.0f);
    buf.Download(dst.data(), uint32_t(dst.size() * sizeof(float))); // GPU copy back
    for (int i = 0; i < 16; ++i) EXPECT_FLOAT_EQ(dst[i], src[i]);
}

// Plain Allocate is device-local: no persistent mapping.
TEST(Buffer, PlainAllocateHasNoMappedPtr) {
    Engine::Core::Context ctx;
    Engine::Core::Buffer buf(ctx);
    buf.Allocate(16);
    EXPECT_EQ(buf.MappedPtr(), nullptr);
}
