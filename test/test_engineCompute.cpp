#include <gtest/gtest.h>

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Compute/StagingBuffer.h"
#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"

#include <cstring>
#include <vector>

using namespace Engine::Core;
using namespace Engine::Compute;

TEST(StagingBufferTest, MappedPointerRoundTrips) {
    Context ctx;
    StagingBuffer staging(ctx, 256, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    ASSERT_NE(staging.Mapped(), nullptr);
    EXPECT_EQ(staging.Size(), 256u);

    auto *p = static_cast<uint32_t *>(staging.Mapped());
    for (int i = 0; i < 64; ++i) p[i] = uint32_t(i * 7);
    for (int i = 0; i < 64; ++i) EXPECT_EQ(p[i], uint32_t(i * 7));
}

TEST(CommandBatchTest, TwoDispatchesWithBarrierRunInOneSubmit) {
    Context ctx;
    constexpr uint32_t N = 4096u;

    Buffer buf(ctx, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    buf.Allocate(N * sizeof(uint32_t));
    std::vector<uint32_t> zeros(N, 0u);
    buf.Upload(zeros.data(), N * sizeof(uint32_t));

    // Program A: v[i] = i.  Program B: v[i] = v[i] * 2.  Separate pipeline objects.
    static const char *kFill = R"(
        #version 450
        layout(local_size_x = 256) in;
        layout(binding = 0) buffer Buf { uint v[]; } b;
        layout(push_constant) uniform PC { uint n; } pc;
        void main(){ uint i=gl_GlobalInvocationID.x; if(i<pc.n) b.v[i]=i; })";
    static const char *kDouble = R"(
        #version 450
        layout(local_size_x = 256) in;
        layout(binding = 0) buffer Buf { uint v[]; } b;
        layout(push_constant) uniform PC { uint n; } pc;
        void main(){ uint i=gl_GlobalInvocationID.x; if(i<pc.n) b.v[i]=b.v[i]*2u; })";
    struct PC { uint32_t n; };

    ComputePipeline fill(ctx);
    fill.Build(kFill, ShaderInput::GlslSrc).Bind(0, buf).Args(PC{N});
    ComputePipeline dbl(ctx);
    dbl.Build(kDouble, ShaderInput::GlslSrc).Bind(0, buf).Args(PC{N});

    CommandBatch batch(ctx);
    batch.DispatchElements(fill, N).Barrier().DispatchElements(dbl, N).Submit();

    std::vector<uint32_t> result(N, 0xFFFFFFFFu);
    buf.Download(result.data(), N * sizeof(uint32_t));
    for (uint32_t i = 0; i < N; ++i) EXPECT_EQ(result[i], i * 2u) << "index " << i;
}

TEST(CommandBatchTest, FillAndCopyRecordInOneSubmit) {
    Context ctx;
    constexpr uint32_t N = 64u;

    Buffer src(ctx);
    Buffer dst(ctx);
    src.Allocate(N * sizeof(uint32_t));
    dst.Allocate(N * sizeof(uint32_t));

    CommandBatch batch(ctx);
    batch.FillBuffer(src.Handle(), 0, VK_WHOLE_SIZE, 0xABABABABu).Barrier();
    std::vector<VkBufferCopy> regions(1);
    regions[0].srcOffset = 0;
    regions[0].dstOffset = 0;
    regions[0].size = N * sizeof(uint32_t);
    batch.CopyBuffer(src.Handle(), dst.Handle(), regions).Submit();

    std::vector<uint32_t> out(N, 0u);
    dst.Download(out.data(), N * sizeof(uint32_t));
    for (uint32_t v : out) EXPECT_EQ(v, 0xABABABABu);
}
