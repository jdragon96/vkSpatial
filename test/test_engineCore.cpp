#include <gtest/gtest.h>

#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"
#include "Engine/Core/Image.h"
#include "Engine/Core/OneShotCommands.h"

#include <numeric>
#include <vector>

using namespace Engine::Core;

TEST(ContextTest, ComputeOnlyConstructionProducesValidHandles) {
    Context ctx;  // enablePresent=false by default

    EXPECT_NE(ctx.instance, VK_NULL_HANDLE);
    EXPECT_NE(ctx.physicalDevice, VK_NULL_HANDLE);
    EXPECT_NE(ctx.device, VK_NULL_HANDLE);
    EXPECT_NE(ctx.computeQueue, VK_NULL_HANDLE);
    EXPECT_NE(ctx.cmdPool, VK_NULL_HANDLE);
    EXPECT_NE(ctx.allocator, VK_NULL_HANDLE);

    EXPECT_EQ(ctx.graphicsQueue, VK_NULL_HANDLE);
    EXPECT_EQ(ctx.graphicsCmdPool, VK_NULL_HANDLE);
    EXPECT_EQ(ctx.surface, VK_NULL_HANDLE);
}

TEST(OneShotCommandsTest, FillBufferRoundTripsThroughSubmit) {
    Context ctx;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = 256;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo allocationInfo{};
    ASSERT_EQ(vmaCreateBuffer(ctx.allocator, &bufferInfo, &allocInfo, &buffer, &allocation, &allocationInfo),
              VK_SUCCESS);

    // Zero the buffer up front so the post-fill check below is meaningful.
    std::memset(allocationInfo.pMappedData, 0xFF, 256);

    SubmitOneShot(ctx, QueueRole::Compute, [&](VkCommandBuffer cmd) {
        vkCmdFillBuffer(cmd, buffer, 0, 256, 0x2A2A2A2Au);
    });

    const uint32_t *data = static_cast<const uint32_t *>(allocationInfo.pMappedData);
    for (int i = 0; i < 64; ++i)
        EXPECT_EQ(data[i], 0x2A2A2A2Au) << "word " << i;

    vmaDestroyBuffer(ctx.allocator, buffer, allocation);
}

TEST(BufferTest, AllocateUploadDownloadRoundTrip) {
    Context ctx;
    Buffer buffer(ctx);

    constexpr uint32_t kCount = 1024;
    std::vector<float> source(kCount);
    std::iota(source.begin(), source.end(), 0.0f);

    buffer.Allocate(kCount * sizeof(float));
    buffer.Upload(source.data(), kCount * sizeof(float));

    std::vector<float> result(kCount, -1.0f);
    buffer.Download(result.data(), kCount * sizeof(float));

    for (uint32_t i = 0; i < kCount; ++i)
        EXPECT_FLOAT_EQ(result[i], source[i]) << "index " << i;
}

TEST(BufferTest, DownloadBeyondCapacityThrows) {
    Context ctx;
    Buffer buffer(ctx);
    buffer.Allocate(16);

    std::vector<uint8_t> dst(64);
    EXPECT_THROW(buffer.Download(dst.data(), 64), std::runtime_error);
}

TEST(ImageTest, CreateDepth2DProducesValidHandles) {
    Context ctx;
    Image image(ctx, ImageDescriptor::Depth2D({256, 256}));

    EXPECT_TRUE(image.Valid());
    EXPECT_TRUE(image.HasView());
    EXPECT_EQ(image.Format(), VK_FORMAT_D32_SFLOAT);
    EXPECT_EQ(image.AspectMask(), static_cast<VkImageAspectFlags>(VK_IMAGE_ASPECT_DEPTH_BIT));
    EXPECT_TRUE(image.Matches({256, 256}, VK_FORMAT_D32_SFLOAT));
    EXPECT_FALSE(image.Matches({512, 512}));
}

TEST(ComputePipelineTest, DispatchSimpleShaderProducesExpectedOutput) {
    Context ctx;

    constexpr uint32_t N = 10001u;  // sum(0..N-1) = 50,005,000
    std::vector<uint32_t> input(N);
    std::iota(input.begin(), input.end(), 0u);

    Buffer inputBuffer(ctx, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    inputBuffer.Allocate(N * sizeof(uint32_t));
    inputBuffer.Upload(input.data(), N * sizeof(uint32_t));

    Buffer outputBuffer(ctx, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    outputBuffer.Allocate(sizeof(uint32_t));
    uint32_t zero = 0;
    outputBuffer.Upload(&zero, sizeof(uint32_t));

    static const char *kShader = R"(
        #version 450
        layout(local_size_x = 256) in;
        layout(binding = 0) readonly buffer InputBuf { uint values[]; } inputBuf;
        layout(binding = 1) buffer OutputBuf { uint total; } outputBuf;
        layout(push_constant) uniform PC { uint count; } pc;
        void main() {
            uint idx = gl_GlobalInvocationID.x;
            if (idx >= pc.count) return;
            atomicAdd(outputBuf.total, inputBuf.values[idx]);
        }
    )";

    struct PushConstants { uint32_t count; };

    ComputePipeline pipeline(ctx);
    pipeline.Build(kShader, ShaderInput::GlslSrc)
            .Bind(0, inputBuffer)
            .Bind(1, outputBuffer)
            .Args(PushConstants{N})
            .DispatchElements(N);

    uint32_t result = 0;
    outputBuffer.Download(&result, sizeof(uint32_t));

    EXPECT_EQ(result, 50005000u);
}
