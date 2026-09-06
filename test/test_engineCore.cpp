#include <gtest/gtest.h>

#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"
#include "Engine/Core/Image.h"
#include "Engine/Core/OneShotCommands.h"
#include "Engine/Core/Sampler.h"
#include "Engine/Compute/StagingBuffer.h"

#include <cstring>
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

TEST(ComputePipelineTest, RecordDispatchIntoExternalCommandBufferProducesSameResult) {
    Context ctx;

    constexpr uint32_t N = 4096u;
    std::vector<uint32_t> input(N, 3u);

    Buffer inputBuffer(ctx, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    inputBuffer.Allocate(N * sizeof(uint32_t));
    inputBuffer.Upload(input.data(), N * sizeof(uint32_t));

    static const char *kShader = R"(
        #version 450
        layout(local_size_x = 256) in;
        layout(binding = 0) buffer Buf { uint v[]; } b;
        layout(push_constant) uniform PC { uint count; } pc;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= pc.count) return;
            b.v[i] = b.v[i] * 2u;
        }
    )";
    struct PC { uint32_t count; };

    ComputePipeline pipe(ctx);
    pipe.Build(kShader, ShaderInput::GlslSrc).Bind(0, inputBuffer).Args(PC{N});

    // Drive submission ourselves via a one-shot command buffer + RecordDispatch.
    const uint32_t gridX = (N + pipe.GetLocalSize().width - 1) / pipe.GetLocalSize().width;
    SubmitOneShot(ctx, QueueRole::Compute, [&](VkCommandBuffer cmd) {
        pipe.RecordDispatch(cmd, gridX);
    });

    std::vector<uint32_t> result(N, 0u);
    inputBuffer.Download(result.data(), N * sizeof(uint32_t));
    for (uint32_t i = 0; i < N; ++i) EXPECT_EQ(result[i], 6u) << "index " << i;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Combined image samplers in a compute pipeline. Every kernel in this repository read its inputs
// from storage buffers until kernel_ValidationScore, which samples a RealSense Z16 depth image
// directly -- half the upload of an unpacked float buffer, and no unpacking arithmetic per pixel.
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

    // Stages `texels` into a fresh sampled image and leaves it in SHADER_READ_ONLY_OPTIMAL.
    // There is no Engine::Core helper for this yet; a kernel that samples its input every frame
    // will want one, but a test that runs the copy once should not be what defines its shape.
    void UploadSampledImage(Context &context, Image &image, uint32_t width, uint32_t height,
                            const std::vector<uint16_t> &texels) {
        ImageDescriptor descriptor = ImageDescriptor::Color2D(
                VkExtent2D{width, height}, VK_FORMAT_R16_UINT,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        descriptor.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        image.Create(descriptor);

        const VkDeviceSize bytes = VkDeviceSize(texels.size() * sizeof(uint16_t));
        Engine::Compute::StagingBuffer staging(context, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        std::memcpy(staging.Mapped(), texels.data(), std::size_t(bytes));

        SubmitOneShot(context, QueueRole::Compute, [&](VkCommandBuffer cmd) {
            image.TransitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);

            VkBufferImageCopy region{};
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {width, height, 1};
            vkCmdCopyBufferToImage(cmd, staging.Handle(), image.Handle(),
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            image.TransitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        });
    }

} // namespace

// A texel-for-texel copy out of a usampler2D. The fixture is 5x3 rather than square, and the
// values encode row * 100 + column, so a kernel that transposed its coordinates or walked the
// image with the wrong row stride cannot produce the expected buffer.
TEST(ComputePipelineTest, SamplesAUsampler2DAlongsideAStorageBuffer) {
    Context context;

    const uint32_t width = 5, height = 3;
    std::vector<uint16_t> texels(std::size_t(width) * height);
    for (uint32_t row = 0; row < height; ++row)
        for (uint32_t column = 0; column < width; ++column)
            texels[std::size_t(row) * width + column] = uint16_t(row * 100 + column);

    Image image(context);
    UploadSampledImage(context, image, width, height, texels);

    Sampler sampler(context, SamplerDescriptor{});

    Buffer output(context, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    output.Allocate(uint32_t(texels.size() * sizeof(uint32_t)));

    static const char *kShader = R"(
        #version 450
        layout(local_size_x = 8, local_size_y = 8) in;
        layout(set = 0, binding = 0) uniform usampler2D source;
        layout(std430, set = 0, binding = 1) writeonly buffer Output { uint values[]; };
        layout(push_constant) uniform PC { int width; int height; };
        void main() {
            ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
            if (pixel.x >= width || pixel.y >= height) return;
            values[pixel.y * width + pixel.x] = texelFetch(source, pixel, 0).x;
        }
    )";

    struct PushConstants { int32_t width; int32_t height; };

    ComputePipeline pipeline(context);
    pipeline.Build(kShader, ShaderInput::GlslSrc)
            .Bind(0, image, sampler)
            .Bind(1, output)
            .Args(PushConstants{int32_t(width), int32_t(height)})
            .Dispatch(1, 1, 1);

    std::vector<uint32_t> readback(texels.size());
    output.Download(readback.data(), uint32_t(readback.size() * sizeof(uint32_t)));

    for (std::size_t i = 0; i < texels.size(); ++i)
        EXPECT_EQ(readback[i], uint32_t(texels[i])) << "texel " << i;
}

// Two samplers and a buffer in one descriptor set, which is the shape kernel_ValidationScore uses.
//
// It does NOT pin the per-type descriptor pool sizing, though that is what the code does and what
// the spec requires: mutating updateDescriptors() to request a pool of storage-buffer descriptors
// only leaves this test green, because MoltenVK does not enforce the pool's type accounting and
// never returns VK_ERROR_OUT_OF_POOL_MEMORY. A driver that does -- or a validation layer, which
// this Context cannot enable -- would reject it. The sizing stays correct on purpose; there is
// simply no local test that would go red if it regressed.
TEST(ComputePipelineTest, TwoSamplersAndABufferShareOneDescriptorSet) {
    Context context;

    const uint32_t width = 2, height = 2;
    const std::vector<uint16_t> texels{7, 7, 7, 7};

    Image first(context), second(context);
    UploadSampledImage(context, first, width, height, texels);
    UploadSampledImage(context, second, width, height, texels);

    Sampler sampler(context, SamplerDescriptor{});

    Buffer output(context, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    output.Allocate(uint32_t(texels.size() * sizeof(uint32_t)));

    static const char *kShader = R"(
        #version 450
        layout(local_size_x = 2, local_size_y = 2) in;
        layout(set = 0, binding = 0) uniform usampler2D first;
        layout(set = 0, binding = 1) uniform usampler2D second;
        layout(std430, set = 0, binding = 2) writeonly buffer Output { uint values[]; };
        void main() {
            ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
            values[pixel.y * 2 + pixel.x] =
                texelFetch(first, pixel, 0).x + texelFetch(second, pixel, 0).x;
        }
    )";

    ComputePipeline pipeline(context);
    pipeline.Build(kShader, ShaderInput::GlslSrc)
            .Bind(0, first, sampler)
            .Bind(1, second, sampler)
            .Bind(2, output)
            .Dispatch(1, 1, 1);

    std::vector<uint32_t> readback(texels.size());
    output.Download(readback.data(), uint32_t(readback.size() * sizeof(uint32_t)));
    for (std::size_t i = 0; i < readback.size(); ++i) EXPECT_EQ(readback[i], 14u) << "texel " << i;
}
