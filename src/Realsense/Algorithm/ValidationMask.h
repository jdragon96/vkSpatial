#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Compute/StagingBuffer.h"
#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"
#include "Engine/Core/Image.h"
#include "Engine/Core/Sampler.h"

#include "Realsense/RealSenseTypes.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace Realsense {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Per-pixel depth confidence for a D400-series frame.
    //
    //     c = [z > 0] * c_range * c_ir * c_nb
    //
    // The D4 VPU already scores its own confidence over forty-odd parameters and then discards it,
    // publishing only the verdict as depth == 0; RS2_STREAM_CONFIDENCE is L515 and will not open on
    // a D435. So the continuous confidence a weighted fusion wants has to be synthesised, which is
    // what the kernel beside this header does.
    //
    // ONE STAGE, not the whole front end. This class uploads the frame and scores it; thresholding,
    // back-projection, normals and compaction belong to RealSensePipeline, which owns the buffers
    // that flow between stages and the order they run in. The two things it publishes downstream
    // are the depth image -- already uploaded, so the threshold pass re-reads it rather than paying
    // for a second copy -- and the property buffer.
    ///////////////////////////////////////////////////////////////////////////////////////////////
    class ValidationMask {
    public:
        ValidationMask(Engine::Core::Context &context, int width, int height)
            : m_context(context), m_width(width), m_height(height) {
            if (width <= 0 || height <= 0)
                throw std::runtime_error("Realsense::ValidationMask::ValidationMask: the frame is " +
                                         std::to_string(width) + "x" + std::to_string(height));

            const std::size_t pixels = std::size_t(width) * std::size_t(height);

            createSampledImage(m_depthImage, VK_FORMAT_R16_UINT);
            createSampledImage(m_infraredImage, VK_FORMAT_R8_UINT);

            Engine::Core::SamplerDescriptor samplerDescriptor;
            samplerDescriptor.magFilter = VK_FILTER_NEAREST;
            samplerDescriptor.minFilter = VK_FILTER_NEAREST;
            samplerDescriptor.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            samplerDescriptor.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerDescriptor.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerDescriptor.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            m_sampler = std::make_unique<Engine::Core::Sampler>(context, samplerDescriptor);

            m_depthStaging = std::make_unique<Engine::Compute::StagingBuffer>(
                    context, VkDeviceSize(pixels * sizeof(std::uint16_t)),
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            m_infraredStaging = std::make_unique<Engine::Compute::StagingBuffer>(
                    context, VkDeviceSize(pixels), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

            m_counters = std::make_unique<Engine::Core::Buffer>(context);
            m_counters->AllocateHostVisibleReadback(std::uint32_t(sizeof(ValidationScoreCounters)));

            m_properties = std::make_unique<Engine::Core::Buffer>(context);
            m_properties->AllocateHostVisibleReadback(
                    std::uint32_t(pixels * sizeof(ValidationMaskProperty)));

            // Build the default score variant now so a broken shader surfaces here rather than at
            // the first frame; the other three are compiled the first time they are asked for.
            scoreKernel(ValidationScoreOptions{});
        }

        // Uploads the frame, resets the counters and scores it.
        void Execute(Engine::Compute::CommandBatch &batch,
                     const std::uint16_t *depthZ16,
                     const ValidationScoreOptions &options,
                     const std::uint8_t *infraredY8 = nullptr) {
            ValidateOptions(options);
            if (!depthZ16)
                throw std::runtime_error("Realsense::ValidationMask::Execute: depthZ16 is null");
            if (options.useInfrared && !infraredY8)
                throw std::runtime_error("Realsense::ValidationMask::Execute: useInfrared is set "
                                         "but no infrared frame was given");

            const std::size_t pixels = std::size_t(m_width) * std::size_t(m_height);
            std::memcpy(m_depthStaging->Mapped(), depthZ16, pixels * sizeof(std::uint16_t));
            batch.CopyBufferToImage(m_depthStaging->Handle(), *m_depthImage);

            if (options.useInfrared) {
                std::memcpy(m_infraredStaging->Mapped(), infraredY8, pixels);
                batch.CopyBufferToImage(m_infraredStaging->Handle(), *m_infraredImage);
            }

            batch.FillBuffer(m_counters->Handle(), 0, sizeof(ValidationScoreCounters), 0u);
            batch.Barrier();

            RecordScore(batch, options);
        }

        void RecordScore(Engine::Compute::CommandBatch &batch,
                         const ValidationScoreOptions &options) {
            ValidateOptions(options);

            const ScorePushConstants pushConstants{m_width,
                                                   m_height,
                                                   options.depthScale,
                                                   options.subpixelRms,
                                                   options.focalLengthPixels,
                                                   options.baselineMeters,
                                                   options.sameSurfaceSigmaMultiplier,
                                                   options.nearFadeStart,
                                                   options.nearFadeEnd,
                                                   options.farFadeStart,
                                                   options.farFadeEnd,
                                                   options.infraredFloor,
                                                   options.infraredReference,
                                                   options.infraredSaturation};

            Engine::Core::ComputePipeline &kernel = scoreKernel(options);
            kernel.Bind(0, *m_depthImage, *m_sampler)
                    .Bind(1, *m_properties)
                    .Bind(2, *m_counters);
            if (options.useInfrared) {
                kernel.Bind(3, *m_infraredImage, *m_sampler);
            }
            kernel.Args(pushConstants);

            const VkExtent3D localSize = kernel.GetLocalSize();
            if (localSize.width == 0 || localSize.height == 0)
                throw std::runtime_error("Realsense::ValidationMask::RecordScore: the kernel "
                                         "reported a zero local size");
            batch.Dispatch(kernel,
                           (std::uint32_t(m_width) + localSize.width - 1) / localSize.width,
                           (std::uint32_t(m_height) + localSize.height - 1) / localSize.height, 1);
        }

        // The dense score image, one float per pixel in row-major order.
        //
        // Gathered rather than memcpy'd: the score is the third member of ValidationMaskProperty,
        // so the device stride is 12 bytes and a flat copy would read `valid` and `emitted` bits as
        // floats. The whole property buffer has to be made visible, not just the float count.
        std::vector<float> DownloadScores() const {
            const std::size_t pixels = std::size_t(m_width) * std::size_t(m_height);
            m_properties->MakeVisibleToCPU(std::uint32_t(pixels * sizeof(ValidationMaskProperty)));
            const auto *entries = static_cast<const ValidationMaskProperty *>(m_properties->MappedPtr());
            std::vector<float> out(pixels);
            for (std::size_t i = 0; i < pixels; ++i) out[i] = entries[i].score;
            return out;
        }

        ValidationScoreCounters DownloadCounters() const {
            m_counters->MakeVisibleToCPU(std::uint32_t(sizeof(ValidationScoreCounters)));
            ValidationScoreCounters out;
            std::memcpy(&out, m_counters->MappedPtr(), sizeof out);
            return out;
        }

        int Width() const { return m_width; }
        int Height() const { return m_height; }

        // What the next stage consumes. Handed out rather than copied: the threshold pass writes
        // `emitted` into the same property buffer this stage wrote `score` into, and re-uploading
        // the depth image for it would double the per-frame transfer for nothing.
        Engine::Core::Buffer &Properties() { return *m_properties; }
        Engine::Core::Image &DepthImage() { return *m_depthImage; }
        Engine::Core::Image &InfraredImage() { return *m_infraredImage; }
        Engine::Core::Sampler &ImageSampler() { return *m_sampler; }

        static void ValidateOptions(const ValidationScoreOptions &options) {
            const auto refuse = [](const std::string &reason) {
                throw std::runtime_error("Realsense::ValidationMask: " + reason);
            };
            if (!(options.depthScale > 0.0f)) refuse("depthScale must be positive");
            if (!(options.subpixelRms > 0.0f)) refuse("subpixelRms must be positive");
            if (!(options.focalLengthPixels > 0.0f))
                refuse("focalLengthPixels must be set from the stream's intrinsics");
            if (!(options.baselineMeters > 0.0f)) refuse("baselineMeters must be positive");
            if (!(options.sameSurfaceSigmaMultiplier > 0.0f))
                refuse("sameSurfaceSigmaMultiplier must be positive");
            if (!(options.nearFadeStart < options.nearFadeEnd))
                refuse("nearFadeStart must be strictly below nearFadeEnd");
            if (!(options.farFadeStart < options.farFadeEnd))
                refuse("farFadeStart must be strictly below farFadeEnd");
            if (!(options.nearFadeEnd <= options.farFadeStart))
                refuse("the near and far fades overlap, so no depth scores 1");
            if (options.useInfrared && !(options.infraredFloor < options.infraredReference))
                refuse("infraredFloor must be strictly below infraredReference");
        }

    private:
        // One pipeline per macro combination, compiled on first use and cached. The two switches
        // are -D definitions rather than uniforms, so they cannot share a module; caching rather
        // than rebuilding matters because a rebuilt pipeline re-allocates descriptors, and the lab
        // re-scores the frozen frame on every slider drag.
        Engine::Core::ComputePipeline &scoreKernel(const ValidationScoreOptions &options) {
            std::string key;
            if (options.useInfrared) key += "infrared;";
            if (options.countRejections) key += "counters;";

            auto found = m_scoreKernels.find(key);
            if (found != m_scoreKernels.end()) return *found->second;

            auto pipeline = std::make_unique<Engine::Core::ComputePipeline>(m_context);
            if (options.useInfrared) pipeline->Define("VALIDATION_SCORE_WITH_INFRARED");
            if (options.countRejections) pipeline->Define("VALIDATION_SCORE_WITH_COUNTERS");
            pipeline->Build("Realsense/Algorithm/ValidationMask.CalculateScore.glsl");
            return *m_scoreKernels.emplace(key, std::move(pipeline)).first->second;
        }

        void createSampledImage(std::unique_ptr<Engine::Core::Image> &image, VkFormat format) {
            Engine::Core::ImageDescriptor descriptor = Engine::Core::ImageDescriptor::Color2D(
                    VkExtent2D{std::uint32_t(m_width), std::uint32_t(m_height)},
                    format,
                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
            descriptor.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            image = std::make_unique<Engine::Core::Image>(m_context, descriptor);
        }

        Engine::Core::Context &m_context;
        int m_width = 0;
        int m_height = 0;

        std::unique_ptr<Engine::Core::Image> m_depthImage;
        std::unique_ptr<Engine::Core::Image> m_infraredImage;
        std::unique_ptr<Engine::Core::Sampler> m_sampler;
        std::unique_ptr<Engine::Compute::StagingBuffer> m_depthStaging;
        std::unique_ptr<Engine::Compute::StagingBuffer> m_infraredStaging;
        std::unique_ptr<Engine::Core::Buffer> m_counters;
        std::unique_ptr<Engine::Core::Buffer> m_properties;

        std::map<std::string, std::unique_ptr<Engine::Core::ComputePipeline>> m_scoreKernels;
    };

} // namespace Realsense
