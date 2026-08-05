#pragma once

#include "Engine/Pipeline/Reconstruction/DepthCameraFrameSource.h" // CameraIntrinsics (shared)
#include "Engine/Pipeline/Reconstruction/ReconstructionSource.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace Engine::Pipeline {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Structured-light strategy (extensible) — reconstruct a point cloud from a set of projected-
    // pattern images. The DECODE/triangulation algorithm varies (gray-code, phase-shift, ...), so it
    // is itself a pluggable strategy (Command Pattern), exactly like the ICP AlignmentCommand: a new
    // decoder is a new IStructuredLightDecoder, not a new source. The source grabs pattern sets from
    // an IPatternProvider and reconstructs each set with the chosen decoder.
    ///////////////////////////////////////////////////////////////////////////////////////////////

    struct PatternSet {
        std::vector<std::vector<uint8_t>> images; // captured images under each projected pattern
        int width = 0, height = 0;
    };

    class IPatternProvider {
    public:
        virtual ~IPatternProvider() = default;
        virtual const CameraIntrinsics &Intrinsics() const = 0;
        virtual bool Grab(PatternSet &out) = 0;
    };

    // e.g. GrayCodeDecoder, PhaseShiftDecoder (add later). Each is the reconstruction core for one
    // structured-light method.
    class IStructuredLightDecoder {
    public:
        virtual ~IStructuredLightDecoder() = default;
        virtual const char *Name() const = 0;
        virtual bool Reconstruct(const PatternSet &patterns, const CameraIntrinsics &k, Frame &out) = 0;
    };

    class StructuredLightFrameSource : public IFrameSource {
    public:
        StructuredLightFrameSource(std::unique_ptr<IPatternProvider> device,
                                   std::unique_ptr<IStructuredLightDecoder> decoder)
            : m_device(std::move(device)), m_decoder(std::move(decoder)) {}

        EAcquisitionType Type() const override { return EAcquisitionType::StructuredLight; }
        const char *Name() const override { return "structured-light"; }

        bool Next(Frame &out) override {
            PatternSet ps;
            if (!m_device || !m_decoder || !m_device->Grab(ps)) return false;
            return m_decoder->Reconstruct(ps, m_device->Intrinsics(), out);
        }

    private:
        std::unique_ptr<IPatternProvider> m_device;
        std::unique_ptr<IStructuredLightDecoder> m_decoder;
    };

} // namespace Engine::Pipeline
