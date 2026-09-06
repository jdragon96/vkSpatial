#pragma once

#include "Pipeline/Types.h"

#include <cstdint>

namespace Pipeline {

    class FusionGate {
    public:
        explicit FusionGate(FusionGateConfig config = {});

        bool Admit(bool trackValid, ETrackFailure failure, float fitness, float rmse);

        bool IsArmed() const { return m_armed; }

        std::uint64_t BootstrapHeldFrames() const { return m_bootstrapHeldFrames; }
        std::uint64_t FusionRejectedByFitness() const { return m_fusionRejectedByFitness; }
        std::uint64_t FusionRejectedByRmse() const { return m_fusionRejectedByRmse; }

    private:
        FusionGateConfig m_config;
        bool m_seeded = false;
        bool m_armed = false;
        int m_consecutiveGoodFrames = 0;
        std::uint64_t m_bootstrapHeldFrames = 0;
        std::uint64_t m_fusionRejectedByFitness = 0;
        std::uint64_t m_fusionRejectedByRmse = 0;
    };

} // namespace Pipeline
