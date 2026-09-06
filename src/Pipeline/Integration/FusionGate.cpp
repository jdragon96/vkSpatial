#include "Pipeline/Integration/FusionGate.h"

namespace Pipeline {

    FusionGate::FusionGate(FusionGateConfig config)
        : m_config(config),
          // Bootstrapping off means there is nothing to wait for, and callers must not have to
          // special-case that: the gate reports itself armed from construction.
          m_armed(config.bootstrapConsecutiveFrames <= 0) {}

    bool FusionGate::Admit(bool trackValid, ETrackFailure failure, float fitness, float rmse) {
        if (!m_seeded) {
            m_seeded = true;
            return true;
        }

        if (!m_armed) {
            if (fitness >= m_config.bootstrapMinFitness) ++m_consecutiveGoodFrames;
            else
                m_consecutiveGoodFrames = 0;
            if (m_consecutiveGoodFrames >= m_config.bootstrapConsecutiveFrames) m_armed = true;
            ++m_bootstrapHeldFrames;
            return false;
        }

        if (!trackValid && (failure == ETrackFailure::NoModel || failure == ETrackFailure::NoLocalTarget)) return true;

        if (m_config.minimumFusionFitness > 0.0f && fitness < m_config.minimumFusionFitness) {
            ++m_fusionRejectedByFitness;
            return false;
        }
        if (m_config.maximumFusionRmse > 0.0f && rmse > m_config.maximumFusionRmse) {
            ++m_fusionRejectedByRmse;
            return false;
        }
        return true;
    }

} // namespace Pipeline
