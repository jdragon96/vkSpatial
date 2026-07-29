#pragma once

#include "Engine/Spatial/AdvancedTSDF.h"
#include "Engine/Spatial/TiledDirectionalTSDF.h"

namespace Engine::Spatial {

    // AdvancedTSDF-backed tiled TSDF: same geometry/routing/extraction as TiledDirectionalTSDF, but
    // each tile is an AdvancedTSDF (compact hash + point-to-plane + stored-gradient). Adds the A1
    // (confidence weight) and A2 (Hermite position) setters, forwarded to every tile via the
    // configure hook. Configure these BEFORE Integrate -- the hook is applied when a tile is first
    // created (during Integrate).
    class TiledAdvancedTSDF : public TiledDirectionalTSDF<AdvancedTSDF> {
    public:
        TiledAdvancedTSDF() {
            m_configureHook = [this](AdvancedTSDF &tile) {
                tile.SetConfidenceWeight(m_confWeight);
                tile.SetHermitePosition(m_hermite);
            };
        }

        // A1: surface-proximity confidence weight lambda in [0,1]; 0 = off. Matches AdvancedTSDF.
        void SetConfidenceWeight(float lambda) { m_confWeight = lambda; }

        // A2: cubic-Hermite zero-crossing position instead of linear. false = linear (default).
        void SetHermitePosition(bool on) { m_hermite = on; }

    private:
        float m_confWeight = 0.5f; // A1 default matches AdvancedTSDF
        bool m_hermite = false;    // A2 default matches AdvancedTSDF
    };

} // namespace Engine::Spatial
