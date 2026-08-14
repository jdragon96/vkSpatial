#pragma once

#include "TSDF/Backends/AdvancedTSDF.h"
#include "TSDF/Backends/TiledDirectionalTSDF.h"

#include <algorithm>
#include <cstring>

namespace TSDF {

    class TiledAdvancedTSDF : public TiledDirectionalTSDF<AdvancedTSDF> {
    public:
        TiledAdvancedTSDF() {
            m_configureHook = [this](AdvancedTSDF &tile) {
                tile.SetConfidenceWeight(m_confWeight);
                tile.SetHermitePosition(m_hermite);
            };
        }

        void SetConfidenceWeight(float lambda) { m_confWeight = lambda; }

        void SetHermitePosition(bool on) { m_hermite = on; }

        void DownloadEntries(std::vector<AdvancedEntry> &out) const {
            if (this->TileCount() == 0) {
                out.resize(0);
                return;
            }

            ensureShared(m_sharedCapacity == 0 ? kInitialShared : m_sharedCapacity);
            uint32_t total = compactAllTiles();
            if (total > m_sharedCapacity) {
                ensureShared(total + total / 2u);
                total = compactAllTiles();
            }
            const uint32_t n = std::min(total, m_sharedCapacity);
            out.reserve(m_sharedCapacity);
            out.resize(n);
            if (n > 0) {
                m_sharedOut->MakeVisibleToCPU(n * uint32_t(sizeof(AdvancedEntry)));
                std::memcpy(out.data(), m_sharedOut->MappedPtr(), n * sizeof(AdvancedEntry));
            }
        }

        // Convenience wrapper (tests / one-off callers). The per-frame hot path uses the reusing form.
        std::vector<AdvancedEntry> DownloadEntries() const {
            std::vector<AdvancedEntry> out;
            DownloadEntries(out);
            return out;
        }

    private:
        float m_confWeight = 0.5f;
        bool m_hermite = false;

        uint32_t compactAllTiles() const {
            auto *countPtr = static_cast<uint32_t *>(m_sharedCount->MappedPtr());
            *countPtr = 0;
            m_sharedCount->MakeVisibleToGPU(sizeof(uint32_t));

            Engine::Compute::CommandBatch batch(*this->contextPtr());
            this->forEachTileCore([&](const AdvancedTSDF &tile,
                                      const Eigen::Vector3i &coreMin,
                                      const Eigen::Vector3i &coreMax) {
                tile.RecordCompact(*m_sharedOut, *m_sharedCount, batch, coreMin, coreMax);
            });
            batch.Submit();
            m_sharedCount->MakeVisibleToCPU(sizeof(uint32_t));
            return *countPtr;
        }

        static constexpr uint32_t kInitialShared = 1u << 23; // ~8 M entries up front -> no mid-scan growth
        mutable std::unique_ptr<Engine::Core::Buffer> m_sharedOut;
        mutable std::unique_ptr<Engine::Core::Buffer> m_sharedCount;
        mutable uint32_t m_sharedCapacity = 0;

        void ensureShared(uint32_t entries) const {
            if (m_sharedOut && m_sharedCapacity >= entries) return;
            m_sharedCapacity = entries;
            m_sharedOut = std::make_unique<Engine::Core::Buffer>(*this->contextPtr());
            m_sharedCount = std::make_unique<Engine::Core::Buffer>(*this->contextPtr());
            m_sharedOut->AllocateHostVisibleReadback(m_sharedCapacity * uint32_t(sizeof(AdvancedEntry)));
            m_sharedCount->AllocateHostVisibleReadback(sizeof(uint32_t));
        }
    };

} // namespace TSDF
