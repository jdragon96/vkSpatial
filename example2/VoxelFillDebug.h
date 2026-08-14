#pragma once

#include "TSDF/Backends/AdvancedTSDF.h" // Engine::Spatial::AdvancedEntry

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace voxdbg {

    using Rgba = std::array<uint8_t, 4>;

    // Quantized voxel-direction key: round(center/voxel) per axis + direction. Stable across
    // repeated DownloadEntries() (same voxel centre -> same key).
    struct VoxelKey {
        int x, y, z;
        uint8_t dir;
        bool operator==(const VoxelKey &o) const {
            return x == o.x && y == o.y && z == o.z && dir == o.dir;
        }
    };
    struct VoxelKeyHash {
        std::size_t operator()(const VoxelKey &k) const {
            std::size_t h = std::hash<int>()(k.x);
            h ^= std::hash<int>()(k.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.z) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(int(k.dir)) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };

    inline VoxelKey keyOf(const Engine::Spatial::AdvancedEntry &e, float voxel) {
        return VoxelKey{int(std::lround(e.center.x() / voxel)),
                        int(std::lround(e.center.y() / voxel)),
                        int(std::lround(e.center.z() / voxel)), uint8_t(e.direction)};
    }

    // Tracks the first frame that filled each voxel-direction. update() marks entries whose key is
    // seen for the first time and records firstFrame[key]=frameIdx for them.
    class FillTracker {
    public:
        explicit FillTracker(float voxel) : m_voxel(voxel) {}
        void reset() { m_firstFrame.clear(); }
        std::vector<char> update(const std::vector<Engine::Spatial::AdvancedEntry> &entries,
                                 int frameIdx) {
            std::vector<char> isNew(entries.size(), 0);
            for (std::size_t i = 0; i < entries.size(); ++i) {
                const VoxelKey k = keyOf(entries[i], m_voxel);
                if (m_firstFrame.find(k) == m_firstFrame.end()) {
                    m_firstFrame.emplace(k, frameIdx);
                    isNew[i] = 1;
                }
            }
            return isNew;
        }
        int firstFrame(const VoxelKey &k) const {
            auto it = m_firstFrame.find(k);
            return it == m_firstFrame.end() ? -1 : it->second;
        }
        std::size_t size() const { return m_firstFrame.size(); }

    private:
        float m_voxel;
        std::unordered_map<VoxelKey, int, VoxelKeyHash> m_firstFrame;
    };

    enum class ColorMode { TsdfSign, Weight, FillFrame, Direction };

    // tsdf in world units: +d red, -d blue, |d| < 0.1*trunc white (surface band); fades to white
    // toward the surface.
    inline Rgba tsdfColor(float tsdf, float trunc) {
        const float band = 0.1f * trunc;
        if (std::fabs(tsdf) < band) return {255, 255, 255, 255};
        const float t = std::min(1.0f, std::fabs(tsdf) / std::max(1e-6f, trunc));
        const uint8_t c = uint8_t(60 + 195 * (1.0f - t));
        return tsdf > 0.0f ? Rgba{255, c, c, 255} : Rgba{c, c, 255, 255};
    }
    // weight heat: 0 -> blue, wMax -> red.
    inline Rgba weightColor(float w, float wMax) {
        const float t = std::min(1.0f, std::max(0.0f, w / std::max(1e-6f, wMax)));
        return {uint8_t(255.0f * t), 40, uint8_t(255.0f * (1.0f - t)), 255};
    }
    // fill-frame rainbow: frame 0 red -> mid green -> last blue; unseen (-1) grey.
    inline Rgba fillFrameColor(int frame, int nFrames) {
        if (frame < 0) return {90, 90, 90, 255};
        const float h = float(frame) / float(std::max(1, nFrames - 1));
        const float r = std::max(0.0f, 1.0f - 2.0f * h);
        const float g = 1.0f - std::fabs(2.0f * h - 1.0f);
        const float b = std::max(0.0f, 2.0f * h - 1.0f);
        return {uint8_t(255.0f * r), uint8_t(255.0f * g), uint8_t(255.0f * b), 255};
    }
    // 6 axis colours: +X,-X,+Y,-Y,+Z,-Z.
    inline Rgba directionColor(uint8_t dir) {
        static const Rgba lut[6] = {{230, 60, 60, 255}, {120, 20, 20, 255}, {60, 230, 60, 255},
                                    {20, 120, 20, 255},  {60, 60, 230, 255}, {20, 20, 120, 255}};
        return lut[dir < 6 ? dir : 0];
    }
    inline bool belowThreshold(float weight, float wThresh) { return weight < wThresh; }

} // namespace voxdbg
