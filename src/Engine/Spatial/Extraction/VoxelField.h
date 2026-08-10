#pragma once

// Sparse, resolution-agnostic voxel field: the common input every isosurface extraction
// strategy in Engine::Spatial::Extraction consumes. Backed by a sparse hash map keyed on
// integer grid coordinates (matching the shared MarchingCubesCore's IVec3Hash), so a field
// can be built from an implicit analytic function (FromImplicit, used by tests/fixtures) or
// from a real TSDF's downloaded entries (FromAdvancedEntries).

#include "Engine/Spatial/Extraction/MarchingCubesCore.h"

#include <Eigen/Core>
#include <array>
#include <functional>
#include <unordered_map>
#include <vector>

namespace Engine::Spatial {
    struct AdvancedEntry; // forward declaration; full definition in AdvancedTSDF.h
}

namespace Engine::Spatial::Extraction {

    class VoxelField {
    public:
        void Insert(const std::array<int, 3> &coord, float value);
        void Insert(const std::array<int, 3> &coord, float value, const Eigen::Vector3f &gradient);

        bool Sample(const std::array<int, 3> &coord, float &outValue) const;
        bool Gradient(const std::array<int, 3> &coord, Eigen::Vector3f &outNormal) const;
        bool HasGradients() const { return m_hasGradients; }

        float CellSize() const { return m_cellSize; }
        void SetCellSize(float cellSize) { m_cellSize = cellSize; }

        const std::vector<std::array<int, 3>> &OccupiedCoords() const { return m_occupied; }

    private:
        struct Cell {
            float value = 0.0f;
            Eigen::Vector3f gradient = Eigen::Vector3f::Zero();
            bool hasGradient = false;
        };

        std::unordered_map<std::array<int, 3>, Cell, core::IVec3Hash> m_cells;
        std::vector<std::array<int, 3>> m_occupied;
        float m_cellSize = 1.0f;
        bool m_hasGradients = false;
    };

    // Samples `valueFunction` (and, if provided, `gradientFunction`) at every integer coord in
    // the inclusive box [minCoord, maxCoord], and inserts a coord into the returned field iff
    // its value lies within a narrow band `|value| <= 2*cellSize` of the isosurface -- wide
    // enough to fully bracket the surface (so the eight corners of every cube straddling the
    // zero-crossing are resolvable) while staying sparse.
    VoxelField FromImplicit(const std::array<int, 3> &minCoord, const std::array<int, 3> &maxCoord, float cellSize,
                             const std::function<float(const Eigen::Vector3f &)> &valueFunction,
                             const std::function<Eigen::Vector3f(const Eigen::Vector3f &)> *gradientFunction = nullptr);

    // Adapts a TSDF's downloaded (voxel,direction) entries into a VoxelField: coord recovered
    // from the entry's world-space centre (lround(center/cellSize - 0.5) per axis, matching
    // AdaptiveVoxelGrid's latticeCoordOf convention), value = tsdf, gradient = normal.
    VoxelField FromAdvancedEntries(const std::vector<Engine::Spatial::AdvancedEntry> &entries, float cellSize);

} // namespace Engine::Spatial::Extraction
