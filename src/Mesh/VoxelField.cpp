#include "Mesh/VoxelField.h"
#include "TSDF/Backends/AdvancedTSDF.h" // AdvancedEntry (full definition, forward-declared in VoxelField.h)

#include <cmath>

namespace Mesh {

    void VoxelField::Insert(const std::array<int, 3> &coord, float value) {
        const auto it = m_cells.find(coord);
        if (it == m_cells.end()) {
            m_cells.emplace(coord, Cell{value, Eigen::Vector3f::Zero(), false});
            m_occupied.push_back(coord);
        } else {
            it->second.value = value;
        }
    }

    void VoxelField::Insert(const std::array<int, 3> &coord, float value, const Eigen::Vector3f &gradient) {
        const auto it = m_cells.find(coord);
        if (it == m_cells.end()) {
            m_cells.emplace(coord, Cell{value, gradient, true});
            m_occupied.push_back(coord);
        } else {
            it->second.value = value;
            it->second.gradient = gradient;
            it->second.hasGradient = true;
        }
        m_hasGradients = true;
    }

    bool VoxelField::Sample(const std::array<int, 3> &coord, float &outValue) const {
        const auto it = m_cells.find(coord);
        if (it == m_cells.end()) return false;
        outValue = it->second.value;
        return true;
    }

    bool VoxelField::Gradient(const std::array<int, 3> &coord, Eigen::Vector3f &outNormal) const {
        const auto it = m_cells.find(coord);
        if (it == m_cells.end() || !it->second.hasGradient) return false;
        outNormal = it->second.gradient;
        return true;
    }

    VoxelField FromImplicit(const std::array<int, 3> &minCoord, const std::array<int, 3> &maxCoord, float cellSize,
                             const std::function<float(const Eigen::Vector3f &)> &valueFunction,
                             const std::function<Eigen::Vector3f(const Eigen::Vector3f &)> *gradientFunction) {
        VoxelField field;
        field.SetCellSize(cellSize);
        const float band = 2.0f * cellSize;

        for (int x = minCoord[0]; x <= maxCoord[0]; ++x)
            for (int y = minCoord[1]; y <= maxCoord[1]; ++y)
                for (int z = minCoord[2]; z <= maxCoord[2]; ++z) {
                    const std::array<int, 3> coord{x, y, z};
                    const Eigen::Vector3f p(float(x) * cellSize, float(y) * cellSize, float(z) * cellSize);
                    const float value = valueFunction(p);
                    if (std::abs(value) > band) continue;
                    if (gradientFunction)
                        field.Insert(coord, value, (*gradientFunction)(p));
                    else
                        field.Insert(coord, value);
                }
        return field;
    }

    VoxelField FromAdvancedEntries(const std::vector<TSDF::AdvancedEntry> &entries, float cellSize) {
        VoxelField field;
        field.SetCellSize(cellSize);
        for (const auto &entry : entries) {
            const std::array<int, 3> coord{
                    static_cast<int>(std::lround(entry.center.x() / cellSize - 0.5f)),
                    static_cast<int>(std::lround(entry.center.y() / cellSize - 0.5f)),
                    static_cast<int>(std::lround(entry.center.z() / cellSize - 0.5f))};
            field.Insert(coord, entry.tsdf, entry.normal);
        }
        return field;
    }

    VoxelField FromVoxels(const std::vector<TSDFVoxel> &voxels, float cellSize) {
        VoxelField field;
        field.SetCellSize(cellSize);
        for (const TSDFVoxel &voxel : voxels) {
            const std::array<int, 3> coord{
                    static_cast<int>(std::lround(voxel.center.x() / cellSize - 0.5f)),
                    static_cast<int>(std::lround(voxel.center.y() / cellSize - 0.5f)),
                    static_cast<int>(std::lround(voxel.center.z() / cellSize - 0.5f))};
            field.Insert(coord, voxel.tsdf, voxel.normal);
        }
        return field;
    }

} // namespace Mesh
