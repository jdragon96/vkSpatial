#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"
#include "Registration/RegistrationTypes.h"

#include <Eigen/Dense>
#include <cstdint>
#include <memory>
#include <vector>

namespace Pipeline {

    class LocalGrid {
    public:
        LocalGrid(const std::vector<Eigen::Vector3f> &points, float cell);

        int Nearest(const Eigen::Vector3f &query, float radius) const;

        Eigen::Vector3f m_origin = Eigen::Vector3f::Zero(); // AABB min (zeroed: left indeterminate on
                                                            // empty input otherwise, and this is a
                                                            // public GPU-upload member)
        Eigen::Vector3i m_dims{1, 1, 1};                    // cells per axis
        float m_cell = 1.0f;
        std::vector<uint32_t> m_bucketStart;       // size dims.prod()+1 (prefix sum)
        std::vector<uint32_t> m_bucketIdx;         // size pts.size() (pt indices grouped by cell)
        const std::vector<Eigen::Vector3f> &m_pts; // caller-owned; must outlive this LocalGrid

    private:
        int cellIndex(const Eigen::Vector3i &cellCoordinate) const {
            return (cellCoordinate.z() * m_dims.y() + cellCoordinate.y()) * m_dims.x() + cellCoordinate.x();
        }
        Eigen::Vector3i cellOf(const Eigen::Vector3f &point) const {
            return Eigen::Vector3i(int(std::floor((point.x() - m_origin.x()) / m_cell)),
                                   int(std::floor((point.y() - m_origin.y()) / m_cell)),
                                   int(std::floor((point.z() - m_origin.z()) / m_cell)));
        }
    };


} // namespace Pipeline