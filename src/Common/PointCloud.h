#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace Common {

    // How Save writes the file. Load does not take one -- it reads whichever the file declares.
    //
    // Spelled here rather than reusing util::PlyFormat::EDataType so this header keeps depending on
    // nothing but Eigen: it is included by ~20 translation units, several of them hot, and pulling
    // <fstream> and <unordered_map> into all of them to name two enumerators is a bad trade.
    enum class EPointCloudFormat {
        Binary, // lossless and compact; what Save uses unless told otherwise
        Ascii,  // inspectable by hand, and the only thing util::LoadPly reads
    };

    struct PointCloud {
        std::vector<Eigen::Vector3f> points, normals;

        // PLY, through util::PlyFormat. Normals are written only when there is one per point, and
        // read back only when the file carries them. False on a file that cannot be opened, is not
        // a PLY, or ends early -- never a partially-filled cloud.
        //
        // Load REPLACES the current contents rather than appending, unlike util::LoadPly.
        bool Save(const std::string &path,
                  EPointCloudFormat format = EPointCloudFormat::Binary) const;
        bool Load(const std::string &path);

        // Canonical(좌표 기준) Order: X -> Y -> Z 순서로 비교하여 정렬
        inline void SortTargetIntoCanonicalOrder() {
            const std::size_t n = points.size();
            if (normals.size() != n) return;

            struct PointWithNormal {
                Eigen::Vector3f point, normal;
            };
            std::vector<PointWithNormal> records(n);
            for (std::size_t i = 0; i < n; ++i) records[i] = {points[i], normals[i]};
            std::sort(records.begin(), records.end(), [](const PointWithNormal &a, const PointWithNormal &b) {
                if (a.point.x() != b.point.x()) return a.point.x() < b.point.x();
                if (a.point.y() != b.point.y()) return a.point.y() < b.point.y();
                return a.point.z() < b.point.z();
            });
            for (std::size_t i = 0; i < n; ++i) {
                points[i] = records[i].point;
                normals[i] = records[i].normal;
            }
        }
    };

} // namespace Common
