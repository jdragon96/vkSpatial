#include "Common/PointCloud.h"

#include "utilities/PlyFormat.h"

// Out of line on purpose: PlyFormat pulls in <fstream>, <sstream> and <unordered_map>, and
// PointCloud.h is included by ~20 translation units that want none of them.

namespace Common {

    bool PointCloud::Save(const std::string &path, EPointCloudFormat format) const {
        util::PlyFormat ply;
        ply.SetDataType(format == EPointCloudFormat::Ascii ? util::PlyFormat::EDataType::Ascii
                                                           : util::PlyFormat::EDataType::Binary);

        // A cloud whose normals do not pair one-to-one with its points has no normals as far as
        // the file is concerned; writing the shorter array would silently mis-pair every point
        // after the first gap.
        const bool writeNormals = normals.size() == points.size();
        for (std::size_t i = 0; i < points.size(); ++i) {
            ply.AddPoint(points[i].x(), points[i].y(), points[i].z());
            if (writeNormals) ply.AddNormal(normals[i].x(), normals[i].y(), normals[i].z());
        }

        return ply.Serialize(path);
    }

    bool PointCloud::Load(const std::string &path) {
        util::PlyFormat ply;
        if (!ply.Deserialize(path)) return false;

        const std::vector<float> &flatPoints = ply.GetPoints();
        const std::vector<float> &flatNormals = ply.GetNormals();
        const std::size_t pointCount = ply.GetPointCount();

        // Replace, never append: a Load onto a non-empty cloud otherwise concatenates two scans
        // into one, which reads as a plausible cloud and is wrong everywhere.
        points.clear();
        normals.clear();
        points.reserve(pointCount);

        for (std::size_t i = 0; i < pointCount; ++i)
            points.emplace_back(flatPoints[3 * i], flatPoints[3 * i + 1], flatPoints[3 * i + 2]);

        if (flatNormals.size() == flatPoints.size()) {
            normals.reserve(pointCount);
            for (std::size_t i = 0; i < pointCount; ++i)
                normals.emplace_back(flatNormals[3 * i], flatNormals[3 * i + 1],
                                     flatNormals[3 * i + 2]);
        }

        return true;
    }

} // namespace Common
