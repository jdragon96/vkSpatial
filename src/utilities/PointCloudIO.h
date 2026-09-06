#pragma once

#include "utilities/PlyFormat.h" // the one PLY implementation; these are thin adapters over it

#include <Eigen/Core>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// Oriented-point-cloud I/O (points + optional per-point normals) for PLY and OBJ, shared by the
// example tools (voxel_fill_debugger, tsdf_folder_eval, ...). Meshes (vertices + faces) live in
// utilities/PlyMesh.h instead — this file is points only.
namespace util {

    inline bool EndsWithLower(const std::string &s, const std::string &suffix) {
        if (s.size() < suffix.size()) return false;
        std::string tail = s.substr(s.size() - suffix.size());
        std::transform(tail.begin(), tail.end(), tail.begin(),
                       [](unsigned char c) { return char(std::tolower(c)); });
        return tail == suffix;
    }

    // ---- PLY (adapters over util::PlyFormat) ----

    // Appends to `pts` / `nrm` so callers can accumulate several files, which is why this is not
    // simply Common::PointCloud::Load (that one replaces). Reads ASCII and binary_little_endian
    // alike -- the hand-rolled reader this replaced returned false on anything but ASCII.
    inline bool LoadPly(const std::string &path, std::vector<Eigen::Vector3f> &pts,
                        std::vector<Eigen::Vector3f> &nrm) {
        PlyFormat ply;
        if (!ply.Deserialize(path)) return false;

        const std::vector<float> &flatPoints = ply.GetPoints();
        const std::vector<float> &flatNormals = ply.GetNormals();
        const std::size_t count = ply.GetPointCount();
        if (count == 0) return false;

        pts.reserve(pts.size() + count);
        for (std::size_t i = 0; i < count; ++i)
            pts.emplace_back(flatPoints[3 * i], flatPoints[3 * i + 1], flatPoints[3 * i + 2]);

        if (flatNormals.size() == flatPoints.size()) {
            nrm.reserve(nrm.size() + count);
            for (std::size_t i = 0; i < count; ++i)
                nrm.emplace_back(flatNormals[3 * i], flatNormals[3 * i + 1], flatNormals[3 * i + 2]);
        }
        return true;
    }

    // ASCII, deliberately: this is what the repo's PLY assets are written as, and what a human
    // opens to check a scan. Binary output goes through Common::PointCloud::Save.
    inline bool SavePly(const std::string &path, const std::vector<Eigen::Vector3f> &pts,
                        const std::vector<Eigen::Vector3f> &nrm) {
        PlyFormat ply;
        ply.SetDataType(PlyFormat::EDataType::Ascii);
        const bool hasNormals = !pts.empty() && nrm.size() == pts.size();
        for (std::size_t i = 0; i < pts.size(); ++i) {
            ply.AddPoint(pts[i].x(), pts[i].y(), pts[i].z());
            if (hasNormals) ply.AddNormal(nrm[i].x(), nrm[i].y(), nrm[i].z());
        }
        return ply.Serialize(path);
    }

    // ---- OBJ (as a point cloud: v -> points, vn -> normals; faces ignored) ----

    inline bool LoadObj(const std::string &path, std::vector<Eigen::Vector3f> &pts,
                        std::vector<Eigen::Vector3f> &nrm) {
        std::ifstream f(path);
        if (!f) return false;
        std::string line;
        std::vector<Eigen::Vector3f> vn;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            std::string tok;
            ss >> tok;
            if (tok == "v") {
                float x, y, z;
                ss >> x >> y >> z;
                pts.emplace_back(x, y, z);
            } else if (tok == "vn") {
                float x, y, z;
                ss >> x >> y >> z;
                vn.emplace_back(x, y, z);
            }
        }
        if (vn.size() == pts.size()) nrm.insert(nrm.end(), vn.begin(), vn.end()); // 1:1 pairing
        return !pts.empty();
    }

    inline bool SaveObj(const std::string &path, const std::vector<Eigen::Vector3f> &pts,
                        const std::vector<Eigen::Vector3f> &nrm) {
        std::ofstream f(path);
        if (!f) return false;
        for (const Eigen::Vector3f &p : pts) f << "v " << p.x() << ' ' << p.y() << ' ' << p.z() << '\n';
        if (nrm.size() == pts.size())
            for (const Eigen::Vector3f &n : nrm) f << "vn " << n.x() << ' ' << n.y() << ' ' << n.z() << '\n';
        return bool(f);
    }

    // ---- extension dispatch (.obj -> OBJ, else PLY) ----

    inline bool LoadPointCloud(const std::string &path, std::vector<Eigen::Vector3f> &pts,
                               std::vector<Eigen::Vector3f> &nrm) {
        return EndsWithLower(path, ".obj") ? LoadObj(path, pts, nrm) : LoadPly(path, pts, nrm);
    }
    inline bool SavePointCloud(const std::string &path, const std::vector<Eigen::Vector3f> &pts,
                               const std::vector<Eigen::Vector3f> &nrm) {
        return EndsWithLower(path, ".obj") ? SaveObj(path, pts, nrm) : SavePly(path, pts, nrm);
    }

} // namespace util
