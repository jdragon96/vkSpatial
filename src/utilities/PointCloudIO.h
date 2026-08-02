#pragma once

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

    // ---- PLY ----

    // Tolerant ASCII-PLY reader: x y z always; nx ny nz if the header declares them (appends to the
    // vectors so callers can accumulate; returns false on open error, non-ascii, or no points).
    inline bool LoadPly(const std::string &path, std::vector<Eigen::Vector3f> &pts,
                        std::vector<Eigen::Vector3f> &nrm) {
        std::ifstream f(path);
        if (!f) return false;
        std::string line;
        std::size_t count = 0;
        bool ascii = false, hasN = false;
        std::vector<std::string> props;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            std::string tok;
            ss >> tok;
            if (tok == "format") {
                std::string fmt;
                ss >> fmt;
                ascii = (fmt == "ascii");
            } else if (tok == "element") {
                std::string e;
                ss >> e;
                if (e == "vertex") ss >> count;
            } else if (tok == "property") {
                std::string t, name;
                ss >> t >> name;
                props.push_back(name);
            } else if (tok == "end_header")
                break;
        }
        if (!ascii) return false;
        hasN = std::find(props.begin(), props.end(), "nx") != props.end();
        const std::size_t stride = props.size();
        pts.reserve(pts.size() + count);
        for (std::size_t i = 0; i < count && std::getline(f, line); ++i) {
            std::istringstream ss(line);
            std::vector<float> vals(stride, 0.0f);
            for (std::size_t j = 0; j < stride; ++j) ss >> vals[j];
            pts.emplace_back(vals[0], vals[1], vals[2]);
            if (hasN && stride >= 6) nrm.emplace_back(vals[3], vals[4], vals[5]);
        }
        return !pts.empty();
    }

    // ASCII-PLY writer: x y z (+ nx ny nz when a normal per point is provided).
    inline bool SavePly(const std::string &path, const std::vector<Eigen::Vector3f> &pts,
                        const std::vector<Eigen::Vector3f> &nrm) {
        std::ofstream f(path);
        if (!f) return false;
        const bool hasN = !pts.empty() && nrm.size() == pts.size();
        f << "ply\nformat ascii 1.0\nelement vertex " << pts.size()
          << "\nproperty float x\nproperty float y\nproperty float z\n";
        if (hasN) f << "property float nx\nproperty float ny\nproperty float nz\n";
        f << "end_header\n";
        for (std::size_t i = 0; i < pts.size(); ++i) {
            f << pts[i].x() << ' ' << pts[i].y() << ' ' << pts[i].z();
            if (hasN) f << ' ' << nrm[i].x() << ' ' << nrm[i].y() << ' ' << nrm[i].z();
            f << '\n';
        }
        return bool(f);
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
