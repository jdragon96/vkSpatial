#pragma once

// Triangle-mesh PLY reader: vertex x/y/z and face vertex_indices (polygons fan-triangulated).
// A thin adapter over util::PlyFormat, which does the parsing for ASCII and binary_little_endian
// alike and skips every element/property this does not model (e.g. Artec multi_texture_*).
//
// Kept as its own entry point because a mesh is vertices + faces, while PointCloudIO deals in
// points + normals -- the two callers want different halves of the same file.

#include "utilities/PlyFormat.h"

#include <Eigen/Core>
#include <Eigen/Geometry> // Vector3f::cross

#include <stdexcept>
#include <string>
#include <vector>

namespace util {

    struct TriMesh {
        std::vector<Eigen::Vector3f> vertices;
        std::vector<Eigen::Vector3i> faces;
    };

    // Throws std::runtime_error rather than returning a flag: every caller treats a missing or
    // malformed mesh as fatal, and a half-filled TriMesh is worse than no mesh.
    inline void LoadPlyMesh(const std::string &path, TriMesh &mesh) {
        PlyFormat ply;
        if (!ply.Deserialize(path))
            throw std::runtime_error("LoadPlyMesh: cannot read " + path);

        const std::vector<float> &flatVertices = ply.GetPoints();
        const std::size_t vertexCount = ply.GetPointCount();
        mesh.vertices.clear();
        mesh.vertices.reserve(vertexCount);
        for (std::size_t i = 0; i < vertexCount; ++i)
            mesh.vertices.emplace_back(flatVertices[3 * i], flatVertices[3 * i + 1],
                                       flatVertices[3 * i + 2]);

        const std::vector<unsigned int> &corners = ply.GetTriangleIndices();
        mesh.faces.clear();
        mesh.faces.reserve(corners.size() / 3);
        for (std::size_t i = 0; i + 2 < corners.size(); i += 3)
            mesh.faces.emplace_back(int(corners[i]), int(corners[i + 1]), int(corners[i + 2]));

        if (mesh.vertices.empty()) throw std::runtime_error("LoadPlyMesh: no vertices in " + path);
    }

    // Area-weighted vertex normals: each face's un-normalised cross product is proportional to
    // twice its area, so summing them weights large faces more before the final normalise.
    inline std::vector<Eigen::Vector3f> ComputeVertexNormals(const TriMesh &m) {
        std::vector<Eigen::Vector3f> n(m.vertices.size(), Eigen::Vector3f::Zero());
        const int nv = int(m.vertices.size());
        for (const Eigen::Vector3i &f: m.faces) {
            if (f[0] < 0 || f[1] < 0 || f[2] < 0 || f[0] >= nv || f[1] >= nv || f[2] >= nv) continue;
            const Eigen::Vector3f fn = (m.vertices[f[1]] - m.vertices[f[0]])
                                               .cross(m.vertices[f[2]] - m.vertices[f[0]]);
            n[f[0]] += fn;
            n[f[1]] += fn;
            n[f[2]] += fn;
        }
        for (Eigen::Vector3f &v: n) {
            const float len = v.norm();
            v = len > 1e-12f ? Eigen::Vector3f(v / len) : Eigen::Vector3f(0, 0, 1);
        }
        return n;
    }

} // namespace util
