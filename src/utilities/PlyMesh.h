#pragma once

// Minimal triangle-mesh PLY reader (ASCII or binary_little_endian) + per-vertex normals.
// Extracts vertex x/y/z and face vertex_indices (polygons fan-triangulated); every other
// element/property (e.g. Artec multi_texture_*) is parsed only to be skipped. Host is assumed
// little-endian (true on macOS / x86 / arm64).

#include <Eigen/Core>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace util {

    struct TriMesh {
        std::vector<Eigen::Vector3f> vertices;
        std::vector<Eigen::Vector3i> faces;
    };

    namespace ply_detail {
        inline int typeSize(const std::string &t) {
            if (t == "char" || t == "uchar" || t == "int8" || t == "uint8") return 1;
            if (t == "short" || t == "ushort" || t == "int16" || t == "uint16") return 2;
            if (t == "int" || t == "uint" || t == "int32" || t == "uint32" ||
                t == "float" || t == "float32")
                return 4;
            if (t == "double" || t == "float64") return 8;
            return 0;
        }

        struct Prop {
            std::string name, type, listCountType, listItemType;
            bool isList = false;
        };
        struct Elem {
            std::string name;
            size_t count = 0;
            std::vector<Prop> props;
        };

        // Read one scalar of PLY type `t` from a little-endian byte cursor; returns it as double.
        inline double readScalar(const char *&cur, const char *end, const std::string &t) {
            const int n = typeSize(t);
            if (cur + n > end) throw std::runtime_error("LoadPlyMesh: unexpected end of data");
            double out = 0.0;
            if (t == "float" || t == "float32") { float v; std::memcpy(&v, cur, 4); out = v; }
            else if (t == "double" || t == "float64") { double v; std::memcpy(&v, cur, 8); out = v; }
            else if (t == "uchar" || t == "uint8" || t == "char" || t == "int8") { out = double(uint8_t(*cur)); }
            else if (t == "short" || t == "int16") { int16_t v; std::memcpy(&v, cur, 2); out = v; }
            else if (t == "ushort" || t == "uint16") { uint16_t v; std::memcpy(&v, cur, 2); out = v; }
            else if (t == "int" || t == "int32") { int32_t v; std::memcpy(&v, cur, 4); out = v; }
            else if (t == "uint" || t == "uint32") { uint32_t v; std::memcpy(&v, cur, 4); out = v; }
            cur += n;
            return out;
        }
    } // namespace ply_detail

    inline void AddPolygon(TriMesh &m, const std::vector<int> &idx) {
        for (size_t k = 1; k + 1 < idx.size(); ++k)
            m.faces.emplace_back(idx[0], idx[k], idx[k + 1]); // fan triangulation
    }

    inline void LoadPlyMesh(const std::string &path, TriMesh &mesh) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("LoadPlyMesh: cannot open " + path);

        // ── header ──
        std::string line;
        bool ascii = false, binaryLE = false;
        std::vector<ply_detail::Elem> elems;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream ss(line);
            std::string tok;
            ss >> tok;
            if (tok == "format") {
                std::string fmt;
                ss >> fmt;
                ascii = (fmt == "ascii");
                binaryLE = (fmt == "binary_little_endian");
            } else if (tok == "element") {
                ply_detail::Elem e;
                ss >> e.name >> e.count;
                elems.push_back(e);
            } else if (tok == "property") {
                ply_detail::Prop p;
                std::string t;
                ss >> t;
                if (t == "list") { p.isList = true; ss >> p.listCountType >> p.listItemType >> p.name; }
                else { p.type = t; ss >> p.name; }
                if (!elems.empty()) elems.back().props.push_back(p);
            } else if (tok == "end_header") {
                break;
            }
        }
        if (!ascii && !binaryLE)
            throw std::runtime_error("LoadPlyMesh: unsupported PLY format (need ascii or binary_little_endian): " + path);

        mesh.vertices.clear();
        mesh.faces.clear();

        if (ascii) {
            for (const auto &e : elems) {
                const bool isVertex = e.name == "vertex", isFace = e.name == "face";
                for (size_t i = 0; i < e.count && std::getline(f, line); ++i) {
                    std::istringstream ss(line);
                    if (isVertex) {
                        float x, y, z;
                        ss >> x >> y >> z; // x/y/z are the first three properties
                        mesh.vertices.emplace_back(x, y, z);
                    } else if (isFace) {
                        int cnt = 0;
                        ss >> cnt;
                        std::vector<int> idx(std::max(0, cnt));
                        for (int &v : idx) ss >> v;
                        AddPolygon(mesh, idx);
                    }
                }
            }
            return;
        }

        // ── binary_little_endian: slurp the rest and parse from a cursor ──
        std::string blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const char *cur = blob.data();
        const char *end = cur + blob.size();
        for (const auto &e : elems) {
            const bool isVertex = e.name == "vertex", isFace = e.name == "face";
            if (isVertex) mesh.vertices.reserve(e.count);
            for (size_t i = 0; i < e.count; ++i) {
                Eigen::Vector3f v(0, 0, 0);
                for (const auto &p : e.props) {
                    if (p.isList) {
                        const long long cnt = (long long) ply_detail::readScalar(cur, end, p.listCountType);
                        if (isFace && p.name == "vertex_indices") {
                            std::vector<int> idx(size_t(std::max<long long>(0, cnt)));
                            for (auto &vi : idx) vi = int(ply_detail::readScalar(cur, end, p.listItemType));
                            AddPolygon(mesh, idx);
                        } else {
                            const long long skip = cnt * ply_detail::typeSize(p.listItemType);
                            if (cur + skip > end) throw std::runtime_error("LoadPlyMesh: truncated list");
                            cur += skip;
                        }
                    } else {
                        const double val = ply_detail::readScalar(cur, end, p.type);
                        if (isVertex) {
                            if (p.name == "x") v.x() = float(val);
                            else if (p.name == "y") v.y() = float(val);
                            else if (p.name == "z") v.z() = float(val);
                        }
                    }
                }
                if (isVertex) mesh.vertices.push_back(v);
            }
        }
    }

    // Area-weighted per-vertex normals (accumulate un-normalised face normals, then normalise).
    inline std::vector<Eigen::Vector3f> ComputeVertexNormals(const TriMesh &m) {
        std::vector<Eigen::Vector3f> n(m.vertices.size(), Eigen::Vector3f::Zero());
        const int nv = int(m.vertices.size());
        for (const Eigen::Vector3i &f : m.faces) {
            if (f[0] < 0 || f[1] < 0 || f[2] < 0 || f[0] >= nv || f[1] >= nv || f[2] >= nv) continue;
            const Eigen::Vector3f fn = (m.vertices[f[1]] - m.vertices[f[0]])
                                               .cross(m.vertices[f[2]] - m.vertices[f[0]]);
            n[f[0]] += fn;
            n[f[1]] += fn;
            n[f[2]] += fn;
        }
        for (Eigen::Vector3f &v : n) {
            const float len = v.norm();
            v = len > 1e-12f ? Eigen::Vector3f(v / len) : Eigen::Vector3f(0, 0, 1);
        }
        return n;
    }

} // namespace util
