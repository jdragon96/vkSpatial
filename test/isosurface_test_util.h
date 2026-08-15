#pragma once
#include "Mesh/VoxelField.h"
#include "Mesh/SurfaceMesh.h"
#include <Eigen/Core>
#include <array>
#include <cmath>
#include <map>
#include <vector>

namespace isotest {
    using Mesh::SurfaceMesh;
    using Mesh::VoxelField;

    // Analytic sphere signed field (value = |p| - radius), exact gradient = p/|p|.
    inline VoxelField SphereField(float radius, float cellSize, int halfN) {
        const std::array<int,3> lo{-halfN,-halfN,-halfN}, hi{halfN,halfN,halfN};
        auto value = [radius](const Eigen::Vector3f& p){ return p.norm() - radius; };
        auto grad  = [](const Eigen::Vector3f& p){ float n=p.norm(); return n>1e-6f?Eigen::Vector3f(p/n):Eigen::Vector3f(0,0,1);} ;
        std::function<Eigen::Vector3f(const Eigen::Vector3f&)> g = grad;
        return Mesh::FromImplicit(lo, hi, cellSize, value, &g);
    }
    // Axis-aligned box SDF centred at origin, exact gradient (sharp edges/corners).
    inline VoxelField BoxField(const Eigen::Vector3f& halfExtents, float cellSize, int halfN) {
        const std::array<int,3> lo{-halfN,-halfN,-halfN}, hi{halfN,halfN,halfN};
        auto value = [halfExtents](const Eigen::Vector3f& p){
            Eigen::Vector3f q = p.cwiseAbs() - halfExtents;
            Eigen::Vector3f qm = q.cwiseMax(0.0f);
            return qm.norm() + std::min(std::max(q.x(),std::max(q.y(),q.z())), 0.0f);
        };
        auto grad = [halfExtents](const Eigen::Vector3f& p){
            const float e = 1e-3f; Eigen::Vector3f g;
            auto f=[&](const Eigen::Vector3f& x){ Eigen::Vector3f q=x.cwiseAbs()-halfExtents; Eigen::Vector3f qm=q.cwiseMax(0.0f);
                return qm.norm()+std::min(std::max(q.x(),std::max(q.y(),q.z())),0.0f); };
            g.x()=f(p+Eigen::Vector3f(e,0,0))-f(p-Eigen::Vector3f(e,0,0));
            g.y()=f(p+Eigen::Vector3f(0,e,0))-f(p-Eigen::Vector3f(0,e,0));
            g.z()=f(p+Eigen::Vector3f(0,0,e))-f(p-Eigen::Vector3f(0,0,e));
            float n=g.norm(); return n>1e-6f?Eigen::Vector3f(g/n):Eigen::Vector3f(0,0,1);
        };
        std::function<Eigen::Vector3f(const Eigen::Vector3f&)> gg = grad;
        return Mesh::FromImplicit(lo, hi, cellSize, value, &gg);
    }

    // Undirected-edge incidence count over the mesh's triangles.
    inline std::map<std::pair<int,int>,int> EdgeCounts(const SurfaceMesh& m) {
        std::map<std::pair<int,int>,int> counts;
        auto add=[&](int a,int b){ counts[{std::min(a,b),std::max(a,b)}]++; };
        for (const auto& t : m.triangles){ add(t[0],t[1]); add(t[1],t[2]); add(t[2],t[0]); }
        return counts;
    }
    inline bool IsEdgeManifold(const SurfaceMesh& m){ for (auto& kv:EdgeCounts(m)) if (kv.second>2) return false; return true; }
    inline bool IsWatertight(const SurfaceMesh& m){ auto c=EdgeCounts(m); if(c.empty()) return false; for(auto& kv:c) if(kv.second!=2) return false; return true; }
    inline int EulerCharacteristic(const SurfaceMesh& m){ return int(m.vertices.size()) - int(EdgeCounts(m).size()) + int(m.triangles.size()); }

    // Mean alignment of per-vertex normals with the outward radial direction from `center`.
    // > 0 means normals point outward on average; a globally winding-inverted mesh gives < 0.
    inline double MeanOutwardNormalAlignment(const SurfaceMesh& mesh, const Eigen::Vector3f& center) {
        if (mesh.vertices.empty()) return 0.0;
        double sum = 0.0;
        for (size_t i = 0; i < mesh.vertices.size(); ++i) {
            Eigen::Vector3f radial = mesh.vertices[i] - center;
            double n = radial.norm();
            if (n > 1e-6) sum += double(mesh.normals[i].dot(radial / float(n)));
        }
        return sum / double(mesh.vertices.size());
    }
}
