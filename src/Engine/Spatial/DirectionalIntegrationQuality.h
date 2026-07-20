#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace Engine::Spatial {

    struct IntegrationQuality {
        uint32_t maxDirections = 1;   // K; 1 = single dominant (current behavior)
        uint32_t dirExponent = 4;     // p; integer, applied by repeated multiply
        bool viewAngleWeight = false; // multiply weight by max(0, dot(n, viewDir))
    };

    struct DirWeight { uint8_t direction; float relWeight; };

    inline float ipow(float x, uint32_t p) { float r = 1.0f; for (uint32_t i = 0; i < p; ++i) r *= x; return r; }

    // Sign-matched canonical dir per axis: +X=0,-X=1,+Y=2,-Y=3,+Z=4,-Z=5.
    // r_d = |n_axis|^p. Keep dirs with r_d/r_max >= 0.05, sorted desc, tie order x,y,z,
    // up to q.maxDirections. relWeight = r_d/r_max. Returns count (>=1).
    // maxDirections==1 reduces to exactly dominantAxisOf(n) with relWeight 1.
    inline int TopKDirections(const Eigen::Vector3f &n, const IntegrationQuality &q, DirWeight out[6]) {
        const float ax = std::abs(n.x()), ay = std::abs(n.y()), az = std::abs(n.z());
        const uint8_t dx = n.x() >= 0.0f ? 0u : 1u;
        const uint8_t dy = n.y() >= 0.0f ? 2u : 3u;
        const uint8_t dz = n.z() >= 0.0f ? 4u : 5u;
        // candidates in tie order x,y,z, sorted by |component| desc via 3 comparisons
        DirWeight c[3] = {{dx, ax}, {dy, ay}, {dz, az}};
        // stable insertion sort desc by relWeight-field-as-|component| (only 3, keeps x>y>z on ties)
        for (int i = 1; i < 3; ++i) {
            DirWeight key = c[i];
            int j = i - 1;
            while (j >= 0 && c[j].relWeight < key.relWeight) { c[j + 1] = c[j]; --j; }
            c[j + 1] = key;
        }
        const float rmax = ipow(c[0].relWeight, q.dirExponent);
        int cnt = 0;
        const uint32_t K = q.maxDirections < 1 ? 1u : q.maxDirections;
        for (int i = 0; i < 3 && uint32_t(cnt) < K; ++i) {
            const float rd = ipow(c[i].relWeight, q.dirExponent);
            const float rel = rmax > 0.0f ? rd / rmax : 0.0f;
            if (i == 0) { out[cnt++] = {c[0].direction, 1.0f}; continue; }
            if (rel >= 0.05f) out[cnt++] = {c[i].direction, rel};
        }
        return cnt;
    }

} // namespace Engine::Spatial
