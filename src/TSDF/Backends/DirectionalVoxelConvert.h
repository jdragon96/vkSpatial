#pragma once

#include "TSDF/Backends/DirectionalTSDFTypes.h"

#include <cmath>

namespace Engine::Spatial {

    // Single source of truth for Host<->Gpu TSDF voxel conversion, reused by both residency
    // backends and every debug download. Host stores running-average (value,weight)+unit
    // normal; Gpu stores fixed-point accumulators (sumDW,sumW,sumN) the integrate kernel
    // atomicAdds into. sumN = n * weight * scale so normalize(sumN) == n and continued
    // per-observation accumulation blends by weight.
    inline GpuTsdfVoxel HostVoxelToGpu(const HostTsdfVoxel &h) {
        GpuTsdfVoxel g;
        g.sumW = uint32_t(std::lround(double(h.weight) * kTsdfFixedScale));
        g.sumDW = int32_t(std::lround(double(h.value) * double(h.weight) * kTsdfFixedScale));
        g.sumNx = int32_t(std::lround(double(h.nx) * double(h.weight) * kTsdfFixedScale));
        g.sumNy = int32_t(std::lround(double(h.ny) * double(h.weight) * kTsdfFixedScale));
        g.sumNz = int32_t(std::lround(double(h.nz) * double(h.weight) * kTsdfFixedScale));
        return g;
    }

    inline HostTsdfVoxel GpuVoxelToHost(const GpuTsdfVoxel &g) {
        HostTsdfVoxel h;
        h.weight = float(g.sumW) / float(kTsdfFixedScale);
        h.value = g.sumW > 0 ? float(double(g.sumDW) / double(g.sumW)) : 0.0f;
        const double lx = g.sumNx, ly = g.sumNy, lz = g.sumNz;
        const double len = std::sqrt(lx * lx + ly * ly + lz * lz);
        if (len > 1e-6) {
            h.nx = float(lx / len);
            h.ny = float(ly / len);
            h.nz = float(lz / len);
        } else {
            h.nx = h.ny = h.nz = 0.0f;
        }
        return h;
    }

} // namespace Engine::Spatial
