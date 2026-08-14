#pragma once

#include <Eigen/Core>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace Engine::Spatial {

    // Architecture constants from the DirectionalTSDF design doc (§3).
    constexpr uint32_t kGroupDim = 8;                                       // 8x8x8 voxels per group
    constexpr uint32_t kVoxelsPerGroup = kGroupDim * kGroupDim * kGroupDim; // 512
    constexpr uint32_t kLocalGroupGrid = 50;                                // 50x50x50 groups per local window
    constexpr uint32_t kNumDirections = 6;                                  // +X,-X,+Y,-Y,+Z,-Z
    constexpr uint32_t kIndexGridCells =
            kLocalGroupGrid * kLocalGroupGrid * kLocalGroupGrid * kNumDirections; // 750,000
    constexpr uint32_t kInvalidPoolIndex = 0xFFFFFFFFu;
    constexpr int32_t kTsdfFixedScale = 10000; // same fixed-point scale as SimpleTSDF

    enum class SlotState : uint8_t {
        Free = 0,
        ResidentClean = 1,
        ResidentDirty = 2,
        PendingUpload = 3,
        PendingWriteBack = 4,
    };

    struct DirectionalGroupKey {
        int32_t gx = 0;
        int32_t gy = 0;
        int32_t gz = 0;
        uint8_t direction = 0; // 0..5 = +X,-X,+Y,-Y,+Z,-Z

        bool operator==(const DirectionalGroupKey &o) const {
            return gx == o.gx && gy == o.gy && gz == o.gz && direction == o.direction;
        }
        bool operator!=(const DirectionalGroupKey &o) const { return !(*this == o); }
    };

    struct DirectionalGroupKeyHash {
        size_t operator()(const DirectionalGroupKey &k) const {
            uint64_t h = uint64_t(uint32_t(k.gx)) * 73856093ull;
            h ^= uint64_t(uint32_t(k.gy)) * 19349663ull;
            h ^= uint64_t(uint32_t(k.gz)) * 83492791ull;
            h ^= uint64_t(k.direction) * 2654435761ull;
            h ^= h >> 33;
            return size_t(h);
        }
    };

    // Host store / wire format: running-average form (value = avg SDF, weight = total
    // weight) plus the finalized unit surface normal (v1: 3×f32; oct compression deferred).
    struct HostTsdfVoxel {
        float value = 0.0f;
        float weight = 0.0f;
        float nx = 0.0f;
        float ny = 0.0f;
        float nz = 0.0f;
    }; // 20B

    // GPU active-pool format: fixed-point accumulators so the integrate kernel can atomicAdd
    // (GLSL has no float atomics). Conversion: sumW = weight*scale, sumDW = value*weight*scale.
    struct GpuTsdfVoxel {
        int32_t sumDW = 0;
        uint32_t sumW = 0;
        int32_t sumNx = 0; // Σ n·w·TSDF_SCALE per direction layer (finalized to HostTsdfVoxel normal on write-back, rehydrated on upload)
        int32_t sumNy = 0;
        int32_t sumNz = 0;
    }; // 20B

    // GLSL std430 has no 8-bit members, so direction/state/dirty/valid live in one packed uint.
    struct ActiveGroupMeta {
        int32_t gx = 0;
        int32_t gy = 0;
        int32_t gz = 0;
        uint32_t packed = 0; // bits 0-7 direction, 8-15 state, 16-23 dirty, 24-31 valid
    }; // 16B

    constexpr uint32_t PackMeta(uint8_t direction, SlotState state, bool dirty, bool valid) {
        return uint32_t(direction) | (uint32_t(state) << 8) |
               (uint32_t(dirty ? 1 : 0) << 16) | (uint32_t(valid ? 1 : 0) << 24);
    }
    constexpr uint8_t MetaDirection(uint32_t packed) { return uint8_t(packed & 0xFFu); }
    constexpr SlotState MetaState(uint32_t packed) { return SlotState((packed >> 8) & 0xFFu); }
    constexpr bool MetaDirty(uint32_t packed) { return ((packed >> 16) & 0xFFu) != 0; }
    constexpr bool MetaValid(uint32_t packed) { return ((packed >> 24) & 0xFFu) != 0; }

    // CPU mirror of the shader-side indexGrid addressing. lx/ly/lz are local group coords
    // (global - localBase), all in [0, kLocalGroupGrid).
    constexpr uint32_t IndexGridOffset(uint32_t lx, uint32_t ly, uint32_t lz, uint32_t direction) {
        return ((lz * kLocalGroupGrid + ly) * kLocalGroupGrid + lx) * kNumDirections + direction;
    }

    // Extraction candidate produced on the GPU (Phase 3); declared here so the GPU layout
    // is asserted alongside the other shared structs.
    struct DirectionalCandidate {
        float px = 0, py = 0, pz = 0;
        float nx = 0, ny = 0, nz = 0;
        int32_t gx = 0, gy = 0, gz = 0;
        uint32_t direction = 0;
    }; // 40B

    // CPU-side merged output point (Phase 3).
    struct ExtractedPoint {
        Eigen::Vector3f position = Eigen::Vector3f::Zero();
        Eigen::Vector3f normal = Eigen::Vector3f::Zero();
        int32_t ownerGx = 0, ownerGy = 0, ownerGz = 0;
        uint8_t dirMask = 0;
    };

    static_assert(std::is_standard_layout_v<HostTsdfVoxel>);
    static_assert(sizeof(HostTsdfVoxel) == 20);
    static_assert(offsetof(HostTsdfVoxel, nx) == 8);
    static_assert(std::is_standard_layout_v<GpuTsdfVoxel>);
    static_assert(sizeof(GpuTsdfVoxel) == 20);
    static_assert(offsetof(GpuTsdfVoxel, sumW) == 4);
    static_assert(offsetof(GpuTsdfVoxel, sumNx) == 8);
    static_assert(offsetof(GpuTsdfVoxel, sumNz) == 16);
    static_assert(std::is_standard_layout_v<ActiveGroupMeta>);
    static_assert(sizeof(ActiveGroupMeta) == 16);
    static_assert(offsetof(ActiveGroupMeta, packed) == 12);
    static_assert(std::is_standard_layout_v<DirectionalCandidate>);
    static_assert(sizeof(DirectionalCandidate) == 40);

} // namespace Engine::Spatial
