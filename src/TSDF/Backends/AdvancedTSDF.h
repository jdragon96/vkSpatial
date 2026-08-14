#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"
#include "TSDF/Backends/DirectionalIntegrationQuality.h"
#include "Engine/Core/OrientedPointCloud.h"

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace Engine::Spatial {

    // Per-(voxel,direction) hash entry for AdvancedTSDF. 24 bytes: distance/weight
    // accumulators + a stored-gradient (observed normal) accumulator. Layout must match
    // DirEntry in advanced_tsdf_{integrate,extract}.vert.glsl exactly.
    struct AdvDirEntry {
        uint32_t key;  // packDirKey(voxel, dir); 0xFFFFFFFF = empty
        int32_t sumDW; // Σ tsdf · w · 10000
        uint32_t sumW; // Σ w · 10000
        int32_t sumNx; // Σ n · w · 10000  (stored gradient; normalized at extraction)
        int32_t sumNy;
        int32_t sumNz;
    };
    static_assert(sizeof(AdvDirEntry) == 24, "AdvDirEntry must be 24 bytes");
    static_assert(offsetof(AdvDirEntry, sumNx) == 12);

    // Per-(voxel,direction) readback: world-space voxel centre + recovered tsdf/weight and the
    // denoised stored-gradient normal (normalize(sumN); zero if degenerate).
    struct AdvancedEntry {
        Eigen::Vector3f center;
        uint32_t direction;
        float tsdf;
        float weight;
        Eigen::Vector3f normal;
        int32_t firstFrame; // frame that first filled this (voxel,dir); GPU-stamped, -1 if never stamped
    };

    static_assert(sizeof(AdvancedEntry) == 40, "AdvancedEntry must be 10 packed 4-byte scalars (== OutEntry)");

    // AdvancedTSDF — the best-of-all directional TSDF distilled from this repo's measurements:
    //   - compact per-(voxel,direction) flat hash: block-DirectionalTSDF accuracy at roughly an
    //     order of magnitude less memory (no 512-voxel block waste),
    //   - point-to-plane integration (near-exact on flat surfaces, better edges — measured),
    //   - stored-gradient mode-3 extraction (denoised normals + legacy sub-voxel zero-crossing
    //     position),
    // implemented over the user-style advanced_tsdf_{integrate,extract}.vert.glsl shaders.
    //
    // 32-bit key => a movable 512^3-voxel window. Unlike CompactDirectionalTSDF's fixed-corner
    // default (only correct at voxelSize=0.1), the default window here is CENTRED on the world
    // origin at ANY voxelSize (originVoxel = -256). Pass an explicit windowMinCorner to
    // re-centre elsewhere. Scenes larger than one 512^3 window need tiling.
    class AdvancedTSDF {
    public:
        AdvancedTSDF() = default;

        void Build(Engine::Core::Context &ctx,
                   float voxelSize = 0.01f,
                   float truncation = 0.03f,
                   uint32_t hashCapacity = 1u << 20,
                   uint32_t maxPoints = 1u << 15,
                   const Eigen::Vector3f &windowMinCorner =
                           Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN()));

        // Voxel-space origin of the movable window (exposed for tests/diagnostics).
        Eigen::Vector3i OriginVoxel() const { return m_originVoxel; }

        void SetIntegrationQuality(const IntegrationQuality &q) { m_quality = q; }

        // Integrate SDF form. true (default) = point-to-plane (removes grazing bias, measured
        // best); false = projective ray distance. Flows into the integrate push-constant.
        void SetPointToPlane(bool on) { m_pointToPlane = on; }

        // A1 (measure-first): surface-proximity confidence weight lambda in [0,1] applied to
        // each observation (down-weights band voxels far from the surface). 0 = off/uniform.
        void SetConfidenceWeight(float lambda) { m_confWeight = lambda; }

        // A2 (measure-first): cubic-Hermite (gradient-augmented) zero-crossing position instead
        // of linear. Uses the stored gradient at both endpoints. false = linear (default).
        void SetHermitePosition(bool on) { m_hermite = on; }

        // Frame index stamped into a slot the first time it is filled (read back per entry as
        // AdvancedEntry::firstFrame). Set it before each integrate; lets the caller recover
        // "first-seen frame" + "new this frame" without a CPU tracker re-hashing the whole model.
        void SetCurrentFrame(int frame) { m_currentFrame = frame; }

        // Integrate a normal-carrying point cloud observed from cameraPos. Normals drive the
        // dominant-direction selection, the view-angle weight, and the stored gradient.
        void Integrate(const std::vector<Eigen::Vector3f> &points,
                       const std::vector<Eigen::Vector3f> &normals,
                       const Eigen::Vector3f &cameraPos = Eigen::Vector3f::Zero());

        // Records upload (memcpy into mapped buffers) + the integrate dispatch into `batch` WITHOUT
        // submitting, so many tiles batch into ONE submit. Same result as Integrate; the caller owns
        // and submits the batch.
        void RecordIntegrate(const std::vector<Eigen::Vector3f> &points,
                             const std::vector<Eigen::Vector3f> &normals,
                             const Eigen::Vector3f &cameraPos,
                             Engine::Compute::CommandBatch &batch);

        // Real-time integrate: like Integrate, but sizes the upload buffers to the WHOLE frame (growing
        // them with slack when a frame arrives larger than any before) instead of clamping to the
        // Build-time maxPoints. No point is dropped and no frame size need be known up front -- for
        // streaming a live camera/reconstruction. Grown buffers are reused, so a reallocation is paid
        // only when a frame sets a new high-water mark.
        void IntegrateGPU(const std::vector<Eigen::Vector3f> &points,
                          const std::vector<Eigen::Vector3f> &normals,
                          const Eigen::Vector3f &cameraPos = Eigen::Vector3f::Zero());

        // Batched IntegrateGPU (grow + upload + record into `batch`, no submit) -- lets a tiled
        // coordinator fuse many tiles into ONE submit. Mirror of RecordIntegrate without the clamp.
        void RecordIntegrateGPU(const std::vector<Eigen::Vector3f> &points,
                                const std::vector<Eigen::Vector3f> &normals,
                                const Eigen::Vector3f &cameraPos,
                                Engine::Compute::CommandBatch &batch);

        // Record an integrate dispatch reading a CALLER-provided whole-cloud buffer (n points) instead
        // of this tile's own upload buffer -- no memcpy, no CPU routing. The integrate shader's window
        // filter keeps only the points inside this tile's window, so a tiled coordinator uploads the
        // cloud ONCE and dispatches every tile over it. `points`/`normals` are tight float3 arrays.
        void RecordIntegrateShared(Engine::Core::Buffer &points, Engine::Core::Buffer &normals,
                                   uint32_t n, const Eigen::Vector3f &cameraPos,
                                   Engine::Compute::CommandBatch &batch);

        // Mode-3 hybrid extraction → oriented point cloud. merge=true clusters/dedups the raw
        // candidates on the CPU (corner-preserving), via MergeCandidates.
        OrientedPointCloud ExtractPointCloud(uint32_t maxCandidates = 1u << 19,
                                             bool merge = false) const;

        uint32_t FilledCount() const;

        // Current slot count. Doubles on every growHash, so a caller that cached the Build-time
        // capacity would report a stale load factor -- read it here instead.
        uint32_t HashCapacity() const { return m_hashCapacity; }

        // Unpack every occupied, sufficiently-observed entry (world centre, direction, tsdf,
        // weight, stored-gradient normal). GPU-compacts the hash so only filled entries cross back.
        std::vector<AdvancedEntry> DownloadEntries() const;

        // RECORD (no reset, no submit) this tile's compaction into `batch`, appending into caller-shared
        // (out, count). The kernel emits only voxels in [coreMinWorld, coreMaxWorld) (world voxel
        // coords) -- so a tiled coordinator gives each tile its OWN core and batches EVERY tile into one
        // submit writing ONE shared buffer, with no cross-tile ghost duplicates and no CPU decode (the
        // kernel writes ready-made world-space AdvancedEntry records). The caller resets `count` once
        // before recording, submits once, then reads `count` (the exact required size) and the entries.
        // Pass the whole window ([originVoxel, originVoxel+512)) to append everything (standalone use).
        void RecordCompact(Engine::Core::Buffer &out, Engine::Core::Buffer &count,
                           Engine::Compute::CommandBatch &batch,
                           const Eigen::Vector3i &coreMinWorld,
                           const Eigen::Vector3i &coreMaxWorld) const;

        void Reset();

        // Corner-preserving cluster/dedup of raw {position,normal} candidates (shared utility).
        static OrientedPointCloud MergeCandidates(const std::vector<Eigen::Vector3f> &points,
                                                  const std::vector<Eigen::Vector3f> &normals,
                                                  float voxelSize);

    private:
        // Upload the first n points/normals into the mapped buffers (n <= current capacity) and record
        // the integrate dispatch into `batch`. Shared by RecordIntegrate (clamped) and RecordIntegrateGPU.
        void recordUpload(const std::vector<Eigen::Vector3f> &points,
                          const std::vector<Eigen::Vector3f> &normals,
                          const Eigen::Vector3f &cameraPos, uint32_t n,
                          Engine::Compute::CommandBatch &batch);

        // Grow the mapped point/normal upload buffers to hold >= n points (with slack), re-binding the
        // kernel to the new handles. Only ever grows; a no-op once capacity suffices.
        void ensureUploadCapacity(uint32_t n);

        // Keep the hash load factor bounded so probe chains -- and thus per-frame integrate cost -- stay
        // ~constant as the map accumulates (and no insert ever overflows MAX_PROBE and silently drops a
        // voxel). Reads the GPU fill count; if load >= ~0.5, doubles the hash and GPU-rehashes into it.
        // Called at the start of each integrate; a no-op until a tile actually fills up. Only the tiles
        // that fill grow, so total memory tracks real occupancy (not a big fixed per-tile pre-allocation).
        void maybeGrow();
        void growHash(uint32_t newCapacity); // allocate + clear + rehash into a larger hash, then rebind

        Engine::Core::Context *m_ctx = nullptr;
        float m_voxelSize = 0.01f;
        float m_truncation = 0.03f;
        uint32_t m_hashCapacity = 0;
        uint32_t m_maxPoints = 0;
        Eigen::Vector3i m_originVoxel = Eigen::Vector3i::Constant(-256); // centred default
        IntegrationQuality m_quality;
        bool m_pointToPlane = true; // measured-best default
        float m_confWeight = 0.5f;  // A1: adopted (measured cube RMSE -29%); 0 disables
        bool m_hermite = false;     // A2: off by default (no measured gain on synthetic fixtures)
        int m_currentFrame = 0;     // stamped into a slot on its first fill (see SetCurrentFrame)

        std::unique_ptr<Engine::Core::Buffer> m_hashBuffer;
        std::unique_ptr<Engine::Core::Buffer> m_pointBuffer;
        std::unique_ptr<Engine::Core::Buffer> m_normalBuffer;
        std::unique_ptr<Engine::Core::Buffer> m_statBuffer;
        std::unique_ptr<Engine::Core::Buffer> m_firstFrameBuffer; // per-slot first-fill frame (int32)
        mutable std::unique_ptr<Engine::Core::Buffer> m_compactBuffer;
        mutable std::unique_ptr<Engine::Core::Buffer> m_compactCountBuffer;

        std::unique_ptr<Engine::Core::ComputePipeline> kernel_integratePoints;
        std::unique_ptr<Engine::Core::ComputePipeline> kernel_clearVoxel;
        std::unique_ptr<Engine::Core::ComputePipeline> kernel_rehashTable;
        std::unique_ptr<Engine::Core::ComputePipeline> kernel_compactTable;
    };

} // namespace Engine::Spatial
