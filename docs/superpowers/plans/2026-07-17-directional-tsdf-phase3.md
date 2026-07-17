# DirectionalTSDF Phase 3 (integration/extraction 연동) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the full per-frame reconstruction pipeline to `DirectionalTSDF` — dominant-direction TSDF integration (design doc §13, dirty marking included), direction-layer-aware point extraction (§15), CPU candidate merge (§15 표), recomputeMask 기반 old-point 병합 (§14) — plus the interproximal demo with per-frame CSV streaming stats and stage timings.

**Architecture:** Two new compute shaders. `directional_tsdf_integrate.comp` ray-marches each point sample along the camera ray (SimpleTSDF의 검증된 패턴을 direction layer로 확장), writes fixed-point normalized TSDF into resident pool slots only (invariant #7), and marks touched slots dirty via `atomicOr`. `directional_tsdf_extract.comp` runs one thread per voxel of each recompute group, finds zero-crossings against +axis neighbours **in the same direction layer only** (invariant #10), and appends `DirectionalCandidate`s with gradient normals. The new public `Integrate()` orchestrates: BeginFrame → CPU write-set/halo 계산 → EnsureResident → integrate dispatch → extract dispatch → CPU merge → old-point 병합, timing each stage with `std::chrono` (동기 제출이므로 wall-clock이 정확).

**Tech Stack:** C++17, Vulkan 1.3 via `Engine::Core`, GLSL compute, Eigen, GoogleTest.

**Spec:** `docs/superpowers/specs/2026-07-17-directional-tsdf-design.md` §4.3/§4.4/§5(9~17단계)/§6/§7.

## Global Constraints

- Phase 1/2 plan의 Global Constraints 전부 유지. 기존 Directional 테스트 20개는 시그니처 변경 없이 통과 유지 (`Build`는 default 인자 2개 추가만 — 기존 호출부 컴파일 유지).
- **Dirty 표현**: GPU는 `atomicOr(packed, 1u<<16)`로 dirty 비트만 세운다. `SlotState::ResidentDirty`로의 state 전이는 GPU에서 atomic하게 표현 불가(1|2=3=PendingUpload 오염)하므로 **dirty 비트가 유일한 진실**이며 classify 커널은 이미 dirty 비트로 write-back을 판정한다. state는 ResidentClean으로 남는다.
- **TSDF 값 정규화**: 커널은 `clamp(sdf/truncation, -1, 1)`을 저장한다 (문서 §13). Phase 1의 host↔GPU 고정소수점 변환은 이 정규화 값 기준으로 이미 일관됨 (`HostTsdfVoxel.value` 주석과 일치).
- **Voxel↔group 좌표**: 음수 좌표에서 `v >> 3`(arithmetic shift)이 floor-division, `v & 7`이 그룹 내 좌표. CPU/GLSL 동일 수식 사용. 그룹 내 선형 인덱스는 `(lz*8+ly)*8+lx` — integrate/extract 두 커널이 반드시 동일해야 한다.
- **recomputeMask는 spatial**: 추출 대상은 "write-set의 spatial group 위치에 존재하는 **모든 direction layer의 resident 그룹**"이다. 쓰인 direction만 재추출하면 같은 위치의 다른 layer 표면 점이 old-point drop으로 소실된다 (계획 수립 중 발견한 함정 — 아래 Task 2 참조).
- `dominantAxis`의 비교 순서/타이브레이크는 CPU(write-set 계산)와 GLSL(적분 direction)이 **완전히 동일**해야 한다.
- 테스트 baseline: 71 tests, 70 pass (`WideBVHTest.RadiusMatchesCpuReference` 기존 실패).
- 빌드:
  ```bash
  export VULKAN_SDK=/Users/sjy/VulkanSDK/1.4.328.1/macOS
  cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)"
  ```

---

### Task 1: integrate 커널 + `Integrate()` 전반부 (residency + 적분 + dirty)

**Files:**
- Create: `src/shader/directional_tsdf_integrate.comp`
- Modify: `src/Engine/Spatial/DirectionalTSDF.h`
- Modify: `src/Engine/Spatial/DirectionalTSDF.cpp`
- Modify: `test/test_directionalTSDF.cpp`

**Interfaces:**
- Produces (Task 2가 확장):
  ```cpp
  void Build(Engine::Core::Context &ctx, float voxelSize = 0.1f, float truncation = 0.3f,
             uint32_t poolCapacity = 32768,
             uint32_t maxPoints = 1u << 15, uint32_t maxCandidates = 1u << 18);
  void Integrate(const std::vector<Eigen::Vector3f> &points,
                 const std::vector<Eigen::Vector3f> &normals,
                 const Eigen::Vector3f &cameraPos,
                 const Eigen::Vector3f &aabbCenterHint);
  // Stats에 추가: float beginFrameMs, ensureResidentMs, integrateMs, extractMs, mergeMs;
  ```
- Shader contract (`directional_tsdf_integrate.comp`): binding 0 = `PointSample{float px,py,pz,nx,ny,nz}[]` (readonly), 1 = indexGrid (readonly), 2 = `GpuVoxel{int sumDW; uint sumW}[]` pool (atomicAdd), 3 = meta (atomicOr dirty). Push `{uint numPoints; float voxelSize; float truncation; int baseX,baseY,baseZ; float camX,camY,camZ;}` (36B).

- [ ] **Step 1: integrate 셰이더 작성**

Create `src/shader/directional_tsdf_integrate.comp`:

```glsl
#version 460
layout(local_size_x = 256) in;

// Dominant-direction TSDF integration (design doc §13). Extends SimpleTSDF's
// voxel_tsdf_integrate.comp ray-march: each sample writes only into the direction
// layer chosen by its normal's dominant axis, and only into resident groups
// (invariant #7 — indexGrid miss → skip). Touched slots are marked dirty.

#define LOCAL_GRID 50
#define TSDF_SCALE 10000.0

struct PointSample { float px, py, pz, nx, ny, nz; };
struct GpuVoxel { int sumDW; uint sumW; };
struct ActiveGroupMeta { int gx; int gy; int gz; uint packed; };

layout(push_constant) uniform PC {
    uint  g_numPoints;
    float g_voxelSize;
    float g_truncation;
    int   g_baseX;
    int   g_baseY;
    int   g_baseZ;
    float g_camX;
    float g_camY;
    float g_camZ;
};

layout(std430, set = 0, binding = 0) readonly buffer Points { PointSample g_points[]; };
layout(std430, set = 0, binding = 1) readonly buffer IndexGrid { uint g_indexGrid[]; };
layout(std430, set = 0, binding = 2) buffer Pool { GpuVoxel g_pool[]; };
layout(std430, set = 0, binding = 3) buffer Meta { ActiveGroupMeta g_meta[]; };

// Must match dominantAxisOf() in DirectionalTSDF.cpp exactly (comparison order + ties).
uint dominantAxis(vec3 n) {
    vec3 a = abs(n);
    if (a.x >= a.y && a.x >= a.z) return n.x >= 0.0 ? 0u : 1u;
    if (a.y >= a.x && a.y >= a.z) return n.y >= 0.0 ? 2u : 3u;
    return n.z >= 0.0 ? 4u : 5u;
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= g_numPoints) return;

    PointSample s = g_points[gid];
    vec3 p = vec3(s.px, s.py, s.pz);
    vec3 cam = vec3(g_camX, g_camY, g_camZ);

    vec3 diff = p - cam;
    float depth = length(diff);
    if (depth < 1e-6) return;
    vec3 rayDir = diff / depth;
    uint dir = dominantAxis(vec3(s.nx, s.ny, s.nz));

    int steps = int(ceil(g_truncation / g_voxelSize)) + 1;
    for (int t = -steps; t <= steps; t++) {
        vec3 samplePos = p + rayDir * (float(t) * g_voxelSize);
        ivec3 v = ivec3(floor(samplePos / g_voxelSize));
        vec3 vCenter = (vec3(v) + vec3(0.5)) * g_voxelSize;

        float sdf = depth - dot(vCenter - cam, rayDir);
        if (abs(sdf) > g_truncation) continue;

        ivec3 g = v >> 3; // floor division by 8, negatives included
        int lx = g.x - g_baseX;
        int ly = g.y - g_baseY;
        int lz = g.z - g_baseZ;
        if (lx < 0 || ly < 0 || lz < 0 ||
            lx >= LOCAL_GRID || ly >= LOCAL_GRID || lz >= LOCAL_GRID)
            continue;

        uint cell = ((uint(lz) * uint(LOCAL_GRID) + uint(ly)) * uint(LOCAL_GRID) + uint(lx)) * 6u + dir;
        uint poolIndex = g_indexGrid[cell];
        if (poolIndex == 0xFFFFFFFFu) continue; // not resident → not in the write set

        ivec3 lv = v & 7;
        uint voxelIdx = (uint(lv.z) * 8u + uint(lv.y)) * 8u + uint(lv.x);
        uint addr = poolIndex * 512u + voxelIdx;

        float newValue = clamp(sdf / g_truncation, -1.0, 1.0);
        atomicAdd(g_pool[addr].sumDW, int(newValue * TSDF_SCALE));
        atomicAdd(g_pool[addr].sumW, uint(TSDF_SCALE));
        atomicOr(g_meta[poolIndex].packed, 1u << 16); // dirty flag (see Global Constraints)
    }
}
```

- [ ] **Step 2: 헤더 수정**

`DirectionalTSDF.h`:

(a) `Stats`에 타이밍 필드 추가:

```cpp
        struct Stats {
            uint32_t residentCount = 0;
            uint32_t missingCount = 0;
            uint32_t writeBackCount = 0;
            uint32_t h2dBytes = 0;
            uint32_t d2hBytes = 0;
            float overlapRatio = 0.0f;
            // Stage timings (ms). Valid because every GPU submission is synchronous.
            float beginFrameMs = 0.0f;
            float ensureResidentMs = 0.0f;
            float integrateMs = 0.0f;
            float extractMs = 0.0f;
            float mergeMs = 0.0f;
        };
```

(b) `Build` 시그니처 확장:

```cpp
        void Build(Engine::Core::Context &ctx,
                   float voxelSize = 0.1f,
                   float truncation = 0.3f,
                   uint32_t poolCapacity = 32768,
                   uint32_t maxPoints = 1u << 15,
                   uint32_t maxCandidates = 1u << 18);
```

(c) `EnsureResident` 선언 아래 public에 추가:

```cpp
        // Full frame pipeline (design doc §5): BeginFrame → write-set/halo residency →
        // GPU integration → extraction over the recompute mask → candidate merge →
        // old-point replacement. points/normals must be the same length; at most
        // maxPoints samples are used.
        void Integrate(const std::vector<Eigen::Vector3f> &points,
                       const std::vector<Eigen::Vector3f> &normals,
                       const Eigen::Vector3f &cameraPos,
                       const Eigen::Vector3f &aabbCenterHint);

        const std::vector<ExtractedPoint> &PointCloud() const { return m_pointCloud; }
        void ExportPointCloud(const std::string &path) const; // ASCII PLY with normals
```

(d) private 멤버 추가 (`m_classifyKernel` 아래):

```cpp
        std::unique_ptr<Engine::Core::Buffer> m_pointBuffer;      // PointSample[maxPoints]
        std::unique_ptr<Engine::Core::Buffer> m_candidateBuffer;  // DirectionalCandidate[maxCandidates]
        std::unique_ptr<Engine::Core::Buffer> m_candidateCounter; // uint32
        std::unique_ptr<Engine::Core::ComputePipeline> m_integrateKernel;
        std::unique_ptr<Engine::Core::ComputePipeline> m_extractKernel;
        uint32_t m_maxPoints = 0;
        uint32_t m_maxCandidates = 0;
        std::vector<ExtractedPoint> m_pointCloud;
```

(e) private 메서드 추가:

```cpp
        std::vector<ExtractedPoint> mergeCandidates(
                const std::vector<DirectionalCandidate> &candidates) const;
```

- [ ] **Step 3: 실패하는 테스트 작성**

`test/test_directionalTSDF.cpp` 끝에 추가:

```cpp
namespace {
    // Grid of samples on the plane x = planeX with the given normal.
    void makePlane(float planeX, float extent, float step, const Eigen::Vector3f &normal,
                   std::vector<Eigen::Vector3f> &points, std::vector<Eigen::Vector3f> &normals) {
        for (float y = -extent; y <= extent + 1e-4f; y += step)
            for (float z = -extent; z <= extent + 1e-4f; z += step) {
                points.emplace_back(planeX, y, z);
                normals.push_back(normal);
            }
    }
} // namespace

TEST(DirectionalTSDFPhase3Test, IntegrationWritesSignedBandAndMarksDirty) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, 4096);

    std::vector<Eigen::Vector3f> points, normals;
    makePlane(0.0f, 0.4f, 0.05f, Eigen::Vector3f(1, 0, 0), points, normals);

    tsdf.Integrate(points, normals, /*cam=*/Eigen::Vector3f(2, 0, 0),
                   /*hint=*/Eigen::Vector3f::Zero());
    EXPECT_GT(tsdf.LastFrameStats().missingCount, 0u);

    // Camera on +X: voxel (0,0,0) (centre x=+0.05) is in front → value > 0.
    auto front = tsdf.DebugDownloadGroupVoxels({0, 0, 0, 0}); // +X layer
    EXPECT_GT(front[0].weight, 0.0f);
    EXPECT_GT(front[0].value, 0.0f);

    // Voxel (-1,0,0) lives in group gx=-1 at local (7,0,0) → index 7; behind → value < 0.
    auto behind = tsdf.DebugDownloadGroupVoxels({-1, 0, 0, 0});
    EXPECT_GT(behind[7].weight, 0.0f);
    EXPECT_LT(behind[7].value, 0.0f);

    // Dirty marking: move the window away → dirty groups fall outside → WriteBackList.
    tsdf.BeginFrame(Eigen::Vector3f(80.0f, 0.0f, 0.0f));
    EXPECT_GT(tsdf.DebugLastClassifyCounts().writeBack, 0u);
}
```

- [ ] **Step 4: 구현 — Build 확장 + Integrate() 전반부**

`DirectionalTSDF.cpp`:

(a) include 추가: `#include <algorithm>`, `#include <chrono>`, `#include <fstream>`.

(b) 익명 namespace에 추가:

```cpp
        struct IntegratePC {
            uint32_t numPoints;
            float voxelSize;
            float truncation;
            int32_t baseX;
            int32_t baseY;
            int32_t baseZ;
            float camX;
            float camY;
            float camZ;
        };

        struct ExtractPC {
            uint32_t numGroups;
            float voxelSize;
            uint32_t maxCandidates;
            int32_t baseX;
            int32_t baseY;
            int32_t baseZ;
        };

        // Must match dominantAxis() in directional_tsdf_integrate.comp exactly.
        uint8_t dominantAxisOf(const Eigen::Vector3f &n) {
            const float ax = std::fabs(n.x()), ay = std::fabs(n.y()), az = std::fabs(n.z());
            if (ax >= ay && ax >= az) return n.x() >= 0.0f ? 0 : 1;
            if (ay >= ax && ay >= az) return n.y() >= 0.0f ? 2 : 3;
            return n.z() >= 0.0f ? 4 : 5;
        }

        uint64_t spatialKey(int32_t x, int32_t y, int32_t z) {
            return (uint64_t(uint32_t(x) & 0x1FFFFFu) << 42) |
                   (uint64_t(uint32_t(y) & 0x1FFFFFu) << 21) |
                   uint64_t(uint32_t(z) & 0x1FFFFFu);
        }
```

(c) `Build` 시그니처를 헤더와 맞추고, `m_countsBuffer->Allocate(...)` 다음에 추가:

```cpp
        m_maxPoints = maxPoints;
        m_maxCandidates = maxCandidates;
        m_pointBuffer = std::make_unique<Engine::Core::Buffer>(ctx);
        m_candidateBuffer = std::make_unique<Engine::Core::Buffer>(ctx);
        m_candidateCounter = std::make_unique<Engine::Core::Buffer>(ctx);
        m_pointBuffer->Allocate(maxPoints * 6u * sizeof(float));
        m_candidateBuffer->Allocate(maxCandidates * sizeof(DirectionalCandidate));
        m_candidateCounter->Allocate(sizeof(uint32_t));
```

그리고 classify 커널 생성 다음에:

```cpp
        m_integrateKernel = std::make_unique<Engine::Core::ComputePipeline>(ctx);
        m_integrateKernel->Build("directional_tsdf_integrate.comp")
                .Bind(0, *m_pointBuffer)
                .Bind(1, *m_indexGrid)
                .Bind(2, *m_poolVoxels)
                .Bind(3, *m_metaBuffer);

        m_extractKernel = std::make_unique<Engine::Core::ComputePipeline>(ctx);
        m_extractKernel->Build("directional_tsdf_extract.comp")
                .Bind(0, *m_slotListBuffer) // reused as the recompute-group list
                .Bind(1, *m_metaBuffer)
                .Bind(2, *m_poolVoxels)
                .Bind(3, *m_indexGrid)
                .Bind(4, *m_candidateBuffer)
                .Bind(5, *m_candidateCounter);

        m_pointCloud.clear();
```

주의: extract 커널이 아직 없으므로 이 Step에서는 `m_extractKernel` 블록을 **제외**하고 Task 2에서 추가한다. (Build 시 셰이더 파일이 없으면 throw.)

(d) `Integrate` 정의 추가 (Task 1 범위 — 추출/병합은 Task 2에서 채움):

```cpp
    void DirectionalTSDF::Integrate(const std::vector<Eigen::Vector3f> &points,
                                    const std::vector<Eigen::Vector3f> &normals,
                                    const Eigen::Vector3f &cameraPos,
                                    const Eigen::Vector3f &aabbCenterHint) {
        if (points.size() != normals.size())
            throw std::runtime_error("DirectionalTSDF: points/normals size mismatch");
        if (points.empty()) return;

        using Clock = std::chrono::steady_clock;
        auto msBetween = [](Clock::time_point a, Clock::time_point b) {
            return std::chrono::duration<float, std::milli>(b - a).count();
        };

        const auto t0 = Clock::now();
        BeginFrame(aabbCenterHint);
        const auto t1 = Clock::now();

        const uint32_t N = std::min(uint32_t(points.size()), m_maxPoints);

        // IntegrationWriteSet: per-sample conservative box over the truncation-band
        // ray segment (design doc §7/§13), keyed by the sample's dominant direction.
        std::unordered_set<DirectionalGroupKey, DirectionalGroupKeyHash> writeSet;
        for (uint32_t i = 0; i < N; ++i) {
            Eigen::Vector3f diff = points[i] - cameraPos;
            float depth = diff.norm();
            if (depth < 1e-6f) continue;
            Eigen::Vector3f dir = diff / depth;
            const uint8_t d = dominantAxisOf(normals[i]);
            const float band = m_truncation + m_voxelSize;
            Eigen::Vector3f a = points[i] - dir * band;
            Eigen::Vector3f b = points[i] + dir * band;
            Eigen::Vector3i vmin, vmax;
            for (int c = 0; c < 3; ++c) {
                const float lo = std::min(a[c], b[c]);
                const float hi = std::max(a[c], b[c]);
                vmin[c] = int(std::floor(lo / m_voxelSize)) - 1;
                vmax[c] = int(std::floor(hi / m_voxelSize)) + 1;
            }
            for (int gz = vmin.z() >> 3; gz <= (vmax.z() >> 3); ++gz)
                for (int gy = vmin.y() >> 3; gy <= (vmax.y() >> 3); ++gy)
                    for (int gx = vmin.x() >> 3; gx <= (vmax.x() >> 3); ++gx)
                        writeSet.insert({gx, gy, gz, d});
        }

        // ResidentRequiredSet = writeSet + 1-group halo (extraction neighbourhood, §7).
        std::vector<DirectionalGroupKey> required;
        {
            std::unordered_set<DirectionalGroupKey, DirectionalGroupKeyHash> requiredSet;
            for (const auto &k : writeSet)
                for (int dz = -1; dz <= 1; ++dz)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx)
                            requiredSet.insert(
                                    {k.gx + dx, k.gy + dy, k.gz + dz, k.direction});
            required.assign(requiredSet.begin(), requiredSet.end());
        }
        EnsureResident(required);
        const auto t2 = Clock::now();

        // Upload samples and integrate.
        std::vector<float> samples(size_t(N) * 6u);
        for (uint32_t i = 0; i < N; ++i) {
            samples[size_t(i) * 6 + 0] = points[i].x();
            samples[size_t(i) * 6 + 1] = points[i].y();
            samples[size_t(i) * 6 + 2] = points[i].z();
            samples[size_t(i) * 6 + 3] = normals[i].x();
            samples[size_t(i) * 6 + 4] = normals[i].y();
            samples[size_t(i) * 6 + 5] = normals[i].z();
        }
        m_pointBuffer->Upload(samples.data(), uint32_t(samples.size() * sizeof(float)));
        IntegratePC ipc{N,
                        m_voxelSize,
                        m_truncation,
                        m_localBase.x(),
                        m_localBase.y(),
                        m_localBase.z(),
                        cameraPos.x(),
                        cameraPos.y(),
                        cameraPos.z()};
        m_integrateKernel->Args(ipc).DispatchElements(N);
        const auto t3 = Clock::now();

        extractAndMerge(writeSet, t3, msBetween); // Task 2에서 정의; Task 1에서는 아래 참고

        m_stats.beginFrameMs = msBetween(t0, t1);
        m_stats.ensureResidentMs = msBetween(t1, t2);
        m_stats.integrateMs = msBetween(t2, t3);
    }
```

**Task 1에서는** `extractAndMerge(...)` 호출 줄을 넣지 말고 (Task 2에서 추가), 타이밍 3개만 기록한다. `ExportPointCloud`도 Task 2 몫.

- [ ] **Step 5: 빌드 + 테스트**

```bash
cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)"
./build/test/vkspatial_tests --gtest_filter="DirectionalTSDFPhase3Test.*"
```
Expected: `IntegrationWritesSignedBandAndMarksDirty` PASS. 이어서 회귀:
```bash
./build/test/vkspatial_tests --gtest_filter="Directional*"
```
Expected: 21 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/shader/directional_tsdf_integrate.comp src/Engine/Spatial/DirectionalTSDF.h src/Engine/Spatial/DirectionalTSDF.cpp test/test_directionalTSDF.cpp
git commit -m "Add dominant-direction TSDF integration kernel (Phase 3)"
```

---

### Task 2: extract 커널 + candidate merge + old-point 병합

**Files:**
- Create: `src/shader/directional_tsdf_extract.comp`
- Modify: `src/Engine/Spatial/DirectionalTSDF.cpp` (extract 커널 생성/디스패치 + merge + PLY export)
- Modify: `test/test_directionalTSDF.cpp`

**Interfaces:**
- Produces: 동작하는 `Integrate()` 전체 파이프라인, `PointCloud()`, `ExportPointCloud(path)`.
- Shader contract (`directional_tsdf_extract.comp`): binding 0 = 추출 대상 그룹의 poolIndex 목록(readonly, `m_slotListBuffer` 재사용), 1 = meta, 2 = pool, 3 = indexGrid, 4 = `DirectionalCandidate[]` out, 5 = `uint counter`. Push `{uint numGroups; float voxelSize; uint maxCandidates; int baseX,baseY,baseZ;}` (24B). Thread k → group k/512의 voxel k%512. **같은 direction layer끼리만** +axis 이웃과 sign crossing 비교 (invariant #10); crossing 평균 위치 + central-difference gradient normal로 candidate append.

- [ ] **Step 1: extract 셰이더 작성**

Create `src/shader/directional_tsdf_extract.comp`:

```glsl
#version 460
layout(local_size_x = 256) in;

// Direction-layer point extraction (design doc §15). One thread per voxel of each
// recompute group. Sign crossings are compared ONLY within the same direction layer
// (invariant #10). Candidates carry a finite-difference gradient normal.

#define LOCAL_GRID 50

struct ActiveGroupMeta { int gx; int gy; int gz; uint packed; };
struct GpuVoxel { int sumDW; uint sumW; };
struct Candidate {
    float px; float py; float pz;
    float nx; float ny; float nz;
    int gx; int gy; int gz;
    uint direction;
};

layout(push_constant) uniform PC {
    uint  g_numGroups;
    float g_voxelSize;
    uint  g_maxCandidates;
    int   g_baseX;
    int   g_baseY;
    int   g_baseZ;
};

layout(std430, set = 0, binding = 0) readonly buffer GroupList { uint g_groupSlots[]; };
layout(std430, set = 0, binding = 1) readonly buffer Meta { ActiveGroupMeta g_meta[]; };
layout(std430, set = 0, binding = 2) readonly buffer Pool { GpuVoxel g_pool[]; };
layout(std430, set = 0, binding = 3) readonly buffer IndexGrid { uint g_indexGrid[]; };
layout(std430, set = 0, binding = 4) buffer Candidates { Candidate g_candidates[]; };
layout(std430, set = 0, binding = 5) buffer Counter { uint g_candidateCount; };

bool fetchValue(ivec3 v, uint dir, out float value) {
    ivec3 g = v >> 3;
    int lx = g.x - g_baseX;
    int ly = g.y - g_baseY;
    int lz = g.z - g_baseZ;
    value = 0.0;
    if (lx < 0 || ly < 0 || lz < 0 ||
        lx >= LOCAL_GRID || ly >= LOCAL_GRID || lz >= LOCAL_GRID)
        return false;
    uint cell = ((uint(lz) * uint(LOCAL_GRID) + uint(ly)) * uint(LOCAL_GRID) + uint(lx)) * 6u + dir;
    uint poolIndex = g_indexGrid[cell];
    if (poolIndex == 0xFFFFFFFFu) return false;
    ivec3 lv = v & 7;
    uint addr = poolIndex * 512u + (uint(lv.z) * 8u + uint(lv.y)) * 8u + uint(lv.x);
    GpuVoxel vox = g_pool[addr];
    if (vox.sumW == 0u) return false;
    value = float(vox.sumDW) / float(vox.sumW);
    return true;
}

void main() {
    uint k = gl_GlobalInvocationID.x;
    if (k >= g_numGroups * 512u) return;
    uint slotIdx = k / 512u;
    uint voxelIdx = k % 512u;
    uint poolIndex = g_groupSlots[slotIdx];
    ActiveGroupMeta m = g_meta[poolIndex];
    uint dir = m.packed & 0xFFu;

    ivec3 lv = ivec3(int(voxelIdx & 7u), int((voxelIdx >> 3u) & 7u), int(voxelIdx >> 6u));
    ivec3 v = ivec3(m.gx, m.gy, m.gz) * 8 + lv;

    GpuVoxel center = g_pool[poolIndex * 512u + voxelIdx];
    if (center.sumW == 0u) return;
    float c = float(center.sumDW) / float(center.sumW);

    // Zero crossings against the three +axis neighbours (same layer only).
    vec3 posSum = vec3(0.0);
    int crossings = 0;
    for (int axis = 0; axis < 3; axis++) {
        ivec3 nv = v;
        nv[axis] += 1;
        float nVal;
        if (!fetchValue(nv, dir, nVal)) continue;
        if ((c > 0.0) == (nVal > 0.0)) continue;
        if (c == nVal) continue;
        float t = c / (c - nVal);
        vec3 cross = (vec3(v) + vec3(0.5)) * g_voxelSize;
        cross[axis] += t * g_voxelSize;
        posSum += cross;
        crossings++;
    }
    if (crossings == 0) return;

    // Normal: central-difference gradient in the same layer (one-sided fallback).
    vec3 grad = vec3(0.0);
    for (int axis = 0; axis < 3; axis++) {
        ivec3 vp = v; vp[axis] += 1;
        ivec3 vm = v; vm[axis] -= 1;
        float fp, fm;
        bool hp = fetchValue(vp, dir, fp);
        bool hm = fetchValue(vm, dir, fm);
        if (hp && hm) grad[axis] = (fp - fm) * 0.5;
        else if (hp) grad[axis] = fp - c;
        else if (hm) grad[axis] = c - fm;
    }
    float len = length(grad);
    if (len < 1e-6) return;

    uint idx = atomicAdd(g_candidateCount, 1u);
    if (idx >= g_maxCandidates) return;

    Candidate cand;
    vec3 pos = posSum / float(crossings);
    vec3 n = grad / len;
    cand.px = pos.x; cand.py = pos.y; cand.pz = pos.z;
    cand.nx = n.x;   cand.ny = n.y;   cand.nz = n.z;
    cand.gx = m.gx;  cand.gy = m.gy;  cand.gz = m.gz;
    cand.direction = dir;
    g_candidates[idx] = cand;
}
```

- [ ] **Step 2: 실패하는 테스트 작성**

`test/test_directionalTSDF.cpp` 끝에 추가:

```cpp
TEST(DirectionalTSDFPhase3Test, SinglePlaneExtractsSurfacePoints) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, 4096);

    std::vector<Eigen::Vector3f> points, normals;
    makePlane(0.0f, 0.5f, 0.05f, Eigen::Vector3f(1, 0, 0), points, normals);
    tsdf.Integrate(points, normals, Eigen::Vector3f(2, 0, 0), Eigen::Vector3f::Zero());

    const auto &cloud = tsdf.PointCloud();
    ASSERT_GT(cloud.size(), 50u);
    for (const auto &pt : cloud) {
        EXPECT_NEAR(pt.position.x(), 0.0f, 0.15f);
        EXPECT_GT(pt.normal.x(), 0.7f); // gradient points toward the camera (+X)
    }
}

// The doc's core quality claim (§2): two opposing surfaces 0.4 apart with truncation
// 0.3 have overlapping bands in [-0.1, 0.1] — a single SDF field cancels there, but
// separate direction layers must keep both surfaces intact.
TEST(DirectionalTSDFPhase3Test, OpposingSurfacesRemainSeparate) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, 4096);

    std::vector<Eigen::Vector3f> lp, ln, rp, rn;
    makePlane(-0.2f, 0.4f, 0.05f, Eigen::Vector3f(1, 0, 0), lp, ln);  // left, faces +X
    makePlane(+0.2f, 0.4f, 0.05f, Eigen::Vector3f(-1, 0, 0), rp, rn); // right, faces -X

    tsdf.Integrate(lp, ln, Eigen::Vector3f(3, 0, 0), Eigen::Vector3f::Zero());
    tsdf.Integrate(rp, rn, Eigen::Vector3f(-3, 0, 0), Eigen::Vector3f::Zero());

    int nearLeft = 0, nearRight = 0;
    for (const auto &pt : tsdf.PointCloud()) {
        if (std::fabs(pt.position.x() + 0.2f) < 0.1f) {
            ++nearLeft;
            EXPECT_GT(pt.normal.x(), 0.5f);
        }
        if (std::fabs(pt.position.x() - 0.2f) < 0.1f) {
            ++nearRight;
            EXPECT_LT(pt.normal.x(), -0.5f);
        }
    }
    EXPECT_GT(nearLeft, 30);
    EXPECT_GT(nearRight, 30);
}

TEST(DirectionalTSDFPhase3Test, ReintegrationDoesNotDuplicatePoints) {
    Engine::Core::Context ctx;
    DirectionalTSDF tsdf;
    tsdf.Build(ctx, 0.1f, 0.3f, 4096);

    std::vector<Eigen::Vector3f> points, normals;
    makePlane(0.0f, 0.5f, 0.05f, Eigen::Vector3f(1, 0, 0), points, normals);

    tsdf.Integrate(points, normals, Eigen::Vector3f(2, 0, 0), Eigen::Vector3f::Zero());
    const size_t count1 = tsdf.PointCloud().size();
    ASSERT_GT(count1, 0u);

    tsdf.Integrate(points, normals, Eigen::Vector3f(2, 0, 0), Eigen::Vector3f::Zero());
    const size_t count2 = tsdf.PointCloud().size();

    // recomputeMask replaces old points instead of accumulating them (invariant #9).
    EXPECT_GT(count2, count1 / 2);
    EXPECT_LT(count2, count1 * 3 / 2);
}
```

- [ ] **Step 3: 구현**

`DirectionalTSDF.cpp`:

(a) Task 1 Step 4(c)에서 보류한 `m_extractKernel` 생성 블록을 `Build()`에 추가 (integrate 커널 생성 바로 아래).

(b) `Integrate()`의 `const auto t3 = Clock::now();` 다음, 타이밍 기록 앞에 추출/병합 로직 추가:

```cpp
        // recomputeMask is SPATIAL (design doc §14): re-extract every resident direction
        // layer at the written spatial locations, otherwise other-layer surface points
        // at those locations would be dropped by the old-point merge and never rebuilt.
        std::unordered_set<uint64_t> recomputeSpatial;
        for (const auto &k : writeSet)
            recomputeSpatial.insert(spatialKey(k.gx, k.gy, k.gz));

        std::vector<uint32_t> groupSlots;
        for (const auto &entry : m_residentIndex)
            if (recomputeSpatial.count(
                        spatialKey(entry.first.gx, entry.first.gy, entry.first.gz)) > 0)
                groupSlots.push_back(entry.second);

        uint32_t candidateCount = 0;
        if (!groupSlots.empty()) {
            m_slotListBuffer->Upload(groupSlots.data(),
                                     uint32_t(groupSlots.size() * sizeof(uint32_t)));
            const uint32_t zero = 0;
            m_candidateCounter->Upload(&zero, sizeof(zero));
            ExtractPC epc{uint32_t(groupSlots.size()),
                          m_voxelSize,
                          m_maxCandidates,
                          m_localBase.x(),
                          m_localBase.y(),
                          m_localBase.z()};
            m_extractKernel->Args(epc).DispatchElements(
                    uint32_t(groupSlots.size()) * kVoxelsPerGroup);
            m_candidateCounter->Download(&candidateCount, sizeof(candidateCount));
            candidateCount = std::min(candidateCount, m_maxCandidates);
        }
        std::vector<DirectionalCandidate> candidates(candidateCount);
        if (candidateCount > 0)
            m_candidateBuffer->Download(candidates.data(),
                                        candidateCount * sizeof(DirectionalCandidate));
        const auto t4 = Clock::now();

        // Candidate merge + old-point replacement (§14/§15, invariant #9).
        std::vector<ExtractedPoint> fresh = mergeCandidates(candidates);
        m_pointCloud.erase(
                std::remove_if(m_pointCloud.begin(), m_pointCloud.end(),
                               [&](const ExtractedPoint &pt) {
                                   return recomputeSpatial.count(spatialKey(
                                                  pt.ownerGx, pt.ownerGy, pt.ownerGz)) > 0;
                               }),
                m_pointCloud.end());
        m_pointCloud.insert(m_pointCloud.end(), fresh.begin(), fresh.end());
        const auto t5 = Clock::now();

        m_stats.extractMs = msBetween(t3, t4);
        m_stats.mergeMs = msBetween(t4, t5);
```

(주의: Task 1에서 만든 타이밍 기록 3줄은 유지하고 `extractMs`/`mergeMs` 두 줄이 추가되는 형태다. Task 1의 `extractAndMerge(...)` placeholder 줄은 애초에 넣지 않았으므로 삭제할 것은 없다.)

(c) `mergeCandidates` 정의 추가 (문서 §15 merge/split 표 — merge 조건: position < 0.6 voxel && normal angle < 30°; 그 외 별도 cluster, voxel당 최대 6개):

```cpp
    std::vector<ExtractedPoint> DirectionalTSDF::mergeCandidates(
            const std::vector<DirectionalCandidate> &candidates) const {
        struct Cluster {
            Eigen::Vector3f posSum = Eigen::Vector3f::Zero();
            Eigen::Vector3f nSum = Eigen::Vector3f::Zero();
            int count = 0;
            uint8_t dirMask = 0;
            int32_t gx = 0, gy = 0, gz = 0;
        };
        auto voxelKey = [](int x, int y, int z) {
            return (uint64_t(uint32_t(x) & 0x1FFFFFu) << 42) |
                   (uint64_t(uint32_t(y) & 0x1FFFFFu) << 21) |
                   uint64_t(uint32_t(z) & 0x1FFFFFu);
        };
        const float posThresh = 0.6f * m_voxelSize; // positionMergeThreshold (§15)
        const float cosThresh = 0.866f;             // normalMergeThreshold = 30° (§15)

        std::unordered_map<uint64_t, std::vector<Cluster>> buckets;
        for (const auto &c : candidates) {
            Eigen::Vector3f pos(c.px, c.py, c.pz);
            Eigen::Vector3f nrm(c.nx, c.ny, c.nz);
            const int vx = int(std::floor(pos.x() / m_voxelSize));
            const int vy = int(std::floor(pos.y() / m_voxelSize));
            const int vz = int(std::floor(pos.z() / m_voxelSize));
            auto &clusters = buckets[voxelKey(vx, vy, vz)];
            bool merged = false;
            for (auto &cl : clusters) {
                const Eigen::Vector3f mean = cl.posSum / float(cl.count);
                const Eigen::Vector3f meanN = cl.nSum.normalized();
                if ((pos - mean).norm() < posThresh && nrm.dot(meanN) > cosThresh) {
                    cl.posSum += pos;
                    cl.nSum += nrm;
                    cl.count++;
                    cl.dirMask |= uint8_t(1u << c.direction);
                    merged = true;
                    break;
                }
            }
            if (!merged && clusters.size() < kNumDirections)
                clusters.push_back({pos, nrm, 1, uint8_t(1u << c.direction),
                                    c.gx, c.gy, c.gz});
        }

        std::vector<ExtractedPoint> out;
        for (auto &bucket : buckets)
            for (auto &cl : bucket.second) {
                ExtractedPoint pt;
                pt.position = cl.posSum / float(cl.count);
                pt.normal = cl.nSum.normalized();
                pt.ownerGx = cl.gx;
                pt.ownerGy = cl.gy;
                pt.ownerGz = cl.gz;
                pt.dirMask = cl.dirMask;
                out.push_back(pt);
            }
        return out;
    }
```

(d) `ExportPointCloud` 정의 추가:

```cpp
    void DirectionalTSDF::ExportPointCloud(const std::string &path) const {
        std::ofstream f(path);
        if (!f.is_open())
            throw std::runtime_error("DirectionalTSDF::ExportPointCloud: cannot open " + path);

        f << "ply\nformat ascii 1.0\n"
          << "element vertex " << m_pointCloud.size() << "\n"
          << "property float x\nproperty float y\nproperty float z\n"
          << "property float nx\nproperty float ny\nproperty float nz\n"
          << "end_header\n";
        for (const auto &pt : m_pointCloud)
            f << pt.position.x() << ' ' << pt.position.y() << ' ' << pt.position.z() << ' '
              << pt.normal.x() << ' ' << pt.normal.y() << ' ' << pt.normal.z() << '\n';
    }
```

- [ ] **Step 4: 빌드 + 테스트**

```bash
cmake --build build --target vkspatial_tests -j"$(sysctl -n hw.ncpu)"
./build/test/vkspatial_tests --gtest_filter="DirectionalTSDFPhase3Test.*"
```
Expected: 4 tests PASS. 이어서:
```bash
./build/test/vkspatial_tests 2>&1 | tail -6
```
Expected: 75 tests, 74 pass (기존 실패 1건).

- [ ] **Step 5: Commit**

```bash
git add src/shader/directional_tsdf_extract.comp src/Engine/Spatial/DirectionalTSDF.cpp test/test_directionalTSDF.cpp
git commit -m "Add direction-layer extraction, candidate merge, and old-point replacement (Phase 3)"
```

---

### Task 3: interproximal 데모 + CSV 스트리밍 통계

**Files:**
- Create: `example2/directional_tsdf_demo.cpp`
- Modify: `example2/CMakeLists.txt`

**Interfaces:**
- Consumes: `DirectionalTSDF` 전체 API + `SimpleTSDF` (단일 SDF 비교용).
- Produces: 실행 파일 `directional_tsdf_demo` — `directional_tsdf_stats.csv`(프레임별 스트리밍/타이밍 지표), `directional_result.ply`, `simple_result_mc.ply`.

- [ ] **Step 1: 데모 작성**

Create `example2/directional_tsdf_demo.cpp`:

```cpp
// Interproximal scenario demo: two opposing thin surfaces 0.4mm apart, scanned by a
// scripted camera path sliding along z. Logs per-frame streaming stats + stage timings
// as CSV, then exports the directional point cloud and a single-SDF (SimpleTSDF)
// marching-cubes mesh of the same data for visual comparison.
#include "Engine/Core/Context.h"
#include "Engine/Spatial/DirectionalTSDF.h"
#include "Engine/Spatial/SimpleTSDF.h"

#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <vector>

int main() {
    Engine::Core::Context ctx;

    constexpr float voxelSize = 0.1f;
    constexpr float truncation = 0.3f;
    constexpr float gapHalf = 0.2f;    // planes at x = ±0.2 → 0.4 gap; bands overlap in ±0.1
    constexpr float extentY = 1.0f;
    constexpr float extentZ = 6.0f;
    constexpr float step = 0.05f;
    constexpr float footprint = 1.5f;  // scanner footprint half-width along z
    constexpr int numFrames = 40;

    // Full synthetic surfaces (left faces +X, right faces -X).
    struct Sample { Eigen::Vector3f p, n; };
    std::vector<Sample> surface;
    for (float y = -extentY; y <= extentY + 1e-4f; y += step)
        for (float z = -extentZ; z <= extentZ + 1e-4f; z += step) {
            surface.push_back({{-gapHalf, y, z}, {1, 0, 0}});
            surface.push_back({{+gapHalf, y, z}, {-1, 0, 0}});
        }
    std::cout << "surface samples: " << surface.size() << "\n";

    Engine::Spatial::DirectionalTSDF tsdf;
    tsdf.Build(ctx, voxelSize, truncation);

    Engine::Spatial::SimpleTSDF simple;
    simple.Build(ctx, voxelSize, truncation, 1u << 20, 1u << 15);

    std::ofstream csv("directional_tsdf_stats.csv");
    csv << "frame,points,resident,missing,overlapPct,h2dKB,writeBack,"
           "beginMs,ensureMs,integrateMs,extractMs,mergeMs,cloudPoints\n";

    for (int f = 0; f < numFrames; ++f) {
        const float camZ = -5.0f + 0.25f * float(f);
        const Eigen::Vector3f cam(0.0f, 2.5f, camZ);

        // Frame patch: samples inside the scanner footprint.
        std::vector<Eigen::Vector3f> points, normals;
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        for (const auto &s : surface) {
            if (std::fabs(s.p.z() - camZ) > footprint) continue;
            points.push_back(s.p);
            normals.push_back(s.n);
            centroid += s.p;
        }
        if (points.empty()) continue;
        centroid /= float(points.size());

        tsdf.Integrate(points, normals, cam, centroid);
        simple.Integrate(points, cam);

        const auto st = tsdf.LastFrameStats();
        csv << f << ',' << points.size() << ',' << st.residentCount << ','
            << st.missingCount << ',' << st.overlapRatio * 100.0f << ','
            << st.h2dBytes / 1024 << ',' << st.writeBackCount << ','
            << st.beginFrameMs << ',' << st.ensureResidentMs << ','
            << st.integrateMs << ',' << st.extractMs << ',' << st.mergeMs << ','
            << tsdf.PointCloud().size() << '\n';

        std::cout << "frame " << f << ": pts=" << points.size()
                  << " missing=" << st.missingCount
                  << " overlap=" << st.overlapRatio * 100.0f << "%"
                  << " h2d=" << st.h2dBytes / 1024 << "KB"
                  << " cloud=" << tsdf.PointCloud().size() << "\n";
    }

    tsdf.ExportPointCloud("directional_result.ply");
    simple.ExportMC("simple_result_mc.ply");
    std::cout << "wrote directional_tsdf_stats.csv, directional_result.ply, "
                 "simple_result_mc.ply\n";
    return 0;
}
```

- [ ] **Step 2: CMake 등록**

`example2/CMakeLists.txt`의 `voxel_tsdf_mc` 블록 다음에 추가:

```cmake
add_executable(directional_tsdf_demo directional_tsdf_demo.cpp)
target_link_libraries(directional_tsdf_demo PRIVATE Engine::Spatial)
target_compile_definitions(directional_tsdf_demo PRIVATE VKBVH_SHADER_DIR=\"${VKBVH_SHADER_DIR}\")
```

- [ ] **Step 3: 빌드 + 실행 + 결과 확인**

```bash
export VULKAN_SDK=/Users/sjy/VulkanSDK/1.4.328.1/macOS
cmake -S . -B build -DVULKAN_SDK="$VULKAN_SDK"
cmake --build build --target directional_tsdf_demo -j"$(sysctl -n hw.ncpu)"
cd build/example2 && ./directional_tsdf_demo
```
Expected:
- 프레임 로그에서 첫 프레임 missing이 크고(전부 신규), 이후 프레임의 overlap%가 높게(≈80~95%) 유지 — 문서의 steady-state 주장 재현.
- `directional_result.ply`의 점들이 x≈±0.2 두 면으로 분리 유지 (검사: `awk`로 x 좌표 분포 확인 가능).
- `directional_tsdf_stats.csv` 생성.

```bash
# x 좌표 분포 sanity check: 두 클러스터(-0.2 근처 / +0.2 근처)가 모두 존재해야 한다
# (PLY 헤더는 10줄: ply/format/element/property×6/end_header)
awk 'NR>10 { if ($1 < 0) neg++; else pos++ } END { print "x<0:", neg, " x>0:", pos }' directional_result.ply
```
Expected: 양쪽 모두 수백 개 이상.

- [ ] **Step 4: Commit**

```bash
git add example2/directional_tsdf_demo.cpp example2/CMakeLists.txt
git commit -m "Add interproximal DirectionalTSDF demo with CSV streaming stats"
```

---

## Phase 3 completion checklist

- [ ] 단일 평면 적분 → 부호 있는 band + dirty 마킹 (WriteBackList 관찰로 검증)
- [ ] 단일 평면 추출 → 표면 위치/normal 정확
- [ ] 마주보는 두 면(band 겹침 구간 존재)이 direction layer 분리로 모두 보존 — 문서 §2 핵심 주장 검증
- [ ] 재적분 시 old point 교체(중복 누적 없음, invariant #9)
- [ ] 데모: overlap%/missing/h2dKB/타이밍 CSV + 두 면 분리 PLY + SimpleTSDF 비교 산출물
- [ ] 전체 스위트 75/74 유지
- [ ] Phase 4 계획(dirty write-back/eviction) 착수 전 리뷰
