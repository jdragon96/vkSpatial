# AdaptiveVoxelGrid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Runtime 2-level variance-adaptive single-field TSDF grid: fine GPU integration (compose `SimpleTSDF`) → CPU variance merge → multi-resolution CPU Marching Cubes mesh.

**Architecture:** `Engine::Spatial::AdaptiveVoxelGrid` owns one `SimpleTSDF m_fine`. `DownloadVoxels()` feeds a CPU variance-merge that builds a mixed-resolution voxel set; a CPU multi-res Marching Cubes (tables transcribed from `voxel_common.glsl`) extracts the mesh.

**Tech Stack:** C++17, Eigen, Vulkan compute (via `SimpleTSDF`), gtest.

## Global Constraints
- Single-field TSDF; **2-level** only (fine `h`, coarse `2h`); coarse block = `2×2×2` fine voxels.
- Variance source = `SimpleTSDF::DownloadVoxels().variance` (= `E[d²]−E[d]²`, ≡ paper Welford for `w=1`). Do NOT re-derive on GPU.
- Merge is **CPU**; MC is **CPU**. No new GPU shader.
- Coarse voxel value = weight-averaged fine value `Σ(W_i D_i)/Σ W_i`.
- Fine voxel integer coord recovered from center: `v = lround(center/h − 0.5)`; coarse coord = `floorDiv(v, 2)` per axis (floor division, correct for negatives).
- Build env: `VULKAN_SDK=/usr/local`; configure `cmake -S . -B build -DGTest_DIR=/opt/homebrew/lib/cmake/GTest -Dgflags_DIR=/opt/homebrew/lib/cmake/gflags`; build+run **`vkspatial_tests`**. macOS has no `timeout`. Run cmake/build as SEPARATE simple commands (rtk hook mangles `tail` in chains).
- Fixtures: `example2/shape_fixtures.h` — `fixtures::SampleViews(Shape::Cube|Cylinder, voxel)` → `vector<fixtures::View>` with `.points`, `.camPos`; region classify helper used by `test/test_simpletsdf_variance.cpp` (reuse its edge/flat classification pattern).

---

### Task 1: Scaffold AdaptiveVoxelGrid (compose SimpleTSDF) + CMake + fine passthrough

**Files:**
- Create: `src/Engine/Spatial/AdaptiveVoxelGrid.h`, `src/Engine/Spatial/AdaptiveVoxelGrid.cpp`
- Modify: `src/Engine/CMakeLists.txt` (add `Spatial/AdaptiveVoxelGrid.cpp` to the `EngineSpatial` sources — find the list containing `Spatial/SimpleTSDF.cpp` and add alongside)
- Create: `test/test_adaptiveVoxelGrid.cpp`; Modify: `test/CMakeLists.txt` (add `test_adaptiveVoxelGrid.cpp` to the `vkspatial_tests` sources — mirror how `test_simpletsdf_variance.cpp` is listed)

**Interfaces:**
- Produces: `class AdaptiveVoxelGrid { void Build(Engine::Core::Context&, float fineVoxelSize, float truncation, uint32_t hashCapacity=1u<<20, uint32_t maxPoints=1u<<17); void Integrate(const std::vector<Eigen::Vector3f>&, const Eigen::Vector3f& cameraPos=Eigen::Vector3f::Zero()); void Reset(); size_t FineCount() const; };` internally `SimpleTSDF m_fine; float m_h; float m_trunc;`

- [ ] **Step 1: Write the failing test** — `test/test_adaptiveVoxelGrid.cpp`

```cpp
#include "Engine/Spatial/AdaptiveVoxelGrid.h"
#include "Engine/Spatial/SimpleTSDF.h"
#include "Engine/Core/Context.h"
#include "../example2/shape_fixtures.h"
#include <gtest/gtest.h>

using Engine::Spatial::AdaptiveVoxelGrid;
namespace { const float kVoxel = 0.05f, kTrunc = 0.15f; }

TEST(AdaptiveVoxelGrid, FineMatchesSimpleTSDF) {
    Engine::Core::Context ctx;                       // skip pattern: see test_simpletsdf_variance.cpp if Vulkan unavailable
    const auto views = fixtures::SampleViews(Shape::Cube, kVoxel);
    ASSERT_FALSE(views.empty());

    Engine::Spatial::SimpleTSDF ref; ref.Build(ctx, kVoxel, kTrunc);
    for (const auto& v : views) ref.Integrate(v.points, v.camPos);

    AdaptiveVoxelGrid avg; avg.Build(ctx, kVoxel, kTrunc);
    for (const auto& v : views) avg.Integrate(v.points, v.camPos);

    EXPECT_EQ(avg.FineCount(), size_t(ref.FilledCount()));
}
```

- [ ] **Step 2: Run test to verify it fails** — `VULKAN_SDK=/usr/local cmake --build build --target vkspatial_tests` (expect: compile error, AdaptiveVoxelGrid.h missing)

- [ ] **Step 3: Implement `AdaptiveVoxelGrid.h`**

```cpp
#pragma once
#include "Engine/Spatial/SimpleTSDF.h"
#include <Eigen/Core>
#include <vector>
namespace Engine::Spatial {
    struct MixedVoxel { Eigen::Vector3f center; float tsdf; float weight; float size; uint8_t level; };
    struct AdaptiveMesh {
        std::vector<Eigen::Vector3f> vertices;
        std::vector<Eigen::Vector3i> triangles;
        std::vector<Eigen::Vector3f> normals;
    };
    class AdaptiveVoxelGrid {
    public:
        void Build(Engine::Core::Context& ctx, float fineVoxelSize, float truncation,
                   uint32_t hashCapacity = 1u<<20, uint32_t maxPoints = 1u<<17);
        void Integrate(const std::vector<Eigen::Vector3f>& points,
                       const Eigen::Vector3f& cameraPos = Eigen::Vector3f::Zero());
        void Reset();
        void SetVarianceThreshold(float sigma2);      // Task 2
        void SetVariancePercentile(float p);          // Task 2
        void SetMinOccupancy(uint32_t n);             // Task 2
        std::vector<MixedVoxel> DownloadMixedVoxels();// Task 2
        size_t FineCount() const;                     // Task 1: fine voxels
        size_t CoarseCount() const;                   // Task 2
        AdaptiveMesh ExtractMesh();                   // Task 4
    private:
        SimpleTSDF m_fine;
        float m_h = 0.05f, m_trunc = 0.15f;
        float m_threshold = -1.0f;   // <0 => percentile mode
        float m_percentile = 0.5f;
        uint32_t m_minOcc = 4;
        void buildMixed();           // Task 2
        std::vector<MixedVoxel> m_mixed;   // Task 2 cache
        bool m_mixedDirty = true;
        size_t m_fineCount = 0, m_coarseCount = 0;
    };
}
```

- [ ] **Step 4: Implement `AdaptiveVoxelGrid.cpp` (Task-1 subset)** — `Build`/`Integrate`/`Reset` delegate to `m_fine`; `FineCount()` returns `m_fine.FilledCount()` for now (Task 2 replaces with mixed count). Stub the Task-2/4 methods to `{}`/`return {};` with a `// Task N` note.

- [ ] **Step 5: Run test to verify it passes** — build + `VULKAN_SDK=/usr/local ./build/test/vkspatial_tests --gtest_filter='AdaptiveVoxelGrid.*'` (expect PASS)

- [ ] **Step 6: Commit** — `git add src/Engine/Spatial/AdaptiveVoxelGrid.* src/Engine/CMakeLists.txt test/test_adaptiveVoxelGrid.cpp test/CMakeLists.txt && git commit -m "feat(spatial): AdaptiveVoxelGrid scaffold composing SimpleTSDF (fine passthrough)"`

---

### Task 2: Variance merge → mixed-resolution grid

**Files:** Modify `src/Engine/Spatial/AdaptiveVoxelGrid.{h,cpp}` (implement `buildMixed`, `DownloadMixedVoxels`, `Set*`, `FineCount`/`CoarseCount`), `test/test_adaptiveVoxelGrid.cpp` (add tests).

**Interfaces:**
- Consumes: `m_fine.DownloadVoxels()` → `vector<VoxelStat{center,tsdf,weight,variance}>`.
- Produces: `m_mixed` (vector<MixedVoxel>), `m_fineCount`/`m_coarseCount`; `FineCount()`/`CoarseCount()` read them after `buildMixed()`.

- [ ] **Step 1: Write failing tests** — three tests:

```cpp
static void integrateCube(Engine::Core::Context& ctx, AdaptiveVoxelGrid& a, float voxel) {
    for (const auto& v : fixtures::SampleViews(Shape::Cube, voxel)) a.Integrate(v.points, v.camPos);
}
TEST(AdaptiveVoxelGrid, CoarsensFlatKeepsMemoryLower) {
    Engine::Core::Context ctx;
    AdaptiveVoxelGrid a; a.Build(ctx, 0.05f, 0.15f);
    integrateCube(ctx, a, 0.05f);
    a.SetVariancePercentile(0.6f);                     // coarsen the low-variance 60%
    const auto mixed = a.DownloadMixedVoxels();
    ASSERT_FALSE(mixed.empty());
    EXPECT_GT(a.CoarseCount(), 0u);                    // some flat blocks coarsened
    EXPECT_LT(a.FineCount() + a.CoarseCount(),          // total voxels reduced vs all-fine
              /*all-fine*/ [&]{ Engine::Spatial::SimpleTSDF s; s.Build(ctx,0.05f,0.15f);
                  integrateCubeSimple(ctx,s,0.05f); return size_t(s.FilledCount()); }());
}
TEST(AdaptiveVoxelGrid, CoarseVoxelsAreFlat) {   // coarse voxels have low variance by construction
    Engine::Core::Context ctx; AdaptiveVoxelGrid a; a.Build(ctx,0.05f,0.15f);
    integrateCube(ctx,a,0.05f); a.SetVariancePercentile(0.5f);
    for (const auto& mv : a.DownloadMixedVoxels()) if (mv.level==1) EXPECT_FLOAT_EQ(mv.size, 0.10f);
    SUCCEED();
}
```
(Add a small `integrateCubeSimple` helper mirroring `integrateCube` for `SimpleTSDF`.)

- [ ] **Step 2: Run to verify failure** — build+run filter (expect FAIL: CoarseCount 0 / mixed empty).

- [ ] **Step 3: Implement `buildMixed()`**

```cpp
// pseudo: recover fine coords, bucket into 2^3 blocks, decide fine/coarse
void AdaptiveVoxelGrid::buildMixed() {
    const auto vox = m_fine.DownloadVoxels();
    // 1. recover integer fine coord + choose threshold
    auto vcoord=[&](const Eigen::Vector3f& c){ return Eigen::Vector3i(
        (int)std::lround(c.x()/m_h-0.5f),(int)std::lround(c.y()/m_h-0.5f),(int)std::lround(c.z()/m_h-0.5f)); };
    float theta = m_threshold;
    if (theta < 0.f) { std::vector<float> s; s.reserve(vox.size());
        for (auto& v: vox) s.push_back(v.variance); std::sort(s.begin(),s.end());
        theta = s.empty()?0.f:s[std::min(s.size()-1,(size_t)(m_percentile*s.size()))]; }
    // 2. bucket by coarse coord (floorDiv by 2)
    auto fdiv=[](int a,int b){ int q=a/b, r=a%b; return (r!=0 && ((r<0)!=(b<0)))? q-1: q; };
    struct Blk{ std::vector<size_t> idx; }; std::map<std::array<int,3>,Blk> blocks;
    std::vector<Eigen::Vector3i> vc(vox.size());
    for (size_t i=0;i<vox.size();++i){ vc[i]=vcoord(vox[i].center);
        blocks[{fdiv(vc[i].x(),2),fdiv(vc[i].y(),2),fdiv(vc[i].z(),2)}].idx.push_back(i); }
    // 3. decide per block
    m_mixed.clear(); m_fineCount=0; m_coarseCount=0;
    for (auto& [ck,blk]: blocks) {
        double vs=0; float wsum=0, dwsum=0;
        for (size_t i: blk.idx){ vs+=vox[i].variance; wsum+=vox[i].weight; dwsum+=vox[i].weight*vox[i].tsdf; }
        const float meanVar = vs/blk.idx.size();
        if (blk.idx.size()>=m_minOcc && meanVar < theta) {   // coarse
            Eigen::Vector3f cc((ck[0]*2+1)*m_h,(ck[1]*2+1)*m_h,(ck[2]*2+1)*m_h);  // center of 2^3 block
            m_mixed.push_back({cc, dwsum/std::max(wsum,1e-6f), wsum, 2.0f*m_h, 1}); m_coarseCount++;
        } else for (size_t i: blk.idx){                       // keep fine
            m_mixed.push_back({vox[i].center, vox[i].tsdf, vox[i].weight, m_h, 0}); m_fineCount++;
        }
    }
    m_mixedDirty=false;
}
```
`DownloadMixedVoxels()`/`FineCount()`/`CoarseCount()` call `buildMixed()` if `m_mixedDirty`, then return the members. `Set*`/`Integrate`/`Reset` set `m_mixedDirty=true`. `FineCount()` in Task 1 test still passes when no `Set*` called + default percentile: to keep Task-1 semantics, have `FineCount()` trigger `buildMixed()`; the cube at 0.05 with default percentile 0.5 will coarsen some — SO update the Task-1 test to call `a.SetVarianceThreshold(0.0f)` (nothing coarsens → all fine) before asserting equality with SimpleTSDF. Make that one-line edit to the Task-1 test in this task.

- [ ] **Step 4: Run tests to verify pass** — build+run filter (expect PASS all 3 + the adjusted Task-1 test).

- [ ] **Step 5: Commit** — `feat(spatial): AdaptiveVoxelGrid variance merge -> mixed-resolution grid`

---

### Task 3: CPU Marching Cubes tables + single-resolution CPU MC (validated vs GPU MC)

**Files:** Create `src/Engine/Spatial/MarchingCubesTables.h`; Modify `AdaptiveVoxelGrid.cpp` (add a static/file-local `cpuMarchingCubes` over a uniform grid) + `test_adaptiveVoxelGrid.cpp`.

**Interfaces:**
- Produces: `namespace mc { extern const int CORNER[8][3]; extern const int INV_CORNER[8]; extern const int edgeTable[256]; extern const int triTable[4096]; }` and a file-local `AdaptiveMesh cpuMarchingCubes(const std::function<bool(int,int,int,float&)>& sample, float voxelSize)` iterating cells over the sampled voxel set.

- [ ] **Step 1: Transcribe tables** — copy `CORNER[8]` (as `int[8][3]`), `INV_CORNER[8]`, `edgeTable[256]`, `triTable[4096]` VERBATIM from `src/shader/voxel_common.glsl` (lines ~30–…) into `MarchingCubesTables.h` as `constexpr int`, converting GLSL `ivec3[8](...)`/`int[N](...)` to C++ `[N]`/`[N][3]` initializers. No value changes.

- [ ] **Step 2: Write failing test** — CPU-MC-vs-GPU-MC equivalence:

```cpp
TEST(AdaptiveVoxelGrid, CpuMcMatchesGpuMcAllFine) {
    Engine::Core::Context ctx; AdaptiveVoxelGrid a; a.Build(ctx,0.05f,0.15f);
    integrateCube(ctx,a,0.05f);  a.SetVarianceThreshold(0.0f);   // force all-fine
    const AdaptiveMesh m = a.ExtractMesh();
    Engine::Spatial::SimpleTSDF s; s.Build(ctx,0.05f,0.15f); integrateCubeSimple(ctx,s,0.05f);
    const auto ref = s.ExtractPointCloud().points;   // GPU MC vertices
    ASSERT_FALSE(m.vertices.empty()); ASSERT_FALSE(ref.empty());
    // one-directional Chamfer: every CPU vertex has a close GPU vertex
    const double d = maxNearestDist(m.vertices, ref);   // helper: max over m of min dist to ref
    EXPECT_LT(d, 0.05f);   // within one fine voxel
}
```
(Add `maxNearestDist` brute-force helper in the test; cube MC vertex count is small.)

- [ ] **Step 3: Run to verify failure** — `ExtractMesh` still a stub → empty vertices → FAIL.

- [ ] **Step 4: Implement `cpuMarchingCubes` + wire an all-fine `ExtractMesh`** — build a hash map `(vx,vy,vz)->tsdf` from the fine mixed voxels; for each fine cell iterate 8 `CORNER` offsets, `getTSDF` from the map (skip cell if any corner missing), `cubeIndex` from `sdf[c]<0`, `edgeTable`/`triTable` lookup, linear edge interpolation `p = a + t*(b-a)`, `t=sdf_a/(sdf_a-sdf_b)`; accumulate triangles; per-vertex normals = area-weighted face normals (mirror `SimpleTSDF::ExtractPointCloud`'s welding+normal step). For Task 3, `ExtractMesh` runs `buildMixed()` then MC over **fine voxels only** (coarse handled in Task 4).

- [ ] **Step 5: Run test to verify pass** — expect PASS (CPU MC ≈ GPU MC on all-fine cube).

- [ ] **Step 6: Commit** — `feat(spatial): CPU marching cubes (tables from voxel_common.glsl) validated vs GPU MC`

---

### Task 4: Multi-resolution ExtractMesh + transitional-voxel handling

**Files:** Modify `AdaptiveVoxelGrid.cpp` (extend `cpuMarchingCubes`/`ExtractMesh` to mixed resolution) + `test_adaptiveVoxelGrid.cpp`.

**Interfaces:**
- Consumes: `m_mixed` (fine `level==0` + coarse `level==1`), `MarchingCubesTables`.
- Produces: `ExtractMesh()` over the mixed grid with transitional handling; vertices welded within `0.25*m_h`.

- [ ] **Step 1: Write failing tests** — accuracy + transitional integrity:

```cpp
TEST(AdaptiveVoxelGrid, MeshAccuracyVsAllFine) {
    Engine::Core::Context ctx;
    AdaptiveVoxelGrid fineOnly; fineOnly.Build(ctx,0.05f,0.15f); integrateCube(ctx,fineOnly,0.05f);
    fineOnly.SetVarianceThreshold(0.0f); const auto refM = fineOnly.ExtractMesh();
    AdaptiveVoxelGrid adp;    adp.Build(ctx,0.05f,0.15f); integrateCube(ctx,adp,0.05f);
    adp.SetVariancePercentile(0.6f);      const auto m = adp.ExtractMesh();
    ASSERT_GT(adp.CoarseCount(), 0u);
    EXPECT_LT(maxNearestDist(m.vertices, refM.vertices), 0.10f);  // within one coarse voxel
}
TEST(AdaptiveVoxelGrid, NoDegenerateOrDuplicateVerts) {
    Engine::Core::Context ctx; AdaptiveVoxelGrid a; a.Build(ctx,0.05f,0.15f);
    integrateCube(ctx,a,0.05f); a.SetVariancePercentile(0.6f); const auto m = a.ExtractMesh();
    for (const auto& t : m.triangles) {                          // no zero-area triangles
        const auto& A=m.vertices[t.x()]; const auto& B=m.vertices[t.y()]; const auto& C=m.vertices[t.z()];
        EXPECT_GT((B-A).cross(C-A).norm(), 1e-9f);
    }
    // no two welded vertices closer than the collapse epsilon (0.25*h) except identical
    // (spot-check via a grid hash; see helper).
    EXPECT_TRUE(noNearDuplicates(m.vertices, 0.25f*0.05f));
}
```

- [ ] **Step 2: Run to verify failure** — coarse cells not yet meshed / transitional cracks → FAIL (accuracy or duplicates).

- [ ] **Step 3: Implement multi-res MC** — extend `ExtractMesh`:
  - Sample function `getSDF(level, cx,cy,cz)`: look up value at the requested level; **finer-favoring** — when a coarse-cell corner coincides with a fine region, prefer the fine value; when a fine-cell corner falls in a coarse region, sample the coarse voxel value (nearest coarse center).
  - Iterate fine cells over fine voxels and coarse cells over coarse voxels (own resolution), each via `cpuMarchingCubes` at that cell size.
  - **Transitional cells** at a fine↔coarse boundary: for a coarse cell adjacent to finer neighbors, **truncate** the coarse cell along the shared face (do not emit MC geometry into the sub-region already covered by fine cells) — implement by skipping the coarse cell's contribution on faces whose neighbor coarse-coord is fine-occupied, per Fig 6.
  - **Vertex collapse:** weld all emitted vertices within `0.25*m_h` (grid-snap hash) before building the final index list; recompute area-weighted normals after welding.

- [ ] **Step 4: Run tests to verify pass** — expect PASS (accuracy within one coarse voxel; no degenerate/duplicate). If transitional cracks persist, apply the spec §8 fallback (uniformly refine boundary coarse cells to fine) and note it in the commit.

- [ ] **Step 5: Commit** — `feat(spatial): multi-resolution marching cubes with transitional-voxel handling`

---

### Task 5: adaptive_voxel_grid_demo (measured memory + accuracy)

**Files:** Create `example2/adaptive_voxel_grid_demo.cpp`; Modify `example2/CMakeLists.txt` (`add_spatial_example(adaptive_voxel_grid_demo adaptive_voxel_grid_demo.cpp)` + `target_link_libraries(... PRIVATE Engine::Spatial Engine::Core)`).

**Interfaces:** Consumes the full `AdaptiveVoxelGrid` API.

- [ ] **Step 1: Write the demo** — integrate cube AND cylinder (`fixtures::SampleViews`), sweep 3 percentiles (0.3/0.5/0.7); print per shape: `FineCount`, `CoarseCount`, total-voxel reduction vs all-fine `SimpleTSDF::FilledCount`, and `maxNearestDist(mesh, allFineMesh)` (accuracy). Export `adaptive_<shape>_<pct>.ply` (write `AdaptiveMesh` as an OBJ/PLY with faces).

- [ ] **Step 2: Build** — `VULKAN_SDK=/usr/local cmake -S . -B build ...` then `--build build --target adaptive_voxel_grid_demo` (reconfigure needed for the new target).

- [ ] **Step 3: Run + record** — `VULKAN_SDK=/usr/local ./build/example2/adaptive_voxel_grid_demo`; paste the memory-reduction + accuracy table into the commit body. Acceptance: reduction > 1× with accuracy within one coarse voxel (else report honestly).

- [ ] **Step 4: Commit** — `feat(example2): adaptive_voxel_grid_demo — measured memory reduction + mesh accuracy`

---

## Self-Review
- **Spec coverage:** §3 data model → Task 1/2; §2 variance+merge → Task 2; §5 multi-res MC + transitional → Task 3 (CPU MC foundation) + Task 4; §7 tests 1-5 → Task 2 (1-3), Task 3 (4 baseline), Task 4 (4 accuracy + 5 integrity); §6 API → Tasks 1/2/4; measurement → Task 5.
- **Placeholders:** none — each step has concrete code or a precise transcription/mechanical instruction.
- **Type consistency:** `MixedVoxel{center,tsdf,weight,size,level}` and `AdaptiveMesh{vertices,triangles,normals}` defined Task 1, used identically Tasks 2/4/5; `buildMixed`/`m_mixed`/`m_fineCount`/`m_coarseCount` defined Task 1 header, implemented Task 2; MC tables namespace `mc::` defined Task 3, used Task 3/4.
- **Ordering note:** Task 1's equality test needs `SetVarianceThreshold(0.0f)` once Task 2 lands (the plan states the one-line edit in Task 2 Step 3).
