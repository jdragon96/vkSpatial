# Isosurface Extraction Strategies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A strategy-pattern isosurface-extraction framework (mirroring `Tracker`/`TrackerRegistry`) with the classical Marching-Cubes family (`mc`, `mc33`, `mtet`, `emc`, `dc`, `dmc`, `cms`) as pluggable, name-switchable CPU extractors over a shared `VoxelField`.

**Architecture:** New `Engine::Spatial::Extraction` unit: `VoxelField` (sparse signed field + optional stored gradient), `SurfaceMesh` (verts/tris/normals), `IsoSurfaceExtractor` (abstract), `ExtractorRegistry` (name→factory, `Default()`), and a shared `MarchingCubesCore` extracted from the primitives currently private to `AdaptiveVoxelGrid.cpp`. Each algorithm is one phase = one file + registration + its test.

**Tech Stack:** C++17, Eigen, GoogleTest (`vkspatial_tests`), macOS/MoltenVK (CPU-only here, no shaders).

## Global Constraints

- Full descriptive names, NO abbreviations in all new/edited code (`triangleTable`, `edgeCornerPairs`, `cubeIndex` — not `tt`/`ec`/`ci`). Existing transcribed tables in `MarchingCubesTables.h` keep their published names, with a provenance comment.
- CPU-only. No shader changes. Do NOT modify `src/shader/*` or `voxel_tsdf_mc.comp`.
- Do NOT touch the user's uncommitted WIP: `src/Engine/Spatial/AdvancedTSDF.cpp`, `src/shader/advanced_tsdf_compact.comp.glsl`, `src/shader/advanced_tsdf_integrate.comp.glsl`, `src/Engine/Pipeline/Registration/RegistrationThread.cpp`, and the `lib/SPIRV-Reflect` submodule. Each task `git add`s ONLY its own files — never `git add .` / `git add -A`.
- Build/run the `vkspatial_tests` target. Adding a new `.cpp` requires a CMake reconfigure (both `src/Engine/CMakeLists.txt` and `test/CMakeLists.txt` use `file(GLOB ...)`, evaluated at configure time): run `cmake -S . -B build -DGTest_DIR=/opt/homebrew/lib/cmake/GTest -Dgflags_DIR=/opt/homebrew/lib/cmake/gflags -Dglog_DIR=/opt/homebrew/lib/cmake/glog` then `cmake --build build --target vkspatial_tests -j`. NEVER `rm -rf build`.
- Commit trailer on every commit: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Do NOT push to origin.
- Lookup tables transcribed from a published source carry a comment citing that source and a note that one wrong entry silently corrupts the mesh (as `MarchingCubesTables.h` does).
- Baseline suite is green (241 pass / 1 skip / 0 fail). It must stay green; the `AdaptiveVoxelGrid` bit-exact test (`test_adaptiveVoxelGrid.cpp`) is the regression guard for the Task-1 core extraction.

## Reference: existing code to reuse (read before Task 1)

- `src/Engine/Spatial/MarchingCubesTables.h` — `Engine::Spatial::mc`: `CORNER[8][3]`, `INV_CORNER[8]`, `edgeTable[256]`, `triTable[4096]`.
- `src/Engine/Spatial/AdaptiveVoxelGrid.cpp` anonymous namespace (lines ~15–239): `IVec3Hash`, `using VoxelValueMap = std::unordered_map<std::array<int,3>, float, IVec3Hash>`, `vertInterp(p1,p2,v1,v2)`, `kEdgeCorners[12][2]`, `struct RawTri { Eigen::Vector3f a,b,c; }`, `generateRawTriangles(bases, sampleAt, cellSize, out)` (template on sampler `bool(const std::array<int,3>&, float&)`), `candidateBases(VoxelValueMap)`, `weldAndNormal(tris, weld)` → `AdaptiveMesh`.
- `src/Engine/Spatial/AdaptiveVoxelGrid.h` — `struct AdaptiveMesh { std::vector<Eigen::Vector3f> vertices; std::vector<Eigen::Vector3i> triangles; std::vector<Eigen::Vector3f> normals; }`, `AdaptiveMesh ExtractMesh();`.
- `src/Engine/Eval/RmseMetrics.h` — `float Engine::Eval::NearestNeighbourRMSE(const std::vector<Eigen::Vector3f>& from, const std::vector<Eigen::Vector3f>& to)`.
- `src/Engine/Spatial/AdvancedTSDF.h` — `struct AdvancedEntry { Eigen::Vector3f center; uint32_t direction; float tsdf; float weight; Eigen::Vector3f normal; int32_t firstFrame; }`.
- Pattern to mirror: `src/Engine/Pipeline/Registration/Tracker.h` (abstract + `TrackerRegistry`) and `TrackerRegistry.cpp` (`Default()`).

## File Structure

```
src/Engine/Spatial/Extraction/
  SurfaceMesh.h              SurfaceMesh struct (verts/tris/normals)
  VoxelField.h / .cpp        sparse signed field + optional gradient; adapters
  MarchingCubesCore.h / .cpp shared: IVec3Hash, VoxelValueMap, RawTri, vertInterp, kEdgeCornerPairs,
                             candidateBases, generateRawTriangles (template), weldAndNormal
  IsoSurfaceExtractor.h      abstract strategy
  ExtractorRegistry.h / .cpp Register/Create/Has/Default
  MarchingCubesExtractor.cpp          "mc"     (Task 1)
  MarchingCubes33Tables.h             (Task 2, transcribed)
  MarchingCubes33Extractor.cpp        "mc33"   (Task 2)
  MarchingTetrahedraExtractor.cpp     "mtet"   (Task 3)
  ExtendedMarchingCubesExtractor.cpp  "emc"    (Task 4)
  DualContouringExtractor.cpp         "dc"     (Task 5)
  DualMarchingCubesExtractor.cpp      "dmc"    (Task 6)
  CubicalMarchingSquaresExtractor.cpp "cms"    (Task 7)
  QuadraticErrorFunction.h / .cpp     shared QEF solver (Task 4, reused by dc/dmc/cms)
test/
  isosurface_test_util.h     analytic VoxelField builders + mesh predicates (Task 1)
  test_isosurface_mc.cpp     Task 1
  test_isosurface_mc33.cpp   Task 2
  test_isosurface_mtet.cpp   Task 3
  test_isosurface_emc.cpp    Task 4
  test_isosurface_dc.cpp     Task 5
  test_isosurface_dmc.cpp    Task 6
  test_isosurface_cms.cpp    Task 7
docs/ISOSURFACE_EXTRACTION.md   companion doc (Task 7, final)
```

---

### Task 1: Framework + shared core + `"mc"`

**Files:**
- Create: `src/Engine/Spatial/Extraction/SurfaceMesh.h`, `VoxelField.h`, `VoxelField.cpp`, `MarchingCubesCore.h`, `MarchingCubesCore.cpp`, `IsoSurfaceExtractor.h`, `ExtractorRegistry.h`, `ExtractorRegistry.cpp`, `MarchingCubesExtractor.cpp`
- Create: `test/isosurface_test_util.h`, `test/test_isosurface_mc.cpp`
- Modify: `src/Engine/Spatial/AdaptiveVoxelGrid.h` (make `AdaptiveMesh` an alias of `SurfaceMesh`), `src/Engine/Spatial/AdaptiveVoxelGrid.cpp` (call shared core instead of the private primitives)

**Interfaces:**
- Produces (later tasks consume these EXACT signatures):
  - `namespace Engine::Spatial::Extraction`
  - `struct SurfaceMesh { std::vector<Eigen::Vector3f> vertices; std::vector<Eigen::Vector3i> triangles; std::vector<Eigen::Vector3f> normals; };`
  - Coordinates are `std::array<int,3>` (grid indices), matching the shared core.
  - `class VoxelField` with: `void Insert(const std::array<int,3>& coord, float value);` · `void Insert(const std::array<int,3>& coord, float value, const Eigen::Vector3f& gradient);` · `bool Sample(const std::array<int,3>& coord, float& outValue) const;` · `bool Gradient(const std::array<int,3>& coord, Eigen::Vector3f& outNormal) const;` · `bool HasGradients() const;` · `float CellSize() const;` · `void SetCellSize(float);` · `const std::vector<std::array<int,3>>& OccupiedCoords() const;`
  - Free adapters (in `VoxelField.h`): `VoxelField FromImplicit(const std::array<int,3>& minCoord, const std::array<int,3>& maxCoord, float cellSize, const std::function<float(const Eigen::Vector3f&)>& valueFunction, const std::function<Eigen::Vector3f(const Eigen::Vector3f&)>* gradientFunction = nullptr);` and `VoxelField FromAdvancedEntries(const std::vector<Engine::Spatial::AdvancedEntry>& entries, float cellSize);`
  - `MarchingCubesCore.h` (namespace `Engine::Spatial::Extraction::core`): `IVec3Hash`, `using VoxelValueMap = std::unordered_map<std::array<int,3>, float, IVec3Hash>;`, `struct RawTriangle { Eigen::Vector3f a,b,c; };`, `Eigen::Vector3f VertexInterpolate(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, float value1, float value2);`, `extern const int kEdgeCornerPairs[12][2];`, `std::unordered_set<std::array<int,3>, IVec3Hash> CandidateBases(const std::vector<std::array<int,3>>& occupied);`, `template<typename SampleFunction> void GenerateRawTriangles(const std::unordered_set<std::array<int,3>, IVec3Hash>& bases, SampleFunction&& sampleAt, float cellSize, std::vector<RawTriangle>& out);`, `SurfaceMesh WeldAndComputeNormals(const std::vector<RawTriangle>& triangles, float weldDistance);`
  - `IsoSurfaceExtractor.h`: `struct ExtractParams { float isoLevel = 0.0f; float weldFraction = 0.25f; float featureAngleCosineThreshold = 0.9f; };` and `class IsoSurfaceExtractor { public: virtual ~IsoSurfaceExtractor() = default; virtual const char* Name() const = 0; virtual SurfaceMesh Extract(const VoxelField& field, const ExtractParams& params) const = 0; };`
  - `ExtractorRegistry.h`: `class ExtractorRegistry { public: using Factory = std::function<std::unique_ptr<IsoSurfaceExtractor>()>; void Register(const std::string& name, Factory factory); std::unique_ptr<IsoSurfaceExtractor> Create(const std::string& name) const; bool Has(const std::string& name) const; static ExtractorRegistry Default(); private: std::unordered_map<std::string, Factory> m_factories; };`
- Consumes: existing `mc::edgeTable`, `mc::triTable`, `mc::CORNER` from `MarchingCubesTables.h`; `Engine::Eval::NearestNeighbourRMSE`.

- [ ] **Step 1: Write the test utility** `test/isosurface_test_util.h`

```cpp
#pragma once
#include "Engine/Spatial/Extraction/VoxelField.h"
#include "Engine/Spatial/Extraction/SurfaceMesh.h"
#include <Eigen/Core>
#include <array>
#include <cmath>
#include <map>
#include <vector>

namespace isotest {
    using Engine::Spatial::Extraction::SurfaceMesh;
    using Engine::Spatial::Extraction::VoxelField;

    // Analytic sphere signed field (value = |p| - radius), exact gradient = p/|p|.
    inline VoxelField SphereField(float radius, float cellSize, int halfN) {
        const std::array<int,3> lo{-halfN,-halfN,-halfN}, hi{halfN,halfN,halfN};
        auto value = [radius](const Eigen::Vector3f& p){ return p.norm() - radius; };
        auto grad  = [](const Eigen::Vector3f& p){ float n=p.norm(); return n>1e-6f?Eigen::Vector3f(p/n):Eigen::Vector3f(0,0,1);} ;
        std::function<Eigen::Vector3f(const Eigen::Vector3f&)> g = grad;
        return Engine::Spatial::Extraction::FromImplicit(lo, hi, cellSize, value, &g);
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
        return Engine::Spatial::Extraction::FromImplicit(lo, hi, cellSize, value, &gg);
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
}
```

- [ ] **Step 2: Write the failing test** `test/test_isosurface_mc.cpp`

```cpp
#include "Engine/Spatial/Extraction/ExtractorRegistry.h"
#include "Engine/Eval/RmseMetrics.h"
#include "isosurface_test_util.h"
#include <gtest/gtest.h>

using namespace Engine::Spatial::Extraction;

TEST(Isosurface, RegistryCreatesMcAndRejectsUnknown) {
    auto reg = ExtractorRegistry::Default();
    EXPECT_TRUE(reg.Has("mc"));
    ASSERT_NE(reg.Create("mc"), nullptr);
    EXPECT_EQ(reg.Create("nonexistent"), nullptr);
    EXPECT_STREQ(reg.Create("mc")->Name(), "mc");
}

TEST(Isosurface, MarchingCubesSphereIsAccurateAndManifold) {
    const float cellSize = 0.1f, radius = 0.7f;
    VoxelField field = isotest::SphereField(radius, cellSize, 12);
    auto mc = ExtractorRegistry::Default().Create("mc");
    SurfaceMesh mesh = mc->Extract(field, ExtractParams{});
    ASSERT_GT(mesh.triangles.size(), 100u);
    EXPECT_TRUE(isotest::IsEdgeManifold(mesh));
    EXPECT_TRUE(isotest::IsWatertight(mesh));               // closed surface
    // accuracy: every vertex lies within ~one cell of the true sphere
    std::vector<Eigen::Vector3f> onSphere;
    for (const auto& v : mesh.vertices) onSphere.push_back(v.normalized() * radius);
    EXPECT_LT(Engine::Eval::NearestNeighbourRMSE(mesh.vertices, onSphere), cellSize);
}
```

- [ ] **Step 3: Run to verify it fails** — Reconfigure + build; expect COMPILE failure (headers/types absent).
Run: `cmake -S . -B build -DGTest_DIR=/opt/homebrew/lib/cmake/GTest -Dgflags_DIR=/opt/homebrew/lib/cmake/gflags -Dglog_DIR=/opt/homebrew/lib/cmake/glog && cmake --build build --target vkspatial_tests -j`
Expected: FAIL to compile (`Extraction/ExtractorRegistry.h` not found).

- [ ] **Step 4: Implement `SurfaceMesh.h`**

```cpp
#pragma once
#include <Eigen/Core>
#include <vector>
namespace Engine::Spatial::Extraction {
    struct SurfaceMesh {
        std::vector<Eigen::Vector3f> vertices;
        std::vector<Eigen::Vector3i> triangles;   // vertex indices
        std::vector<Eigen::Vector3f> normals;      // per-vertex
    };
}
```

- [ ] **Step 5: Implement `MarchingCubesCore.{h,cpp}`** — MOVE (do not modify the logic of) `IVec3Hash`, `VoxelValueMap`, `RawTriangle` (renamed from `RawTri`), `VertexInterpolate` (renamed from `vertInterp`), `kEdgeCornerPairs` (renamed from `kEdgeCorners`), `GenerateRawTriangles` (template, in header), `CandidateBases` (take `const std::vector<std::array<int,3>>&`), and `WeldAndComputeNormals` (renamed from `weldAndNormal`, returns `SurfaceMesh`) out of `AdaptiveVoxelGrid.cpp` into this unit, namespace `Engine::Spatial::Extraction::core`. `GenerateRawTriangles` keeps using `mc::edgeTable`/`mc::triTable`/`mc::CORNER` from `MarchingCubesTables.h`, with the SAME winding swap (`out.push_back({ev[ei0], ev[ei2], ev[ei1]})`).

- [ ] **Step 6: Implement `VoxelField.{h,cpp}`** — sparse `std::unordered_map<std::array<int,3>, Cell, core::IVec3Hash>` where `Cell{ float value; Eigen::Vector3f gradient; bool hasGradient; }`, plus `m_occupied` vector kept in sync on `Insert`. `FromImplicit` iterates the integer box `[minCoord,maxCoord]`, sets `value = valueFunction(coord*cellSize)` and (if `gradientFunction`) the gradient; inserts ONLY cells within a narrow band `|value| <= 2*cellSize` PLUS enough to bracket the surface (insert a coord if it is within `2*cellSize`; the sphere/box surfaces are then fully bracketed). `FromAdvancedEntries` sets `coord = { lround(center.x/cellSize - 0.5f), ... }`, `value = tsdf`, `gradient = normal`.

- [ ] **Step 7: Implement `IsoSurfaceExtractor.h` + `ExtractorRegistry.{h,cpp}` + `MarchingCubesExtractor.cpp`** — Registry mirrors `TrackerRegistry`. `MarchingCubesExtractor : IsoSurfaceExtractor` with `Name()=="mc"`; `Extract` builds `VoxelValueMap` from the field's occupied coords, runs `core::CandidateBases` → `core::GenerateRawTriangles(bases, sampler, field.CellSize(), raw)` where `sampler(coord,out) = field.Sample(coord,out)`, → `core::WeldAndComputeNormals(raw, params.weldFraction * field.CellSize())`. `ExtractorRegistry::Default()` registers `"mc"` now; later tasks add their names to this same function.

- [ ] **Step 8: Update `AdaptiveVoxelGrid`** — in `AdaptiveVoxelGrid.h` replace the `AdaptiveMesh` struct with `using AdaptiveMesh = Engine::Spatial::Extraction::SurfaceMesh;` (include `Extraction/SurfaceMesh.h`). In `AdaptiveVoxelGrid.cpp` delete the moved primitives and call `core::` equivalents (its multi-resolution `ExtractMesh` keeps its own fine/coarse combine + single weld, now via `core::WeldAndComputeNormals` and `core::GenerateRawTriangles`).

- [ ] **Step 9: Reconfigure, build, run** — expect PASS + the existing `AdaptiveVoxelGrid` bit-exact test still green.
Run: `cmake -S . -B build -D... && cmake --build build --target vkspatial_tests -j && ./build/test/vkspatial_tests --gtest_filter='Isosurface.*:*AdaptiveVoxelGrid*'`
Expected: PASS; then full `./build/test/vkspatial_tests` = 243+ pass / 1 skip / 0 fail.

- [ ] **Step 10: Commit**
```bash
git add src/Engine/Spatial/Extraction/ test/isosurface_test_util.h test/test_isosurface_mc.cpp src/Engine/Spatial/AdaptiveVoxelGrid.h src/Engine/Spatial/AdaptiveVoxelGrid.cpp
git commit -m "feat(extraction): strategy framework + shared MC core + mc extractor

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `"mc33"` — topologically-correct Marching Cubes 33

**Files:**
- Create: `src/Engine/Spatial/Extraction/MarchingCubes33Tables.h`, `MarchingCubes33Extractor.cpp`, `test/test_isosurface_mc33.cpp`
- Modify: `src/Engine/Spatial/Extraction/ExtractorRegistry.cpp` (register `"mc33"`)

**Interfaces:**
- Consumes: Task-1 `VoxelField`, `SurfaceMesh`, `IsoSurfaceExtractor`, `ExtractParams`, `core::CandidateBases`, `core::WeldAndComputeNormals`, `core::VertexInterpolate`, `core::kEdgeCornerPairs`, `mc::CORNER`.
- Produces: `"mc33"` registered strategy.

**Approach:** Implement Chernyaev's MC33 via the Lewiner et al. (2003) case/subcase resolution. `MarchingCubes33Tables.h` transcribes, WITH provenance (cite: Lewiner, Lopes, Vieira, Tavares, "Efficient Implementation of Marching Cubes' Cases with Topological Guarantees", JGT 2003; reference source `http://thomas.lewiner.org/pdfs/marching_cubes_jgt.pdf` / the widely-mirrored `LookUpTable.h`): `cases[256][2]`, `tiling1..tiling14`, `test3/test4/test6/test7/test10/test12/test13` subconfig tables, and `subconfig13`. The extractor computes, per cube, the case + subcase using the **asymptotic decider** for face tests (`FaceTest(cube, face)` = sign of `A*C - B*D` per the bilinear-saddle formula) and the interior test for the ambiguous internal cases, then emits triangles on the 12 standard edges with `core::VertexInterpolate`, and welds via `core::WeldAndComputeNormals`. Vertex placement on edges is identical to `"mc"`; only the triangle TOPOLOGY differs.

- [ ] **Step 1: Write the failing test** `test/test_isosurface_mc33.cpp`

```cpp
#include "Engine/Spatial/Extraction/ExtractorRegistry.h"
#include "Engine/Eval/RmseMetrics.h"
#include "isosurface_test_util.h"
#include <gtest/gtest.h>
using namespace Engine::Spatial::Extraction;

// A single cube carrying a face-ambiguous configuration (two diagonally-opposite
// negative corners on one face). Original 15-case MC can leave a boundary hole here;
// MC33 must produce a topologically consistent (edge-manifold) patch.
static VoxelField AmbiguousCube() {
    VoxelField f; f.SetCellSize(1.0f);
    // corner signs (MC corner order): negatives at 0 and 2 (a face diagonal), rest positive
    const float s[8] = {-1,+1,-1,+1, +1,+1,+1,+1};
    for (int c=0;c<8;++c) f.Insert({Engine::Spatial::mc::CORNER[c][0],Engine::Spatial::mc::CORNER[c][1],Engine::Spatial::mc::CORNER[c][2]}, s[c]);
    return f;
}

TEST(Isosurface, Mc33RegisteredAndSphereAccurate) {
    auto reg = ExtractorRegistry::Default();
    ASSERT_TRUE(reg.Has("mc33"));
    VoxelField field = isotest::SphereField(0.7f, 0.1f, 12);
    SurfaceMesh mesh = reg.Create("mc33")->Extract(field, ExtractParams{});
    EXPECT_TRUE(isotest::IsEdgeManifold(mesh));
    EXPECT_TRUE(isotest::IsWatertight(mesh));
    std::vector<Eigen::Vector3f> on; for (auto& v:mesh.vertices) on.push_back(v.normalized()*0.7f);
    EXPECT_LT(Engine::Eval::NearestNeighbourRMSE(mesh.vertices, on), 0.1f);
}

TEST(Isosurface, Mc33ResolvesFaceAmbiguityEdgeManifold) {
    SurfaceMesh mesh = ExtractorRegistry::Default().Create("mc33")->Extract(AmbiguousCube(), ExtractParams{});
    ASSERT_GT(mesh.triangles.size(), 0u);
    EXPECT_TRUE(isotest::IsEdgeManifold(mesh)); // no >2-incident edges from a bad ambiguity choice
}
```

- [ ] **Step 2: Run to verify it fails** — `... --gtest_filter='Isosurface.Mc33*'` → FAIL (`Has("mc33")` false / not registered).
- [ ] **Step 3: Implement `MarchingCubes33Tables.h`** (transcribe with provenance comment).
- [ ] **Step 4: Implement `MarchingCubes33Extractor.cpp`** (case+subcase resolution, asymptotic decider, edge triangles).
- [ ] **Step 5: Register `"mc33"`** in `ExtractorRegistry.cpp`.
- [ ] **Step 6: Reconfigure/build/run** — `Isosurface.Mc33*` PASS; full suite green.
- [ ] **Step 7: Commit** (`git add` the 3 new/modified files only).

---

### Task 3: `"mtet"` — Marching Tetrahedra

**Files:** Create `src/Engine/Spatial/Extraction/MarchingTetrahedraExtractor.cpp`, `test/test_isosurface_mtet.cpp`; Modify `ExtractorRegistry.cpp`.

**Interfaces:** Consumes Task-1 core; Produces `"mtet"`.

**Approach:** For each candidate cube (reuse `core::CandidateBases`), split into 6 tetrahedra using a fixed decomposition of the 8 corners (the standard 6-tet split sharing the main diagonal 0–6: tets `{0,5,1,6},{0,1,2,6},{0,2,3,6},{0,3,7,6},{0,7,4,6},{0,4,5,6}`). For each tetra, the 4 corner signs give `2^4` cases (2 unique up to symmetry): 0/4 negative → no triangle; 1 or 3 negative → one triangle; 2 negative → two triangles (a quad). Place vertices on the tetra edges with `core::VertexInterpolate` at world corner positions `(base+CORNER[c])*cellSize`. Emit into `RawTriangle`, then `core::WeldAndComputeNormals`. No trilinear ambiguity by construction.

- [ ] **Step 1: Write the failing test** `test/test_isosurface_mtet.cpp`

```cpp
#include "Engine/Spatial/Extraction/ExtractorRegistry.h"
#include "Engine/Eval/RmseMetrics.h"
#include "isosurface_test_util.h"
#include <gtest/gtest.h>
using namespace Engine::Spatial::Extraction;

TEST(Isosurface, MarchingTetrahedraSphereClosedManifold) {
    auto reg = ExtractorRegistry::Default();
    ASSERT_TRUE(reg.Has("mtet"));
    VoxelField field = isotest::SphereField(0.7f, 0.1f, 12);
    SurfaceMesh mesh = reg.Create("mtet")->Extract(field, ExtractParams{});
    ASSERT_GT(mesh.triangles.size(), 100u);
    EXPECT_TRUE(isotest::IsEdgeManifold(mesh));
    EXPECT_TRUE(isotest::IsWatertight(mesh));
    EXPECT_EQ(isotest::EulerCharacteristic(mesh), 2); // topological sphere
    std::vector<Eigen::Vector3f> on; for (auto& v:mesh.vertices) on.push_back(v.normalized()*0.7f);
    EXPECT_LT(Engine::Eval::NearestNeighbourRMSE(mesh.vertices, on), 0.1f);
}
```

- [ ] **Step 2: Run to verify it fails** → FAIL (`Has("mtet")` false).
- [ ] **Step 3: Implement `MarchingTetrahedraExtractor.cpp`.**
- [ ] **Step 4: Register `"mtet"`.**
- [ ] **Step 5: Reconfigure/build/run** — PASS; full suite green.
- [ ] **Step 6: Commit** (3 files only).

---

### Task 4: `"emc"` — Extended Marching Cubes + shared QEF

**Files:** Create `src/Engine/Spatial/Extraction/QuadraticErrorFunction.{h,cpp}`, `ExtendedMarchingCubesExtractor.cpp`, `test/test_isosurface_emc.cpp`; Modify `ExtractorRegistry.cpp`.

**Interfaces:**
- Produces: `QuadraticErrorFunction` — `class QuadraticErrorFunction { public: void Add(const Eigen::Vector3f& point, const Eigen::Vector3f& normal); Eigen::Vector3f Solve(const Eigen::Vector3f& centroidBias, float regularization = 1e-3f) const; private: Eigen::Matrix3f m_ata = Eigen::Matrix3f::Zero(); Eigen::Vector3f m_atb = Eigen::Vector3f::Zero(); };` (each `Add` accumulates `A^T A += n n^T`, `A^T b += n (n·point)`; `Solve` = SVD/Tikhonov of `(A^T A + reg·I) x = A^T b + reg·centroidBias`, biased to `centroidBias`); and `"emc"`.
- Consumes Task-1 core + `VoxelField::Gradient`.

**Approach:** Per candidate cube, compute the 12 possible edge crossings (`core::VertexInterpolate`) and their normals (interpolate `VoxelField::Gradient` at the two edge corners; if the field lacks gradients, central-difference the field). Determine "feature cell" = the crossing normals span an angle whose min pairwise `cos < params.featureAngleCosineThreshold`. Non-feature cells: identical to `"mc"` (reuse the mc topology). Feature cells: compute ONE feature vertex `v* = QEF.Solve(cellCentroid)` from the crossing (point,normal) pairs, and replace the cell's MC triangles by a fan from `v*` to the cell's boundary edge-crossing polygon (walk the mc triangle edges on the cube boundary). Weld with `core::WeldAndComputeNormals`.

- [ ] **Step 1: Write the failing test** `test/test_isosurface_emc.cpp`

```cpp
#include "Engine/Spatial/Extraction/ExtractorRegistry.h"
#include "isosurface_test_util.h"
#include <gtest/gtest.h>
#include <algorithm>
using namespace Engine::Spatial::Extraction;

// Distance from a mesh's vertices to the nearest true box edge (the sharp 90-deg creases).
// Feature-preserving extractors put vertices ON the crease; rounded MC pulls them inward.
static float MinVertexToBoxEdgeGap(const SurfaceMesh& m, const Eigen::Vector3f& he) {
    // sample the 12 edges of the box [-he,he]; measure how close the closest mesh vertex gets
    float best = 1e9f;
    for (const auto& v : m.vertices) {
        Eigen::Vector3f a = v.cwiseAbs();
        // near an edge means TWO coords ~ he
        float d1 = std::abs(a.x()-he.x()), d2 = std::abs(a.y()-he.y()), d3 = std::abs(a.z()-he.z());
        std::array<float,3> d{d1,d2,d3}; std::sort(d.begin(), d.end());
        best = std::min(best, d[0]+d[1]); // two smallest ~0 on an edge
    }
    return best;
}

TEST(Isosurface, ExtendedMcPreservesBoxEdgesBetterThanMc) {
    auto reg = ExtractorRegistry::Default();
    ASSERT_TRUE(reg.Has("emc"));
    const Eigen::Vector3f he(0.5f,0.5f,0.5f); const float cell=0.1f;
    VoxelField field = isotest::BoxField(he, cell, 10);
    SurfaceMesh mcMesh  = reg.Create("mc")->Extract(field, ExtractParams{});
    SurfaceMesh emcMesh = reg.Create("emc")->Extract(field, ExtractParams{});
    EXPECT_TRUE(isotest::IsEdgeManifold(emcMesh));
    // emc should land a vertex closer to the true crease than mc does
    EXPECT_LT(MinVertexToBoxEdgeGap(emcMesh, he), MinVertexToBoxEdgeGap(mcMesh, he));
}
```

- [ ] **Step 2: Run to verify it fails** → FAIL (`Has("emc")` false).
- [ ] **Step 3: Implement `QuadraticErrorFunction.{h,cpp}`.**
- [ ] **Step 4: Implement `ExtendedMarchingCubesExtractor.cpp`.**
- [ ] **Step 5: Register `"emc"`.**
- [ ] **Step 6: Reconfigure/build/run** — PASS; full suite green.
- [ ] **Step 7: Commit** (4 files only).

---

### Task 5: `"dc"` — Dual Contouring of Hermite data

**Files:** Create `src/Engine/Spatial/Extraction/DualContouringExtractor.cpp`, `test/test_isosurface_dc.cpp`; Modify `ExtractorRegistry.cpp`.

**Interfaces:** Consumes Task-1 core + `QuadraticErrorFunction` (Task 4) + `VoxelField::Gradient`; Produces `"dc"`.

**Approach:** For every cell with a sign change on ≥1 edge, compute one dual vertex = `QEF.Solve(cellCentroid)` over that cell's sign-changing edges' Hermite data (crossing point + normal). Store dual vertices in a `map<cell coord, vertex index>`. Then for each sign-changing edge of the grid, the 4 cells sharing that edge each contributed a dual vertex; connect those 4 dual vertices into a quad (orient by the edge's sign direction), split into 2 triangles. Compute per-vertex normals by area-weighting (reuse the accumulation pattern; DC vertices are already placed, so just accumulate face normals — you can call a small local normal pass or reuse `core::WeldAndComputeNormals` with `weldDistance` tiny so nothing merges). No `mc` topology is used.

- [ ] **Step 1: Write the failing test** `test/test_isosurface_dc.cpp`

```cpp
#include "Engine/Spatial/Extraction/ExtractorRegistry.h"
#include "Engine/Eval/RmseMetrics.h"
#include "isosurface_test_util.h"
#include <gtest/gtest.h>
using namespace Engine::Spatial::Extraction;

TEST(Isosurface, DualContouringSphereAccurateAndManifold) {
    auto reg = ExtractorRegistry::Default();
    ASSERT_TRUE(reg.Has("dc"));
    const float cell=0.1f, r=0.7f;
    VoxelField field = isotest::SphereField(r, cell, 12);
    SurfaceMesh mesh = reg.Create("dc")->Extract(field, ExtractParams{});
    ASSERT_GT(mesh.triangles.size(), 100u);
    EXPECT_TRUE(isotest::IsEdgeManifold(mesh));
    std::vector<Eigen::Vector3f> on; for (auto& v:mesh.vertices) on.push_back(v.normalized()*r);
    EXPECT_LT(Engine::Eval::NearestNeighbourRMSE(mesh.vertices, on), cell);
}

TEST(Isosurface, DualContouringPreservesBoxEdges) {
    const Eigen::Vector3f he(0.5f,0.5f,0.5f); const float cell=0.1f;
    VoxelField field = isotest::BoxField(he, cell, 10);
    SurfaceMesh dcMesh = ExtractorRegistry::Default().Create("dc")->Extract(field, ExtractParams{});
    SurfaceMesh mcMesh = ExtractorRegistry::Default().Create("mc")->Extract(field, ExtractParams{});
    // DC's QEF vertices sit on the crease; compare the tightest crease approach (helper reused from emc test file is fine to duplicate locally)
    auto gap=[&](const SurfaceMesh& m){ float best=1e9f; for(auto&v:m.vertices){Eigen::Vector3f a=v.cwiseAbs();
        std::array<float,3> d{std::abs(a.x()-he.x()),std::abs(a.y()-he.y()),std::abs(a.z()-he.z())}; std::sort(d.begin(),d.end()); best=std::min(best,d[0]+d[1]);} return best; };
    EXPECT_LT(gap(dcMesh), gap(mcMesh));
}
```

- [ ] **Step 2: Run to verify it fails** → FAIL (`Has("dc")` false).
- [ ] **Step 3: Implement `DualContouringExtractor.cpp`.**
- [ ] **Step 4: Register `"dc"`.**
- [ ] **Step 5: Reconfigure/build/run** — PASS; full suite green.
- [ ] **Step 6: Commit** (3 files only).

---

### Task 6: `"dmc"` — Dual Marching Cubes

**Files:** Create `src/Engine/Spatial/Extraction/DualMarchingCubesExtractor.cpp`, `test/test_isosurface_dmc.cpp`; Modify `ExtractorRegistry.cpp`.

**Interfaces:** Consumes Task-1 core + `QuadraticErrorFunction` + `"dc"` vertex logic; Produces `"dmc"`.

**Approach:** Build the grid dual: the DC dual vertices (one per primal cell, Task-5 placement) become the corners of dual cells; each primal grid VERTEX with its 8 surrounding primal cells forms one dual cube whose 8 corners are those cells' dual vertices, and whose corner scalar values are the primal field value at that shared primal vertex. Run the standard `"mc"` topology (`core::GenerateRawTriangles`-style case lookup) on these dual cubes, interpolating along dual-cube edges. This yields a crack-free polygonalization that reproduces thin features. Weld with a tiny `weldDistance`.

- [ ] **Step 1: Write the failing test** `test/test_isosurface_dmc.cpp`

```cpp
#include "Engine/Spatial/Extraction/ExtractorRegistry.h"
#include "isosurface_test_util.h"
#include <gtest/gtest.h>
using namespace Engine::Spatial::Extraction;

// Thin slab: |z| - t  intersected as two sheets at z = +/- t, spacing 2t ~ one cell.
static VoxelField ThinSlab(float halfThickness, float cell, int halfN) {
    const std::array<int,3> lo{-halfN,-halfN,-halfN}, hi{halfN,halfN,halfN};
    auto value=[halfThickness](const Eigen::Vector3f& p){ return std::abs(p.z()) - halfThickness; };
    return FromImplicit(lo,hi,cell,value,nullptr);
}

TEST(Isosurface, DualMarchingCubesRegisteredAndSphereManifold) {
    auto reg = ExtractorRegistry::Default();
    ASSERT_TRUE(reg.Has("dmc"));
    VoxelField field = isotest::SphereField(0.7f, 0.1f, 12);
    SurfaceMesh mesh = reg.Create("dmc")->Extract(field, ExtractParams{});
    ASSERT_GT(mesh.triangles.size(), 100u);
    EXPECT_TRUE(isotest::IsEdgeManifold(mesh));
}

TEST(Isosurface, DualMarchingCubesResolvesThinSlabAsTwoSheets) {
    VoxelField field = ThinSlab(0.06f, 0.1f, 8); // slab thinner than a cell
    SurfaceMesh mesh = ExtractorRegistry::Default().Create("dmc")->Extract(field, ExtractParams{});
    // both z=+/-0.06 sheets present: vertices exist above AND below z=0
    bool above=false, below=false;
    for (auto& v: mesh.vertices){ if (v.z()>0.02f) above=true; if (v.z()<-0.02f) below=true; }
    EXPECT_TRUE(above && below);
}
```

- [ ] **Step 2: Run to verify it fails** → FAIL (`Has("dmc")` false).
- [ ] **Step 3: Implement `DualMarchingCubesExtractor.cpp`.**
- [ ] **Step 4: Register `"dmc"`.**
- [ ] **Step 5: Reconfigure/build/run** — PASS; full suite green.
- [ ] **Step 6: Commit** (3 files only).

---

### Task 7: `"cms"` — Cubical Marching Squares + companion doc

**Files:** Create `src/Engine/Spatial/Extraction/CubicalMarchingSquaresExtractor.cpp`, `test/test_isosurface_cms.cpp`, `docs/ISOSURFACE_EXTRACTION.md`; Modify `ExtractorRegistry.cpp`.

**Interfaces:** Consumes Task-1 core + `QuadraticErrorFunction`; Produces `"cms"`.

**Approach:** For each cube, run 2D marching squares on each of the 6 faces (each face = 4 corners → segments on face edges), resolving per-face ambiguity with the 2D asymptotic decider so adjacent cubes agree (crack-free). Stitch the per-face segments into one or more closed loops around the cube; for each loop, if it is a feature cell (Task-4 angle test), insert the QEF feature vertex and fan-triangulate the loop to it; otherwise fan-triangulate the loop to its centroid. Weld with `core::WeldAndComputeNormals`.

- [ ] **Step 1: Write the failing test** `test/test_isosurface_cms.cpp`

```cpp
#include "Engine/Spatial/Extraction/ExtractorRegistry.h"
#include "Engine/Eval/RmseMetrics.h"
#include "isosurface_test_util.h"
#include <gtest/gtest.h>
using namespace Engine::Spatial::Extraction;

TEST(Isosurface, CubicalMarchingSquaresSphereManifoldAccurate) {
    auto reg = ExtractorRegistry::Default();
    ASSERT_TRUE(reg.Has("cms"));
    const float cell=0.1f, r=0.7f;
    VoxelField field = isotest::SphereField(r, cell, 12);
    SurfaceMesh mesh = reg.Create("cms")->Extract(field, ExtractParams{});
    ASSERT_GT(mesh.triangles.size(), 100u);
    EXPECT_TRUE(isotest::IsEdgeManifold(mesh));
    std::vector<Eigen::Vector3f> on; for (auto& v:mesh.vertices) on.push_back(v.normalized()*r);
    EXPECT_LT(Engine::Eval::NearestNeighbourRMSE(mesh.vertices, on), cell);
}

TEST(Isosurface, CubicalMarchingSquaresPreservesBoxEdges) {
    const Eigen::Vector3f he(0.5f,0.5f,0.5f); const float cell=0.1f;
    VoxelField field = isotest::BoxField(he, cell, 10);
    SurfaceMesh cms = ExtractorRegistry::Default().Create("cms")->Extract(field, ExtractParams{});
    SurfaceMesh mc  = ExtractorRegistry::Default().Create("mc")->Extract(field, ExtractParams{});
    auto gap=[&](const SurfaceMesh& m){ float best=1e9f; for(auto&v:m.vertices){Eigen::Vector3f a=v.cwiseAbs();
        std::array<float,3> d{std::abs(a.x()-he.x()),std::abs(a.y()-he.y()),std::abs(a.z()-he.z())}; std::sort(d.begin(),d.end()); best=std::min(best,d[0]+d[1]);} return best; };
    EXPECT_LT(gap(cms), gap(mc));
}
```

- [ ] **Step 2: Run to verify it fails** → FAIL (`Has("cms")` false).
- [ ] **Step 3: Implement `CubicalMarchingSquaresExtractor.cpp`.**
- [ ] **Step 4: Register `"cms"`.**
- [ ] **Step 5: Write `docs/ISOSURFACE_EXTRACTION.md`** — framework structure (VoxelField/SurfaceMesh/IsoSurfaceExtractor/ExtractorRegistry + shared core), and for EACH of the 7 strategies: principle, primal-vs-dual, quads-vs-triangles, topology/feature guarantees, and when to pick it. Link `MARCHING_CUBES_SURVEY.md`.
- [ ] **Step 6: Reconfigure/build/run** — PASS; full suite green.
- [ ] **Step 7: Commit** (4 files only).

---

## Self-review notes

- **Spec coverage:** framework (Task 1) · mc/mc33/mtet/emc/dc/dmc/cms (Tasks 1–7) · analytic-field + manifold/watertight/feature/topology tests (all tasks, util in Task 1) · shared-core refactor guarded by AdaptiveVoxelGrid bit-exact test (Task 1 Step 9) · companion doc (Task 7). Neural + Transvoxel/Flying Edges correctly out of scope (survey doc). All spec sections map to a task.
- **Type consistency:** `SurfaceMesh`, `VoxelField`, `ExtractParams`, `IsoSurfaceExtractor`, `ExtractorRegistry`, `core::{VertexInterpolate,CandidateBases,GenerateRawTriangles,WeldAndComputeNormals,kEdgeCornerPairs,RawTriangle,VoxelValueMap,IVec3Hash}`, `QuadraticErrorFunction` — names used identically across all tasks. Coordinates are `std::array<int,3>` throughout the Extraction unit (matches the reused core + hashing).
- **Table-heavy tasks (2, 7):** the plan cannot inline the multi-hundred-entry MC33/CMS published tables; it names the provenance source and relies on the manifold/watertight/ambiguity predicates as the correctness gate. Implementers transcribe from the cited reference.
