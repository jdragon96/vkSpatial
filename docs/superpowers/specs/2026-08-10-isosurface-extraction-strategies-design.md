# Isosurface Extraction Strategies — Design

## Problem

The engine has exactly one isosurface extractor: the standard 15-case Marching Cubes,
implemented twice (GPU `voxel_tsdf_mc.comp`, and a CPU transcription buried in
`AdaptiveVoxelGrid.cpp`'s anonymous namespace). There is no way to select a *different*
extraction algorithm, and the classical family (topologically-correct MC33, Marching
Tetrahedra, feature-preserving Extended MC / Dual Contouring, dual-grid methods) is absent.
The reusable primitives (`generateRawTriangles`, `candidateBases`, `weldAndNormal`,
`vertInterp`, the lookup tables) exist but are not shared, so every new algorithm would
otherwise re-implement cube iteration and vertex welding.

We want a **strategy-pattern isosurface-extraction framework** — mirroring the existing
`Engine::Pipeline::Tracker` / `TrackerRegistry` pattern — so extraction algorithms are
pluggable, swappable by name, and independently testable, with the classical family
implemented on top of a shared core.

## Goals

- A framework in a new `Engine::Spatial::Extraction` unit: `VoxelField` (input),
  `SurfaceMesh` (output), `IsoSurfaceExtractor` (abstract strategy), `ExtractorRegistry`
  (name → factory, `Default()` hub), and a shared `MarchingCubesCore` extracted from the
  existing `AdaptiveVoxelGrid.cpp` primitives.
- The full **classical** family as pluggable strategies, each an independently testable
  phase, applied CPU-side over the shared `VoxelField`:
  1. `"mc"` — standard Marching Cubes (wraps the existing 15-case tables).
  2. `"mc33"` — topologically-correct Marching Cubes 33 (Lewiner tables).
  3. `"mtet"` — Marching Tetrahedra (cube → 6 tetrahedra; no ambiguity).
  4. `"emc"` — Extended Marching Cubes (feature vertex via QEF over gradients).
  5. `"dc"` — Dual Contouring of Hermite data (per-cell dual vertex via QEF).
  6. `"dmc"` — Dual Marching Cubes (primal contouring of the dual grid).
  7. `"cms"` — Cubical Marching Squares (per-face 2D marching squares + features).
- Output is a triangle `SurfaceMesh` (vertices / triangles / normals) for every strategy
  (dual methods that naturally produce quads triangulate their quads).
- Validated by a deterministic test harness over **analytic implicit fields** (sphere,
  plane, box) with exact gradients: accuracy (reuse `Engine::Eval::NearestNeighbourRMSE`),
  edge-manifoldness, watertightness, and feature preservation (dihedral sharpness).

## Non-goals

- **No neural / differentiable extractors** (Deep MC, Neural MC, DMTet, Neural DC,
  FlexiCubes) — they require a training pipeline / ML framework absent from this C++/Vulkan
  engine. They are covered in `docs/MARCHING_CUBES_SURVEY.md` as documentation only.
- **No GPU compute strategies** in this effort — CPU-first over the shared `VoxelField`.
  GPU parity per strategy is a later effort. (The existing `voxel_tsdf_mc.comp` is untouched.)
- **No Transvoxel / Flying Edges** here — Transvoxel is a multi-resolution LOD stitching
  concern (see the survey doc §7 and `AdaptiveVoxelGrid`'s BFS boundary refinement); Flying
  Edges is a parallel reimplementation of MC with identical output. Both are follow-ups.
- **No rewrite of `AdaptiveVoxelGrid`'s multi-resolution logic.** Its special fine+coarse
  combined-weld path stays; it only starts sharing the low-level primitives it already owns.

## Architecture

```
Engine::Spatial::Extraction
 ├─ VoxelField          input: signed scalar + optional stored gradient, cellSize, occupied coords
 ├─ SurfaceMesh         output: vertices / triangles / normals
 ├─ IsoSurfaceExtractor abstract strategy: Name(), Extract(field, params)
 ├─ ExtractorRegistry   Register / Create / Has / Default  (switch by string name)
 ├─ MarchingCubesCore   shared: RawTri, candidateBases, generateRawTriangles, weldAndNormal, vertInterp
 └─ strategies/         one file per strategy (mc, mc33, mtet, emc, dc, dmc, cms)
```

The framework and hub mirror `src/Engine/Pipeline/Registration/Tracker.h` +
`TrackerRegistry.cpp` exactly (abstract base with `Name()`; registry with
`Register`/`Create`/`Has`/`static Default()`; one file per concrete strategy).

### Interfaces (full descriptive names, no abbreviations)

```cpp
namespace Engine::Spatial::Extraction {

    struct SurfaceMesh {
        std::vector<Eigen::Vector3f> vertices;
        std::vector<Eigen::Vector3i> triangles;
        std::vector<Eigen::Vector3f> normals;   // per-vertex, area-weighted
    };

    // Sparse voxel field the extractors read. Coordinates are integer grid indices; world
    // position of a corner is coord * cellSize (single world frame, as generateRawTriangles).
    class VoxelField {
    public:
        bool  Sample(const Eigen::Vector3i& coord, float& outValue) const;      // signed field value
        bool  Gradient(const Eigen::Vector3i& coord, Eigen::Vector3f& outNormal) const; // stored gradient; false if none
        float CellSize() const;
        const std::vector<Eigen::Vector3i>& OccupiedCoords() const;
        // Adapters (free functions):
        //   VoxelField FromAdvancedEntries(const std::vector<AdvancedEntry>&, float truncationDistance);
        //   VoxelField FromImplicit(bounds, cellSize, std::function<float(Vector3f)>, optional gradientFn);  // tests
    };

    struct ExtractParams {
        float isoLevel = 0.0f;
        float weldFraction = 0.25f;             // weld grid = weldFraction * cellSize (matches AdaptiveVoxelGrid)
        float featureAngleCosineThreshold = 0.9f; // emc/dc/cms: treat a cell as a feature when edge normals diverge past this
    };

    class IsoSurfaceExtractor {
    public:
        virtual ~IsoSurfaceExtractor() = default;
        virtual const char* Name() const = 0;
        virtual SurfaceMesh Extract(const VoxelField& field, const ExtractParams& params) const = 0;
    };

    class ExtractorRegistry {
    public:
        using Factory = std::function<std::unique_ptr<IsoSurfaceExtractor>()>;
        void Register(const std::string& name, Factory factory);
        std::unique_ptr<IsoSurfaceExtractor> Create(const std::string& name) const;
        bool Has(const std::string& name) const;
        static ExtractorRegistry Default();     // registers mc, mc33, mtet, emc, dc, dmc, cms
    private:
        std::unordered_map<std::string, Factory> m_factories;
    };
}
```

### Shared core refactor

Extract from `AdaptiveVoxelGrid.cpp`'s anonymous namespace into `MarchingCubesCore.{h,cpp}`:
`RawTri`, `vertInterp`, `candidateBases(field)`, `generateRawTriangles(bases, sampler,
cellSize, out)`, `weldAndNormal(tris, weld)`. `AdaptiveVoxelGrid` then calls these shared
functions instead of its private copies. Its existing **bit-exact-vs-GPU** test
(`test_adaptiveVoxelGrid.cpp`) is the regression guard proving the extraction is unchanged.

## The algorithms

Each is one phase (task): the strategy file, its registration, and its test.

### Task 1 — Framework + `"mc"`
`VoxelField`, `SurfaceMesh`, `IsoSurfaceExtractor`, `ExtractorRegistry`, `MarchingCubesCore`
(extracted), and `MarchingCubesExtractor` = `candidateBases` → `generateRawTriangles`
(existing edgeTable/triTable) → `weldAndNormal`. Register `"mc"`.
- **Test:** analytic sphere field → mesh; `NearestNeighbourRMSE(vertices, true sphere) <
  cellSize`; edge-manifold (every edge shared by ≤2 triangles); watertight (every edge
  shared by exactly 2). AND the existing `AdaptiveVoxelGrid` bit-exact test still passes.

### Task 2 — `"mc33"` (Marching Cubes 33)
Topologically-correct MC using the Chernyaev/Lewiner case table + subcase resolution
(asymptotic-decider face test + interior test). Tables transcribed from the published
Lewiner reference implementation into `MarchingCubes33Tables.h` (documented provenance).
- **Test:** a fixture cube whose corner signs form a **face-ambiguous** case (the classic
  configuration where 15-case MC leaves a hole): assert `"mc33"` produces a watertight
  (edge-manifold, every boundary edge shared by 2) mesh where `"mc"` does not; both still
  match a plane/sphere within tolerance.

### Task 3 — `"mtet"` (Marching Tetrahedra)
Split each cube into 6 tetrahedra; per-tetra 2^4 = 16-case (2 unique) triangulation on the
tetra edges. No trilinear ambiguity by construction.
- **Test:** analytic sphere → watertight, edge-manifold, closed (Euler characteristic V−E+F
  = 2 for a topological sphere), `NearestNeighbourRMSE < cellSize`.

### Task 4 — `"emc"` (Extended Marching Cubes)
Run MC connectivity, but when a cell is a feature cell (edge-intersection normals diverge
past `featureAngleCosineThreshold`), place ONE feature vertex at the QEF minimum of the
tangent planes (position + gradient at each edge crossing) and fan-triangulate to it,
instead of the rounded MC vertices. Gradients come from `VoxelField::Gradient` (fallback:
central differences of the field).
- **Test:** analytic box (SDF with exact gradients) → the sharp edges/corners are preserved:
  measure the max dihedral angle near an edge; assert `"emc"` reproduces the ~90° edge
  (within tolerance) while `"mc"` rounds it (measurably larger min radius). Flat faces match.

### Task 5 — `"dc"` (Dual Contouring of Hermite data)
One dual vertex per cell at the QEF minimum of its sign-changing edges' Hermite data
(intersection + normal); for each sign-changing edge, connect the dual vertices of the 4
cells sharing that edge into a quad (triangulated). Uses a numerically-stable QEF (SVD /
regularized normal equations) with bias to the cell centroid.
- **Test:** analytic box → sharp features preserved (dihedral like emc) AND the sphere/plane
  accuracy holds; edge-manifold. (DC can be non-manifold at some configs; the test asserts
  edge-manifoldness on the smooth fixtures, documents the sharp-config caveat.)

### Task 6 — `"dmc"` (Dual Marching Cubes)
Primal contouring on the grid **dual** to the voxel grid: build dual cells whose corners are
the DC-style vertices of the primal cells, then run MC on the dual cells. Produces a
crack-free polygonalization that reproduces thin features without extra subdivision.
- **Test:** a thin-slab analytic fixture (two close parallel surfaces) → `"dmc"` reproduces
  both sheets where a coarse `"mc"` merges/misses them; watertight on the sphere.

### Task 7 — `"cms"` (Cubical Marching Squares)
Process each cube face as a 2D marching-squares contour, resolve face ambiguity locally,
stitch the per-face segments into cell polygons, and insert sharp-feature vertices (like
emc) before triangulating. Adaptive/feature-preserving with no cracks.
- **Test:** analytic box → sharp features preserved AND watertight/edge-manifold on the
  sphere; face-ambiguous fixture resolved (no holes).

## Testing framework

A shared test utility (`test/isosurface_test_util.h`) provides:
- **Analytic `VoxelField` builders**: `SphereField(radius, cellSize)`, `PlaneField(...)`,
  `BoxField(halfExtents, cellSize)` — each with exact gradients for the feature methods.
- **Mesh predicates**: `IsEdgeManifold(mesh)` (each undirected edge in ≤2 triangles),
  `IsWatertight(mesh)` (each edge in exactly 2), `EulerCharacteristic(mesh)`,
  `MaxDihedralSharpness(mesh, nearPoint)` (for feature tests).
- Accuracy via the existing `Engine::Eval::NearestNeighbourRMSE`.

Each strategy's test asserts the properties appropriate to it (accuracy for all; watertight
for closed fields; feature preservation for emc/dc/cms; topology resolution for mc33/cms).

## Cross-cutting / Global Constraints

- **Full descriptive names, no abbreviations** in all new/edited code (`maxCorrespondence`-
  style spelled-out identifiers; `triangleTable` not `triTable` in NEW code, though existing
  transcribed tables keep their published names with a provenance comment).
- **CPU-only** here; no shader changes; MoltenVK atomics constraint is not engaged.
- **Do NOT touch the user's uncommitted WIP**: `src/Engine/Spatial/AdvancedTSDF.cpp`,
  `src/shader/advanced_tsdf_compact.comp.glsl`, `src/shader/advanced_tsdf_integrate.comp.glsl`,
  `src/Engine/Pipeline/Registration/RegistrationThread.cpp`, and the `lib/SPIRV-Reflect`
  submodule. Each implementer `git add`s ONLY its own task files — never `git add .`.
- **Build:** reuse the warm `build/` cache; never `rm -rf build`. If a reconfigure is needed,
  `cmake -S . -B build -DGTest_DIR=/opt/homebrew/lib/cmake/GTest -Dgflags_DIR=/opt/homebrew/lib/cmake/gflags -Dglog_DIR=/opt/homebrew/lib/cmake/glog`.
  Build/run the `vkspatial_tests` target.
- **Commit trailer:** `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Do NOT push to origin.
- **Lookup-table provenance:** every transcribed table (MC33, tetra) carries a comment citing
  its published source and a note that a single wrong entry silently corrupts the mesh (as
  `MarchingCubesTables.h` already does).

## Acceptance / Testing

- The existing suite stays green (`vkspatial_tests`, currently 241 pass / 1 skip); the
  `AdaptiveVoxelGrid` bit-exact test still passes after the core extraction (proves no
  regression in the shared primitives).
- Each new strategy adds its test(s); all pass. Selecting a strategy by name through
  `ExtractorRegistry::Default().Create(name)` returns a working extractor for each of the
  seven names, and `nullptr` for an unknown name (mirrors `TrackerRegistry`).
- Companion doc `docs/ISOSURFACE_EXTRACTION.md`: framework structure + each implemented
  strategy's principle, output character (primal/dual, quads/triangles), feature/topology
  guarantees, and selection guidance.

## Risks

- **MC33 / CMS table complexity** — the topological subcase logic is intricate; a wrong
  table entry silently corrupts topology. Mitigation: transcribe from a published reference
  with provenance; the face-ambiguous fixture + manifold predicates catch corruption.
- **QEF numerical stability** (emc/dc/dmc) — degenerate/near-parallel normals blow up the
  solve. Mitigation: SVD or Tikhonov-regularized solve biased to the cell centroid; test on
  analytic fields with known features.
- **Shared-core refactor regressing `AdaptiveVoxelGrid`** — Mitigation: its bit-exact-vs-GPU
  test is the gate; the refactor is "move, don't modify" the primitives.
- **Dual-method manifoldness** — DC can produce non-manifold edges at sharp configs.
  Mitigation: assert edge-manifoldness on smooth fixtures; document the caveat (Manifold DC
  is a noted future refinement, not in scope).
