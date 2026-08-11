# Isosurface: unified output + mesh connectivity + GPU extractors — Design

Three related additions to the `Engine::Spatial::Extraction` framework, all keeping the existing
`SurfaceMesh` as the single output type:

1. **Unified output** — every extractor, CPU and GPU, returns the same `SurfaceMesh`.
2. **Mesh connectivity** — a half-edge–based non-manifold / boundary detection facility over
   `SurfaceMesh`, plus a viewer visualization that highlights the flagged edges.
3. **GPU extractors** — GPU-compute implementations of the three per-cube-independent algorithms
   (`mc`, `mc33`, `mtet`), each returning the same `SurfaceMesh` and validated against its CPU twin.

Scope decisions (confirmed): keep the `SurfaceMesh` name (no rename); GPU only for `mc`/`mc33`/`mtet`
(the dual/feature methods `emc`/`dc`/`dmc`/`cms` stay CPU — their cross-cell connectivity / variable
per-cell stitching is research-hard on MoltenVK, which also lacks GPU float atomics).

## Part 1 — Unified `SurfaceMesh` output

`SurfaceMesh { vertices, triangles(Vector3i indices), normals }` (unchanged) is the single output of
every extractor. GPU extractors (Part 3) implement the SAME `IsoSurfaceExtractor` contract
(`SurfaceMesh Extract(const VoxelField&, const ExtractParams&) const`), producing a `SurfaceMesh`
indistinguishable in type from the CPU path. No rename; `AdaptiveMesh` stays an alias. This part is
realized by Part 3's GPU extractors returning `SurfaceMesh` — there is no separate work item.

## Part 2 — Mesh connectivity (half-edge, non-manifold detection) + visualization

### Connectivity facility (`Engine/Spatial/Extraction/MeshConnectivity.{h,cpp}`)

Build a half-edge structure from the triangle soup and classify topology:

```cpp
namespace Engine::Spatial::Extraction {

    // Half-edge adjacency built from a SurfaceMesh's triangles. Each directed edge (a->b) is a
    // half-edge; its twin is the half-edge (b->a) of an adjacent triangle. An undirected edge is
    // MANIFOLD when it has exactly 2 incident half-edges (one twin pair), BOUNDARY when 1,
    // NON-MANIFOLD when > 2.
    struct HalfEdge {
        int origin;   // vertex index this half-edge starts at
        int face;     // triangle index it belongs to
        int next;     // next half-edge around the same face (CCW)
        int twin;     // opposing half-edge index, or -1 if boundary/non-manifold
    };

    struct HalfEdgeMesh {
        std::vector<HalfEdge> halfEdges;              // 3 per triangle
        std::vector<int> vertexHalfEdge;              // one outgoing half-edge per vertex (-1 if isolated)
    };
    HalfEdgeMesh BuildHalfEdgeMesh(const SurfaceMesh& mesh);

    struct ConnectivityReport {
        std::vector<std::pair<int,int>> boundaryEdges;     // undirected (v0<v1), 1 incident triangle
        std::vector<std::pair<int,int>> nonManifoldEdges;  // undirected, > 2 incident triangles
        std::vector<int> nonManifoldVertices;              // umbrella is not a single closed/open fan (bowtie)
        int degenerateTriangles = 0;                       // repeated index or zero-area
        bool IsEdgeManifold() const { return nonManifoldEdges.empty(); }
        bool IsClosed() const { return boundaryEdges.empty() && nonManifoldEdges.empty(); }
    };
    ConnectivityReport AnalyzeConnectivity(const SurfaceMesh& mesh);
}
```

- **Edge classification:** group directed half-edges by undirected key `(min(a,b), max(a,b))`; count
  incident triangles → boundary (1) / manifold (2) / non-manifold (>2). Set `twin` only for the unique
  2-incidence case.
- **Non-manifold vertex (bowtie):** a vertex whose incident triangles do not form a single edge-connected
  fan (walk the one-ring via twins; if it splits into ≥2 components, the vertex is non-manifold).
- **Degenerate:** any triangle with a repeated vertex index or (post-position) near-zero area.
- This supersedes the ad-hoc `isotest::IsEdgeManifold/IsWatertight` predicates (the tests may keep using
  theirs; the facility is the production one and the viewer/consumers use it).

**Tests** (`test/test_mesh_connectivity.cpp`):
- Closed analytic sphere mesh (via `mc`) → `IsClosed()` true, no boundary/non-manifold, 0 degenerate.
- A hand-built **non-manifold fixture**: 3 triangles sharing one edge → that edge in `nonManifoldEdges`.
- A hand-built **boundary fixture**: a single triangle → its 3 edges all boundary.
- A hand-built **bowtie fixture**: two triangles sharing only one vertex → that vertex in
  `nonManifoldVertices`.
- The `dc` ambiguous-face fixture (from `test_isosurface_dc.cpp`'s DISABLED test) → `AnalyzeConnectivity`
  reports the non-manifold edge (documents dc's known limitation with the production facility).

### Visualization (extends the viewer)

Extend `example2/isosurface_viewer` + a small line-overlay in `IsosurfaceMeshPass` (or a sibling
`EdgeOverlayPass`): after each `rebuild()`, run `AnalyzeConnectivity` and upload the flagged edges as
`VK_PRIMITIVE_TOPOLOGY_LINE_LIST` segments — **non-manifold edges red, boundary edges yellow** — drawn
over the mesh (slight depth bias / drawn after, always visible). ImGui: a `Checkbox("show non-manifold /
boundary edges")` toggle, and add the counts (non-manifold edge count, boundary edge count, bowtie vertex
count) to the existing stats block. So switching to `dc` on a box visibly lights up the ambiguous-face
cracks in red.

## Part 3 — GPU extractors (`mc`, `mc33`, `mtet`)

### Interface + registry

GPU extractors implement `IsoSurfaceExtractor` but need a GPU `Context`, so they are constructed with one
and registered through a Context-aware registry (mirrors how `GpuPointToPlaneIcp` takes a `Context&`):

```cpp
// Engine/Spatial/Extraction/GpuExtractorRegistry.{h,cpp}
namespace Engine::Spatial::Extraction {
    class GpuExtractorRegistry {
    public:
        using Factory = std::function<std::unique_ptr<IsoSurfaceExtractor>(Engine::Core::Context&)>;
        void Register(const std::string& name, Factory);
        std::unique_ptr<IsoSurfaceExtractor> Create(const std::string& name, Engine::Core::Context&) const;
        bool Has(const std::string& name) const;
        static GpuExtractorRegistry Default();   // registers "mc-gpu", "mc33-gpu", "mtet-gpu"
    private:
        std::unordered_map<std::string, Factory> m_factories;
    };
}
```

Names: `"mc-gpu"`, `"mc33-gpu"`, `"mtet-gpu"` (so both CPU and GPU are selectable side by side). Each
returns a `SurfaceMesh` (Part 1).

### GPU compute pipeline (shared across the three)

Each GPU extractor:
1. **Upload** the `VoxelField` to GPU buffers: the occupied voxel coordinates + their values (and, for a
   forward-compatible layout, a slot for gradients even though mc/mc33/mtet do not use them). Provide
   corner sampling on the GPU via a compact hash of `coord -> value` (mirror the engine's existing hash
   patterns) OR, simplest first cut, upload a dense value grid over the field's integer bounding box (the
   viewer's analytic/scan fields are bounded). Choose per implementation; document the choice.
2. **Dispatch** a compute shader, one invocation per candidate cube (the 8-neighbour sweep over occupied
   coords, as `core::CandidateBases`). Each invocation samples its 8 corners, computes `cubeIndex`, and
   emits triangles into an output vertex buffer using an **int `atomicAdd` counter** (MoltenVK supports
   integer atomics; no float atomics are needed — vertices are written to distinct, counter-reserved
   slots, not accumulated). Reuse the MC edge/triangle tables (already in `voxel_common.glsl` /
   `MarchingCubesTables.h`); `mc33` adds the MC33 case/subcase tables + asymptotic-decider in GLSL;
   `mtet` uses the 6-tetra split.
3. **Download** the emitted triangles and **weld** on the CPU via `core::WeldAndComputeNormals` (identical
   post-processing to the CPU extractors → same `SurfaceMesh`, same weld semantics). (A GPU weld is a
   future optimization; not in scope.)

The existing `voxel_tsdf_mc.comp` (per-voxel MC with corner de-dup) is the structural reference for the
`mc-gpu` shader; adapt it to the Extraction `VoxelField` sampling + the raw-triangle output buffer.

### Consistency requirement (the gate)

Each GPU extractor must produce a mesh that **matches its CPU counterpart** on the same `VoxelField` —
mirroring the existing `AdaptiveVoxelGrid` CPU-MC ≡ GPU-MC bit-exact test. Because both paths use the same
tables, the same `VertexInterpolate`, and the same `core::WeldAndComputeNormals`, the vertex sets should
match to floating-point tolerance (allow tiny ordering/rounding differences: assert
`NearestNeighbourRMSE(gpuVerts, cpuVerts) < 1e-5` both directions AND equal triangle counts within a small
tolerance; document any GPU int-quantization used for the emit).

**Tests** (`test/test_gpu_extractors.cpp`, one per algorithm):
- `mc-gpu` vs `mc` on an analytic sphere `VoxelField` → vertex-set NN RMSE ~0, edge-manifold, watertight.
- `mc33-gpu` vs `mc33` on sphere + the face-ambiguous cube → same topology (edge-manifold), matching verts.
- `mtet-gpu` vs `mtet` on sphere → matching verts, `EulerCharacteristic == 2`.
- Registry: `GpuExtractorRegistry::Default().Has("mc-gpu")` etc.; `Create` returns nullptr for unknown.

## Cross-cutting / Global Constraints

- Full descriptive names, no abbreviations (`cubeIndex`, `edgeCrossingPosition`, not `ci`/`ev`); GLSL
  follows the repo's `///`-banner / Allman / numbered-step style; transcribed tables keep published names
  + provenance comments.
- **MoltenVK: no GPU float atomics** — the emit uses an **int** `atomicAdd` slot counter (reserve-then-
  write), never a float accumulation. Measure any GPU perf only in `build-rel`.
- Reuse existing infrastructure: `MarchingCubesCore` (weld / tables / VertexInterpolate), the compute
  patterns of `voxel_tsdf_mc.comp` / `GpuPointToPlaneIcp`, the viewer's render passes. Do not reinvent.
- Do NOT touch the user's uncommitted WIP (`AdvancedTSDF.cpp/.h`, `advanced_tsdf_*.comp.glsl`, the ICP
  registration files, `Submap/Tiled*.h`, `lib/SPIRV-Reflect`). Each task `git add`s only its own files.
- Build the `vkspatial_tests` target (+ `isosurface_viewer` for the viz). New `.cpp`/shader needs a
  reconfigure. Never `rm -rf build`. Commit trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
  Do not push.

## Acceptance / Testing

- The suite stays green; the new connectivity + GPU-extractor tests pass. Each GPU extractor matches its
  CPU twin (the consistency gate). The viewer compiles and the connectivity overlay + counts are wired
  (visual output needs a display; the automated gate is compile + the connectivity/consistency tests).

## Non-goals

- No GPU `emc`/`dc`/`dmc`/`cms` (CPU only — deferred; connectivity/QEF on MoltenVK is out of scope here).
- No GPU weld / GPU connectivity analysis (CPU post-process; a future optimization).
- No new isosurface algorithm; no `SurfaceMesh` rename.
