# AdaptiveVoxelGrid Design (variance-adaptive multi-resolution TSDF)

**Goal:** A runtime 2-level variance-adaptive single-field TSDF grid — integrate at a fine
resolution, measure per-voxel TSDF variance, downsample low-variance blocks to a coarse
level, and extract a multi-resolution surface mesh via Marching Cubes with transitional-voxel
handling. Implements the core of *"Resolution Where It Counts: Hash-based GPU-Accelerated 3D
Reconstruction via Variance-Adaptive Voxel Grids"* (De Rebotti et al., 2025, "MrHash").

**Architecture:** GPU fine integration (reuse `SimpleTSDF`) + CPU-orchestrated variance merge
+ CPU multi-resolution Marching Cubes. New class `Engine::Spatial::AdaptiveVoxelGrid` that
*composes* a `SimpleTSDF` for the fine level.

**Non-goals (deferred):** N-level (>2) resolution heaps; full-GPU block reallocation; GPU
multi-res MC; streaming; 3D Gaussian Splatting rendering; directional × resolution combine.

---

## 1. Background

### 1.1 What the paper does (relevant core)
- Single flat hash table storing **multi-resolution voxel blocks** (not an octree). Voxels of
  the same size live in 8³ blocks; a per-block resolution level indexes into that level's heap.
- Each voxel stores TSDF `D`, weight `W`, color, and **variance σ²** of the running mean.
- Variance drives resolution: **start finest**, aggregate per-block variance, and **downsample
  (merge) low-variance blocks to a coarser level** (Fig 4); high-variance blocks stay fine.
- Multi-resolution Marching Cubes with **transitional-voxel** handling at fine↔coarse
  boundaries (Fig 5/6): finer-favoring SDF interpolation, coarse-voxel truncation along shared
  faces, and a vertex-collapse step.

### 1.2 Current codebase state
- `SimpleTSDF` (`src/Engine/Spatial/SimpleTSDF.{h,cpp}`): single-field, single-resolution,
  per-voxel hash. Integrate kernel `voxel_tsdf_integrate.comp` already accumulates the second
  moment (`TSDFEntry {key, sumDW, sumW, sumD2}`). Exposes:
  - `Build(ctx, voxelSize, truncation, hashCapacity, maxPoints)`
  - `Integrate(points, cameraPos)` (unweighted; also a normal-weighted overload)
  - `DownloadVoxels() -> std::vector<VoxelStat{ Eigen::Vector3f center; float tsdf; float weight; float variance; }>`
  - `ExtractPointCloud(maxTris) -> OrientedPointCloud` (existing single-res Marching Cubes)
  - `Reset()`, `FilledCount()`
- The variance-adaptive concept is otherwise **offline only** (benchmark selection); there is
  **no runtime multi-resolution storage** (`SimpleTSDF.h`: "no multi-resolution storage yet").
  `AdaptiveVoxelGrid` is exactly that missing runtime structure.

---

## 2. Math

### 2.1 Per-voxel variance (parallel-atomic ≡ paper's Welford)
The fine integrate accumulates, per voxel, unweighted (`w=1`, as the paper fixes `w_k=1`):

$$\text{sumDW}=\sum_k d_k,\quad \text{sumW}=\sum_k 1 = N,\quad \text{sumD2}=\sum_k d_k^2$$

$$\bar D = \frac{\text{sumDW}}{\text{sumW}},\qquad \sigma^2 = \frac{\text{sumD2}}{\text{sumW}} - \bar D^2 = \mathbb{E}[d^2]-\mathbb{E}[d]^2$$

This equals the paper's Welford result $\sigma_i^2 = S_{2,i,k}/k$ (Eq 5–6) exactly for `w=1`;
the parallel-atomic form is chosen because Welford's running update is not atomic-safe under
many threads writing one voxel. (Documented equivalence, not an algorithm change.)

### 2.2 Downsample (merge) criterion
Partition fine space into coarse blocks of `B=2×2×2` fine voxels (coarse voxel size `2h`).
For coarse block `c` with occupied fine voxels `F_c`:

$$\bar\sigma^2_c = \frac{1}{|F_c|}\sum_{i\in F_c}\sigma_i^2$$

If $\bar\sigma^2_c < \theta$ (variance threshold) **and** `|F_c|` ≥ `minOccupancy` → the block is
**coarse**: replace its fine voxels with one coarse voxel whose TSDF is the weight-averaged
fine value $\bar D_c = \frac{\sum_i W_i D_i}{\sum_i W_i}$. Otherwise the block stays **fine**.
`θ` is settable directly or as a percentile of the observed `σ²` distribution.

---

## 3. Architecture & data model

### 3.1 Composition
`AdaptiveVoxelGrid` owns one internal `SimpleTSDF m_fine` (voxel size `h`). It adds:
- CPU **mixed grid** assembly (variance merge, §2.2),
- CPU **multi-resolution Marching Cubes** (§5).

No new GPU shader is required for integration — the fine pass reuses `voxel_tsdf_integrate.comp`.

### 3.2 Mixed grid (CPU representation)
```cpp
struct MixedVoxel {
    Eigen::Vector3f center;   // world-space center
    float tsdf;               // signed distance value (world units)
    float weight;             // accumulated weight
    float size;               // voxel edge length: h (fine) or 2h (coarse)
    uint8_t level;            // 0 = fine, 1 = coarse
};
```
Built once per `ExtractMesh()` (or on demand + cached, invalidated by `Integrate`/threshold
change). Fine voxels are keyed by integer fine-coordinate; coarse voxels by coarse-coordinate.

---

## 4. Pipeline / data flow

1. **Fine integrate (GPU):** `m_fine.Integrate(points, cameraPos)` per frame — accumulates
   `sumDW/sumW/sumD2`.
2. **Readback (GPU→CPU):** `m_fine.DownloadVoxels()` → `{center, tsdf, weight, variance}`.
3. **Variance merge (CPU):** bucket fine voxels into `2³` coarse blocks; compute `σ̄²_c`;
   decide fine vs coarse (§2.2); assemble the `MixedVoxel` set.
4. **Multi-res MC (CPU):** run Marching Cubes over the mixed grid with transitional handling
   (§5) → `Mesh{ vertices, triangles, normals }`.

---

## 5. Multi-resolution Marching Cubes + transitional voxels

Base: extend the existing `SimpleTSDF` MC (`ExtractPointCloud`) which already does per-cell
zero-crossing + area-weighted vertex normals. The multi-res additions:

- **Fine region:** standard MC per fine cell (8 fine-voxel corners), trilinear SDF, unchanged.
- **Coarse region:** MC per coarse cell (8 coarse-voxel corners at `2h`).
- **Transitional cells (fine↔coarse boundary):** a corner may lack a same-level neighbor.
  - **SDF at a missing corner:** finer-favoring interpolation (Fig 5) — if a fine value exists
    at/near the corner use it; else interpolate from available same-level neighbors; else fall
    back to the coarse value. Never leave a corner undefined (skip cell if unresolvable).
  - **Coarse-voxel truncation (Fig 6):** on faces shared with finer neighbors, clip the coarse
    cell to the finer partition so MC does not emit overlapping faces in the shared region.
  - **Vertex collapse:** weld MC vertices within a small epsilon (`0.25·h`) after emission to
    remove near-duplicate boundary vertices and reduce cracks.
- **First-cut fidelity target:** "no gross cracks / no duplicate faces" at boundaries, not a
  formal watertight guarantee (documented limitation; full GPU multi-res MC deferred).

---

## 6. Public API (`src/Engine/Spatial/AdaptiveVoxelGrid.{h,cpp}`)

```cpp
namespace Engine::Spatial {
struct AdaptiveMesh {                       // extraction output
    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector3i> triangles;
    std::vector<Eigen::Vector3f> normals;   // per-vertex
};

class AdaptiveVoxelGrid {
public:
    void Build(Engine::Core::Context &ctx, float fineVoxelSize, float truncation,
               uint32_t hashCapacity = 1u<<20, uint32_t maxPoints = 1u<<17);
    void Integrate(const std::vector<Eigen::Vector3f> &points,
                   const Eigen::Vector3f &cameraPos = Eigen::Vector3f::Zero());
    void SetVarianceThreshold(float sigma2);            // absolute
    void SetVariancePercentile(float p /*0..1*/);       // or percentile of observed σ²
    void SetMinOccupancy(uint32_t n);                   // block coarsen gate (default 4)

    AdaptiveMesh ExtractMesh();                         // multi-res MC over the mixed grid
    std::vector<MixedVoxel> DownloadMixedVoxels();      // inspection/visualization
    size_t FineCount() const;                           // fine voxels kept
    size_t CoarseCount() const;                         // coarse voxels
    void Reset();
};
} // namespace Engine::Spatial
```
`ExtractMesh`/`DownloadMixedVoxels` build (and cache) the mixed grid; `Integrate` and any
`Set*` invalidate the cache.

---

## 7. Testing strategy (measure-first)

`test/test_adaptiveVoxelGrid.cpp` (gtest, in `vkspatial_tests`), synthetic fixtures
(reuse `example2/shape_fixtures.h` sampler or the existing cube/cylinder generators):

1. **Variance localization:** integrate a cube; flat-face blocks have `σ̄²` below a curvature/
   edge block's `σ̄²` by a clear margin (reuses the established finding; assert ratio > 3× at a
   fine voxel size where the signal is reliable).
2. **Adaptive coarsening:** with a mid threshold, flat regions coarsen while edges stay fine:
   `CoarseCount() > 0`, and edge-adjacent blocks remain fine (checked by querying mixed voxels
   near a known edge → all `level==0`).
3. **Memory reduction:** `FineCount() + CoarseCount()` < all-fine `FilledCount()`; on a mostly-
   flat fixture, coarse blocks replace ~8× their fine voxels (assert total voxel reduction).
4. **Mesh accuracy:** Chamfer distance between `ExtractMesh()` vertices and the all-fine
   `SimpleTSDF` MC vertices is small in flat regions (within ~`0.5·h`), and edge vertices are
   preserved (no rounding beyond the fine baseline).
5. **Transitional integrity:** at fine↔coarse boundaries there are no duplicate vertices beyond
   the collapse epsilon and no zero-area / flipped triangles (assert triangle-quality + vertex
   uniqueness within epsilon).

Deterministic synthetic input (fixed sample budget) keeps these reproducible, per the repo's
measure-first discipline (cf. the FD-gradient regression that measurement caught).

---

## 8. Risks / open items
- **Transitional watertightness** is the hard part; first cut targets "no gross artifacts,"
  not formal watertight. If test 5 cannot be met with truncation+collapse, fall back to
  uniformly refining coarse cells at boundaries (more memory, but crack-free) and document it.
- **CPU MC cost** on large scenes; acceptable for the bounded high-precision use case, GPU
  multi-res MC is the deferred follow-on.
- **Percentile threshold** requires the full σ² distribution (already downloaded); absolute
  threshold is the simpler default.

---

## 9. Self-review
- **Placeholders:** none — every component has a concrete interface and formula.
- **Consistency:** `MixedVoxel`/`AdaptiveMesh` used identically in §3/§5/§6; `σ²` definition in
  §2.1 matches the `DownloadVoxels().variance` source; threshold `θ` used in §2.2 and §6.
- **Scope:** single implementation plan (one subsystem: the grid + merge + MC). 2-level and
  CPU-merge keep it bounded; N-level/GPU-merge/streaming/3DGS explicitly deferred (§ non-goals).
- **Ambiguity:** downsample criterion fully specified (block = 2³, mean σ², `θ` + minOccupancy,
  weight-averaged coarse value); transitional rules enumerated (finer-favoring interp,
  truncation, collapse epsilon `0.25h`).
