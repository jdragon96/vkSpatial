# SubmapAdvancedTSDF — Density-Adaptive 2-Level Detail Submaps (Design)

**Date:** 2026-08-02 · **Status:** approved (design), spec for implementation

## Goal
Recover fine detail where points are dense: keep a **base** TSDF at `baseVoxel` everywhere, and overlay a **detail** TSDF at `baseVoxel/2` only in dense sub-blocks. Both levels reuse the existing `TiledAdvancedTSDF` (the detail level's half-voxel tiles are automatically the "smaller windows"). Batch pipeline (fits `tsdf_folder_eval`).

## Why two tiled levels (not one refined window)
A base tile core is 448 base-voxels = 896 half-voxels/axis > the 512 key-encoding cap, so a whole tile cannot be refined inside one detail window. Using a second `TiledAdvancedTSDF` at half voxel lets the detail level tile itself (each detail tile = 512×½voxel physical = a smaller window), covering only dense regions.

## Architecture — `src/Engine/Spatial/SubmapAdvancedTSDF.h` (header-only, composes TiledAdvancedTSDF)
- `TiledAdvancedTSDF m_base;`   — voxel `baseVoxel`, integrates ALL points.
- `TiledAdvancedTSDF m_detail;` — voxel `baseVoxel/2`, integrates ONLY dense-block points.
- Density state: during pass 1, per **block** a point count + a set of occupied base-voxel keys; after finalize, only a `std::unordered_set<BlockKey>` of dense blocks is kept.

### Density blocks (trigger)
- A **block** is a cube of `blockVoxels` base-voxels (default 32). `blockWorldSize = baseVoxel * blockVoxels`. `blockOf(p) = floor(p / blockWorldSize)` (ivec3).
- **Dense metric** (scale-invariant): `avgPtsPerOccVoxel = pointCount / max(1, occupiedBaseVoxels) >= k` (default `k = 4`). Rationale: if each occupied base voxel is averaging ≥ k observed points, halving the voxel (~2× per axis) resolves them; flat/sparse blocks stay coarse. `occupiedBaseVoxels` = distinct `floor(p/baseVoxel)` keys seen in the block (memory bounded by occupied surface voxels).

## Data flow (batch, 2-pass)
1. **Pass 1 — density:** for every point of every frame, `AddDensity(pts)`: `block = blockOf(p)`; `++count[block]`; `occ[block].insert(voxelKey(p))`.
2. **FinalizeDensity():** for each block, mark dense if `count/max(1,occ.size()) >= k`; store dense block keys; free the count/occ maps.
3. **Pass 2 — integrate:** per frame, `m_base.Integrate(pts, nrm, cam)` (all); build a sub-list of the points whose `blockOf(p)` is dense and `m_detail.Integrate(subPts, subNrm, cam)`.
4. **ExtractPointCloud(merge):** `out = m_detail.ExtractPointCloud(merge)` (kept whole) + `m_base.ExtractPointCloud(merge)` with each base point **dropped when `blockOf(point)` is dense** (precedence dedup — detail wins). Concatenate. A minor coarse↔fine seam at block borders is accepted in v1.

## API
```cpp
class SubmapAdvancedTSDF {
  void Build(Engine::Core::Context& ctx, float baseVoxel, float truncation,
             int blockVoxels = 32, float detailPtsPerVoxel = 4.0f,
             uint32_t tileHashPerTile = 1u<<20, uint32_t maxPointsPerFrame = 1u<<17);
  void SetIntegrationQuality(const IntegrationQuality&);  // forwarded to base+detail
  void SetPointToPlane(bool);                              // forwarded
  void SetConfidenceWeight(float);                         // forwarded
  void SetHermitePosition(bool);                           // forwarded
  void AddDensity(const std::vector<Eigen::Vector3f>& pts);   // pass 1
  void FinalizeDensity();                                     // mark dense; frees accumulators
  void Integrate(const std::vector<Eigen::Vector3f>& pts,
                 const std::vector<Eigen::Vector3f>& nrm,
                 const Eigen::Vector3f& cam = Eigen::Vector3f::Zero());  // base+detail
  OrientedPointCloud ExtractPointCloud(bool merge = true) const;
  uint32_t DenseBlockCount() const;
  uint32_t BaseTileCount() const;   // m_base.TileCount()
  uint32_t DetailTileCount() const; // m_detail.TileCount()
};
```
`Integrate` before `FinalizeDensity` is a usage error (no dense set yet) — assert/guard (detail gets nothing). `Build` sets `m_detail` voxel = `baseVoxel/2`, same truncation (trunc spans the same physical band; C+2G≤512 still holds at half voxel since G scales with trunc/voxel).

## Testing — `test/test_submapAdvancedTsdf.cpp` (GoogleTest, GPU)
1. **DenseRegionGetsDetail:** a coarse plane patch + a locally 4× denser sub-patch. After `AddDensity`/`Finalize`/`Integrate`, `DenseBlockCount() > 0` and `DetailTileCount() > 0`; the dense sub-region's extracted points have ~half the nearest-neighbour spacing of the coarse region (finer). 
2. **SparseSceneNoDetail:** a uniformly sparse plane → `DenseBlockCount() == 0`, `DetailTileCount() == 0`, extract ≈ base-only.
3. **NoDoubleSurface (dedup):** in a dense region, count extracted points whose voxel is a base voxel inside a dense block == 0 (base correctly dropped there); no base+detail overlap. Total ≈ base(non-dense) + detail(dense).
4. **ForwardsSettings / FinalizeGate:** setters reach both levels (measurable), and `Integrate` before `FinalizeDensity` leaves detail empty.

## Consumers
- `example2/tsdf_folder_eval.cpp`: add `--submap` (with `--block N`, `--detail-k K`). When set: pass-1 `AddDensity` over all frames → `FinalizeDensity` → pass-2 `Integrate` → `ExtractPointCloud`; print `dense blocks: D, base tiles: B, detail tiles: T`. RMSE path unchanged.
- (Follow-up, NOT v1) `voxel_fill_debugger`: draw base tiles (cyan) + detail tiles (magenta) + dense-block boxes.

## Global constraints
- Header-only, composes `TiledAdvancedTSDF`; no engine changes to existing classes. No new third-party deps.
- 2 levels only (base + ½); recursion (¼) and online-streaming density are out of scope (v2).
- Build/test: `VULKAN_SDK=/usr/local`; tests via `vkspatial_tests` with the conda-workaround flags.
- Git: commit only new/modified files for this feature; leave the repo's other uncommitted changes untouched.

## Out of scope (follow-ups)
- Seamless coarse↔fine boundary (transitional voxels / MrHash-style) — v1 uses precedence dedup.
- Recursive/multi-level (¼, ⅛) refinement; online (per-frame) density with re-integration; per-tile streaming to bound VRAM.
