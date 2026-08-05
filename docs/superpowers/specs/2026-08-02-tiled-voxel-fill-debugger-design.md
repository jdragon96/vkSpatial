# Tiled Voxel-Fill Debugger (Design)

**Date:** 2026-08-02 · **Status:** approved (design), spec for implementation

## Goal
Let `voxel_fill_debugger` reconstruct/visualize scenes that exceed one 512³ window (e.g. `scans/dragon --voxel 0.2`, which currently errors) by using **`TiledAdvancedTSDF`** instead of a single `AdvancedTSDF`. Draw **one window box per touched tile** so the user sees the "window is smaller → more tiles allocated" behavior directly.

## Background / why it errors today
The single `AdvancedTSDF` window is a hard **512-voxels-per-axis** cap (32-bit key packs 9 bits/axis, `& 0x1FF`; out-of-range voxels are dropped in `packDirKey`). At voxel 0.2 the dragon spans 967 voxels/axis > 512 → the debugger's single-window guard refuses. `TiledAdvancedTSDF` already lifts this by tiling (touched tiles only), but it lacks `DownloadEntries()` (the debugger's per-frame voxel readout) and `Reset()` (scrub-back), and exposes no per-tile boxes.

## Engine changes — `TiledDirectionalTSDF<Backend>` (base template) + `TiledAdvancedTSDF`
Additive only; the existing `TiledCompactDirectionalTSDF` alias is unaffected (new base members are either backend-agnostic or a protected template only instantiated when used).

1. **`void Reset()`** (base, public): `m_tiles.clear();` — drops all tiles (scrub-back replay).
2. **`std::vector<std::pair<Eigen::Vector3f,Eigen::Vector3f>> CoreBoxes() const`** (base, public): for each tile, its world-space **core** AABB `[O + tile·C, O + tile·C + C)·voxel`. One box per touched tile (non-overlapping grid; the ghost margin is intentionally not drawn). Used for the per-tile window boxes.
3. **`template<class Fn> void forEachTileCore(Fn&& fn) const`** (base, protected): calls `fn(const Backend& tile, Eigen::Vector3i coreMinVox, Eigen::Vector3i coreMaxVox)` per tile. Lets the subclass aggregate with the same core-only dedup as `ExtractPointCloud`.
4. **`float voxelSize() const`** (base, protected): expose `m_voxelSize` for the subclass's floor.
5. **`std::vector<AdvancedEntry> DownloadEntries() const`** (`TiledAdvancedTSDF`, public): via `forEachTileCore`, download each tile's `AdvancedTSDF::DownloadEntries()` and keep only entries whose voxel `floor(center/voxel)` lies in that tile's core `[coreMin,coreMax)` → concatenate. Core-only filter = no cross-tile duplicate voxel-direction keys (mirrors ExtractPointCloud).

## Debugger change — `example2/voxel_fill_debugger.cpp` (always tiled)
- Replace the single `AdvancedTSDF tsdf` with `TiledAdvancedTSDF tiled`. **Remove** the single-window sizing block and the `axisVox > 512` error (tiling handles any size). `tiled.Build(ctx, voxel, trunc, tileHash, maxPts)` (no windowMinCorner — tiles auto-place on the global grid). Forward quality/p2p/conf/hermite via the tiled setters.
- `--tile-hash <N>` (default `1<<20`, ~24 MB/tile) so fine-voxel/large scenes can be sized to VRAM; `maxPts = nextPow2(maxFramePts)`.
- Scrub (`rebuildTo`): `tiled.Integrate` / `tiled.Reset` / `tiled.DownloadEntries`. `FillTracker` runs unchanged on the aggregated entries (global centers → globally unique keys after core dedup).
- Boxes (`refreshSets`): set 4 (cyan) = **all** `tiled.CoreBoxes()` rendered as `boxEdges` concatenated into one point set (N tile windows); set 5 (orange) = allocated box from aggregated entries (unchanged `entriesAabb`). Point sets 0–3 (occupied/new/input/camera) unchanged.
- `--dump`: print `tiles: N` and keep per-frame `occupied/new/below-wthresh/allocBox`; add tile count.
- ImGui: "window box (512^3)" label → "tile windows (N)"; stat shows tile count.

## Testing
- Unit (`test/test_tiledAdvancedTsdf.cpp`, add):
  - `DownloadEntriesAggregatesTilesCoreOnly`: plane spanning ≥2 tiles → `DownloadEntries()` returns >0 entries with **no duplicate `voxdbg::VoxelKey`** (core dedup); count ≈ sum over tiles of core entries.
  - `ResetClearsTiles`: integrate → `Reset()` → `TileCount()==0`, `DownloadEntries().empty()`.
  - `CoreBoxesMatchTileCount`: `CoreBoxes().size() == TileCount()`.
- Headless smoke: `voxel_fill_debugger --dir scans/dragon --voxel 0.2 --dump` now succeeds (no error), prints `tiles: N (>1)`, occupied non-decreasing, final > 0.

## Global constraints
- Additive engine changes; `TiledCompactDirectionalTSDF` behavior unchanged (its tests stay green).
- Point rendering only (reuse `boxEdges`/PointCloudPass; kMaxSets already 6).
- Build: `VULKAN_SDK=/usr/local`; tests target `vkspatial_tests` with the conda-workaround flags (`-DCMAKE_IGNORE_PREFIX_PATH=/opt/homebrew/anaconda3 -U{gflags,glog,GTest,Ceres}_DIR`).
- Git: commit only the touched files; leave the repo's other uncommitted changes (user's `.comp.glsl` shader rename, block-A1, etc.) untouched.

## Out of scope
- Tiled backend for `TiledCompactDirectionalTSDF::DownloadEntries` (only Advanced needs it).
- Streaming/eviction (per-tile hash all resident, as today).
