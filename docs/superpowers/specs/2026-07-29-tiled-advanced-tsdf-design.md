# Tiled AdvancedTSDF — Incrementally-Scalable TSDF (Design)

**Date:** 2026-07-29
**Status:** approved (design), spec for implementation

## Goal

Lift `AdvancedTSDF`'s single 512³-voxel window limit with **lazy spatial tiling**, so a scan
can grow into new regions indefinitely while only *touched* tiles are allocated. All of
AdvancedTSDF's quality — compact flat-hash storage, point-to-plane integration, stored-gradient
extraction, and the A1 (confidence weight) / A2 (Hermite position) toggles — is preserved
per tile. **VRAM-only**: no host/disk residency tier (explicitly out of scope, see §7).

## Background

The repo already has `TiledCompactDirectionalTSDF` — a CPU coordinator that tiles space over
`CompactDirectionalTSDF` tiles (each a single 512³ window), with lazy allocation, ghost-margin
routing, and core-only extraction. It works and is tested (common-voxel validation + chair demo).

Two facts make generalization trivial:

1. `CompactDirectionalTSDF::Build` and `AdvancedTSDF::Build` have **identical signatures**:
   `Build(ctx, voxel, trunc, hashCapacity, maxPoints, windowMinCorner)`. Likewise
   `ExtractPointCloud(maxCandidates, merge)`, `Integrate(points, normals, cameraPos)`,
   `SetIntegrationQuality`, `SetPointToPlane`, `FilledCount`, `Reset` all match.
2. The **only** API delta is AdvancedTSDF's two extra setters: `SetConfidenceWeight(float)` (A1)
   and `SetHermitePosition(bool)` (A2).

So the tiling coordinator can be a template parameterized on the backend, and the A1/A2 delta is
handled by a per-tile configuration hook set by an AdvancedTSDF-specific subclass — no `if
constexpr`/detection idiom, no C++20 dependency.

## Architecture

```
template<class Backend> class TiledDirectionalTSDF        // new: generalized CPU coordinator
using   TiledCompactDirectionalTSDF = TiledDirectionalTSDF<CompactDirectionalTSDF>;   // migrated
class   TiledAdvancedTSDF : public TiledDirectionalTSDF<AdvancedTSDF>                 // + A1/A2
```

- **`TiledDirectionalTSDF<Backend>`** owns the tiling: tile map, geometry, ghost routing, core
  extraction, and the common config (`SetIntegrationQuality`, `SetPointToPlane`). Header-only.
- Tile creation runs a **configuration hook** `std::function<void(Backend&)> m_configureHook`
  (protected, default empty) after applying the common config. This is the single extension
  point subclasses use to forward backend-specific settings.
- **`TiledAdvancedTSDF`** is a thin subclass that stores A1/A2 state and, in its constructor, sets
  `m_configureHook` to forward `SetConfidenceWeight(m_confWeight)` and
  `SetHermitePosition(m_hermite)` to each new tile. It exposes `SetConfidenceWeight` /
  `SetHermitePosition` (which also re-apply to already-created tiles, mirroring how
  `SetPointToPlane` is stored-then-applied).
- **`TiledCompactDirectionalTSDF`** becomes a `using` alias of the template. Its old `.cpp`
  tiling logic moves verbatim into the template header; behavior is unchanged and its existing
  tests now validate the shared template.

### Tile geometry (unchanged from the proven scheme)

- Core side **C = kCore = 448** voxels (the tile's owned region).
- Ghost margin **G = ⌈trunc / voxel⌉ + 1** voxels; per-tile 512³ window = core + 2G. Enforced
  `C + 2G ≤ 512` (throws in `Build` if a too-large truncation/voxel ratio violates it).
- Global voxel origin **O = (0,0,0)**. Tile of voxel `v` = `floorDiv(v − O, C)` per axis.
- Tile window: `originVoxel = O + tile·C − G`; `windowMinCorner = originVoxel · voxel`.

### Data flow

- **`Build(ctx, voxel, trunc, hashCapPerTile = 1<<22, maxPtsPerFrame = 1<<17)`** — stores params,
  computes G, enforces `C + 2G ≤ 512`. Tiles are lazy (none created yet).
- **`SetIntegrationQuality(q)` / `SetPointToPlane(on)`** — stored; applied to each tile on
  creation, and re-applied to existing tiles (so calling after some integration still takes
  effect). `TiledAdvancedTSDF::SetConfidenceWeight/SetHermitePosition` behave the same way.
- **`Integrate(points, normals, cameraPos)`** — for each point: `v = floor(point/voxel)`, find
  home tile `tileOf(v)`, and route to the home tile **plus** any axis-neighbor tile whose ghost
  band the point falls into (`local[a] < G` → −1 neighbor; `local[a] ≥ C − G` → +1 neighbor;
  up to 8 tiles for a corner point). Accumulate per-tile point/normal sublists, lazily
  create+`Build`+configure each touched tile, then `tile.Integrate(sublist, cameraPos)`. Ghost
  routing guarantees each tile's core carries the full truncation band → no seam.
- **`ExtractPointCloud(merge = true)`** — for each tile: `tile.ExtractPointCloud(1<<21, merge)`,
  keep only points whose voxel lies in that tile's **core** `[tile·C, tile·C + C)` (drops ghost
  duplicates), concatenate.
- **`FilledCount()`** — sum over tiles (slightly over-counts due to ghost overlap; documented).
- **`TileCount()`** — number of allocated tiles (diagnostic).

### Backend contract (compile-time, satisfied by both backends)

`Build(ctx, float, float, uint32_t, uint32_t, const Vector3f&)`, `SetIntegrationQuality(const
IntegrationQuality&)`, `SetPointToPlane(bool)`, `Integrate(const vector<Vector3f>&, const
vector<Vector3f>&, const Vector3f&)`, `ExtractPointCloud(uint32_t, bool) → OrientedPointCloud`,
`FilledCount() → uint32_t`, `Reset()`. AdvancedTSDF additionally: `SetConfidenceWeight(float)`,
`SetHermitePosition(bool)` (reached only through `TiledAdvancedTSDF`'s hook).

## Components / file structure

- **Create** `src/Engine/Spatial/TiledDirectionalTSDF.h` — the `template<class Backend>`
  coordinator (all tiling logic, header-only).
- **Create** `src/Engine/Spatial/TiledAdvancedTSDF.h` — `class TiledAdvancedTSDF :
  public TiledDirectionalTSDF<AdvancedTSDF>` with A1/A2 setters + configure hook.
- **Migrate** `src/Engine/Spatial/TiledCompactDirectionalTSDF.h` → `#include
  "TiledDirectionalTSDF.h"` + `using TiledCompactDirectionalTSDF =
  TiledDirectionalTSDF<CompactDirectionalTSDF>;`. **Delete** `TiledCompactDirectionalTSDF.cpp`
  (logic moved to the header); drop it from any CMake source list / update globs.
- **Create** `test/test_tiledAdvancedTsdf.cpp` — new correctness tests (§ Testing).
- **Modify** `example2/tsdf_folder_eval.cpp` — auto-tiled fallback (§ eval integration).

## Data flow correctness — invariants

1. **Bit-identical where a single window fits.** With a scene inside one core and one tile
   allocated, `TiledAdvancedTSDF` extraction equals a standalone `AdvancedTSDF` (same voxel,
   trunc, settings) — the tiling adds no error where it isn't needed.
2. **Seam-free across tiles.** Ghost routing (integrate boundary points into adjacent tiles) +
   core-only extraction (dedup) → a surface spanning ≥2 tiles has no gap and no duplicate band at
   the boundary.
3. **Lazy growth.** Only tiles a point (or its ghost) touches are allocated; `TileCount` reflects
   exactly the touched set.

## Testing

New `test/test_tiledAdvancedTsdf.cpp` (GoogleTest, headless, follows `test_advancedTsdf.cpp`):

1. **`TilingMatchesSingleWindowWhereItFits`** — integrate a small plane/cube patch that fits one
   core; assert `TiledAdvancedTSDF` extract ≈ standalone `AdvancedTSDF` extract (same point count
   within tolerance, max nearest-distance ≈ 0). The bit-for-bit correctness bar.
2. **`PlaneSpanningTilesIsSeamFree`** — integrate a plane wider than one core (≥2 tiles); assert
   extracted points cover the full span with no boundary gap (max nearest-neighbor gap ≤ ~voxel)
   and no duplicate slab (point density near the seam ≈ interior).
3. **`OnlyTouchedTilesAllocated`** — integrate two well-separated patches; assert `TileCount`
   equals the expected touched-tile count (not the bounding-box tile count).
4. **`A1A2SettersReachTiles`** — set `SetConfidenceWeight` to 0 vs default and assert the extracted
   point count / distribution changes (A1 measurably weights out far-band points, as on the single
   window); set `SetHermitePosition(true)` and assert it runs without error and changes positions.
5. Existing `TiledCompactDirectionalTSDF` tests are retargeted onto the template (validates the
   migration is behavior-preserving).

`CMakeLists.txt` for tests is glob-based; the new `.cpp` is picked up on reconfigure.

## eval tool integration

`example2/tsdf_folder_eval.cpp`: replace the current hard `voxel too fine` error (returned when
`axisVox > 512`) with an **automatic tiled path**:

- If the object fits one window (`axisVox ≤ 512`): keep the single `AdvancedTSDF` path (fast, low
  memory) — unchanged.
- If it exceeds one window: build a `TiledAdvancedTSDF` (per-tile hashCap sized as today, placed by
  the tiler), integrate/extract through it, and print a one-line note (`window too small → tiled:
  N tiles`). Forward the same `--voxel/--trunc/--no-p2p/--conf/--hermite` settings.
- `--single` flag forces the single-window path (restores the hard error when too fine), for
  isolating single-window behavior.

## Global constraints

- **No new third-party dependencies.** Header-only coordinator, Eigen + STL only.
- **VRAM-only** — no host store, no residency backend, no disk. (Out of scope; see §7.)
- **Behavior-preserving migration** — `TiledCompactDirectionalTSDF`'s public API and results are
  unchanged; existing tests must pass without modification (beyond include path if needed).
- **User GLSL/code style** — match surrounding Engine::Spatial conventions; no shader changes
  (tiling is CPU-side; AdvancedTSDF's shaders are reused as-is).
- **Git policy** — do not commit unless the user explicitly asks.

## §7 Out of scope (future tiers)

Per the chosen ceiling (VRAM-only), these are explicitly deferred and NOT built here:

- **VRAM↔host residency** — evicting cold tiles to host RAM (ROI/LRU). The next tier when scenes
  exceed VRAM; the repo's `StreamingResidencyBackend`/`UnifiedResidencyBackend` are the reference.
- **Disk at-rest tier** — serializing cold tiles (quantized: oct16 normal + fp16) to disk.
- **Multi-resolution / progressive refinement** — orthogonal to spatial tiling.

The tile is the natural granularity for all three, so this design is forward-compatible: a later
residency layer swaps the `unordered_map<TileKey, unique_ptr<Backend>>` for a tiered store without
touching the routing/extraction logic.
