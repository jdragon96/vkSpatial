# Voxel-Fill Debugger — Per-Frame TSDF Voxel Accumulation Viewer (Design)

**Date:** 2026-08-01
**Status:** approved (design), spec for implementation

## Goal

A live debug tool that shows the TSDF **voxel-filling process frame by frame** during integration, so
coverage gaps and holes are visible and diagnosable: which frame fills which voxels, where weight is
too low to survive extraction, and where the surface never gets covered. Reads a folder of scan-frame
PLYs, integrates them one at a time into `AdvancedTSDF`, and after each frame renders the occupied
voxels (as colored points) with the input frame and camera.

## Scope (v1)

- **Single-window `AdvancedTSDF` only.** Its `DownloadEntries()` returns per-voxel
  `{center, direction, tsdf, weight, normal}` in one call — simple to inspect. The scene must fit one
  512³ window (e.g. `scans/dragon` at voxel ≥ ~0.5). If the requested voxel needs > 512 voxels/axis,
  print the minimum feasible voxel and exit (same guard `tsdf_folder_eval` uses).
- **Tiling is out of scope (follow-up).** `TiledAdvancedTSDF` has no `DownloadEntries` and would need
  per-tile download/merge; deferred.
- **Point rendering only** (reuse `PointCloudPass`); no new instanced-cube render pass.

## Architecture

A new standalone example, `example2/voxel_fill_debugger.cpp`, composed from proven pieces:

- **Data path** (from `tsdf_folder_eval.cpp`): tolerant `readPly` + `estimateCamera` to load
  `frame_*.ply` (points + normals; per-frame camera estimated from centroid + mean-normal).
- **Render scaffold** (from `object_scan_viewer.cpp` / `chair_tsdf_viewer.cpp`): `Engine::Render`
  `Application`/`Scene`/`Camera`/`RenderGraph` + `PointCloudPass` (4 sets) + `ImGuiPass` (`SetUi`),
  trackball mouse/keys, manual frame loop.
- **New logic** (this tool): frame-scrub integrate into `AdvancedTSDF`, `DownloadEntries()` after each
  frame, diff to track per-voxel first-fill frame + new-this-frame, field→color mapping, weight
  thresholding.

The pure host-side logic is factored into a small header `example2/VoxelFillDebug.h` (no Vulkan, no
render deps) so it is unit-testable:

- `struct VoxelKey { int x, y, z; uint8_t dir; }` + hash/eq — quantized `round(center/voxel)` + dir.
- `class FillTracker` — `std::unordered_map<VoxelKey,int> firstFrame`. `update(entries, frameIdx) →
  std::vector<char> isNew` (parallel to `entries`): marks entries whose key was not seen before and
  records `firstFrame[key]=frameIdx`. `reset()` clears it (for scrub-back replay).
- Free functions (pure): `Rgba tsdfColor(float tsdf, float trunc)`, `Rgba weightColor(float w, float
  wMax)`, `Rgba fillFrameColor(int frame, int nFrames)`, `Rgba directionColor(uint8_t dir)`,
  `bool belowThreshold(float weight, float wThresh)`. `Rgba = std::array<uint8_t,4>`.

## Data flow (frame scrub)

`showFrame(N)`:
1. If `N < shownFrame` (scrubbed back): `tsdf.Reset(); tracker.reset(); shownFrame = -1;` then replay.
2. For each frame `f` in `(shownFrame, N]`, in order:
   `tsdf.Integrate(frames[f].pts, frames[f].nrm, frames[f].cam)`; then
   `entries = tsdf.DownloadEntries()`; then `isNew = tracker.update(entries, f)`.
   Running `update` after every step keeps `firstFrame` exact (each voxel's first-fill frame is
   whichever step first added it). The `entries`/`isNew` from the LAST step (frame N) are what get
   rendered, so "new-this-frame" = voxels first filled by frame N.
   (Normal forward playback advances one frame, so the loop runs once; the multi-step path only
   happens on a scrub jump.)
3. Build point sets from the last step's `entries` + `isNew` under the current color mode and weight
   threshold; upload.

`DownloadEntries()` may return up to ~1M entries; diffing + rebuilding sets is O(entries) per frame —
fine for step-through debugging (hundreds of ms). Merge (dedup) is NOT applied — raw per-voxel
entries are the debug subject.

## Point sets (PointCloudPass, kMaxSets=4)

- **0 occupied** — every filled voxel so far; color = current mode; voxels `belowThreshold` are dimmed
  to dark grey (or hidden via a toggle) to show extraction drop-outs.
- **1 new-this-frame** — voxels first filled by frame N (bright highlight, e.g. yellow), overriding
  set 0's color for those points, so each frame's contribution is visible.
- **2 input** — frame N's input points (from the PLY), a distinct dim color.
- **3 camera** — the frame-N camera marker (short segment toward the scene, like object_scan_viewer).

## Color modes (runtime radio; recompute from downloaded fields, no re-integrate)

- **tsdf sign** — `+d` red → `−d` blue, `|d|<0.1·trunc` white (surface band); reveals SDF structure /
  zero-crossing.
- **weight (heat)** — blue (low) → red (high), normalized to observed max; low weight = weakly
  observed = hole candidate.
- **fill-frame** — hue by `firstFrame[key]/nFrames`; shows coverage growth and never-filled regions.
- **direction** — 6 fixed axis colors; shows per-direction storage.

## Hole-debugging device

A **weight-threshold slider** (`wThresh`) matching the extraction's min-weight cutoff: entries with
`weight < wThresh` are dimmed/hidden in set 0. This directly shows *which voxels will be dropped at
extraction → become holes*. A stat shows the below-threshold count and fraction.

## ImGui panel

Frame slider (scrub) · play/pause · restart · fps · color-mode radio · weight-threshold slider ·
layer toggles (occupied/new/input/camera, and "hide below-threshold") · stats: frame N/total,
occupied voxels, new-this-frame, filled fraction (occupied / max-occupied-at-last-frame), below-thresh
count, current voxel/trunc.

## CLI

```
voxel_fill_debugger --dir <folder> [--voxel v] [--trunc t] [--no-p2p] [--conf L] [--hermite]
                    [--wthresh w] [--dump] [--no-view]
```
`--dump` / `--no-view`: headless — integrate frame-by-frame, print per-frame stats (occupied, new,
below-thresh), no window. Same window-fit guard as `tsdf_folder_eval`.

## Testing

- **Unit (`test/test_voxelFillDebug.cpp`, headless, no GPU):** synthetic `AdvancedEntry` vectors →
  - `FillTracker.update` marks exactly the newly-seen keys, records correct `firstFrame`, and
    `reset()` clears it (re-marks everything new on replay).
  - Color functions: tsdf sign boundaries (+/−/surface), weight heat endpoints, fill-frame hue spread,
    direction 6 distinct colors; `belowThreshold` boundary.
- **Headless smoke:** `voxel_fill_debugger --dir scans/dragon --voxel 0.5 --dump` prints monotonically
  non-decreasing occupied counts and a plausible new-per-frame series (verified by eye / a scripted
  check that occupied is non-decreasing and final > 0).
- Live window reuses the verified object_scan_viewer/chair_tsdf_viewer scaffold (not unit-tested).

## Files

- Create `example2/VoxelFillDebug.h` — pure host logic (VoxelKey, FillTracker, color/threshold fns).
- Create `example2/voxel_fill_debugger.cpp` — CLI, folder read, integrate/scrub, download/diff,
  render scaffold, ImGui.
- Create `test/test_voxelFillDebug.cpp` — unit tests for the pure logic.
- Modify `example2/CMakeLists.txt` — add the target inside the glslc guard (links `Engine::Render
  imgui Engine::Core`, `PointCloudPass.cpp ImGuiPass.cpp`, `add_compiled_shaders(... pointcloud.vert
  pointcloud.frag)`), mirroring `object_scan_viewer`.

## Global constraints

- **Reuse, don't re-invent:** copy `readPly`/`estimateCamera` from `tsdf_folder_eval.cpp` and the
  render-loop/trackball/camera-marker pattern from `object_scan_viewer.cpp`; do not modify those files.
- **No new third-party dependencies.**
- **AdvancedTSDF is used read-only via its public API** (`Build`, `Set*`, `Integrate`,
  `DownloadEntries`, `Reset`); no engine changes.
- **Single-window only**; guard `axisVox > 512` with the minimum-voxel message, like `tsdf_folder_eval`.
- **Build/run (macOS):** `VULKAN_SDK=/usr/local cmake -S . -B build`; build the example target;
  conda-on-PATH shells need `-DCMAKE_IGNORE_PREFIX_PATH=/opt/homebrew/anaconda3` only for the test
  target (GTest). See project build-gotchas.
- **Git policy:** commit only when the user asks.

## Out of scope (follow-ups)

- Tiled backend (`TiledAdvancedTSDF` per-tile download).
- Instanced-cube voxel rendering.
- Per-voxel click inspection (sumDW/sumW/sumN readout) and ghost-routing/tile visualization (the other
  two debug targets considered).
