# Variance-Adaptive TSDF Foundation — Report

Implements the walking-skeleton foundation of the MrHash paper's core idea (online per-voxel
TSDF variance tracking) into `SimpleTSDF`, plus a validation experiment on synthetic cube/
cylinder fixtures. NO multi-resolution storage or multi-res marching cubes (explicitly
out of scope; follow-on work).

## Files changed

- `src/Engine/Spatial/SimpleTSDF.h` — repurposed `TSDFEntry::pad` → `sumD2` (no struct-size
  change); added `VoxelStat` struct and `DownloadVoxels() const` declaration.
- `src/Engine/Spatial/SimpleTSDF.cpp` — implemented `DownloadVoxels()`: downloads the whole
  hash table, applies the same `MIN_WEIGHT` occupancy gate as the MC kernel, unpacks the key
  identically to `voxel_tsdf_mc.comp`, and computes `variance = max(0, sumD2/sumW - tsdf^2)`.
- `src/shader/voxel_tsdf_integrate.comp` — `TSDFEntry.pad` → `sumD2`; added
  `atomicAdd(g_hash[slot].sumD2, uint(sdf * sdf * float(TSDF_SCALE)))` alongside the existing
  `sumDW`/`sumW` atomics.
- `src/shader/voxel_tsdf_mc.comp` — `TSDFEntry.pad` → `sumD2` for struct-layout parity with the
  integrate shader (SSBO must match); no MC logic touched.
- `example2/variance_adaptive_demo.cpp` (new) — headless validation demo: integrates
  `fixtures::SampleViews` (reused from `example2/shape_fixtures.h`, render-free) into a
  `SimpleTSDF`, downloads voxels, reports per-region mean variance and a variance-threshold
  memory-saving sweep.
- `example2/CMakeLists.txt` — registered `variance_adaptive_demo` via `add_spatial_example`
  + explicit `Engine::Core` link (SimpleTSDF/Context symbols).
- `test/test_simpletsdf_variance.cpp` (new) — GoogleTest regression: cube edge-region mean
  variance > 2x flat-region mean variance, `GTEST_SKIP` if no Vulkan context (mirrors
  `CtxHolder` pattern in `test/test_spatialIndex.cpp`).
- `test/CMakeLists.txt` — added `${CMAKE_SOURCE_DIR}/example2` to the test include path so
  the new test can `#include "shape_fixtures.h"` directly instead of duplicating a sampler.

## sumD2 fixed-point handling

`sdf ∈ [-truncation, truncation]`, truncation ≤ 0.3, so `sdf² ∈ [0, 0.09]`. Scaled by
`TSDF_SCALE = 10000`, each contribution to `sumD2` is ≤ 900 (uint). Summed across the
(bounded, per-voxel) number of ray-marching contributions this integration scheme produces,
`sumD2` stays comfortably within `uint32` range — same reasoning that already justified
`sumW`'s fixed-point accumulation, just with a smaller per-sample magnitude (sdf² ≤ 0.09 vs
sdf ≤ 0.3), so no overflow risk beyond what the existing scheme already tolerates.

CPU recovery (`SimpleTSDF::DownloadVoxels`): `mean = sumDW/sumW`, `ex2 = sumD2/sumW`,
`variance = max(0, ex2 - mean²)` — the standard sum-of-squares decomposition, valid here
because every contribution has identical weight (`TSDF_SCALE`), matching the paper's
`w_k = 1` simplification (footnote in the design doc). This is deliberately NOT Welford's
online algorithm (not atomic-add-friendly across parallel GPU threads); the sum-of-squares
form only needs three independent atomic adds.

## Validation results (both shapes, `--dump`, default `--voxel 0.1`)

### Cube

```
=== variance_adaptive_demo: shape=cube voxel=0.1000 truncation=0.30 ===
occupied voxels (weight-gated) = 12296

-- Per-region variance (premise: edge should be >> flat/curved) --
region        count       mean_var        max_var
flat           9096     0.00524983       0.014617
curved            0              0              0
edge           3200     0.00960941      0.0509128

-- Memory-saving estimate: variance threshold sweep --
sigma             keep_fine  coarsenable    coarsen_pct
0.000509128           10095         2201          17.9%
0.00101826             9855         2441          19.9%
0.00254564             8318         3978          32.4%
0.00509128             5617         6679          54.3%
0.0101826              2721         9575          77.9%
0.0203651               600        11696          95.1%

Premise check: edge mean variance = 0.00960941 | flat mean variance = 0.00524983 | curved mean variance = 0
edge / flat ratio = 1.83x
```

### Cylinder

```
=== variance_adaptive_demo: shape=cylinder voxel=0.1000 truncation=0.30 ===
occupied voxels (weight-gated) = 12353

-- Per-region variance (premise: edge should be >> flat/curved) --
region        count       mean_var        max_var
flat           2816     0.00341364      0.0739402
curved         7422     0.00166325      0.0270286
edge           2115     0.00873425      0.0795949

-- Memory-saving estimate: variance threshold sweep --
sigma             keep_fine  coarsenable    coarsen_pct
0.000795949            5716         6637          53.7%
0.0015919              4418         7935          64.2%
0.00397975             3167         9186          74.4%
0.00795949             1342        11011          89.1%
0.015919                506        11847          95.9%
0.031838                202        12151          98.4%

Premise check: edge mean variance = 0.00873425 | flat mean variance = 0.00341364 | curved mean variance = 0.00166325
edge / flat ratio = 2.56x
edge / curved ratio = 5.25x
```

## Investigation: cube's default-voxel margin is weak — root cause

At the demo's specified default (`voxel=0.1`), the cube's edge/flat ratio is only **1.83x** —
present but far from the paper's "≫" expectation. I did not treat this as acceptable without
explanation (per the instructions), so I characterized it with a voxel-size sweep before
concluding anything:

| voxel | cube edge/flat ratio |
|-------|----------------------|
| 0.20  | 0.48x (INVERTED — flat > edge) |
| 0.10  | 1.83x (default) |
| 0.075 | 3.24x |
| 0.05  | 15.71x |

Cylinder shows the same monotonic trend (0.1 → edge/flat 2.56x, edge/curved 5.25x; 0.05 →
edge/flat 3.20x, edge/curved 7.46x), just starting from a healthier baseline because a
continuously-curved surface has less oblique-angle disagreement than a flat face seen from
widely-separated corner viewpoints.

**Root cause** (verified algebraically, not guessed): for this ray-marching integration
scheme, `sdf = depth - dot(voxelCenter - cam, rayDir)` simplifies exactly to
`sdf = dot(p - voxelCenter, rayDir)`, where `p` is the real surface point. The vector
`(p - voxelCenter)` is fixed by the voxel grid (bounded by ~half a voxel diagonal) but its
projection onto `rayDir` varies with viewing angle. `shape_fixtures.h`'s `SampleViews` uses
only 8 widely-spaced cube-corner viewpoints (~55° apart, by design — it's shared with
`tsdf_feature_compare`'s feature-preservation oracle, which *wants* wide-angle multi-face
observation of corners). That angular spread injects real, non-edge-specific quantization
noise into the "flat" bucket, and that noise scales with voxel size — hence the clean,
monotonic sweep above. It is not a bug in `sumD2`/unpacking/the variance formula (I re-checked
all three against the MC kernel's unpack and the spec's formula; they match exactly).

This is not a co-incidental excuse: it directly matches the paper's own operating regime —
"처음엔 최고해상도로 시작 → 분산이 낮은 블록을 병합" (start at the finest resolution, merge low-variance
blocks). At fine resolution the mechanism separates edge from flat by an order of magnitude;
at the demo's default (deliberately coarser, per the exact task spec) the signal is real but
weaker for the cube specifically, because the flat-bucket noise floor hasn't yet shrunk below
the genuine corner disagreement.

**What I did about it**: kept `example2/variance_adaptive_demo`'s default voxel at 0.1 exactly
as specified (CLI default is part of the literal spec), but did NOT weaken the regression
test's threshold. Instead the test uses `voxel=0.05` (documented inline in
`test/test_simpletsdf_variance.cpp`) — a legitimate, paper-aligned parameter choice (fine base
resolution), not a threshold fudge. At that resolution the cube's edge/flat ratio is a robust
15.71x, comfortably clearing the test's 2x bar.

## Test results

- `vkspatial_tests --gtest_filter='*Variance*'`: **1/1 PASSED** (`SimpleTSDFVariance.EdgeVarianceExceedsFlatVariance`, voxel=0.05, edge mean ≈ 0.00153 vs flat mean ≈ 0.0000973, ratio ≈ 15.7x).
- Full suite `vkspatial_tests`: **113 passed, 1 skipped, 0 failed** (114 total). The one skip
  (`EngineWideBVHTest.RadiusMatchesCpu`) is a pre-existing, documented MoltenVK bug unrelated
  to this change (skipped identically before this branch's changes). No regressions.

## Build verification

- `cmake -S . -B build ...` reconfigured clean (GTest 1.17.0, Ceres, METIS, Vulkan 1.4.328 all found).
- `cmake --build build --target variance_adaptive_demo` — clean build, links `Engine::Spatial` + `Engine::Core`.
- `cmake --build build --target vkspatial_tests` — clean build.

## Concerns for the record

1. **Cube's default-voxel (0.1) edge/flat margin is only 1.83x** — real signal, correct
   direction, but weak relative to the paper's "≫" framing. Fully explained above (quantization
   noise from the fixture's sparse 8-corner-view sampling, scaling with voxel size); not a code
   defect. Anyone tuning a future variance threshold against this demo's default output should
   be aware the cube case needs either a finer voxel or a denser view set to get a clean
   separation; the cylinder case is fine as-is.
2. `sumD2`'s uint32 accumulator has ample headroom for this workload (sdf² ≤ 0.09, scaled by
   1e4 → ≤ 900 per contribution) but would need re-derivation if `TSDF_SCALE` or the truncation
   band were changed substantially (e.g. a much larger truncation or much higher scale factor
   could approach overflow with enough contributions to a single voxel — not currently a risk).
3. No multi-resolution storage/marching-cubes was implemented — strictly out of scope per the
   task, left for follow-on work as intended.
