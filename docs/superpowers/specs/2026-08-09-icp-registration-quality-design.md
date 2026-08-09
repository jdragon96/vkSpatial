# Local ICP Registration-Quality Improvements — Design

## Problem

The local ICP trackers (`GpuIcpTracker` = `"icp"`, `PointToPlaneIcpTracker` = `"icp-cpu"`) produce poor alignment quality. Root causes, in order of impact:

1. **The target is quantized voxel centers, discarding sub-voxel truth.** Both trackers build the target point cloud from `AdvancedEntry.center` — the voxel center, snapped to the map grid (e.g. 0.5 m). But `AdvancedEntry` also stores `tsdf` (the normalized signed distance) and `normal` (the stored gradient); the true surface sits at `center − tsdf·truncationDistance·normal`. Ignoring `tsdf` caps alignment precision at roughly half a voxel.
2. **Loose, hard-cutoff correspondence gating.** `maxCorrespondenceDistance = 2 × voxel` with a hard binary accept/reject and no robust weighting admits and equally trusts wrong correspondences.
3. **Coarse target normals** from the voxelized model make point-to-plane orientation imprecise.
4. **Prior is the previous pose only** (`RegistrationThread` seeds `Track` with `previousPose`, no motion model), so under real camera motion the ICP starts far from the answer.

There is currently no quantitative quality metric (`RegistrationResult` has only `fitness` = inlier ratio, no residual error), so improvements cannot be measured.

## Goals

Improve local ICP registration quality in three stacked tiers, applied to **both** trackers, each tier measurably validated:

- **Tier 1** — sub-voxel target using the stored `tsdf`.
- **Tier 2** — robust correspondences (Huber weighting, normal-compatibility rejection, tighter gate).
- **Tier 3** — coarse-to-fine correspondence-distance annealing + a constant-velocity motion model.

Measured by a **deterministic perturbation-recovery harness** (recovered-pose error + residual RMSE) plus **live RMSE** in the viewer. Built in order, `#0 measurement → #1 → #2 → #3`, each an independently testable phase.

## Non-goals

- No global↔local composition / relocalization flow (the four trackers stay independent selectable strategies; that composition is a separate future effort).
- No full multi-resolution model pyramid (coarse-to-fine is done via distance annealing — see Tier 3).
- No new dataset ingestion (the perturbation harness provides ground-truth for validation).

## Tier 0 — Measurement (built first)

The validation instrument must exist before the tiers, so each tier's effect is provable.

**RMSE in `RegistrationResult`.** Add `float rmse = 0.0f`. It is `sqrt(sumOfSquaredResiduals / numberOfInliers)` where each residual is the point-to-plane distance `(transformedSourcePoint − targetPoint) · targetNormal`.
- **GPU** (`icp_iterate.comp.glsl`): the per-workgroup fixed-point reduction currently has 28 slots (21 upper-triangular normal-matrix entries + 6 right-hand-side + 1 inlier count). Add a **29th slot** accumulating `Σ residual²` (same fixed-point scale as the matrix entries; magnitudes are tiny in the centred frame, no overflow). `Solve` reads it back and, after the loop, computes `rmse = sqrt(sumOfSquaredResiduals / numberOfInliers)` from the final iteration.
- **CPU** (`AlignPointToPlaneIcp`): accumulate `sumOfSquaredResiduals += residual * residual` alongside the existing inlier count; compute the same `rmse`.

**Perturbation-recovery harness.** A gated benchmark test (mirroring the existing `GpuIcp.DISABLED_BenchmarkVsCpu` style, and the `Pipeline.GpuIcpTrackerRecoversMovingCameraPose` pattern that already builds a `ModelSnapshot` + `Frame` + prior and calls `Track`). It:
- builds a **realistic model**: a multi-plane corner surface as `AdvancedEntry` records whose `center` is **snapped to a coarse voxel grid** but whose `tsdf`/`normal` encode the true sub-voxel surface (+ small additive noise) — so the harness exercises the actual quantization problem *and* the sub-voxel data Tier 1 relies on;
- makes the source frame = the true surface under a **known** SE(3) perturbation (translation + rotation);
- **drives the tracker's `Track(frame, &modelSnapshot, prior)`** (not just `Solve`) so the tracker-level tiers — sub-voxel target construction (Tier 1), robust/normal rejection and annealing (Tiers 2–3, via `Solve`) — are all exercised end to end; reports **recovered-pose error** (translation norm, rotation angle vs the known inverse) **+ residual RMSE**.
- Each later tier must reduce (or hold) these numbers; the harness is the regression gate for the whole effort. The constant-velocity motion model (Tier 3) lives in `RegistrationThread`, not `Track`, so it is validated by a separate small `RegistrationThread`-level check (a straight-line motion sequence where the predicted prior must beat the previous-pose prior), not this harness.

**Live RMSE** shown in the viewer's stats panel (the tracker already carries per-stage timing there; add the latest tracking RMSE).

## Tier 1 — Sub-voxel target (CPU-side; no shader change)

In both trackers' target construction, replace the raw center with the stored sub-voxel surface point:

```
surfacePoint = entry.center − entry.tsdf * truncationDistance * entry.normal
```

- Add `float truncationDistance = 0.0f` to `ModelSnapshot`, populated in `IntegrationThread::buildSnapshot` from the map config's truncation (the same value used by the integrate shader), next to the already-added `voxel`.
- `tsdf` is normalized to [−1, 1] (`voxel2point / truncateDistance` in the integrate shader), so the world offset is `tsdf * truncationDistance`. The **sign** must be verified against the integrate/extract convention during implementation — the harness's recovered-pose error confirms it (a wrong sign makes recovery worse, which the test will flag).
- The GPU path uploads these already-projected surface points, so no shader change is needed for this tier.

## Tier 2 — Robust correspondences

- **Huber robust weighting.** Each correspondence contributes with weight `w(residual)` — `1` for `|residual| ≤ huberScale`, `huberScale / |residual|` beyond — applied to both the normal-matrix and right-hand-side contributions, replacing the current hard binary accept. `huberScale ≈ voxel` (tunable).
- **Normal-compatibility rejection.** Skip a correspondence when `sourceNormal · targetNormal < cos(compatibilityAngle)` (~30°). This requires the source normals on the GPU: add a **source-normals storage buffer** (new binding) that `GpuPointToPlaneIcp` uploads from `frame.nrm`, and the compatibility check in the shader. The CPU path uses `frame`/source normals directly.
- **Tighter base gate:** default `maxCorrespondenceDistance ≈ 1 × voxel` (Tier 3 then replaces the fixed gate with a schedule).

## Tier 3 — Coarse-to-fine annealing + motion model

**Coarse-to-fine annealing** — reconciled with the per-solve grid hoist so it does NOT reintroduce a per-iteration grid rebuild:
- Build the correspondence grid **once** at the **coarsest** cell size (the widest annealed distance), as the current hoist already does.
- Pass a **per-iteration `currentMaxCorrespondenceDistance`** as a push-constant that starts wide (~2–3 × voxel, large convergence basin) and shrinks geometrically toward ~0.5–1 × voxel over the iterations. The grid's neighbor scan stays fixed; only the distance **filter** (and the Huber scale) tighten each iteration. This keeps the single upload/grid-build per solve intact.
- The CPU `AlignPointToPlaneIcp` mirrors this: grid at the coarsest cell, per-iteration distance filter.

**Constant-velocity motion model** in `RegistrationThread::Run`:
- Track `previousPose` and `previousPreviousPose`. When both are available from valid tracks, seed the prior as `previousPose * (previousPreviousPose.inverse() * previousPose)` (apply the last relative motion again) instead of just `previousPose`. First two frames (or after an invalid track) fall back to `previousPose`.
- This lives in `RegistrationThread` (tracker-agnostic), so it benefits every local tracker at once.

## Cross-cutting

- **Both trackers**: every tier applies to the GPU (`GpuIcpTracker` + `GpuPointToPlaneIcp` + `icp_iterate.comp.glsl`) and the CPU (`PointToPlaneIcpTracker` + `AlignPointToPlaneIcp`) paths. The motion model is shared in `RegistrationThread`.
- **MoltenVK**: the new GPU accumulator (Σ residual²) and any per-correspondence weight use the existing int fixed-point reduction pattern (no float atomics).
- **Naming**: all new/edited code uses full descriptive names — `maxCorrespondenceDistance`, `sumOfSquaredResiduals`, `sourceNormal`, `truncationDistance`, `huberScale` — not abbreviations.

## Acceptance / Testing

This changes behavior (it improves quality), so the gate is:
- The perturbation harness shows **each tier reduces or holds** recovered-pose error and residual RMSE versus the prior tier (baseline captured at Tier 0).
- The existing suite stays green (~238 pass / 1 skip); the ICP correctness tests (`GpuIcp.SolveMatchesCpuOnCorner`, the tracker tests) still pass within tolerance — GPU and CPU paths stay numerically consistent with each other.
- Each phase builds the three targets (`vkspatial_tests`, `voxel_fill_debugger`, `registration_chair_demo`).

## Risks

- **Sub-voxel sign** (Tier 1) inverted → worse alignment. Mitigation: the harness immediately shows regression; verify against the integrate/extract convention.
- **Annealing vs grid hoist** — naive per-iteration `maxCorrespondenceDistance` change would force a per-iteration grid rebuild (undoing the perf fix). Mitigation: build the grid at the coarsest cell once; anneal only the distance filter (specified above).
- **GPU/CPU divergence** — the two paths must apply identical robust weighting / rejection / annealing so `SolveMatchesCpuOnCorner` stays within tolerance. Mitigation: keep the residual/weight formulas identical and test both.
- **Motion model overshoot** on erratic motion → worse prior than previous-pose. Mitigation: only apply after consecutive valid tracks; the harness/live RMSE will show if it hurts.
