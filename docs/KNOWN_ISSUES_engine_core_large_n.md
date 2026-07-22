# RESOLVED: Engine::Core compute correctness at scale / under load

**Status:** ✅ **fixed 2026-07-23.** One-line correction in `src/shader/bvh_boundingBox.comp`
(inverted bottom-up-refit arriver condition). Regression tests added to
`test/test_spatialIndex.cpp` (`BinaryLBVHTest.LargeNKnnExactAcrossScales`,
`RepeatedBuildsStayExactUnderLoad`). Kept for the record because the original diagnosis
(below) pointed at the wrong cause.

## Original symptom

`Engine::Spatial::BinaryLBVH` / `WideBVH` returned **wrong, non-deterministic** KNN/Radius
results, degrading with N (KNN recall ~0.97 @ N=512 → ~0.62 @ N=16384) and dipping under
sustained build+query load even at N≤512.

## Actual root cause

`bvh_boundingBox.comp` does the classic Karras bottom-up AABB refit: each leaf walks to the
root; at every node an atomic visitation counter decides which of the two children-threads
merges that node. The correct rule is **the FIRST arriver stops** (its sibling subtree is
not computed yet) **and the SECOND arriver merges** (both children ready). The shader had the
condition inverted:

```glsl
int historicalVisits = atomicAdd(g_constrInfos[nodeIdx].visitationCount, 1); // returns OLD count
if (historicalVisits == 1) return;   // BUG: second arriver (old==1) returned; first merged
```

So the **first** child to reach a node merged it using the sibling's **not-yet-written**
AABB — a data race. It grew worse with N (more concurrent workgroups race), and small-N
unit tests passed only because their specific query points happened not to touch the
corrupted internal nodes. Fix:

```glsl
if (historicalVisits == 0) return;   // first arriver stops; second (old==1) merges
```

## Why the original "memory coherence" hypothesis was wrong (all tested and ruled out)

The earlier writeup blamed MoltenVK cross-workgroup memory coherence / VMA storage mode.
Each was tested on the failing repro (N up to 16384) and made **no meaningful difference**:

- **Shader optimization** — compiling `bvh_*.comp` at `-O0` instead of `-O performance`:
  recall unchanged (4096: 0.81 → 0.88, still failing).
- **Buffer storage mode** — forcing host-visible/coherent memory
  (`VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT`, Metal `shared` instead of `private`):
  no change (4096: 0.81).
- The `coherent` qualifier + `memoryBarrierBuffer()` already in the shader were fine; the
  bug was pure control-flow logic, not visibility.

Flipping the one condition took recall to **1.000 at every N (512…16384) and under load**,
with no other change (Context / Buffer / ComputePipeline all reverted to baseline).

## Impact now

`example2/bvh_benchmark`'s N≤512 cap and `knn_rec%` caveat can be lifted; large-N GPU BVH
build/query is trustworthy. The DirectionalTSDF "keep per-Integrate N ≲ 1000" mitigation is
no longer required for correctness (it was a symptom of this same refit race via the shared
`Engine::Core` compute path).
