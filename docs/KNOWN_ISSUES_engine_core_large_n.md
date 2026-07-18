# Known issue: Engine::Core compute correctness at scale / under load

**Status:** open — surfaced 2026-07-19 by `example2/bvh_benchmark`. Not addressed in the
pluggable-BVH work (out of scope: it is an `Engine::Core` module issue, not a BVH bug).

## Symptom

`Engine::Spatial::BinaryLBVH` and `WideBVH` (both built on `Engine::Core::Buffer` +
`ComputePipeline`) return **wrong, non-deterministic** KNN/RadiusSearch results in two
regimes:

1. **At N ≳ 1000** (single build, fresh Context): results are grossly wrong and vary
   run-to-run (e.g. KNN recall ~3/16, radius returns empty where dozens are expected).
2. **Under repeated build+dispatch load** (the benchmark builds 6 warmup indices + runs
   400 queries per backend on one Context): KNN recall dips below 100% even at N ≤ 512
   (e.g. 62–75% at N=256/512), worsening with N.

## What is NOT the cause (verified)

- **Not the shaders / BVH algorithm.** The old `vkSpatial::vkBVH` (same
  `src/shader/bvh_*.comp`, via `vkComputeBase`/`vkGPUMemory`) is **correct and
  deterministic at N=1000** (temporarily ran `KNNTest.RandomPointsMatchCpuBruteForce`
  with N=1000, 3/3 passed).
- **Not the single-build path at small N.** `test/test_spatialIndex.cpp` builds one index
  per fresh Context at N≤512 and matches a CPU reference **exactly** — all pass.
- **Not cross-dispatch synchronization.** `SubmitOneShot` calls `vkQueueWaitIdle` after
  every dispatch, so build steps are fully serialized.
- **Not buffer zero-init.** Explicitly zeroing the node and construction (visitation-
  counter) buffers before the build did not help.
- **Not shared-Context-across-backends.** A fresh `Context` per backend did not help.

## Leading hypothesis

`Engine::Core::Buffer` allocates device-local storage with VMA flags that yield a Metal
(MoltenVK) storage mode which does **not** provide the cross-workgroup memory coherence
that `bvh_boundingBox.comp`'s bottom-up, atomic-visitation-counter AABB propagation
relies on. `vkGPUMemory` (old path) apparently allocates with coherent-compatible flags.
The bug only manifests once the propagation spans multiple workgroups (larger N) and/or
after sustained GPU activity, which is why the small-N single-build tests never caught it
— and why no old test caught it either (old KNN/radius correctness tests top out at
N≤500; the old "large-N" tests only validate *sorting*).

## Suggested investigation

1. Run with Vulkan validation layers + synchronization validation on a large-N build.
2. Compare the VMA `VmaAllocationCreateInfo`/usage flags and resulting `VkMemoryPropertyFlags`
   between `Engine::Core::Buffer::Allocate` and the old `vkGPUMemory::Allocate`.
3. Check whether `bvh_boundingBox.comp` needs `coherent` SSBO qualifiers +
   `memoryBarrierBuffer()` that the current pipeline setup doesn't honor on MoltenVK, and
   whether inserting an explicit `VkMemoryBarrier` between the hierarchy and bounding-box
   dispatches (or splitting the propagation into level-by-level dispatches) fixes it.
4. Add large-N (N≥4096) correctness tests to `test/test_spatialIndex.cpp` once fixed, so
   the regime is covered going forward.

## Impact on the benchmark

`example2/bvh_benchmark` is capped at N≤512 and reports `knn_rec%` transparently. Its
**structure-size metrics (nodes, mem_KB) and build_ms are trustworthy** and demonstrate
the wide BVH's compactness; query correctness/timing carry the caveat above.
