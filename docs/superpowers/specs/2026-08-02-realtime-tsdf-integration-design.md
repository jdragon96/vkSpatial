# Real-Time TSDF Integration — Design

**Date:** 2026-08-02
**Status:** approved (design), pending implementation plan
**Target demo:** `example2/voxel_fill_debugger` — plays a folder of frame PLYs into a TSDF and renders the fill process. Today it is not real-time.

## Problem

`voxel_fill_debugger` play is far below 30 FPS. Instrumentation (`util::StageProfiler`, commit b95682c) on 15 dragon frames @ voxel 0.5 measured the per-frame TSDF pipeline:

| stage     | avg ms/frame | share |
| --------- | ------------ | ----- |
| integrate | 148          | 6.9%  |
| download  | 1667         | 77.6% |
| tracker   | 333          | 15.5% |

This design targets **integrate first** (the user's directive). Root cause of the ~148 ms integrate (which is _overhead-bound_, not compute-bound — integrate time is ~equal at voxel 0.5 vs 2.0 despite very different point/voxel counts):

- `Buffer::Upload` (`src/Engine/Core/Buffer.cpp`) allocates a **fresh staging buffer every call** (`vmaCreateBuffer`/`vmaDestroyBuffer`) and does a **one-shot submit + `vkQueueWaitIdle`**.
- `ComputePipeline::Dispatch` (`src/Engine/Core/ComputePipeline.cpp:252`) also **submits + `vkQueueWaitIdle`** every call.
- `AdvancedTSDF::Integrate` = 2 uploads + 1 dispatch = **3 blocking round-trips per tile**; `TiledDirectionalTSDF::Integrate` calls it **per touched tile**; `SubmapAdvancedTSDF::Integrate` runs base + detail → **dozens of blocking submits + staging allocations per frame**.

The debugger already runs the TSDF on its own `Engine::Core::Context ctx` and rendering on a separate `appCtx` (two Vulkan devices, bridged over the CPU via `DownloadEntries`), so there is **no shared Vulkan state** between mapping and rendering.

## Goal

Make integration real-time-capable and keep the render/UI thread smooth, via two independent, localized phases:

- **Phase 1 — fast-integrate (engine-localized):** remove the per-op staging + `vkQueueWaitIdle` overhead from the integrate path. Existing engine APIs stay behavior-identical; only additive members are added. GPU results must be **bit-identical** to today.
- **Phase 2 — async mapping:** move integrate/download off the render thread into a worker owning its own device, handing rendered state back via an immutable snapshot. Render/UI stays ≥30 FPS regardless of mapping cost.

Non-goals (separate follow-ups): the `download` (77.6%) and `tracker` (15.5%) costs; GPU-side compaction; engine-wide `Buffer` changes for non-TSDF callers.

## Global Constraints

- Build: `VULKAN_SDK=/usr/local`; `vkspatial_tests` needs the conda workaround `-DCMAKE_IGNORE_PREFIX_PATH=/opt/homebrew/anaconda3 -Ugflags_DIR -Uglog_DIR -UGTest_DIR -UCeres_DIR`.
- Platform: Apple M4 Max, **UMA** — a `HOST_VISIBLE | DEVICE_LOCAL` memory type exists, so a mapped buffer can also be the storage buffer the shader reads (zero-copy upload).
- **Additive only** for shared engine infra (`Buffer`, `ComputePipeline`): existing `Allocate`/`Upload`/`Download`/`Dispatch` signatures and behavior unchanged; all current callers keep working untouched.
- **Bit-identical GPU results:** existing `AdvancedTSDF`, `TiledAdvancedTSDF`, `SubmapAdvancedTSDF` tests must pass unchanged after Phase 1 (the fast path must produce the same hash contents as the slow path).
- GLSL/host style per repo conventions (`///` banners, Allman braces, tabs, descriptive names, numbered step comments where a function has stages).
- Git: work on a branch, fast-forward merge to local `main`, **do not push to origin**. Commit trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Leave all pre-existing uncommitted work (`.comp.glsl` rename, block-A1 experiment, stray PLYs) untouched.

---

## Phase 1 — Fast-Integrate (engine-localized)

### 1.1 `Buffer` host-visible persistent-mapped mode (additive)

New members on `Engine::Core::Buffer`, existing API untouched:

- `void AllocateHostVisible(uint32_t bytes);` — (re)allocates with VMA flags `VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT`, usage `VMA_MEMORY_USAGE_AUTO`, plus the buffer's `m_extraUsage` (so it can still be a storage buffer). Stores the persistent `VmaAllocationInfo::pMappedData`.
- `void *MappedPtr() const;` — the persistent mapped pointer, or `nullptr` if the buffer was allocated with plain `Allocate`.
- `void MakeVisibleToGPU(uint32_t bytes) const;` — `vmaFlushAllocation(0, bytes)` (no-op if the memory is `HOST_COHERENT`; safe to always call).

`free()` clears the mapped pointer. On UMA the allocation is host-visible **and** device-local, so the shader reads it directly — no staging, no copy, no submit.

### 1.2 `AdvancedTSDF` mapped uploads + batched dispatch

- **Buffers:** allocate `m_pointBuffer` / `m_normalBuffer` with `AllocateHostVisible` (they are written every frame, read by the shader). `m_hashBuffer` / `m_statBuffer` stay device-local (`Allocate`; cleared via existing paths).
- **Upload becomes memcpy:** in `Integrate`, `std::memcpy(m_pointBuffer->MappedPtr(), points, N*3*sizeof(float))` + same for normals + `MakeVisibleToGPU` — **no staging, no submit, no wait**.
- **Batched dispatch API (additive):**
  ```cpp
  // Records upload(memcpy) + dispatch into `batch`, no submit. Caller submits the batch.
  void AdvancedTSDF::RecordIntegrate(const std::vector<Eigen::Vector3f> &points,
                                     const std::vector<Eigen::Vector3f> &normals,
                                     const Eigen::Vector3f &cameraPos,
                                     Engine::Compute::CommandBatch &batch);
  ```
  Uses `m_kernel->Args(pc)` then `batch.DispatchElements(*m_kernel, N)` (`ComputePipeline::RecordDispatch` binds pipeline + push constants + descriptor set). Each tile is a distinct pipeline → safe under CommandBatch's "no same pipeline twice" rule.
- **Existing `Integrate` preserved:** reimplemented as `{ CommandBatch b(*m_ctx); RecordIntegrate(...,b); b.Submit(); }` so single-shot callers and behavior are unchanged.

### 1.3 `TiledDirectionalTSDF` / `SubmapAdvancedTSDF` batched path

- Add `void TiledDirectionalTSDF::Integrate(points, normals, cam, CommandBatch &batch)` — routes points to tiles (unchanged routing), then for each touched tile calls `tile->RecordIntegrate(sub, batch)` (no per-tile submit). The **caller** owns the batch and submits once.
- Keep the existing self-submitting `Integrate(points, normals, cam)` (creates a batch, records all tiles, submits once) — so `TiledAdvancedTSDF` on its own also benefits (one submit per Integrate instead of per tile).
- `SubmapAdvancedTSDF::Integrate` records base + detail into **one** `CommandBatch` and submits once per frame (base and detail are distinct tiles/pipelines → safe). Requires a `Barrier()` only if base and detail write buffers that a later op in the same batch reads — they write independent tile hashes, so no intra-batch dependency; no barrier needed.

**Backend requirement:** `TiledDirectionalTSDF<Backend>` now requires `Backend::RecordIntegrate(pts, nrm, cam, CommandBatch&)`. `AdvancedTSDF` provides it. (The base template's other Backend, if any, needs it only if used with the batched path.)

### Phase 1 testing

- `Buffer` host-visible round-trip unit test: `AllocateHostVisible`, write via `MappedPtr`, read back (host-coherent) — values match.
- Existing GPU tests (`test_advancedTsdf`, `test_tiledAdvancedTsdf`, `test_submapAdvancedTsdf`) pass **unchanged** — proves bit-identical results.
- Measure: `voxel_fill_debugger --dump` integrate avg ms before/after (expect large drop).

---

## Phase 2 — Async Mapping (`AsyncTsdfMapper`) + debugger wiring

### 2.1 `MapSnapshot` (immutable handoff)

```cpp
struct MapSnapshot {
    std::vector<AdvancedEntry> entries;   // downloaded occupied voxels (precedence-deduped)
    std::vector<char> isNew;              // per-entry "new this frame" (parallel to entries)
    int processedFrame = -1;              // last frame integrated into this snapshot
    uint32_t baseTiles = 0, detailTiles = 0, denseBlocks = 0;
    Eigen::Vector3f allocMin{0,0,0}, allocMax{0,0,0}; // entries AABB (± half voxel), empty if none
    double integrateMs = 0, downloadMs = 0, trackerMs = 0; // worker stage times for THIS snapshot
};
```

Produced by the worker, consumed read-only by the render thread. Color-mode / weight-threshold coloring is applied on the render thread from `entries` (cheap vs download), so those interactions stay instant without re-integrating.

### 2.2 `AsyncTsdfMapper` (reusable component)

Owns a worker `std::thread`, its **own `Engine::Core::Context`**, and a `SubmapAdvancedTSDF`. (Templating on the backend is a YAGNI-deferred generalization; start concrete on `SubmapAdvancedTSDF`.)

```cpp
class AsyncTsdfMapper {
    // config mirrors SubmapAdvancedTSDF::Build + quality/p2p/conf/hermite + the frame set.
    void Start(const Config &cfg, std::vector<Frame> frames); // frames owned by the mapper
    void Stop();                                              // sentinel + join
    void RequestFrame(int target);                           // coalesced: worker integrates/replays to `target`
    std::shared_ptr<const MapSnapshot> Latest() const;       // most recent published snapshot (may be null early)
    int RequestedFrame() const;  int ProcessedFrame() const; // atomics, for the "worker N behind" UI
};
```

Per-frame worker stage times surface race-free inside `MapSnapshot` (`integrateMs`/`downloadMs`/`trackerMs`); the render thread reads them from the immutable snapshot. The worker also keeps a cumulative `StageProfiler` it prints on `Stop` (never read across threads while running).

- **Command model:** a single coalesced target-frame (not a growing queue). `RequestFrame(t)` stores the latest requested target under a mutex + condvar; the worker always integrates toward the newest target. Backward target → `Reset` + replay from 0 (dense set preserved by `SubmapAdvancedTSDF::Reset`). This matches the debugger's scrub semantics and avoids unbounded queue growth when the user drags the slider.
- **Worker loop:** wait for (target != processed || stop); compute the step (reset+replay or forward-integrate to target); each integrated frame → `DownloadEntries` + diff (owns its own `FillTracker`) → build `MapSnapshot` → publish. Times each stage into a worker-owned `StageProfiler`.
- **Mailbox handoff:** `std::mutex` + `std::shared_ptr<const MapSnapshot>`; `Publish` swaps under lock, `Latest` copies the shared_ptr under lock. Snapshots are immutable once published (safe to read on the render thread while the worker builds the next).
- **Lifecycle/errors:** `Stop` sets a stop flag + notifies + joins. Worker catches exceptions into a `std::exception_ptr`; `Latest`/`Stop` rethrow on the main thread. Destructor calls `Stop`.

### 2.3 `voxel_fill_debugger` wiring

- Build the mapper from the parsed args + loaded frames; `Start`. Remove the inline `rebuildTo`/scrub GPU calls.
- Render loop (60 FPS, never blocks on the mapper):
  - play/slider → `mapper.RequestFrame(state.frame)`.
  - `snap = mapper.Latest()`; if `snap` changed OR color-mode/threshold changed → `buildVoxelSets(snap->entries, ...)` + box sets from `snap` → `SetPointSet` → render.
  - ImGui: stats from `snap`; a "frame N / M — worker K behind" line (`RequestedFrame` vs `ProcessedFrame`); the timing panel shows the worker stage times from `snap` (`integrateMs`/`downloadMs`/`trackerMs`) + a render-thread `StageProfiler` for `render`/`buildSets`/`upload`.
- The debug tool keeps its own `Engine::Core::Context ctx`? No — the mapper owns the TSDF context internally; the debugger keeps only `appCtx` (render). This removes the debugger's direct TSDF context.

### Phase 2 testing

- **Host-only unit tests (no Vulkan):** a `Mailbox<T>` (mutex + shared_ptr swap) publish/consume + latest-wins; the coalesced target-frame command logic (a small pure `stepPlan(shown, target) -> {reset?, from, to}` helper extracted for testability).
- **Headless mapper smoke:** `Start` with N frames, `RequestFrame(N-1)`, poll `Latest()` until `ProcessedFrame()==N-1`, assert snapshot non-empty and counts match a synchronous `SubmapAdvancedTSDF` reference. (Uses Vulkan; lives with the GPU tests.)

---

## Decomposition

Two implementation plans, sequenced:

- **Plan 1 — Phase 1 (fast-integrate).** Standalone: engine additive APIs + AdvancedTSDF/Tiled/Submap batched path; verified by existing GPU tests (bit-identical) + `--dump` integrate-time drop.
- **Plan 2 — Phase 2 (async mapping).** Depends on Phase 1 (for worker throughput but not correctness). `AsyncTsdfMapper` + `Mailbox` + debugger wiring; host unit tests + headless smoke.

Each plan produces working, tested software on its own.
