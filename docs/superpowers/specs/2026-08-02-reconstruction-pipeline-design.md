# Reconstruction Pipeline (Track / Map / Render) — Design

**Date:** 2026-08-02
**Status:** draft for review
**Builds on:** the real-time TSDF work (`2026-08-02-realtime-tsdf-integration-design.md`, Phase 1 fast-integrate + Phase 2 `AsyncTsdfMapper` already merged). This generalizes the single mapping worker into a 3-stage tracking → mapping → rendering pipeline with a shared producer/consumer module and one lifecycle owner.

## Goal

A reusable, real-application-shaped reconstruction pipeline:

- **Stage 1 — Track (ICP):** estimate each incoming frame's pose (frame-to-model against the latest model, frame-to-frame/passthrough bootstrap). CPU.
- **Stage 2 — Map (Integrate + throttled Download):** integrate the posed frame into the TSDF on its own Vulkan device; periodically download an immutable model snapshot.
- **Stage 3 — Render:** the main thread consumes the latest snapshot and draws. Never does blocking GPU readback.

Communication is producer/consumer channels; a single `ReconstructionPipeline` class owns every inter-thread resource (channels, mailbox, worker threads) and their lifecycle.

**Explicit non-goals (deferred):** the per-stage cost fixes — download GPU-compaction (78% of frame time) and the integrate hash-scatter atomic contention (~130 ms). This design is about *structure/responsiveness*, not per-stage speedup (threading is parallelism, not a per-stage speedup: pipeline throughput = the slowest stage). GPU-raycast frame-to-model ICP is a future tracker; v1 tracks against the CPU snapshot.

## Global Constraints

- Build: `VULKAN_SDK=/usr/local`; tests via the conda workaround `-DCMAKE_IGNORE_PREFIX_PATH=/opt/homebrew/anaconda3 -Ugflags_DIR -Uglog_DIR -UGTest_DIR -UCeres_DIR`; `build/test/vkspatial_tests` GLOBs `test/*.cpp` (new test file → reconfigure).
- Platform: Apple M4 Max, UMA. **Vulkan queues are NOT thread-safe** (command pools are per-thread; queue submits need external sync). Each stage that touches the GPU owns its OWN `Engine::Core::Context` (device) — no device shared across threads.
- Reuse the existing `util::Mailbox<T>` (latest-only handoff) and the `AsyncTsdfMapper` mapping approach (own device + `SubmapAdvancedTSDF` + `FillTracker` + `MapSnapshot`); the Map stage generalizes it to take a per-frame pose instead of coalesced frame replay.
- **Additive**: do not change shared engine infra behavior; the pipeline composes existing classes.
- **Coding style — match the Engine C++ code:** `namespace Engine::…`; `m_`-prefixed private members; `Build(...)`/`Start()`/`Stop()` lifecycle methods; RAII with `std::unique_ptr`; pass `Engine::Core::Context &` by reference; `///` section banners and Allman braces (K&R for `for`), tabs, multi-line signatures, numbered `// 1./2.` step comments in multi-step functions; deleted copy ops on owning classes.
- Git: branch, ff-merge to local `main`, do NOT push. Trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Leave pre-existing uncommitted work untouched.

## Architecture

```
 PushFrame(Frame)                                                     LatestModel()
      │                                                                    ▲
      ▼   Channel<Frame>          Channel<TrackedFrame>       Mailbox<ModelSnapshot>
 ┌─────────────┐  (bounded,   ┌──────────────┐  (bounded,   ┌──────────────────────┐
 │  Track      │  drop-oldest)│  Map          │  drop-oldest)│  Render (MAIN thread) │
 │  (ICP,CPU)  │ ───────────► │ (integrate+   │ ───────────► │  consume snapshot,    │
 │             │              │  throttled dl,│   Mailbox     │  draw. no readback    │
 │             │ ◄────────────┤  own device)  │              └──────────────────────┘
 └─────────────┘  latest model└──────────────┘
     ▲   (Mailbox<ModelSnapshot>: Track aligns new frames to the last published model)
```

- **Two worker threads** (Track, Map) owned by `ReconstructionPipeline`. **Render is the main thread** (macOS/GLFW require window + swapchain on the main thread) and is an *external consumer* of `LatestModel()` — the pipeline does not own it.
- **Download lives in the Map stage, throttled** (every K frames or on demand), NOT on the render thread — that is the key correction over "thread #3 = download+render" (download is ~1.6 s; on the render thread it would refreeze the window).
- **The model snapshot is dual-use:** Map publishes it to a mailbox consumed by BOTH Render (display) and Track (frame-to-model alignment). This closes the Track↔Map dependency without sharing a Vulkan device: Track aligns against a CPU snapshot that may be a frame or two stale (PTAM-style asynchronous tracking/mapping).

## Components

### 1. `util::Channel<T>` — producer/consumer module (comms)

Header-only, `src/utilities/Channel.h`. A bounded blocking queue with a real-time drop policy and a close signal. One module used for both inter-worker links; `util::Mailbox<T>` remains the latest-only variant for the model handoff.

```cpp
namespace util {
    template <typename T>
    class Channel {
    public:
        explicit Channel(std::size_t capacity, bool dropOldestWhenFull = true);
        bool Push(T value);              // false if closed; drops oldest (or blocks) when full per policy
        bool Pop(T &out);                // blocks until an item is available or closed; false when closed+empty
        bool TryPop(T &out);             // non-blocking
        void Close();                    // wakes all blocked Push/Pop; subsequent Pop drains then returns false
        std::size_t Size() const;        // current depth (stats/backpressure display)
        std::size_t Dropped() const;     // total dropped (real-time skip count)
    };
}
```

- **drop-oldest** = real-time backpressure: when Map falls behind, old frames are discarded so latency stays bounded (matches live SLAM). `Dropped()` surfaces skipped frames.
- `Close()` is the shutdown signal: `Pop` returns false so worker loops exit cleanly.

### 2. Data types (`Engine::Pipeline` namespace, `src/Engine/Pipeline/PipelineTypes.h`)

```cpp
struct Frame {                              // raw captured cloud (sensor or PLY)
    std::vector<Eigen::Vector3f> points, normals;
    Eigen::Vector3f sensorHint = Eigen::Vector3f::Zero(); // optional prior (e.g. estimateCamera)
    int index = -1;
};
struct TrackedFrame {                       // Frame + the pose Track resolved
    Frame frame;
    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity(); // sensor→world (identity if already world)
    Eigen::Vector3f cameraWorld = Eigen::Vector3f::Zero();  // world camera position for view-angle weight
};
// ModelSnapshot = the existing asyncmap::MapSnapshot (entries/isNew/firstFrame/counts/boxes/allocBox/
// worker-stage-ms), promoted into Engine::Pipeline. Immutable once published.
```

### 3. Alignment commands (Command Pattern) + Track stage

The Track stage resolves each frame's pose via a **swappable alignment command**, so multiple
registration algorithms can be A/B-tested and selected at runtime. This is the Command Pattern: each
algorithm is an object with a uniform `Execute`, registered by name.

```cpp
struct AlignmentResult {
    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity(); // sensor→world
    float fitness = 0.0f;
    std::size_t inliers = 0;
    bool valid = false;
};

class AlignmentCommand { // Command Pattern: one registration algorithm, encapsulated
public:
    virtual ~AlignmentCommand() = default;
    virtual const char *Name() const = 0;
    // Align `frame` to `model` (null before the first map), seeded by `priorPose`. Pure CPU.
    virtual AlignmentResult Execute(const Frame &frame, const ModelSnapshot *model,
                                    const Eigen::Isometry3f &priorPose) = 0;
};
```

**Commands (all real, v1), registered in `AlignmentRegistry` (name → factory) for runtime selection:**
- **`IdentityAlignment`** — `pose = priorPose` (identity when none), `cameraWorld = frame.sensorHint`. For pre-registered world clouds (the debugger's `estimateCamera` output) and as the bootstrap when no model exists yet.
- **`PointToPlaneIcpAlignment`** — real local **point-to-plane ICP** seeded by `priorPose`, target = `model->entries` (centers + stored-gradient normals). Backed by a NEW engine primitive `Engine::Registration::AlignPointToPlaneIcp(src, tgt, priorT, params) -> RegistrationResult`: uniform-grid nearest-neighbour correspondences + small-angle-linearised 6-DoF normal-equation solve, iterated to convergence. The per-frame tracker for a live sensor.
- **`GlobalRegistrationAlignment`** — wraps the existing `Engine::Registration::GlobalRegistration::Estimate` (FPFH + RANSAC + Ceres) for prior-free (re)localisation and as an A/B baseline.

`AlignmentRegistry`: `Register(name, factory)` + `Create(name)`; the debugger/app selects via e.g. `--align icp|identity|global`. The Track stage holds one `AlignmentCommand` + the previous pose (the `priorPose` for the next frame): `Pop` a `Frame` (capture channel) → `Execute(frame, LatestModel().get(), prevPose)` → build `TrackedFrame` (pose + `cameraWorld`) → `Push` (map channel, drop-oldest); update `prevPose` on `valid`.

### 4. Map stage

Generalizes `AsyncTsdfMapper`: owns its own `Engine::Core::Context` + `SubmapAdvancedTSDF` + `FillTracker`. Loop: `Pop` a `TrackedFrame` → transform points to world by `pose` (no-op for identity) → `Integrate(worldPts, worldNrm, cameraWorld)` → **every K frames** (`downloadEveryN`, default 1; raise to decouple the 1.6 s download) `DownloadEntries` + tracker diff → build `ModelSnapshot` → `Publish` to the mailbox. Density precompute (`AddDensity`/`FinalizeDensity`) still runs at Start over the known frame set when available; for a pure live stream it is skipped (dense set grows as `FinalizeDensity` is re-run on a cadence — v1: precompute from the initial frame set, live-incremental deferred).

### 5. `ReconstructionPipeline` — the one owner

`src/Engine/Pipeline/ReconstructionPipeline.{h,cpp}`. Owns all inter-thread resources and both worker threads; the single place lifecycle and resources live.

```cpp
class ReconstructionPipeline {
public:
    struct Config {
        asyncmap::Config map;          // TSDF/submap config (reused)
        std::size_t captureQueue = 8;  // Frame channel capacity
        std::size_t trackQueue = 4;    // TrackedFrame channel capacity (drop-oldest)
        int downloadEveryN = 1;        // Map download cadence
    };

    void Start(const Config &cfg, std::unique_ptr<AlignmentCommand> align); // launches Track + Map
    void Stop();                       // Close() channels, join both threads, surface errors
    void Reset();                      // drain channels + map submap.Reset()/prevPose (keeps dense
                                       //   set); for the debugger's backward scrub
    bool PushFrame(Frame frame);       // external producer (sensor/PLY); false if stopped/closed
    std::shared_ptr<const ModelSnapshot> LatestModel();  // render + Track consume; rethrows worker errors
    // stats for UI: queue depths, dropped counts, processed index, per-stage times (from snapshot)
    Stats GetStats() const;

private:
    // owns: Channel<Frame> m_capture; Channel<TrackedFrame> m_tracked; Mailbox<ModelSnapshot> m_model;
    //       std::thread m_trackThread, m_mapThread; std::unique_ptr<AlignmentCommand> m_align;
    //       Map-stage device/TSDF live inside the map thread; exception_ptr per stage.
};
```

- **Lifecycle:** `Start` builds config, launches Track then Map. `Stop` closes channels (workers drain + exit), joins, then rethrows any captured stage exception. **Destructor joins WITHOUT rethrowing** (no throw during stack unwinding → no `std::terminate`) — same rule as the `AsyncTsdfMapper` fix.
- **Errors:** each worker catches into a per-stage `std::exception_ptr`; `LatestModel`/`Stop` rethrow on the caller (main) thread.
- **Backpressure/stats:** `GetStats()` exposes channel `Size()`/`Dropped()` and the latest snapshot's stage times + processed index, so the UI can show "map N behind / dropped M".

## voxel_fill_debugger wiring (demo)

Replace the direct `AsyncTsdfMapper` use with `ReconstructionPipeline`, alignment selected by
`--align identity|icp|global` (default `identity`, since the debugger's PLYs are already
world-registered; `icp`/`global` exercise the real registration on the same data for A/B):
- Load PLYs → for play/scrub, `PushFrame` the target frame's `Frame` (sensorHint = `estimateCamera`).
- **Scrub-back = reset+replay:** the pipeline is a forward stream, so a backward target restarts the
  map — `ReconstructionPipeline::Reset()` (drain channels, `submap.Reset()` + tracker/prevPose reset
  on the map thread, keep dense set) then re-`PushFrame` 0..target. The debugger detects
  `target < lastPushed` and calls `Reset()` before re-pushing.
- Render loop: `LatestModel()` → build point-sets (unchanged `buildVoxelSets` from the snapshot) →
  draw at 60 fps. ImGui shows pipeline stats (queue depths, dropped, per-stage ms, "map N behind")
  + the active alignment command name.
- `--dump` keeps the synchronous submap path (baseline timing), untouched.

## Testing

- **`util::Channel` host tests:** push/pop FIFO; drop-oldest when full (+ `Dropped()` count); `Close()` unblocks `Pop` and drains; concurrent producer/consumer stress (no loss beyond drops, monotonic).
- **Local ICP host test:** `Engine::Registration::AlignPointToPlaneIcp` on a plane/cloud transformed by a known small SE(3) → recovers the inverse transform within tolerance (CPU, no Vulkan).
- **Alignment command/registry host tests:** `IdentityAlignment` returns `priorPose` + sensorHint; `AlignmentRegistry::Create("icp")` yields a `PointToPlaneIcpAlignment` that recovers a known transform against a synthetic model; unknown name → null/false.
- **Headless pipeline smoke (Vulkan):** `Start` with `IdentityAlignment`, `PushFrame` N pre-registered frames, poll `LatestModel()` until it reaches frame N-1; assert snapshot non-empty and counts match a synchronous `SubmapAdvancedTSDF` reference. A second smoke drives `PointToPlaneIcpAlignment` on sensor-local frames to confirm the tracked path produces a consistent model.

## Decomposition (plans)

1. **`util::Channel<T>`** + host tests (standalone).
2. **`Engine::Registration::AlignPointToPlaneIcp`** (local point-to-plane ICP) + host test (recovers a known transform).
3. **Alignment commands + `AlignmentRegistry`** (`AlignmentCommand`, `IdentityAlignment`, `PointToPlaneIcpAlignment`, `GlobalRegistrationAlignment`) + host tests. Depends on Task 2 + pipeline types.
4. **Pipeline types + Map stage** (generalize `AsyncTsdfMapper` to a posed `MapStage` taking `TrackedFrame`; keep `AsyncTsdfMapper` or refactor into `MapStage`) + headless smoke.
5. **Track stage** (loop around an `AlignmentCommand`, prev-pose prior) — folded into the pipeline task or its own; host-testable step logic.
6. **`ReconstructionPipeline`** (owns channels + mailbox + both threads + `Reset()`/lifecycle) + headless pipeline smoke (Identity + ICP).
7. **Wire `voxel_fill_debugger`** to the pipeline + `--align` selection + stats UI + scrub-back `Reset()`.

Each plan is independently testable. Real ICP (`PointToPlaneIcpAlignment`) is in v1 (Tasks 2–3), not deferred.
