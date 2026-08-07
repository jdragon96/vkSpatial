# Registration & Features Refactor — Design

## Problem

Registration code is split across **two confusingly-parallel homes**, and one file (`Tracker.cpp`) bundles every tracker strategy together:

- `src/Engine/Registration/` (lib `Engine::Registration`, links Eigen + Ceres) mixes **registration algorithms** (`Icp.h` = CPU point-to-plane, `GlobalRegistration`) with **feature algorithms** (`Fpfh`, `FeatureMatching`, `Downsample`) and the shared types (`RegistrationTypes.h`).
- `src/Engine/Pipeline/Registration/` (part of `EnginePipeline`) holds `Tracker.cpp` (all four tracker strategies in one file), `GpuIcp` (GPU point-to-plane), and `RegistrationThread`.

Two "Registration" folders + a monolithic `Tracker.cpp` make it hard to find where an algorithm lives or add a new one.

## Goals

1. **One registration home**: all registration types, algorithms, tracker strategies, and the hub live under `src/Engine/Pipeline/Registration/` (flat).
2. **One file per algorithm / per strategy**: `Tracker.cpp` is split into a hub + one file per tracker strategy; each registration algorithm is its own file.
3. **Features extracted**: `Fpfh`, `FeatureMatching`, `Downsample` move to a new `src/Engine/Features/` module (`Engine::Features`).
4. **Behavior-preserving**: no algorithm changes. The existing test suite (currently green) is the acceptance gate.

Non-goal: changing any registration/feature math, the pipeline threading, or the public tracker names (`identity` / `icp` / `icp-cpu` / `global`).

## Target Structure

### `src/Engine/Features/` — new lib `Engine::Features` (links Eigen only)

Feature algorithms are the lowest layer (registration depends on them, not vice-versa). The shared types live here too, because the feature algorithms take `PointCloud` and `Engine::Features` must not depend on the pipeline.

| File | Contents | Namespace |
|---|---|---|
| `RegistrationTypes.h` | `PointCloud`, `RegistrationResult`, `RegistrationParam` (moved verbatim; `RegistrationParam` **moved out of `Icp.h`**) | `Engine::Registration` (kept — semantic identity + zero call-site churn) |
| `Fpfh.h/.cpp` | FPFH descriptor | `Engine::Features` (renamed from `Engine::Registration`) |
| `FeatureMatching.h/.cpp` | feature matching | `Engine::Features` |
| `Downsample.h/.cpp` | voxel downsample | `Engine::Features` |

The folder holds two namespaces on purpose: the shared **types** keep `Engine::Registration` (they are registration types, referenced everywhere — renaming would be gratuitous churn), while the **feature algorithms** take the new `Engine::Features` namespace that matches their new module.

### `src/Engine/Pipeline/Registration/` — the single registration home (flat, compiled into `EnginePipeline`)

| File | Contents | Namespace | Was |
|---|---|---|---|
| `PointToPlaneIcp.h` | CPU point-to-plane (`AlignPointToPlaneIcp` + `detail::IcpGridNN`), header-only | `Engine::Registration` | `Engine/Registration/Icp.h` |
| `GpuPointToPlaneIcp.h/.cpp` | `LocalGrid` + `GpuPointToPlaneIcp` (GPU point-to-plane) | `Engine::Pipeline` | `Pipeline/Registration/GpuIcp.h/.cpp` (renamed) |
| `GlobalRegistration.h/.cpp` | FPFH + RANSAC + Ceres global registration | `Engine::Registration` | `Engine/Registration/GlobalRegistration.*` |
| `Tracker.h` | `Tracker` interface, `TrackingResult`, `TrackerRegistry` (the hub) | `Engine::Pipeline` | unchanged |
| `TrackerRegistry.cpp` | `TrackerRegistry::Default()` — registers all four strategies centrally | `Engine::Pipeline` | (extracted from `Tracker.cpp`) |
| `IdentityTracker.h/.cpp` | `"identity"` strategy | `Engine::Pipeline` | (split from `Tracker.cpp`) |
| `PointToPlaneIcpTracker.h/.cpp` | `"icp-cpu"` strategy (wraps `AlignPointToPlaneIcp`) | `Engine::Pipeline` | (split from `Tracker.cpp`) |
| `GpuIcpTracker.h/.cpp` | `"icp"` strategy (wraps `GpuPointToPlaneIcp`, crops + voxel-scaled `maxCorrDist`) | `Engine::Pipeline` | (split from `Tracker.cpp`) |
| `GlobalRegistrationTracker.h/.cpp` | `"global"` strategy (wraps `Estimate`) | `Engine::Pipeline` | (split from `Tracker.cpp`) |
| `RegistrationThread.h/.cpp` | pipeline stage | `Engine::Pipeline` | unchanged |

**Namespaces are otherwise preserved** (CPU ICP / global reg / shared types stay `Engine::Registration`; trackers + GPU ICP stay `Engine::Pipeline`). This keeps the refactor mechanical and low-risk — only `#include` paths, file locations, CMake, and the feature-namespace rename change; no registration call sites change. Aligning the surviving `Engine::Registration` namespace to `Engine::Pipeline` is a possible **follow-up**, deliberately out of scope here to keep this change safe.

### The Tracker hub + strategies

`Tracker.h` already defines the interface and `TrackerRegistry`. The refactor:
- Each strategy class (`IdentityTracker`, `PointToPlaneIcpTracker`, `GpuIcpTracker`, `GlobalRegistrationTracker`) moves to its own `.h/.cpp` (they are currently in one anonymous namespace in `Tracker.cpp`; they become named classes in their own headers so the registry file can reference them).
- `TrackerRegistry::Default()` moves to `TrackerRegistry.cpp` and registers the four factories centrally (central registration is reliable with static libs — no static-initializer-in-a-static-lib fragility). The hub behavior is unchanged: `Create(name)` → strategy.

## CMake / Libs

- **New** `Engine::Features`: `file(GLOB_RECURSE Features/*.cpp)`, `target_link_libraries(EngineFeatures PUBLIC Eigen3::Eigen)`, include root `src/`.
- **Dissolve** `Engine::Registration` (delete the `add_library`/alias/link block; the folder `src/Engine/Registration/` is emptied and removed).
- **`EnginePipeline`**: drop `Engine::Registration` from its link line; add `Engine::Features` and `Ceres::ceres` (Ceres arrives with `GlobalRegistration`, which now compiles into `EnginePipeline`).
- All three engine libs already use `GLOB_RECURSE`, so moved files are picked up on **reconfigure** (`cmake -S . -B build`).

**Layering consequence (accepted):** the pure CPU algorithms (`PointToPlaneIcp`, `GlobalRegistration`) now compile into `EnginePipeline` (which links Vulkan via `Engine::Core`). Their tests (`test_icp`, `test_registration`) therefore link `EnginePipeline`. This is the intended cost of a single registration home; unused Vulkan code is never called by those tests.

## Include & Consumer Migration

Every `#include "Engine/Registration/X"` is rewritten to its new home. Consumers to update:

| Consumer | Change |
|---|---|
| `Pipeline/Registration/Tracker.cpp` | split into the strategy/registry files above |
| `Pipeline/Registration/GpuIcp.*` | renamed to `GpuPointToPlaneIcp.*`; the `Icp.h` include (was only for `RegistrationParam`) is **dropped** — `RegistrationParam` now lives in `RegistrationTypes.h`, which this file already includes; `RegistrationTypes.h` include → `Engine/Features/RegistrationTypes.h` |
| `GlobalRegistration.cpp` | `Fpfh/FeatureMatching/Downsample` includes → `Engine/Features/…`; usages → `Engine::Features::` |
| `test/test_icp.cpp` | `Icp.h` → `Pipeline/Registration/PointToPlaneIcp.h`; `RegistrationTypes.h` → `Engine/Features/…` |
| `test/test_gpuIcp.cpp` | `Icp.h` → `Pipeline/Registration/PointToPlaneIcp.h`; `GpuIcp.h` → `GpuPointToPlaneIcp.h`; `RegistrationTypes.h` → `Engine/Features/…` |
| `test/test_registration.cpp` | `Fpfh/FeatureMatching/Downsample` → `Engine/Features/…` + `Engine::Features::`; `GlobalRegistration.h` → `Pipeline/Registration/…`; `RegistrationTypes.h` → `Engine/Features/…` |
| `example2/registration_chair_demo.cpp` | `GlobalRegistration.h` → `Pipeline/Registration/…`; `RegistrationTypes.h` → `Engine/Features/…` |
| `example2/Alignment.h` (legacy) | `Icp.h` → `Pipeline/Registration/PointToPlaneIcp.h`; `GlobalRegistration.h` → `Pipeline/Registration/…`; `RegistrationTypes.h` → `Engine/Features/…` |

## Testing / Acceptance

Behavior-preserving refactor → the **existing full test suite must stay green** (≈240 tests, currently green). No new behavior, so no new tests are required; the suite is the safety net. Each migration step keeps the build green (the implementation plan sequences the moves so the tree compiles between steps). `test_registration` continues to validate FPFH/matching/global-reg; `test_icp`/`test_gpuIcp` continue to validate the ICP algorithms from their new locations.

## Risks

- **Missed include / namespace update** → build break. Mitigation: `grep` for every old path/namespace; the suite build catches stragglers.
- **CMake not re-globbing** → stale sources. Mitigation: reconfigure (`cmake -S . -B build`) after moves.
- **Static-lib symbol drop** if self-registration were used → avoided by central `TrackerRegistry::Default()`.
- **Legacy `example2/Alignment.h`** must keep compiling (it is globbed into the test build via `test_alignment.cpp`).
