# Voxel-Fill Debugger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A live debug tool that plays the TSDF voxel-filling process frame by frame (reading a folder of scan-frame PLYs, integrating into `AdvancedTSDF`), rendering occupied voxels as colored points so coverage gaps and holes are visible.

**Architecture:** A pure, unit-tested host-logic header (`VoxelFillDebug.h`: voxel-key first-fill tracking + field→color mapping + weight thresholding) plus a standalone example (`voxel_fill_debugger.cpp`) that assembles it with the existing `tsdf_folder_eval` PLY reader and the `object_scan_viewer` render scaffold, integrating/downloading per frame via `AdvancedTSDF`'s public API.

**Tech Stack:** C++17, Eigen, Vulkan compute (existing AdvancedTSDF shaders, unchanged), `Engine::Render` (PointCloudPass/ImGuiPass), GoogleTest, CMake (GLOB sources/tests).

## Global Constraints

- **Single-window `AdvancedTSDF` only** (v1). Scene must fit one 512³ window; if `axisVox > 512`, print the minimum feasible voxel and exit (same guard as `tsdf_folder_eval.cpp`). Tiling is out of scope.
- **Point rendering only** — reuse `PointCloudPass` (`PointVertex{float pos[3]; uint8_t rgba[4]}`, `kMaxSets=4`, `SetPointSet(id, vertices)`, `SetVisible(id,bool)`, `SetPointSize(px)`); no new render pass.
- **Reuse, don't re-invent:** copy `readPly`/`estimateCamera`/arg helpers (`strArg`/`floatArg`/`flag`/`nextPow2`) verbatim from `example2/tsdf_folder_eval.cpp`, and the render scaffold (`PointVertex` build, `pushCloud`, `cameraMarker`, `Application`/`Scene`/`Camera`/`RenderGraph`/`PointCloudPass`/`ImGuiPass` setup, mouse/key trackball listeners, manual `while(!ShouldClose())` loop) from `example2/object_scan_viewer.cpp`. Do NOT modify those files.
- **AdvancedTSDF is read-only via its public API:** `Build(ctx,voxel,trunc,hashCap,maxPoints,windowMinCorner)`, `SetIntegrationQuality(const IntegrationQuality&)`, `SetPointToPlane(bool)`, `SetConfidenceWeight(float)`, `SetHermitePosition(bool)`, `Integrate(pts,nrm,cam)`, `std::vector<AdvancedEntry> DownloadEntries() const` where `AdvancedEntry{Eigen::Vector3f center; uint32_t direction; float tsdf; float weight; Eigen::Vector3f normal;}`, `Reset()`. No engine changes.
- **No new third-party dependencies.**
- **Build/test (macOS):** configure `VULKAN_SDK=/usr/local cmake -S . -B build`; the conda-on-PATH shell needs `-DCMAKE_IGNORE_PREFIX_PATH=/opt/homebrew/anaconda3 -Ugflags_DIR -Uglog_DIR -UGTest_DIR -UCeres_DIR` for the `vkspatial_tests` target (GTest); example targets don't need it. `git submodule update --init lib/SPIRV-Reflect` if empty. Build a single target (`--target <name>`), never the whole project.
- **Git policy:** commit only the files each task creates/modifies; do not touch other pre-existing uncommitted changes.

---

### Task 1: `VoxelFillDebug.h` pure logic + unit tests

Pure host-side logic (no Vulkan/render): per-voxel first-fill tracking + field→color mapping + weight thresholding. Unit-tested headlessly.

**Files:**
- Create: `example2/VoxelFillDebug.h`
- Create: `test/test_voxelFillDebug.cpp`

**Interfaces:**
- Consumes: `Engine::Spatial::AdvancedEntry` (`{Eigen::Vector3f center; uint32_t direction; float tsdf; float weight; Eigen::Vector3f normal;}`) from `Engine/Spatial/AdvancedTSDF.h`.
- Produces (namespace `voxdbg`): `using Rgba = std::array<uint8_t,4>;` `struct VoxelKey{int x,y,z; uint8_t dir; bool operator==}`, `struct VoxelKeyHash`, `VoxelKey keyOf(const AdvancedEntry&, float voxel)`, `class FillTracker{ explicit FillTracker(float voxel); void reset(); std::vector<char> update(const std::vector<AdvancedEntry>&, int frameIdx); int firstFrame(const VoxelKey&) const; std::size_t size() const; }`, `enum class ColorMode{TsdfSign,Weight,FillFrame,Direction}`, `Rgba tsdfColor(float tsdf,float trunc)`, `Rgba weightColor(float w,float wMax)`, `Rgba fillFrameColor(int frame,int nFrames)`, `Rgba directionColor(uint8_t dir)`, `bool belowThreshold(float weight,float wThresh)`.

- [ ] **Step 1: Write the failing tests**

Create `test/test_voxelFillDebug.cpp`:

```cpp
#include "VoxelFillDebug.h"

#include <gtest/gtest.h>

using Engine::Spatial::AdvancedEntry;
using Eigen::Vector3f;
using namespace voxdbg;

namespace {
    AdvancedEntry ent(float cx, float cy, float cz, uint32_t dir, float tsdf, float weight) {
        AdvancedEntry e;
        e.center = Vector3f(cx, cy, cz);
        e.direction = dir;
        e.tsdf = tsdf;
        e.weight = weight;
        e.normal = Vector3f(0, 0, 1);
        return e;
    }
} // namespace

TEST(VoxelFillDebug, KeyOfQuantizesToVoxel) {
    // Two centers inside the same 0.05 voxel + same dir -> identical key; different dir -> different.
    const auto a = keyOf(ent(0.101f, 0.0f, 0.0f, 4, 0, 1), 0.05f);
    const auto b = keyOf(ent(0.099f, 0.0f, 0.0f, 4, 0, 1), 0.05f);
    const auto c = keyOf(ent(0.101f, 0.0f, 0.0f, 5, 0, 1), 0.05f);
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
}

TEST(VoxelFillDebug, FillTrackerMarksNewAndRecordsFirstFrame) {
    FillTracker t(0.05f);
    std::vector<AdvancedEntry> f0 = {ent(0, 0, 0, 4, 0, 1), ent(1, 0, 0, 4, 0, 1)};
    auto n0 = t.update(f0, 0);
    EXPECT_EQ(n0.size(), 2u);
    EXPECT_EQ(n0[0], 1);
    EXPECT_EQ(n0[1], 1);
    EXPECT_EQ(t.size(), 2u);

    // Frame 1: one repeat (0,0,0) + one new (2,0,0). Only the new one is marked; firstFrame kept.
    std::vector<AdvancedEntry> f1 = {ent(0, 0, 0, 4, 0, 1), ent(2, 0, 0, 4, 0, 1)};
    auto n1 = t.update(f1, 1);
    EXPECT_EQ(n1[0], 0); // (0,0,0) already seen at frame 0
    EXPECT_EQ(n1[1], 1); // (2,0,0) new at frame 1
    EXPECT_EQ(t.size(), 3u);
    EXPECT_EQ(t.firstFrame(keyOf(ent(0, 0, 0, 4, 0, 1), 0.05f)), 0);
    EXPECT_EQ(t.firstFrame(keyOf(ent(2, 0, 0, 4, 0, 1), 0.05f)), 1);
}

TEST(VoxelFillDebug, FillTrackerResetReMarksEverything) {
    FillTracker t(0.05f);
    std::vector<AdvancedEntry> f = {ent(0, 0, 0, 4, 0, 1)};
    t.update(f, 0);
    t.reset();
    EXPECT_EQ(t.size(), 0u);
    auto n = t.update(f, 5);
    EXPECT_EQ(n[0], 1);
    EXPECT_EQ(t.firstFrame(keyOf(ent(0, 0, 0, 4, 0, 1), 0.05f)), 5);
}

TEST(VoxelFillDebug, TsdfColorSurfaceBandAndSign) {
    const float trunc = 0.15f;
    const Rgba surf = tsdfColor(0.0f, trunc); // |d| < 0.1*trunc -> white
    EXPECT_EQ(surf, (Rgba{255, 255, 255, 255}));
    const Rgba pos = tsdfColor(0.12f, trunc); // + -> red dominant
    EXPECT_EQ(pos[0], 255);
    EXPECT_LT(pos[2], 255);
    const Rgba neg = tsdfColor(-0.12f, trunc); // - -> blue dominant
    EXPECT_EQ(neg[2], 255);
    EXPECT_LT(neg[0], 255);
}

TEST(VoxelFillDebug, WeightColorEndpoints) {
    const Rgba lo = weightColor(0.0f, 10.0f);  // low -> blue
    EXPECT_EQ(lo[2], 255);
    EXPECT_EQ(lo[0], 0);
    const Rgba hi = weightColor(10.0f, 10.0f); // high -> red
    EXPECT_EQ(hi[0], 255);
    EXPECT_EQ(hi[2], 0);
}

TEST(VoxelFillDebug, FillFrameColorGreyForUnseenAndSpreads) {
    EXPECT_EQ(fillFrameColor(-1, 10), (Rgba{90, 90, 90, 255})); // unseen
    EXPECT_NE(fillFrameColor(0, 10), fillFrameColor(9, 10));    // early != late
}

TEST(VoxelFillDebug, DirectionColorSixDistinct) {
    std::set<Rgba> s;
    for (uint8_t d = 0; d < 6; ++d) s.insert(directionColor(d));
    EXPECT_EQ(s.size(), 6u);
}

TEST(VoxelFillDebug, BelowThresholdBoundary) {
    EXPECT_TRUE(belowThreshold(0.9f, 1.0f));
    EXPECT_FALSE(belowThreshold(1.0f, 1.0f));
    EXPECT_FALSE(belowThreshold(1.5f, 1.0f));
}
```

Add `#include <set>` and `#include <array>` at the top as needed for the test (the `std::set<Rgba>` test).

- [ ] **Step 2: Run tests to verify they FAIL (header missing)**

Run: `VULKAN_SDK=/usr/local cmake -S . -B build -DCMAKE_IGNORE_PREFIX_PATH=/opt/homebrew/anaconda3 -Ugflags_DIR -Uglog_DIR -UGTest_DIR -UCeres_DIR && VULKAN_SDK=/usr/local cmake --build build --target vkspatial_tests -j8`
Expected: BUILD FAILS with `fatal error: 'VoxelFillDebug.h' file not found` (the test's include dir is `example2/`; see Step 3 note).

Note: `test/CMakeLists.txt` compiles `test/*.cpp`; `VoxelFillDebug.h` lives in `example2/`. Add `example2/` to the test target's include path in Step 3.

- [ ] **Step 3: Create `VoxelFillDebug.h` and make it includable from tests**

Create `example2/VoxelFillDebug.h`:

```cpp
#pragma once

#include "Engine/Spatial/AdvancedTSDF.h" // Engine::Spatial::AdvancedEntry

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace voxdbg {

    using Rgba = std::array<uint8_t, 4>;

    // Quantized voxel-direction key: round(center/voxel) per axis + direction. Stable across
    // repeated DownloadEntries() (same voxel centre -> same key).
    struct VoxelKey {
        int x, y, z;
        uint8_t dir;
        bool operator==(const VoxelKey &o) const {
            return x == o.x && y == o.y && z == o.z && dir == o.dir;
        }
    };
    struct VoxelKeyHash {
        std::size_t operator()(const VoxelKey &k) const {
            std::size_t h = std::hash<int>()(k.x);
            h ^= std::hash<int>()(k.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(k.z) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= std::hash<int>()(int(k.dir)) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            return h;
        }
    };

    inline VoxelKey keyOf(const Engine::Spatial::AdvancedEntry &e, float voxel) {
        return VoxelKey{int(std::lround(e.center.x() / voxel)),
                        int(std::lround(e.center.y() / voxel)),
                        int(std::lround(e.center.z() / voxel)), uint8_t(e.direction)};
    }

    // Tracks the first frame that filled each voxel-direction. update() marks entries whose key is
    // seen for the first time and records firstFrame[key]=frameIdx for them.
    class FillTracker {
    public:
        explicit FillTracker(float voxel) : m_voxel(voxel) {}
        void reset() { m_firstFrame.clear(); }
        std::vector<char> update(const std::vector<Engine::Spatial::AdvancedEntry> &entries,
                                 int frameIdx) {
            std::vector<char> isNew(entries.size(), 0);
            for (std::size_t i = 0; i < entries.size(); ++i) {
                const VoxelKey k = keyOf(entries[i], m_voxel);
                if (m_firstFrame.find(k) == m_firstFrame.end()) {
                    m_firstFrame.emplace(k, frameIdx);
                    isNew[i] = 1;
                }
            }
            return isNew;
        }
        int firstFrame(const VoxelKey &k) const {
            auto it = m_firstFrame.find(k);
            return it == m_firstFrame.end() ? -1 : it->second;
        }
        std::size_t size() const { return m_firstFrame.size(); }

    private:
        float m_voxel;
        std::unordered_map<VoxelKey, int, VoxelKeyHash> m_firstFrame;
    };

    enum class ColorMode { TsdfSign, Weight, FillFrame, Direction };

    // tsdf in world units: +d red, -d blue, |d| < 0.1*trunc white (surface band); fades to white
    // toward the surface.
    inline Rgba tsdfColor(float tsdf, float trunc) {
        const float band = 0.1f * trunc;
        if (std::fabs(tsdf) < band) return {255, 255, 255, 255};
        const float t = std::min(1.0f, std::fabs(tsdf) / std::max(1e-6f, trunc));
        const uint8_t c = uint8_t(60 + 195 * (1.0f - t));
        return tsdf > 0.0f ? Rgba{255, c, c, 255} : Rgba{c, c, 255, 255};
    }
    // weight heat: 0 -> blue, wMax -> red.
    inline Rgba weightColor(float w, float wMax) {
        const float t = std::min(1.0f, std::max(0.0f, w / std::max(1e-6f, wMax)));
        return {uint8_t(255.0f * t), 40, uint8_t(255.0f * (1.0f - t)), 255};
    }
    // fill-frame rainbow: frame 0 red -> mid green -> last blue; unseen (-1) grey.
    inline Rgba fillFrameColor(int frame, int nFrames) {
        if (frame < 0) return {90, 90, 90, 255};
        const float h = float(frame) / float(std::max(1, nFrames - 1));
        const float r = std::max(0.0f, 1.0f - 2.0f * h);
        const float g = 1.0f - std::fabs(2.0f * h - 1.0f);
        const float b = std::max(0.0f, 2.0f * h - 1.0f);
        return {uint8_t(255.0f * r), uint8_t(255.0f * g), uint8_t(255.0f * b), 255};
    }
    // 6 axis colours: +X,-X,+Y,-Y,+Z,-Z.
    inline Rgba directionColor(uint8_t dir) {
        static const Rgba lut[6] = {{230, 60, 60, 255}, {120, 20, 20, 255}, {60, 230, 60, 255},
                                    {20, 120, 20, 255},  {60, 60, 230, 255}, {20, 20, 120, 255}};
        return lut[dir < 6 ? dir : 0];
    }
    inline bool belowThreshold(float weight, float wThresh) { return weight < wThresh; }

} // namespace voxdbg
```

Then make it includable from the test target. In `test/CMakeLists.txt`, add `example2` to the test executable's include directories (find the `target_include_directories(vkspatial_tests ...)` block and append `"${CMAKE_SOURCE_DIR}/example2"`; if no such block exists, add one right after `add_executable(vkspatial_tests ...)`):

```cmake
target_include_directories(vkspatial_tests PRIVATE "${CMAKE_SOURCE_DIR}/example2")
```

- [ ] **Step 4: Run tests to verify they PASS**

Run: `VULKAN_SDK=/usr/local cmake -S . -B build -DCMAKE_IGNORE_PREFIX_PATH=/opt/homebrew/anaconda3 -Ugflags_DIR -Uglog_DIR -UGTest_DIR -UCeres_DIR && VULKAN_SDK=/usr/local cmake --build build --target vkspatial_tests -j8 && ./build/test/vkspatial_tests --gtest_filter='VoxelFillDebug.*'`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add example2/VoxelFillDebug.h test/test_voxelFillDebug.cpp test/CMakeLists.txt
git commit -m "feat(example2): VoxelFillDebug.h (fill-frame tracking + field colors) + unit tests"
```
(Append the trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` to the commit body.)

---

### Task 2: `voxel_fill_debugger` example (frame-scrub viewer + `--dump`)

The tool: read frame PLYs, integrate into `AdvancedTSDF` frame by frame, download entries, diff via `FillTracker`, render occupied voxels (colored by mode, weight-thresholded) + new-this-frame + input + camera, with an ImGui debug panel. `--dump` runs headless and prints per-frame stats.

**Files:**
- Create: `example2/voxel_fill_debugger.cpp`
- Modify: `example2/CMakeLists.txt`

**Interfaces:**
- Consumes: `voxdbg::` API from Task 1; `Engine::Spatial::AdvancedTSDF` (Global Constraints); `PointCloudPass`/`ImGuiPass`; the render scaffold + `readPly`/`estimateCamera`/arg helpers referenced in Global Constraints.
- Produces: the `voxel_fill_debugger` executable. No new public interface.

- [ ] **Step 1: Copy the reusable pieces into the new file**

Create `example2/voxel_fill_debugger.cpp`. Start with these includes and copy (verbatim) from `example2/tsdf_folder_eval.cpp`: the arg helpers `strArg`, `floatArg`, `flag`, `nextPow2`, the `readPly` reader, and `estimateCamera`. Copy from `example2/object_scan_viewer.cpp` (verbatim): the `pushCloud` helper and `cameraMarker` helper (they use `PointVertex`/`Vector3f`).

```cpp
#include "ImGuiPass.h"
#include "PointCloudPass.h"
#include "VoxelFillDebug.h"

#include "Engine/Core/Context.h"
#include "Engine/Render/Application.h"
#include "Engine/Render/Camera.h"
#include "Engine/Render/GlfwWindow.h"
#include "Engine/Render/Scene.h"
#include "Engine/Spatial/AdvancedTSDF.h"

#include "imgui.h"

#include <Eigen/Core>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using Eigen::Vector3f;
using Engine::Spatial::AdvancedEntry;
namespace fs = std::filesystem;
// [copy here, verbatim inside an anonymous namespace: strArg, floatArg, flag, nextPow2, readPly,
//  estimateCamera (from tsdf_folder_eval.cpp); pushCloud, cameraMarker (from object_scan_viewer.cpp)]
```

- [ ] **Step 2: Add the voxel-set builder (new logic)**

Inside the same anonymous namespace, add:

```cpp
    // Build the "occupied" set (colored by mode; below-threshold dimmed grey or skipped) and the
    // "new this frame" highlight set from downloaded entries.
    void buildVoxelSets(const std::vector<AdvancedEntry> &entries, const std::vector<char> &isNew,
                        voxdbg::ColorMode mode, float trunc, float wMax, float wThresh, bool hideBelow,
                        const voxdbg::FillTracker &tracker, float voxel, int nFrames,
                        std::vector<PointVertex> &occupied, std::vector<PointVertex> &newThis) {
        occupied.clear();
        newThis.clear();
        for (std::size_t i = 0; i < entries.size(); ++i) {
            const AdvancedEntry &e = entries[i];
            const bool below = voxdbg::belowThreshold(e.weight, wThresh);
            if (below && hideBelow) continue;
            voxdbg::Rgba c;
            switch (mode) {
                case voxdbg::ColorMode::TsdfSign: c = voxdbg::tsdfColor(e.tsdf, trunc); break;
                case voxdbg::ColorMode::Weight: c = voxdbg::weightColor(e.weight, wMax); break;
                case voxdbg::ColorMode::FillFrame:
                    c = voxdbg::fillFrameColor(tracker.firstFrame(voxdbg::keyOf(e, voxel)), nFrames);
                    break;
                case voxdbg::ColorMode::Direction: c = voxdbg::directionColor(uint8_t(e.direction)); break;
            }
            if (below) c = {80, 80, 80, 255}; // extraction drop-out, dimmed
            occupied.push_back({{e.center.x(), e.center.y(), e.center.z()}, {c[0], c[1], c[2], c[3]}});
            if (i < isNew.size() && isNew[i])
                newThis.push_back({{e.center.x(), e.center.y(), e.center.z()}, {255, 240, 40, 255}});
        }
    }
```

- [ ] **Step 3: Write `main` — load, size the window, and the headless `--dump` path**

Add `main`. Parse CLI, load frames (`readPly`+`estimateCamera`), compute bbox → `voxel` (default `extent/200`), `trunc` (`voxel*3`), and the single-window fit guard + sizing (copy the `maxSpan`/`margin`/`axisVox`/`windowMinCorner`/`hashCap`/`maxPts` block from `tsdf_folder_eval.cpp`, and the `axisVox > 512` error `return 3`). Then build `AdvancedTSDF`. Implement `--dump`/`--no-view` headless before any window:

```cpp
int main(int argc, char **argv) {
    try {
        const std::string dir = strArg(argc, argv, "--dir", "");
        if (dir.empty() || !fs::is_directory(dir)) {
            std::cerr << "usage: voxel_fill_debugger --dir <folder> [--voxel v] [--trunc t] "
                         "[--no-p2p] [--conf L] [--hermite] [--wthresh w] [--dump]\n";
            return 2;
        }
        // ---- load frames (copy the frame-collect loop from tsdf_folder_eval.cpp: list frame_*.ply,
        //      readPly -> pts/nrm, estimateCamera -> cam, accumulate bbMin/bbMax, maxFramePts) ----
        // struct Frame { std::vector<Vector3f> pts, nrm; Vector3f cam; };
        // std::vector<Frame> frames;  Vector3f bbMin, bbMax;  size_t maxFramePts;
        // (verbatim from tsdf_folder_eval.cpp)

        const bool p2p = !flag(argc, argv, "--no-p2p");
        const float conf = floatArg(argc, argv, "--conf", 0.5f);
        const bool hermite = flag(argc, argv, "--hermite");
        const Vector3f span = bbMax - bbMin;
        const float extent = span.norm();
        const float voxel = floatArg(argc, argv, "--voxel", extent / 200.0f);
        const float trunc = floatArg(argc, argv, "--trunc", voxel * 3.0f);
        const float wThreshArg = floatArg(argc, argv, "--wthresh", 0.0f);

        const float maxSpan = span.maxCoeff();
        const int margin = int(std::ceil(trunc / voxel)) + 2;
        const long axisVox = long(std::ceil(maxSpan / voxel)) + 2L * margin;
        if (axisVox > 512) {
            const float minVoxel = maxSpan / float(512 - 2 * margin);
            std::fprintf(stderr,
                         "error: voxel %.4f too fine — object spans %ld voxels/axis but a single "
                         "AdvancedTSDF window is 512^3.\n  use --voxel >= %.4f (single-window "
                         "debugger; tiling is out of scope).\n",
                         voxel, axisVox, minVoxel);
            return 3;
        }
        const Vector3f windowMinCorner = bbMin - float(margin) * Vector3f::Constant(voxel);
        const double surf = 2.0 * double(span.x() * span.y() + span.y() * span.z() +
                                         span.z() * span.x());
        const double shell = 2.0 * double(trunc) / double(voxel);
        const double estEntries = (surf / (double(voxel) * double(voxel))) * shell * 1.5;
        const uint32_t hashCap =
                std::max(1u << 20, nextPow2(uint32_t(std::min(estEntries * 2.0, double(1u << 24)))));
        const uint32_t maxPts = nextPow2(uint32_t(std::max<std::size_t>(maxFramePts, 1u << 15)));

        Engine::Core::Context ctx;
        Engine::Spatial::AdvancedTSDF tsdf;
        tsdf.Build(ctx, voxel, trunc, hashCap, maxPts, windowMinCorner);
        tsdf.SetIntegrationQuality({3, 4, true});
        tsdf.SetPointToPlane(p2p);
        tsdf.SetConfidenceWeight(conf);
        tsdf.SetHermitePosition(hermite);

        const int nFrames = int(frames.size());
        voxdbg::FillTracker tracker(voxel);

        std::printf("dir       : %s  (%d frames, extent %.4f)\n", dir.c_str(), nFrames, extent);
        std::printf("advanced  : voxel %.4f, trunc %.4f, hashCap %u, window %ld vox/axis\n", voxel,
                    trunc, hashCap, axisVox);

        if (flag(argc, argv, "--dump") || flag(argc, argv, "--no-view")) {
            for (int f = 0; f < nFrames; ++f) {
                tsdf.Integrate(frames[f].pts, frames[f].nrm, frames[f].cam);
                const auto entries = tsdf.DownloadEntries();
                const auto isNew = tracker.update(entries, f);
                std::size_t below = 0;
                for (const auto &e : entries)
                    if (voxdbg::belowThreshold(e.weight, wThreshArg)) ++below;
                std::size_t nnew = 0;
                for (char c : isNew) nnew += (c != 0);
                std::printf("frame %3d: occupied %zu  new %zu  below-wthresh %zu\n", f,
                            entries.size(), nnew, below);
            }
            std::printf("[--dump] done.\n");
            return 0;
        }
        // ... live viewer (Step 4) ...
```

- [ ] **Step 4: Run the headless smoke to verify the data path**

Build (Step 6 CMake first, or add the target now) and run:
Run: `./build/example2/voxel_fill_debugger --dir scans/dragon --voxel 0.5 --dump`
Expected: prints one `frame N: occupied X new Y below-wthresh Z` line per frame; `occupied` is non-decreasing across frames and the final line's `occupied > 0`. (Sanity: with `--wthresh 0`, `below-wthresh` is 0.)

- [ ] **Step 5: Add the live viewer**

After the `--dump` block, add the windowed viewer. Reuse `object_scan_viewer.cpp`'s scaffold verbatim with these adaptations: create `Application` (`descriptor.window = {1280, 800, "Voxel Fill Debugger"}`), `Context`, `Scene`, `Camera` (`SetPerspective` near/far from `extent`; `SetOrbit(Vec3(center), extent*2)` where `center = 0.5*(bbMin+bbMax)`), `RenderGraph{PointCloudPass(ctx, format, SCAN_VIEWER_SHADER_DIR? -> use a dedicated define VOXDBG_SHADER_DIR), ImGuiPass}`, `pc->SetPointSize(3.0f)`, and the mouse/key trackball listeners + manual `while(!ShouldClose())` loop — all copied from object_scan_viewer.cpp.

Replace object_scan_viewer's per-frame `showFrame` with this scrubbing version (integrates/downloads/diffs, then builds the 4 sets). Keep a `State` struct with `int frame; bool playing; float fps; ColorMode mode; float wThresh; bool hideBelow; bool showOccupied/showNew/showInput/showCamera; int shown=-1; float wMax=1;`:

```cpp
        std::vector<AdvancedEntry> curEntries;
        std::vector<char> curNew;
        auto rebuildTo = [&](int target) {
            target = std::clamp(target, 0, std::max(0, nFrames - 1));
            if (nFrames == 0) return;
            vkDeviceWaitIdle(ctx.device);
            if (target < state.shown) { tsdf.Reset(); tracker.reset(); state.shown = -1; }
            for (int f = state.shown + 1; f <= target; ++f) {
                tsdf.Integrate(frames[f].pts, frames[f].nrm, frames[f].cam);
                curEntries = tsdf.DownloadEntries();
                curNew = tracker.update(curEntries, f);
            }
            state.shown = target;
            state.wMax = 1.0f;
            for (const auto &e : curEntries) state.wMax = std::max(state.wMax, e.weight);
        };
        auto refreshSets = [&]() {
            std::vector<PointVertex> occ, nw;
            buildVoxelSets(curEntries, curNew, state.mode, trunc, state.wMax, state.wThresh,
                           state.hideBelow, tracker, voxel, nFrames, occ, nw);
            pc->SetPointSet(0, occ);
            pc->SetPointSet(1, nw);
            std::vector<PointVertex> in;
            pushCloud(in, frames[std::clamp(state.shown, 0, nFrames - 1)].pts, 100, 110, 120);
            pc->SetPointSet(2, in);
            pc->SetPointSet(3, cameraMarker(frames[std::clamp(state.shown, 0, nFrames - 1)].cam));
        };
```

Drive them from the loop: when `state.frame != state.shown` (scrub/play advance) call `rebuildTo(state.frame)` then `refreshSets()`; when only a color-mode/threshold/layer control changed, call `refreshSets()` alone (no re-integrate). Autoplay advances `state.frame` by 1 every `1/fps` seconds (loop back to 0 at the end), same pattern as object_scan_viewer.

ImGui panel (`imgui->SetUi([&]{ ... })`):

```cpp
            ImGui::Begin("Voxel Fill Debug");
            ImGui::Text("frame %d / %d", state.frame + 1, nFrames);
            if (ImGui::SliderInt("frame", &state.frame, 0, std::max(0, nFrames - 1)))
                state.playing = false;
            ImGui::Checkbox("play", &state.playing);
            ImGui::SameLine();
            if (ImGui::Button("restart")) { state.frame = 0; state.playing = true; }
            ImGui::SliderFloat("fps", &state.fps, 1.0f, 30.0f, "%.0f");
            ImGui::SeparatorText("Color mode");
            int m = int(state.mode);
            bool cm = false;
            cm |= ImGui::RadioButton("tsdf", &m, 0); ImGui::SameLine();
            cm |= ImGui::RadioButton("weight", &m, 1); ImGui::SameLine();
            cm |= ImGui::RadioButton("fill-frame", &m, 2); ImGui::SameLine();
            cm |= ImGui::RadioButton("direction", &m, 3);
            if (cm) { state.mode = voxdbg::ColorMode(m); state.dirty = true; }
            if (ImGui::SliderFloat("weight thresh", &state.wThresh, 0.0f, state.wMax, "%.3f"))
                state.dirty = true;
            if (ImGui::Checkbox("hide below thresh", &state.hideBelow)) state.dirty = true;
            ImGui::SeparatorText("Layers");
            if (ImGui::Checkbox("occupied", &state.showOccupied)) pc->SetVisible(0, state.showOccupied);
            if (ImGui::Checkbox("new this frame", &state.showNew)) pc->SetVisible(1, state.showNew);
            if (ImGui::Checkbox("input", &state.showInput)) pc->SetVisible(2, state.showInput);
            if (ImGui::Checkbox("camera", &state.showCamera)) pc->SetVisible(3, state.showCamera);
            ImGui::SeparatorText("Stats");
            ImGui::Text("occupied voxels: %zu", curEntries.size());
            std::size_t below = 0;
            for (const auto &e : curEntries)
                if (voxdbg::belowThreshold(e.weight, state.wThresh)) ++below;
            ImGui::Text("below thresh: %zu", below);
            ImGui::End();
```

Use a `state.dirty` flag: set it when a color/threshold control changes; in the loop, if `state.dirty` call `refreshSets()` and clear it. (`hideBelow`/mode/threshold change appearance without re-integration.)

- [ ] **Step 6: Add the CMake target**

In `example2/CMakeLists.txt`, inside the `if (GLSLC_EXECUTABLE)` block (next to `object_scan_viewer`), add:

```cmake
    # Voxel-fill debugger: integrate a folder of frame PLYs into AdvancedTSDF frame by frame and
    # render the filling voxels (colored by tsdf/weight/fill-frame/direction, weight-thresholded)
    # to debug coverage/holes. --dump = headless per-frame stats.
    add_spatial_example(voxel_fill_debugger voxel_fill_debugger.cpp PointCloudPass.cpp ImGuiPass.cpp)
    target_link_libraries(voxel_fill_debugger PRIVATE Engine::Render imgui Engine::Core)
    add_compiled_shaders(voxel_fill_debugger VOXDBG_SHADER_DIR Shaders/pointcloud.vert Shaders/pointcloud.frag)
```

(Use `VOXDBG_SHADER_DIR` as the shader-dir define in the .cpp: `const std::string shaderDir = VOXDBG_SHADER_DIR;`.)

- [ ] **Step 7: Build + verify**

Run: `VULKAN_SDK=/usr/local cmake -S . -B build && VULKAN_SDK=/usr/local cmake --build build --target voxel_fill_debugger -j8`
Expected: builds clean.
Run: `./build/example2/voxel_fill_debugger --dir scans/dragon --voxel 0.5 --dump | tail -3`
Expected: final `frame N` line with `occupied > 0`, non-decreasing occupied.
Run: `./build/example2/voxel_fill_debugger 2>&1 | head -1`
Expected: the usage line (no `--dir`).

- [ ] **Step 8: Commit**

```bash
git add example2/voxel_fill_debugger.cpp example2/CMakeLists.txt
git commit -m "feat(example2): voxel_fill_debugger — per-frame TSDF voxel-fill viewer + --dump"
```
(Append the trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` to the commit body.)

---

## Self-Review

**1. Spec coverage:**
- Per-frame accumulation viewer (folder PLY → integrate → download → render) → Task 2. ✓
- Pure logic (FillTracker first-fill, color modes, threshold) + unit tests → Task 1. ✓
- Point rendering, 4 sets (occupied/new/input/camera) → Task 2 buildVoxelSets + refreshSets. ✓
- Color modes tsdf/weight/fill-frame/direction → Task 1 functions, Task 2 radio. ✓
- Weight-threshold hole device + below-thresh stat → Task 1 `belowThreshold`, Task 2 slider/stat/dim. ✓
- Frame scrub (Reset+replay back, integrate-forward) → Task 2 `rebuildTo`. ✓
- Single-window guard (`axisVox>512`) + sizing → Task 2 (copied from tsdf_folder_eval). ✓
- `--dump`/`--no-view` headless + smoke → Task 2 Step 3/4/7. ✓
- CLI (`--dir/--voxel/--trunc/--no-p2p/--conf/--hermite/--wthresh/--dump`) → Task 2 main. ✓
- Tiling / instanced-cube / click-inspection explicitly out of scope → not built. ✓

**2. Placeholder scan:** Task 1 is fully-coded (header + tests). Task 2 references verbatim-copy blocks from two named existing files (arg helpers/readPly/estimateCamera/frame-load from tsdf_folder_eval.cpp; pushCloud/cameraMarker/render-scaffold from object_scan_viewer.cpp) — these are reuse-by-copy of existing committed code, not missing content; all NEW logic (buildVoxelSets, main sizing/dump, rebuildTo/refreshSets, ImGui panel, CMake) is shown in full. No TBD/vague-error placeholders.

**3. Type consistency:** `voxdbg::` names (VoxelKey, FillTracker, keyOf, ColorMode, tsdfColor/weightColor/fillFrameColor/directionColor, belowThreshold, Rgba) match between Task 1 (definitions) and Task 2 (uses). `AdvancedEntry` fields (`center/direction/tsdf/weight/normal`) match the engine header. `PointVertex{pos[3],rgba[4]}`, `SetPointSet/SetVisible/SetPointSize` match PointCloudPass.h. Shader-dir define `VOXDBG_SHADER_DIR` consistent between CMake and .cpp.
