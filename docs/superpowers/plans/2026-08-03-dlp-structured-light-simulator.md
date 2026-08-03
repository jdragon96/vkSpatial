# DLP Structured-Light Simulator — Step 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load a mesh into `Engine::Render` and project structured-light patterns onto it in a real-time interactive window, with projector occlusion (shadows) and all Blender pattern types.

**Architecture:** Two isolated units. (1) `Engine::StructuredLight::PatternGenerators.h` — header-only, pure-CPU port of every Blender pattern generator, output as RGBA8 `PatternImage`, unit-tested with GTest. (2) `example2/dlp_simulator` — a Vulkan app following `example2/ShadowMap.cpp` conventions: `DlpProjectorPass` renders an offscreen projector-POV depth map, then a scene pass that reprojects each fragment into projector clip-space, samples the pattern texture, tests occlusion against the depth map, and modulates by a Lambert term. An ImGui panel drives pattern/projector parameters live.

**Tech Stack:** C++17, Vulkan (dynamic rendering), Eigen (`vkMath`), Dear ImGui, GTest, glslc.

## Global Constraints

- Namespace for the pattern library: `Engine::StructuredLight`. Header-only (no `.cpp`, no new `add_library`).
- Pattern texture is a single format everywhere: `VK_FORMAT_R8G8B8A8_UNORM`; grayscale patterns replicate the value into R,G,B with A=255.
- Follow existing engine conventions: `MakeUnique`/factory + `UniquePtr`, `std::runtime_error("<ClassName>: ...")` on failure, builder chaining on `GraphicsPipelineDescriptor`.
- The projector depth pass and the scene reprojection MUST use the identical `projVP = vkMath::Perspective(...) * vkMath::LookAt(...)`. `vkMath::Perspective` yields Vulkan `[0,1]` NDC-z; compare `ndc.z` directly to the sampled depth (no `*0.5+0.5`).
- Push-constant block is shared by all DLP shaders and is byte-identical to this C++ struct (208 bytes, same size as ShadowMap's, known to work on this MoltenVK/M4 setup):
  ```cpp
  struct DlpPush {
      vkMath::Mat4 model    = vkMath::Mat4::Identity();  // 64
      vkMath::Mat4 viewProj = vkMath::Mat4::Identity();  // 64  (camera; unused by depth vert)
      vkMath::Mat4 projVP   = vkMath::Mat4::Identity();  // 64  (projector)
      float        projPos[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // 16 (xyz + pad)
  };
  ```
- Build/test commands (from repo root, this machine): `VULKAN_SDK=/usr/local`, M4 Max UMA. Configure once, then build specific targets. Tests target is `vkspatial_tests`.

## File Structure

```
src/Engine/StructuredLight/PatternGenerators.h   NEW  header-only pattern library
test/test_patternGenerators.cpp                  NEW  GTest unit tests (globbed)
example2/dlp_depth.vert                           NEW  projector-POV depth vertex shader
example2/dlp_scene.vert                           NEW  scene vertex shader (world pos/normal out)
example2/dlp_scene.frag                           NEW  projective pattern + occlusion + Lambert
example2/DlpProjectorPass.h                        NEW  RenderPass: resources + 2-pass Execute
example2/DlpProjectorPass.cpp                      NEW  RenderPass implementation
example2/dlp_simulator.cpp                         NEW  app: mesh, camera, projector, ImGui, keys
example2/CMakeLists.txt                            MODIFY add dlp_simulator target + shaders
```

Boundaries: the pattern library never touches Vulkan (testable alone). `DlpProjectorPass` owns all GPU resources and both pipelines; the app owns window/camera/projector-pose state and pushes it into the pass through setters. Shaders are the only shared interface between the vertex layout and the frag math.

---

### Task 1: PatternGenerators — types + grayscale generators

**Files:**
- Create: `src/Engine/StructuredLight/PatternGenerators.h`
- Test: `test/test_patternGenerators.cpp`

**Interfaces:**
- Produces (consumed by Task 2, 4, 6):
  - `enum class Engine::StructuredLight::PatternType { PhaseShift, GrayCode, Binary, ColorCoded, ColorPhase };`
  - `enum class Engine::StructuredLight::ColorCodeMode { DeBruijn, Hamming, SelfEqualizing };`
  - `struct PatternParams { PatternType type; int width, height, steps, bits, binaryPatterns; float freq; ColorCodeMode colorMode; int colorK, colorN; float colorPhaseFreq; bool precompensate; float responseMatrix[9]; };` (defaults below)
  - `struct PatternImage { int width, height; std::vector<uint8_t> pixels; };` (pixels = width*height*4, RGBA8)
  - `int PatternCount(const PatternParams&);`
  - `PatternImage MakePattern(const PatternParams&, int step);` (grayscale cases implemented here; color cases added in Task 2)
  - `PatternImage MakeFlat(int width, int height, uint8_t value);`

- [ ] **Step 1: Write the failing test**

Create `test/test_patternGenerators.cpp`:
```cpp
#include "Engine/StructuredLight/PatternGenerators.h"

#include <gtest/gtest.h>

#include <cmath>

using namespace Engine::StructuredLight;

namespace {
    PatternParams basePhase() {
        PatternParams p;
        p.type = PatternType::PhaseShift;
        p.width = 64;
        p.height = 8;
        p.steps = 3;
        p.freq = 4.0f;
        return p;
    }
    // Red channel of pixel (x,y).
    uint8_t R(const PatternImage &img, int x, int y) {
        return img.pixels[(size_t(y) * img.width + x) * 4 + 0];
    }
}

TEST(PatternGenerators, PhaseShiftShapeAndGrayscale) {
    PatternImage img = MakePattern(basePhase(), 0);
    EXPECT_EQ(img.width, 64);
    EXPECT_EQ(img.height, 8);
    ASSERT_EQ(img.pixels.size(), size_t(64) * 8 * 4);
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x) {
            const size_t i = (size_t(y) * img.width + x) * 4;
            EXPECT_EQ(img.pixels[i + 0], img.pixels[i + 1]); // R==G
            EXPECT_EQ(img.pixels[i + 1], img.pixels[i + 2]); // G==B
            EXPECT_EQ(img.pixels[i + 3], 255);               // opaque
        }
}

TEST(PatternGenerators, PhaseShiftRowInvariant) {
    PatternImage img = MakePattern(basePhase(), 1);
    for (int y = 1; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x)
            EXPECT_EQ(R(img, x, y), R(img, x, 0)) << "x=" << x << " y=" << y;
}

TEST(PatternGenerators, PhaseShiftMatchesCosineFormula) {
    PatternParams p = basePhase();
    const int step = 1;
    PatternImage img = MakePattern(p, step);
    const double pi = 3.14159265358979323846;
    for (int x = 0; x < p.width; ++x) {
        const double phase = 2 * pi * p.freq * x / p.width - 2 * pi * step / p.steps;
        double v = 127.5 * (1.0 + std::cos(phase));
        v = std::min(255.0, std::max(0.0, v));
        EXPECT_NEAR(R(img, x, 0), static_cast<uint8_t>(v), 1) << "x=" << x;
    }
}

TEST(PatternGenerators, GrayCodeAdjacentColumnsDifferByOneBit) {
    PatternParams p;
    p.type = PatternType::GrayCode;
    p.width = 256;
    p.height = 2;
    p.bits = 4;
    // Reconstruct the full 4-bit code at each column from all bit-planes.
    std::vector<int> code(p.width, 0);
    for (int bit = 0; bit < p.bits; ++bit) {
        PatternImage img = MakePattern(p, bit);
        for (int x = 0; x < p.width; ++x)
            if (R(img, x, 0) > 127) code[x] |= (1 << (p.bits - 1 - bit));
    }
    for (int x = 1; x < p.width; ++x) {
        const int diff = code[x] ^ code[x - 1];
        const int bitsSet = __builtin_popcount(diff);
        EXPECT_LE(bitsSet, 1) << "gray discontinuity at x=" << x;
    }
}

TEST(PatternGenerators, BinaryIsBinaryValued) {
    PatternParams p;
    p.type = PatternType::Binary;
    p.width = 128;
    p.height = 2;
    p.binaryPatterns = 5;
    PatternImage img = MakePattern(p, 2);
    for (int x = 0; x < p.width; ++x) {
        const uint8_t v = R(img, x, 0);
        EXPECT_TRUE(v == 0 || v == 255) << "x=" << x << " v=" << int(v);
    }
}

TEST(PatternGenerators, PatternCountPerType) {
    PatternParams p;
    p.type = PatternType::PhaseShift; p.steps = 7;          EXPECT_EQ(PatternCount(p), 7);
    p.type = PatternType::GrayCode;   p.bits = 6;           EXPECT_EQ(PatternCount(p), 6);
    p.type = PatternType::Binary;     p.binaryPatterns = 4; EXPECT_EQ(PatternCount(p), 4);
}

TEST(PatternGenerators, FlatIsConstant) {
    PatternImage img = MakeFlat(16, 4, 200);
    ASSERT_EQ(img.pixels.size(), size_t(16) * 4 * 4);
    for (size_t i = 0; i < img.pixels.size(); i += 4) {
        EXPECT_EQ(img.pixels[i + 0], 200);
        EXPECT_EQ(img.pixels[i + 3], 255);
    }
}
```

- [ ] **Step 2: Re-run CMake configure so the new test is globbed, then run to verify it fails**

Run:
```bash
cmake -S . -B build >/dev/null && cmake --build build --target vkspatial_tests 2>&1 | tail -20
```
Expected: FAIL — `fatal error: 'Engine/StructuredLight/PatternGenerators.h' file not found`.

- [ ] **Step 3: Write the header (types + grayscale generators)**

Create `src/Engine/StructuredLight/PatternGenerators.h`:
```cpp
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace Engine::StructuredLight {

    enum class PatternType { PhaseShift, GrayCode, Binary, ColorCoded, ColorPhase };
    enum class ColorCodeMode { DeBruijn, Hamming, SelfEqualizing };

    struct PatternParams {
        PatternType type = PatternType::PhaseShift;
        int width = 1280;
        int height = 720;
        int steps = 3;      // phase-shift N (>=3)
        float freq = 16.0f; // fringe cycles across width
        int bits = 4;       // gray-code bit count
        int binaryPatterns = 5;
        ColorCodeMode colorMode = ColorCodeMode::Hamming;
        int colorK = 5; // De Bruijn / self-equalizing color count
        int colorN = 4; // decode window length
        float colorPhaseFreq = 64.0f;
        bool precompensate = false;
        float responseMatrix[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}; // row-major 3x3
    };

    struct PatternImage {
        int width = 0;
        int height = 0;
        std::vector<uint8_t> pixels; // width*height*4, RGBA8
    };

    namespace detail {

        // grayscale row (width) -> RGBA image (grayscale replicated, alpha 255)
        inline PatternImage GrayRowToImage(const std::vector<uint8_t> &row, int height) {
            const int width = static_cast<int>(row.size());
            PatternImage img;
            img.width = width;
            img.height = height;
            img.pixels.resize(size_t(width) * height * 4);
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x) {
                    const size_t i = (size_t(y) * width + x) * 4;
                    img.pixels[i + 0] = row[x];
                    img.pixels[i + 1] = row[x];
                    img.pixels[i + 2] = row[x];
                    img.pixels[i + 3] = 255;
                }
            return img;
        }

        inline uint8_t ClampU8(double v) {
            if (v < 0.0) v = 0.0;
            if (v > 255.0) v = 255.0;
            return static_cast<uint8_t>(std::lround(v));
        }

        inline std::vector<uint8_t> Sinusoidal(int step, int N, double freq, int W) {
            const double pi = 3.14159265358979323846;
            std::vector<uint8_t> row(W);
            for (int x = 0; x < W; ++x) {
                const double phase = 2 * pi * freq * x / W - 2 * pi * step / N;
                row[x] = ClampU8(127.5 * (1.0 + std::cos(phase)));
            }
            return row;
        }

        inline std::vector<uint8_t> GrayCodeRow(int bit, int nBits, int W) {
            std::vector<uint8_t> row(W);
            for (int x = 0; x < W; ++x) {
                long long gray = (static_cast<long long>(x) * (1LL << nBits)) / W;
                gray ^= (gray >> 1);
                row[x] = ((gray >> (nBits - 1 - bit)) & 1) ? 255 : 0;
            }
            return row;
        }

        inline std::vector<uint8_t> BinaryRow(int step, int W) {
            std::vector<uint8_t> row(W);
            for (int x = 0; x < W; ++x) {
                const long long v = (static_cast<long long>(x) * (1LL << (step + 1))) / W;
                row[x] = (v % 2) ? 255 : 0;
            }
            return row;
        }

    } // namespace detail

    inline PatternImage MakeFlat(int width, int height, uint8_t value) {
        PatternImage img;
        img.width = width;
        img.height = height;
        img.pixels.assign(size_t(width) * height * 4, value);
        for (size_t i = 3; i < img.pixels.size(); i += 4) img.pixels[i] = 255;
        return img;
    }

    inline int PatternCount(const PatternParams &p) {
        switch (p.type) {
            case PatternType::PhaseShift: return p.steps;
            case PatternType::GrayCode:   return p.bits;
            case PatternType::Binary:     return p.binaryPatterns;
            case PatternType::ColorCoded: return 1;
            case PatternType::ColorPhase: return 1;
        }
        return 1;
    }

    // Color cases are defined in Task 2 (same file). For Task 1 they throw.
    PatternImage MakeColorCoded(const PatternParams &p);
    PatternImage MakeColorPhase(const PatternParams &p);

    inline PatternImage MakePattern(const PatternParams &p, int step) {
        switch (p.type) {
            case PatternType::PhaseShift:
                return detail::GrayRowToImage(detail::Sinusoidal(step, p.steps, p.freq, p.width), p.height);
            case PatternType::GrayCode:
                return detail::GrayRowToImage(detail::GrayCodeRow(step, p.bits, p.width), p.height);
            case PatternType::Binary:
                return detail::GrayRowToImage(detail::BinaryRow(step, p.width), p.height);
            case PatternType::ColorCoded: return MakeColorCoded(p);
            case PatternType::ColorPhase: return MakeColorPhase(p);
        }
        throw std::runtime_error("PatternGenerators: unknown pattern type");
    }

} // namespace Engine::StructuredLight
```
Note: `MakeColorCoded`/`MakeColorPhase` are declared but not yet defined — Task 1's tests never hit the color path, so linking the test only needs the definitions once Task 2 lands. To keep Task 1 independently green, add temporary inline stubs at the bottom of the header for now:
```cpp
namespace Engine::StructuredLight {
    inline PatternImage MakeColorCoded(const PatternParams &) {
        throw std::runtime_error("PatternGenerators: color patterns not yet implemented");
    }
    inline PatternImage MakeColorPhase(const PatternParams &) {
        throw std::runtime_error("PatternGenerators: color patterns not yet implemented");
    }
}
```
(Task 2 replaces these stubs with the real implementations.)

- [ ] **Step 4: Build and run the tests to verify they pass**

Run:
```bash
cmake --build build --target vkspatial_tests 2>&1 | tail -5 && \
./build/test/vkspatial_tests --gtest_filter='PatternGenerators.*' 2>&1 | tail -20
```
Expected: all `PatternGenerators.*` tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Engine/StructuredLight/PatternGenerators.h test/test_patternGenerators.cpp
git commit -m "feat(structured-light): pattern generators — grayscale core (phase/gray/binary/flat)"
```

---

### Task 2: PatternGenerators — color patterns + metadata

**Files:**
- Modify: `src/Engine/StructuredLight/PatternGenerators.h` (replace the color stubs with real code)
- Test: `test/test_patternGenerators.cpp` (add color tests)

**Interfaces:**
- Consumes: everything from Task 1.
- Produces (consumed by Task 6): `struct ColorCodeInfo { int logicalStripes; int projectedStripes; int decodeWindow; };` and `ColorCodeInfo ColorCodeMetadata(const PatternParams&);`; real `MakeColorCoded` / `MakeColorPhase`; `MakeSolidColor(int,int,uint8_t,uint8_t,uint8_t)`.

- [ ] **Step 1: Write the failing tests (append to test file)**

Append to `test/test_patternGenerators.cpp`:
```cpp
namespace {
    void RGB(const PatternImage &img, int x, int y, int &r, int &g, int &b) {
        const size_t i = (size_t(y) * img.width + x) * 4;
        r = img.pixels[i + 0]; g = img.pixels[i + 1]; b = img.pixels[i + 2];
    }
}

TEST(PatternGenerators, DeBruijnNoEqualNeighborsAndUniqueWindows) {
    PatternParams p;
    p.type = PatternType::ColorCoded;
    p.colorMode = ColorCodeMode::DeBruijn;
    p.colorK = 5;
    p.colorN = 3;
    p.width = 240;
    p.height = 2;
    PatternImage img = MakePattern(p, 0);
    // Collapse the stripe image back to its per-stripe color index sequence.
    ColorCodeInfo info = ColorCodeMetadata(p);
    std::vector<int> seq;
    int prev = -1;
    for (int x = 0; x < p.width; ++x) {
        int r, g, b; RGB(img, x, 0, r, g, b);
        // map to nearest palette index by exact match against the 5-color palette
        static const int pal[6][3] = {{224,31,31},{31,224,31},{31,31,224},
                                      {31,224,224},{224,31,224},{224,224,31}};
        int idx = 0;
        for (int c = 0; c < p.colorK; ++c)
            if (pal[c][0] == r && pal[c][1] == g && pal[c][2] == b) idx = c;
        if (idx != prev) { seq.push_back(idx); prev = idx; }
    }
    ASSERT_GE(seq.size(), size_t(info.logicalStripes));
    seq.resize(info.logicalStripes);
    for (size_t i = 1; i < seq.size(); ++i)
        EXPECT_NE(seq[i], seq[i - 1]) << "equal neighbor at " << i;
    // all length-n windows unique
    std::vector<std::vector<int>> windows;
    for (size_t i = 0; i + p.colorN <= seq.size(); ++i)
        windows.emplace_back(seq.begin() + i, seq.begin() + i + p.colorN);
    for (size_t a = 0; a < windows.size(); ++a)
        for (size_t b2 = a + 1; b2 < windows.size(); ++b2)
            EXPECT_NE(windows[a], windows[b2]) << "duplicate window " << a << "," << b2;
}

TEST(PatternGenerators, SelfEqualizingChannelPairSumsTo255) {
    PatternParams p;
    p.type = PatternType::ColorCoded;
    p.colorMode = ColorCodeMode::SelfEqualizing;
    p.colorK = 5;
    p.colorN = 3;
    p.width = 480;
    p.height = 1;
    PatternImage img = MakePattern(p, 0);
    ColorCodeInfo info = ColorCodeMetadata(p);
    const int physical = info.projectedStripes; // 2 * logical
    // sample the center of each physical sub-stripe pair and check complement sum
    for (int s = 0; s + 1 < physical; s += 2) {
        const int x0 = (s * p.width) / physical + 1;
        const int x1 = ((s + 1) * p.width) / physical + 1;
        int r0, g0, b0, r1, g1, b1;
        RGB(img, x0, 0, r0, g0, b0);
        RGB(img, x1, 0, r1, g1, b1);
        EXPECT_EQ(r0 + r1, 255);
        EXPECT_EQ(g0 + g1, 255);
        EXPECT_EQ(b0 + b1, 255);
    }
}

TEST(PatternGenerators, ColorPhaseChannelsAreThreeStepPsp) {
    PatternParams p;
    p.type = PatternType::ColorPhase;
    p.width = 96;
    p.height = 1;
    p.colorPhaseFreq = 3.0f;
    PatternImage img = MakePattern(p, 0);
    const double pi = 3.14159265358979323846;
    for (int x = 0; x < p.width; ++x) {
        int r, g, b; RGB(img, x, 0, r, g, b);
        int expect[3];
        for (int step = 0; step < 3; ++step) {
            const double phase = 2 * pi * p.colorPhaseFreq * x / p.width - 2 * pi * step / 3;
            double v = 127.5 * (1.0 + std::cos(phase));
            v = std::min(255.0, std::max(0.0, v));
            expect[step] = static_cast<int>(std::lround(v));
        }
        EXPECT_NEAR(r, expect[0], 1);
        EXPECT_NEAR(g, expect[1], 1);
        EXPECT_NEAR(b, expect[2], 1);
    }
}

TEST(PatternGenerators, PrecompensationIdentityIsNoOpAndSingularThrows) {
    PatternParams p;
    p.type = PatternType::ColorPhase;
    p.width = 32;
    p.height = 1;
    p.colorPhaseFreq = 2.0f;
    PatternImage plain = MakePattern(p, 0);
    p.precompensate = true; // identity responseMatrix by default
    PatternImage same = MakePattern(p, 0);
    for (size_t i = 0; i < plain.pixels.size(); ++i)
        EXPECT_NEAR(plain.pixels[i], same.pixels[i], 1);
    // singular matrix -> throw
    for (int i = 0; i < 9; ++i) p.responseMatrix[i] = 0.0f;
    EXPECT_THROW(MakePattern(p, 0), std::runtime_error);
}
```

- [ ] **Step 2: Run to verify the new tests fail**

Run:
```bash
cmake --build build --target vkspatial_tests 2>&1 | tail -20
```
Expected: FAIL — `ColorCodeMetadata`/`ColorCodeInfo` undefined (and, once compiling, the stub `MakeColorCoded` throws).

- [ ] **Step 3: Replace the color stubs with real implementations**

In `src/Engine/StructuredLight/PatternGenerators.h`, delete the two temporary stub definitions from Task 1 and add the following. Put the helpers inside `namespace detail` (before `MakePattern`) and the public `ColorCodeInfo`/metadata/`MakeSolidColor`/`MakeColorCoded`/`MakeColorPhase` after `MakePattern`. Add `#include <algorithm>`, `#include <array>`, `#include <functional>`, `#include <map>`, `#include <string>` at the top.

```cpp
// ---- inside namespace detail ----

inline const std::array<std::array<int, 3>, 6> &ColorPalette() {
    static const std::array<std::array<int, 3>, 6> pal = {{
        {{224, 31, 31}}, {{31, 224, 31}}, {{31, 31, 224}},
        {{31, 224, 224}}, {{224, 31, 224}}, {{224, 224, 31}}}};
    return pal;
}
inline const std::array<std::array<int, 3>, 8> &HammingPalette() {
    static const std::array<std::array<int, 3>, 8> pal = {{
        {{0, 0, 0}}, {{255, 0, 0}}, {{0, 255, 0}}, {{255, 255, 0}},
        {{0, 0, 255}}, {{255, 0, 255}}, {{0, 255, 255}}, {{255, 255, 255}}}};
    return pal;
}

// B(k,n): every length-n cyclic window appears exactly once. Length k^n.
inline std::vector<int> DeBruijn(int k, int n) {
    std::vector<int> a(size_t(k) * n + 1, 0);
    std::vector<int> seq;
    std::function<void(int, int)> db = [&](int t, int p) {
        if (t > n) {
            if (n % p == 0)
                for (int i = 1; i <= p; ++i) seq.push_back(a[i]);
        } else {
            a[t] = a[t - p];
            db(t + 1, p);
            for (int j = a[t - p] + 1; j < k; ++j) { a[t] = j; db(t + 1, t); }
        }
    };
    db(1, 1);
    return seq;
}

inline std::string TupleKey(const std::vector<int> &v) {
    std::string s;
    for (int x : v) { s += std::to_string(x); s += ','; }
    return s;
}

// Constrained De Bruijn: unique length-n windows AND no two equal neighbors.
// Length k*(k-1)^(n-1). Euler circuit (Hierholzer) over valid (n-1)-tuples.
inline std::vector<int> ConstrainedDeBruijn(int k, int n) {
    std::vector<std::vector<int>> vertices;
    std::vector<int> word(n - 1, 0);
    std::function<void(int)> gen = [&](int pos) {
        if (pos == n - 1) {
            for (int i = 0; i + 1 < n - 1; ++i)
                if (word[i] == word[i + 1]) return;
            vertices.push_back(word);
            return;
        }
        for (int s = 0; s < k; ++s) { word[pos] = s; gen(pos + 1); }
    };
    gen(0);

    std::map<std::string, std::vector<std::vector<int>>> adj;
    for (const auto &v : vertices) {
        std::vector<std::vector<int>> nbrs;
        const int last = v.empty() ? -1 : v.back();
        for (int s = k - 1; s >= 0; --s) {
            if (s == last) continue;
            std::vector<int> nv(v.begin() + 1, v.end());
            nv.push_back(s);
            nbrs.push_back(std::move(nv));
        }
        adj[TupleKey(v)] = std::move(nbrs);
    }

    std::vector<std::vector<int>> stack{vertices.front()};
    std::vector<std::vector<int>> circuit;
    while (!stack.empty()) {
        std::vector<int> &v = stack.back();
        std::vector<std::vector<int>> &nb = adj[TupleKey(v)];
        if (!nb.empty()) {
            std::vector<int> nx = nb.back();
            nb.pop_back();
            stack.push_back(std::move(nx));
        } else {
            circuit.push_back(v);
            stack.pop_back();
        }
    }
    std::reverse(circuit.begin(), circuit.end());

    std::vector<int> linear;
    if (!circuit.empty()) {
        linear = circuit.front();
        for (size_t i = 1; i < circuit.size(); ++i) linear.push_back(circuit[i].back());
    }
    long long edges = k;
    for (int i = 0; i < n - 1; ++i) edges *= (k - 1);
    if (static_cast<long long>(linear.size()) > edges)
        linear.resize(static_cast<size_t>(edges));
    return linear;
}

// Hamming color sequence: adjacent codewords Hamming distance 1, unique n-windows.
// Length 3^(n-1). seq[0]=0b111; seq[i]=seq[i-1] ^ (1<<diffs[i-1]).
inline std::vector<int> HammingColorSequence(int n) {
    std::vector<int> diffs = DeBruijn(3, n - 1);
    std::vector<int> seq{0b111};
    for (size_t i = 0; i + 1 < diffs.size(); ++i)
        seq.push_back(seq.back() ^ (1 << diffs[i]));
    return seq;
}

// pixels: H*W*3 RGB row-tiled from a per-stripe index sequence over a palette.
inline std::vector<uint8_t> StripeRow(const std::vector<int> &seq,
                                      const std::vector<std::array<int, 3>> &palette,
                                      int W) {
    std::vector<uint8_t> row(size_t(W) * 3);
    const int len = static_cast<int>(seq.size());
    for (int x = 0; x < W; ++x) {
        const int stripe = static_cast<int>((static_cast<long long>(x) * len) / W);
        const std::array<int, 3> &c = palette[seq[stripe]];
        row[x * 3 + 0] = static_cast<uint8_t>(c[0]);
        row[x * 3 + 1] = static_cast<uint8_t>(c[1]);
        row[x * 3 + 2] = static_cast<uint8_t>(c[2]);
    }
    return row;
}

inline std::vector<uint8_t> SelfEqualizingRow(const std::vector<int> &seq,
                                              const std::vector<std::array<int, 3>> &palette,
                                              int W) {
    std::vector<uint8_t> row(size_t(W) * 3);
    const int len = static_cast<int>(seq.size());
    const int physical = 2 * len;
    for (int x = 0; x < W; ++x) {
        const int pIndex = static_cast<int>((static_cast<long long>(x) * physical) / W);
        const int logical = pIndex / 2;
        const std::array<int, 3> &c = palette[seq[logical]];
        const bool complement = (pIndex % 2) == 1;
        for (int ch = 0; ch < 3; ++ch)
            row[x * 3 + ch] = static_cast<uint8_t>(complement ? 255 - c[ch] : c[ch]);
    }
    return row;
}

inline PatternImage RgbRowToImage(const std::vector<uint8_t> &row, int height) {
    const int width = static_cast<int>(row.size() / 3);
    PatternImage img;
    img.width = width;
    img.height = height;
    img.pixels.resize(size_t(width) * height * 4);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            const size_t d = (size_t(y) * width + x) * 4;
            img.pixels[d + 0] = row[x * 3 + 0];
            img.pixels[d + 1] = row[x * 3 + 1];
            img.pixels[d + 2] = row[x * 3 + 2];
            img.pixels[d + 3] = 255;
        }
    return img;
}

// 3x3 inverse; throws on (near-)singular matrix. m is row-major length 9.
inline void Invert3x3(const float m[9], double inv[9]) {
    const double a = m[0], b = m[1], c = m[2];
    const double d = m[3], e = m[4], f = m[5];
    const double g = m[6], h = m[7], i = m[8];
    const double A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const double det = a * A + b * B + c * C;
    if (std::abs(det) < 1e-8)
        throw std::runtime_error("PatternGenerators: singular color response matrix");
    const double invDet = 1.0 / det;
    inv[0] = A * invDet;
    inv[1] = -(b * i - c * h) * invDet;
    inv[2] = (b * f - c * e) * invDet;
    inv[3] = B * invDet;
    inv[4] = (a * i - c * g) * invDet;
    inv[5] = -(a * f - c * d) * invDet;
    inv[6] = C * invDet;
    inv[7] = -(a * h - b * g) * invDet;
    inv[8] = (a * e - b * d) * invDet;
}

// corrected = clip( (rgb/255) * inv^T ) * 255  (camera_rgb = M * projector_rgb model)
inline void PrecompensateInPlace(PatternImage &img, const float responseMatrix[9]) {
    double inv[9];
    Invert3x3(responseMatrix, inv);
    for (size_t i = 0; i < img.pixels.size(); i += 4) {
        const double r = img.pixels[i + 0] / 255.0;
        const double g = img.pixels[i + 1] / 255.0;
        const double b = img.pixels[i + 2] / 255.0;
        const double cr = r * inv[0] + g * inv[1] + b * inv[2];
        const double cg = r * inv[3] + g * inv[4] + b * inv[5];
        const double cb = r * inv[6] + g * inv[7] + b * inv[8];
        img.pixels[i + 0] = ClampU8(cr * 255.0);
        img.pixels[i + 1] = ClampU8(cg * 255.0);
        img.pixels[i + 2] = ClampU8(cb * 255.0);
    }
}
```

Then the public functions (after `MakePattern`), replacing the stubs:
```cpp
struct ColorCodeInfo {
    int logicalStripes = 0;
    int projectedStripes = 0;
    int decodeWindow = 0;
};

inline ColorCodeInfo ColorCodeMetadata(const PatternParams &p) {
    ColorCodeInfo info;
    info.decodeWindow = p.colorN;
    if (p.colorMode == ColorCodeMode::Hamming) {
        info.logicalStripes = static_cast<int>(detail::HammingColorSequence(p.colorN).size());
        info.projectedStripes = info.logicalStripes;
    } else {
        const int len = static_cast<int>(detail::ConstrainedDeBruijn(p.colorK, p.colorN).size());
        info.logicalStripes = len;
        info.projectedStripes = (p.colorMode == ColorCodeMode::SelfEqualizing) ? 2 * len : len;
    }
    return info;
}

inline PatternImage MakeSolidColor(int width, int height, uint8_t r, uint8_t g, uint8_t b) {
    PatternImage img;
    img.width = width;
    img.height = height;
    img.pixels.resize(size_t(width) * height * 4);
    for (size_t i = 0; i < img.pixels.size(); i += 4) {
        img.pixels[i + 0] = r;
        img.pixels[i + 1] = g;
        img.pixels[i + 2] = b;
        img.pixels[i + 3] = 255;
    }
    return img;
}

inline PatternImage MakeColorCoded(const PatternParams &p) {
    std::vector<int> seq;
    std::vector<std::array<int, 3>> palette;
    bool selfEq = false;
    if (p.colorMode == ColorCodeMode::Hamming) {
        seq = detail::HammingColorSequence(p.colorN);
        const auto &pal = detail::HammingPalette();
        palette.assign(pal.begin(), pal.end());
    } else {
        seq = detail::ConstrainedDeBruijn(p.colorK, p.colorN);
        const auto &pal = detail::ColorPalette();
        palette.assign(pal.begin(), pal.begin() + p.colorK);
        selfEq = (p.colorMode == ColorCodeMode::SelfEqualizing);
    }
    std::vector<uint8_t> row = selfEq ? detail::SelfEqualizingRow(seq, palette, p.width)
                                      : detail::StripeRow(seq, palette, p.width);
    PatternImage img = detail::RgbRowToImage(row, p.height);
    if (p.precompensate) detail::PrecompensateInPlace(img, p.responseMatrix);
    return img;
}

inline PatternImage MakeColorPhase(const PatternParams &p) {
    std::vector<uint8_t> r0 = detail::Sinusoidal(0, 3, p.colorPhaseFreq, p.width);
    std::vector<uint8_t> r1 = detail::Sinusoidal(1, 3, p.colorPhaseFreq, p.width);
    std::vector<uint8_t> r2 = detail::Sinusoidal(2, 3, p.colorPhaseFreq, p.width);
    PatternImage img;
    img.width = p.width;
    img.height = p.height;
    img.pixels.resize(size_t(p.width) * p.height * 4);
    for (int y = 0; y < p.height; ++y)
        for (int x = 0; x < p.width; ++x) {
            const size_t d = (size_t(y) * p.width + x) * 4;
            img.pixels[d + 0] = r0[x];
            img.pixels[d + 1] = r1[x];
            img.pixels[d + 2] = r2[x];
            img.pixels[d + 3] = 255;
        }
    if (p.precompensate) detail::PrecompensateInPlace(img, p.responseMatrix);
    return img;
}
```

- [ ] **Step 4: Build and run all pattern tests to verify pass**

Run:
```bash
cmake --build build --target vkspatial_tests 2>&1 | tail -5 && \
./build/test/vkspatial_tests --gtest_filter='PatternGenerators.*' 2>&1 | tail -25
```
Expected: all `PatternGenerators.*` tests PASS (grayscale + color).

- [ ] **Step 5: Commit**

```bash
git add src/Engine/StructuredLight/PatternGenerators.h test/test_patternGenerators.cpp
git commit -m "feat(structured-light): color patterns (De Bruijn/Hamming/self-eq/RGB-phase) + precompensation"
```

---

### Task 3: DLP shaders

**Files:**
- Create: `example2/dlp_depth.vert`, `example2/dlp_scene.vert`, `example2/dlp_scene.frag`

**Interfaces:**
- Produces (consumed by Task 4): three GLSL files whose push-constant block matches `DlpPush` and whose scene set 0 has binding 0 = `projectorDepth` sampler2D, binding 1 = `patternTex` sampler2D.

- [ ] **Step 1: Write `example2/dlp_depth.vert`**

```glsl
#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 viewProj;   // camera; unused in the depth pass
    mat4 projVP;     // projector view-projection
    vec4 projPos;
} pc;

void main() {
    gl_Position = pc.projVP * pc.model * vec4(inPosition, 1.0);
}
```

- [ ] **Step 2: Write `example2/dlp_scene.vert`**

```glsl
#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vWorldNormal;

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 viewProj;
    mat4 projVP;
    vec4 projPos;
} pc;

void main() {
    vec4 world = pc.model * vec4(inPosition, 1.0);
    vWorldPos = world.xyz;
    vWorldNormal = normalize(mat3(pc.model) * inNormal);
    gl_Position = pc.viewProj * world;
}
```

- [ ] **Step 3: Write `example2/dlp_scene.frag`**

```glsl
#version 450

layout(set = 0, binding = 0) uniform sampler2D projectorDepth;
layout(set = 0, binding = 1) uniform sampler2D patternTex;

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vWorldNormal;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 viewProj;
    mat4 projVP;
    vec4 projPos;
} pc;

const float kAmbientFloor = 0.08; // matches Blender's max(0.08, ...) Lambert floor
const float kShadowBias   = 0.0015;
const float kFill         = 0.03; // small constant fill so unlit areas aren't pure black

void main() {
    vec3 N = normalize(vWorldNormal);
    vec3 pattern = vec3(0.0);

    vec4 clip = pc.projVP * vec4(vWorldPos, 1.0);
    if (clip.w > 0.0) {
        vec3 ndc = clip.xyz / clip.w;              // Vulkan: ndc.z in [0,1]
        vec2 uv = ndc.xy * 0.5 + 0.5;
        if (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0) {
            float stored = texture(projectorDepth, uv).r;
            if (ndc.z - kShadowBias <= stored) {   // not occluded from the projector
                pattern = texture(patternTex, uv).rgb;
            }
        }
    }

    vec3 toProj = normalize(pc.projPos.xyz - vWorldPos);
    float lambert = max(kAmbientFloor, dot(N, toProj));
    outColor = vec4(pattern * lambert + vec3(kFill), 1.0);
}
```

- [ ] **Step 4: Verify all three compile with glslc**

Run:
```bash
GLSLC="${VULKAN_SDK:-/usr/local}/bin/glslc"
"$GLSLC" -fshader-stage=vertex   example2/dlp_depth.vert -o /tmp/dlp_depth.vert.spv && \
"$GLSLC" -fshader-stage=vertex   example2/dlp_scene.vert -o /tmp/dlp_scene.vert.spv && \
"$GLSLC" -fshader-stage=fragment example2/dlp_scene.frag -o /tmp/dlp_scene.frag.spv && \
echo "ALL SHADERS COMPILED"
```
Expected: prints `ALL SHADERS COMPILED`, no glslc errors.

- [ ] **Step 5: Commit**

```bash
git add example2/dlp_depth.vert example2/dlp_scene.vert example2/dlp_scene.frag
git commit -m "feat(dlp): projector depth + projective-pattern scene shaders"
```

---

### Task 4: DlpProjectorPass + minimal app (built-in sphere)

**Files:**
- Create: `example2/DlpProjectorPass.h`, `example2/DlpProjectorPass.cpp`, `example2/dlp_simulator.cpp`
- Modify: `example2/CMakeLists.txt`

**Interfaces:**
- Consumes: `Engine::StructuredLight::{PatternParams, PatternImage, MakePattern, PatternCount}` (Tasks 1–2); DLP shaders (Task 3); `DlpPush` (Global Constraints).
- Produces (consumed by Tasks 5–6):
  - `DlpProjectorPass(Engine::Core::Context&, VkFormat colorFormat, const std::string& shaderDir, std::vector<Engine::Render::Object<>> objects, int patternWidth=1280, int patternHeight=720)`
  - `void SetPattern(const Engine::StructuredLight::PatternParams&, int step);`
  - `void SetProjector(float fovYRadians, float baselineRadians);`
  - `void SetModel(std::size_t objectIndex, const vkMath::Mat4&);`
  - free function in `dlp_simulator.cpp`: `Engine::Render::Object<> MakeSphere(int stacks, int slices, float radius, vkMath::Vec3 color);`

- [ ] **Step 1: Write `example2/DlpProjectorPass.h`**

```cpp
#pragma once

#include "Engine/Core/Descriptor.h"
#include "Engine/Core/Image.h"
#include "Engine/Core/Sampler.h"
#include "Engine/Render/GraphicsPipeline.h"
#include "Engine/Render/Object.h"
#include "Engine/Render/RenderGraph.h"

#include "Engine/StructuredLight/PatternGenerators.h"
#include "utilities/Math.h"

#include <cstdint>
#include <string>
#include <vector>

class DlpProjectorPass final : public Engine::Render::RenderPass {
public:
    DlpProjectorPass(Engine::Core::Context &context,
                     VkFormat colorFormat,
                     const std::string &shaderDir,
                     std::vector<Engine::Render::Object<>> objects,
                     int patternWidth = 1280,
                     int patternHeight = 720);
    ~DlpProjectorPass() override;

    DlpProjectorPass(const DlpProjectorPass &) = delete;
    DlpProjectorPass &operator=(const DlpProjectorPass &) = delete;

    const char *Name() const override { return "DlpProjectorPass"; }
    void Execute(Engine::Render::RenderContext &ctx) override;

    // Regenerate the pattern texture next frame (CPU generate + GPU re-upload).
    void SetPattern(const Engine::StructuredLight::PatternParams &params, int step);
    void SetProjector(float fovYRadians, float baselineRadians);
    void SetModel(std::size_t objectIndex, const vkMath::Mat4 &model);

private:
    struct Push {
        vkMath::Mat4 model = vkMath::Mat4::Identity();
        vkMath::Mat4 viewProj = vkMath::Mat4::Identity();
        vkMath::Mat4 projVP = vkMath::Mat4::Identity();
        float projPos[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    };

    Engine::Core::Context &m_context;
    std::vector<Engine::Render::Object<>> m_objects;

    int m_patternWidth;
    int m_patternHeight;
    Engine::StructuredLight::PatternParams m_params;
    int m_step = 0;
    bool m_patternDirty = true;
    bool m_patternUploadedOnce = false;

    float m_fovY = 45.0f * 3.14159265358979323846f / 180.0f;
    float m_baseline = 15.0f * 3.14159265358979323846f / 180.0f;

    Engine::Core::Image m_projectorDepth;
    Engine::Core::Sampler m_depthSampler;
    Engine::Core::Image m_patternImage;
    Engine::Core::Sampler m_patternSampler;

    Engine::Core::DescriptorSetLayout m_sceneSetLayout;
    Engine::Core::DescriptorPool m_descriptorPool;
    Engine::Core::DescriptorSet m_sceneSet;

    Engine::Render::GraphicsPipeline m_depthPipeline;
    Engine::Render::GraphicsPipeline m_scenePipeline;

    void CreateResources();
    void CreatePipelines(VkFormat colorFormat, const std::string &shaderDir);
    void UploadPattern();
    void RecordDepthPass(VkCommandBuffer cmd, const vkMath::Mat4 &projVP);
    void RecordScenePass(Engine::Render::RenderContext &ctx,
                         const vkMath::Mat4 &viewProj,
                         const vkMath::Mat4 &projVP,
                         const vkMath::Vec3 &projPos);
};
```

- [ ] **Step 2: Write `example2/DlpProjectorPass.cpp`**

```cpp
#include "DlpProjectorPass.h"

#include "Engine/Core/OneShotCommands.h"
#include "Engine/Render/Rendering.h"
#include "Engine/Render/SwapChain.h"
#include "Engine/Render/View.h"
#include "Engine/Render/Camera.h"
#include "Engine/Compute/StagingBuffer.h"

#include <cstring>
#include <cmath>
#include <stdexcept>

namespace {
    constexpr uint32_t kProjDepthSize = 1024;
}

DlpProjectorPass::DlpProjectorPass(Engine::Core::Context &context,
                                   VkFormat colorFormat,
                                   const std::string &shaderDir,
                                   std::vector<Engine::Render::Object<>> objects,
                                   int patternWidth,
                                   int patternHeight)
    : m_context(context),
      m_objects(std::move(objects)),
      m_patternWidth(patternWidth),
      m_patternHeight(patternHeight),
      m_projectorDepth(context),
      m_depthSampler(context),
      m_patternImage(context),
      m_patternSampler(context),
      m_sceneSetLayout(context),
      m_descriptorPool(context),
      m_depthPipeline(context),
      m_scenePipeline(context) {
    m_params.width = patternWidth;
    m_params.height = patternHeight;
    for (Engine::Render::Object<> &object : m_objects)
        object.Upload(m_context);
    CreateResources();
    CreatePipelines(colorFormat, shaderDir);
    UploadPattern(); // initial pattern → pattern image in SHADER_READ_ONLY before first frame
    m_patternDirty = false;
}

DlpProjectorPass::~DlpProjectorPass() {
    m_scenePipeline.Destroy();
    m_depthPipeline.Destroy();
    m_sceneSet.Reset();
    m_descriptorPool.Destroy();
    m_sceneSetLayout.Destroy();
    m_patternSampler.Destroy();
    m_patternImage.Destroy();
    m_depthSampler.Destroy();
    m_projectorDepth.Destroy();
}

void DlpProjectorPass::SetPattern(const Engine::StructuredLight::PatternParams &params, int step) {
    m_params = params;
    m_params.width = m_patternWidth;   // texture size is fixed at construction
    m_params.height = m_patternHeight;
    const int count = Engine::StructuredLight::PatternCount(m_params);
    m_step = count > 0 ? ((step % count) + count) % count : 0;
    m_patternDirty = true;
}

void DlpProjectorPass::SetProjector(float fovYRadians, float baselineRadians) {
    m_fovY = fovYRadians;
    m_baseline = baselineRadians;
}

void DlpProjectorPass::SetModel(std::size_t objectIndex, const vkMath::Mat4 &model) {
    if (objectIndex < m_objects.size())
        m_objects[objectIndex].SetModel(model);
}

void DlpProjectorPass::CreateResources() {
    // Projector depth target (perspective depth from the projector's POV).
    Engine::Core::ImageDescriptor depthDesc =
            Engine::Core::ImageDescriptor::Depth2D({kProjDepthSize, kProjDepthSize},
                                                   VK_FORMAT_D32_SFLOAT);
    depthDesc.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    m_projectorDepth.Create(depthDesc);
    m_depthSampler.Create(Engine::Core::SamplerDescriptor::ShadowMapManualPCF());

    // Pattern texture (RGBA8, sampled + transfer dst).
    Engine::Core::ImageDescriptor patternDesc = Engine::Core::ImageDescriptor::Color2D(
            {static_cast<uint32_t>(m_patternWidth), static_cast<uint32_t>(m_patternHeight)},
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    m_patternImage.Create(patternDesc);

    Engine::Core::SamplerDescriptor patternSampler;
    patternSampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    patternSampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    patternSampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    patternSampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    m_patternSampler.Create(patternSampler);

    // Scene descriptor set: binding 0 = projector depth, binding 1 = pattern.
    m_sceneSetLayout.Create({
            Engine::Core::DescriptorBinding::CombinedImageSampler(0, VK_SHADER_STAGE_FRAGMENT_BIT),
            Engine::Core::DescriptorBinding::CombinedImageSampler(1, VK_SHADER_STAGE_FRAGMENT_BIT),
    });
    m_descriptorPool.Create({{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2}}, 1);
    m_sceneSet = m_descriptorPool.Allocate(m_sceneSetLayout.Handle());
    m_sceneSet.UpdateCombinedImageSampler(0, m_depthSampler.Handle(), m_projectorDepth.View(),
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_sceneSet.UpdateCombinedImageSampler(1, m_patternSampler.Handle(), m_patternImage.View(),
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void DlpProjectorPass::CreatePipelines(VkFormat colorFormat, const std::string &shaderDir) {
    using RenderVertex = Engine::Render::Vertex;

    Engine::Render::GraphicsPipelineDescriptor depthDesc;
    depthDesc.VertexShader(shaderDir + "/dlp_depth.vert.spv")
            .VertexBinding<RenderVertex>()
            .VertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(RenderVertex, position))
            .VertexAttribute(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(RenderVertex, normal))
            .VertexAttribute(2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(RenderVertex, color))
            .DepthTarget(VK_FORMAT_D32_SFLOAT)
            .DepthBias(1.2f, 1.8f)
            .PushConstant<Push>(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    m_depthPipeline.Build(depthDesc);

    Engine::Render::GraphicsPipelineDescriptor sceneDesc;
    sceneDesc.descriptorSetLayouts.push_back(m_sceneSetLayout.Handle());
    sceneDesc.VertexShader(shaderDir + "/dlp_scene.vert.spv")
            .FragmentShader(shaderDir + "/dlp_scene.frag.spv")
            .VertexBinding<RenderVertex>()
            .VertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(RenderVertex, position))
            .VertexAttribute(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(RenderVertex, normal))
            .VertexAttribute(2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(RenderVertex, color))
            .ColorTarget(colorFormat)
            .DepthTarget(VK_FORMAT_D32_SFLOAT)
            .PushConstant<Push>(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    m_scenePipeline.Build(sceneDesc);
}

void DlpProjectorPass::UploadPattern() {
    const Engine::StructuredLight::PatternImage img =
            Engine::StructuredLight::MakePattern(m_params, m_step);
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(img.pixels.size());

    Engine::Compute::StagingBuffer staging(m_context, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    std::memcpy(staging.Mapped(), img.pixels.data(), static_cast<size_t>(bytes));

    const bool first = !m_patternUploadedOnce;
    Engine::Core::SubmitOneShot(
            m_context, Engine::Core::QueueRole::Graphics, [&](VkCommandBuffer cmd) {
                m_patternImage.TransitionLayout(
                        cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        first ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                              : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        first ? 0 : VK_ACCESS_SHADER_READ_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT);

                VkBufferImageCopy region{};
                region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                region.imageSubresource.layerCount = 1;
                region.imageExtent = {static_cast<uint32_t>(m_patternWidth),
                                      static_cast<uint32_t>(m_patternHeight), 1};
                vkCmdCopyBufferToImage(cmd, staging.Handle(), m_patternImage.Handle(),
                                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

                m_patternImage.TransitionLayout(
                        cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
            });
    m_patternUploadedOnce = true;
}

void DlpProjectorPass::Execute(Engine::Render::RenderContext &ctx) {
    if (!ctx.swapChain || !ctx.depthImage || !ctx.view || !ctx.view->GetCamera())
        throw std::runtime_error("DlpProjectorPass: requires swapchain, depth, view and camera");

    if (m_patternDirty) {
        UploadPattern();
        m_patternDirty = false;
    }

    const float d = 3.0f;
    const vkMath::Vec3 projPos(d * std::sin(m_baseline), 0.6f, d * std::cos(m_baseline));
    const vkMath::Mat4 projView = vkMath::LookAt(projPos, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
    const float aspect = static_cast<float>(m_patternWidth) / static_cast<float>(m_patternHeight);
    const vkMath::Mat4 projProj = vkMath::Perspective(m_fovY, aspect, 0.1f, 20.0f);
    const vkMath::Mat4 projVP = projProj * projView;

    const Engine::Render::Camera *camera = ctx.view->GetCamera();
    const vkMath::Mat4 viewProj = camera->GetProjectionMatrix() * camera->GetViewMatrix();

    RecordDepthPass(ctx.commandBuffer, projVP);
    RecordScenePass(ctx, viewProj, projVP, projPos);
}

void DlpProjectorPass::RecordDepthPass(VkCommandBuffer cmd, const vkMath::Mat4 &projVP) {
    const bool firstUse = m_projectorDepth.CurrentLayout() == VK_IMAGE_LAYOUT_UNDEFINED;
    m_projectorDepth.TransitionLayout(
            cmd, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            firstUse ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            firstUse ? 0 : VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

    Engine::Render::RenderingDescriptor descriptor({kProjDepthSize, kProjDepthSize});
    descriptor.SetDepthAttachment(
            Engine::Render::DepthAttachment(m_projectorDepth.View()).Clear(1.0f).Store(true).Build());
    Engine::Render::RenderingScope rendering(cmd, descriptor);

    m_depthPipeline.Bind(cmd);
    for (const Engine::Render::Object<> &object : m_objects) {
        Push pc{};
        pc.model = object.Model();
        pc.projVP = projVP;
        m_depthPipeline.PushConstants(
                cmd, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, pc);
        object.Render(cmd);
    }
    rendering.End();

    m_projectorDepth.TransitionLayout(
            cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
}

void DlpProjectorPass::RecordScenePass(Engine::Render::RenderContext &ctx,
                                       const vkMath::Mat4 &viewProj,
                                       const vkMath::Mat4 &projVP,
                                       const vkMath::Vec3 &projPos) {
    Engine::Render::ClearOptions clear{};
    clear.color[0] = 0.02f;
    clear.color[1] = 0.02f;
    clear.color[2] = 0.03f;
    clear.color[3] = 1.0f;

    Engine::Render::RenderingDescriptor descriptor = Engine::Render::RenderingDescriptor::ColorDepth(
            ctx.swapChain->Extent(), ctx.swapChain->ImageView(ctx.imageIndex),
            ctx.depthImage->View(), clear);
    Engine::Render::RenderingScope rendering(ctx.commandBuffer, descriptor);

    m_scenePipeline.Bind(ctx.commandBuffer);
    m_sceneSet.Bind(ctx.commandBuffer, m_scenePipeline.Layout());

    for (const Engine::Render::Object<> &object : m_objects) {
        Push pc{};
        pc.model = object.Model();
        pc.viewProj = viewProj;
        pc.projVP = projVP;
        pc.projPos[0] = projPos.x();
        pc.projPos[1] = projPos.y();
        pc.projPos[2] = projPos.z();
        m_scenePipeline.PushConstants(
                ctx.commandBuffer, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, pc);
        object.Render(ctx.commandBuffer);
    }
}
```

- [ ] **Step 3: Write `example2/dlp_simulator.cpp` (minimal: built-in sphere + floor, fixed camera, auto-rotating mesh)**

```cpp
#include "DlpProjectorPass.h"

#include "Engine/Render/Application.h"
#include "Engine/Render/Camera.h"
#include "Engine/Render/RenderGraph.h"
#include "Engine/Render/Scene.h"

#include "Engine/StructuredLight/PatternGenerators.h"
#include "utilities/Math.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

namespace {
    constexpr float kPi = 3.14159265358979323846f;
    using RenderObject = Engine::Render::Object<>;
    using RenderVertex = Engine::Render::Vertex;

    RenderObject MakeSphere(int stacks, int slices, float radius, vkMath::Vec3 color) {
        RenderObject sphere;
        std::vector<RenderVertex> verts;
        for (int i = 0; i <= stacks; ++i) {
            const float v = static_cast<float>(i) / stacks;
            const float phi = v * kPi;
            for (int j = 0; j <= slices; ++j) {
                const float u = static_cast<float>(j) / slices;
                const float theta = u * 2.0f * kPi;
                const vkMath::Vec3 n(std::sin(phi) * std::cos(theta),
                                     std::cos(phi),
                                     std::sin(phi) * std::sin(theta));
                verts.emplace_back(vkMath::Vec3(n * radius), n, color);
            }
        }
        std::vector<RenderObject::Index> indices;
        const int rowLen = slices + 1;
        for (int i = 0; i < stacks; ++i)
            for (int j = 0; j < slices; ++j) {
                const uint32_t a = i * rowLen + j;
                const uint32_t b = a + rowLen;
                indices.insert(indices.end(), {a, b, a + 1, a + 1, b, b + 1});
            }
        sphere.SetGeometry(std::move(verts), std::move(indices));
        return sphere;
    }

    RenderObject MakeFloor(float y, float halfSize, vkMath::Vec3 color) {
        RenderObject plane;
        const vkMath::Vec3 n{0.0f, 1.0f, 0.0f};
        const uint32_t v0 = plane.AddVertex(RenderVertex({-halfSize, y, -halfSize}, n, color));
        const uint32_t v1 = plane.AddVertex(RenderVertex({halfSize, y, -halfSize}, n, color));
        const uint32_t v2 = plane.AddVertex(RenderVertex({halfSize, y, halfSize}, n, color));
        const uint32_t v3 = plane.AddVertex(RenderVertex({-halfSize, y, halfSize}, n, color));
        plane.AddQuad(v0, v1, v2, v3);
        return plane;
    }
}

int main() {
    try {
        Engine::Render::ApplicationDescriptor descriptor;
        descriptor.window = {1280, 800, "DLP Structured-Light Simulator"};
        Engine::Render::Application app(descriptor);

        std::vector<RenderObject> objects;
        objects.push_back(MakeSphere(48, 64, 1.0f, {1.0f, 1.0f, 1.0f}));
        objects.push_back(MakeFloor(-1.1f, 4.0f, {1.0f, 1.0f, 1.0f}));

        const std::string shaderDir = DLP_SHADER_DIR;
        auto passOwned = std::make_unique<DlpProjectorPass>(
                app.GetContext(), app.GetSwapChain().Format(), shaderDir, std::move(objects));
        DlpProjectorPass *pass = passOwned.get();

        Engine::StructuredLight::PatternParams params;
        params.type = Engine::StructuredLight::PatternType::PhaseShift;
        params.freq = 16.0f;
        params.steps = 3;
        pass->SetPattern(params, 0);
        pass->SetProjector(45.0f * kPi / 180.0f, 15.0f * kPi / 180.0f);

        Engine::Render::Scene scene;
        Engine::Render::Camera camera;
        const VkExtent2D extent = app.GetSwapChain().Extent();
        const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
        camera.SetPerspective(55.0f * kPi / 180.0f, aspect, 0.1f, 80.0f);
        camera.LookAt({0.0f, 1.5f, 4.5f}, {0.0f, 0.0f, 0.0f});

        Engine::Render::RenderGraph graph;
        graph.AddPass(std::move(passOwned));

        app.GetView().SetScene(&scene);
        app.GetView().SetCamera(&camera);
        app.GetView().SetRenderGraph(&graph);

        // Auto-rotate the mesh so parallax is visible without input (removed in Task 6).
        Engine::Render::KeyListenerGroup keys(app.GetWindow().Keys());
        keys.Add(Engine::Render::KeyEventType::Press, [&](Engine::Render::KeyEvent &e) {
            if (e.keyCode == Engine::Render::KeyCode::Escape)
                app.GetWindow().RequestClose();
        });

        // Application::Run drives the loop; use a frame counter via a mouse-free tick.
        // Simplest: rotate based on wall-clock inside a per-frame callback is not exposed,
        // so rotate from a static counter updated each Execute is not available here.
        // Instead, set an initial tilt so the fringe is clearly visible.
        pass->SetModel(0, vkMath::RotationY(0.4f));

        app.Run();
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}
```

- [ ] **Step 4: Wire the target into `example2/CMakeLists.txt`**

Inside the `if (GLSLC_EXECUTABLE)` block (next to the `shadow_map` target), add:
```cmake
    add_executable(dlp_simulator dlp_simulator.cpp DlpProjectorPass.cpp)
    target_link_libraries(dlp_simulator PRIVATE Engine::Render Engine::Core Engine::Compute)
    add_compiled_shaders(dlp_simulator DLP_SHADER_DIR
            dlp_depth.vert dlp_scene.vert dlp_scene.frag)
```

- [ ] **Step 5: Configure, build, and run**

Run:
```bash
cmake -S . -B build >/dev/null && cmake --build build --target dlp_simulator 2>&1 | tail -20
```
Expected: builds clean. Then run and observe (visual acceptance):
```bash
./build/example2/dlp_simulator
```
Expected: a window shows a white sphere on a floor with sinusoidal fringes projected across it; the fringe bends over the sphere's curvature; the sphere casts a dark (pattern-free) shadow onto the floor where it occludes the projector. `Esc` quits.

- [ ] **Step 6: Commit**

```bash
git add example2/DlpProjectorPass.h example2/DlpProjectorPass.cpp example2/dlp_simulator.cpp example2/CMakeLists.txt
git commit -m "feat(dlp): DlpProjectorPass + minimal viewer (sphere, projected fringe, projector shadow)"
```

---

### Task 5: Mesh file loading (PLY / OBJ) + centering + CLI

**Files:**
- Modify: `example2/dlp_simulator.cpp`

**Interfaces:**
- Consumes: `util::TriMesh`, `util::LoadPlyMesh`, `util::ComputeVertexNormals` from `utilities/PlyMesh.h`; `DlpProjectorPass` from Task 4.
- Produces: `dlp_simulator <mesh.ply|.obj>` loads and centers a mesh; no arg → built-in sphere.

- [ ] **Step 1: Add mesh-loading helpers and CLI to `dlp_simulator.cpp`**

Add includes near the top:
```cpp
#include "utilities/PlyMesh.h"

#include <array>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
```
Add these helpers into the anonymous namespace (alongside `MakeSphere`):
```cpp
    bool EndsWithLower(const std::string &s, const std::string &suffix) {
        if (s.size() < suffix.size()) return false;
        for (size_t i = 0; i < suffix.size(); ++i)
            if (std::tolower(s[s.size() - suffix.size() + i]) != suffix[i]) return false;
        return true;
    }

    // Minimal OBJ triangle-mesh loader (v + f; triangulates polygons as a fan).
    bool LoadObjMesh(const std::string &path, util::TriMesh &mesh) {
        std::ifstream in(path);
        if (!in) return false;
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream ss(line);
            std::string tag;
            ss >> tag;
            if (tag == "v") {
                Eigen::Vector3f v;
                ss >> v.x() >> v.y() >> v.z();
                mesh.vertices.push_back(v);
            } else if (tag == "f") {
                std::vector<int> idx;
                std::string tok;
                while (ss >> tok) {
                    const int slash = static_cast<int>(tok.find('/'));
                    const int vi = std::stoi(slash < 0 ? tok : tok.substr(0, slash));
                    idx.push_back(vi > 0 ? vi - 1 : static_cast<int>(mesh.vertices.size()) + vi);
                }
                for (size_t i = 2; i < idx.size(); ++i)
                    mesh.faces.emplace_back(idx[0], idx[i - 1], idx[i]);
            }
        }
        return !mesh.vertices.empty() && !mesh.faces.empty();
    }

    // Center the mesh's bbox at the origin and scale the longest axis to ~2 (fits [-1,1]).
    void CenterAndScale(util::TriMesh &mesh) {
        Eigen::Vector3f lo = mesh.vertices.front(), hi = mesh.vertices.front();
        for (const Eigen::Vector3f &v : mesh.vertices) {
            lo = lo.cwiseMin(v);
            hi = hi.cwiseMax(v);
        }
        const Eigen::Vector3f center = 0.5f * (lo + hi);
        const float extent = (hi - lo).maxCoeff();
        const float scale = extent > 1e-6f ? 2.0f / extent : 1.0f;
        for (Eigen::Vector3f &v : mesh.vertices) v = (v - center) * scale;
    }

    RenderObject MeshToObject(const util::TriMesh &mesh, vkMath::Vec3 color) {
        const std::vector<Eigen::Vector3f> normals = util::ComputeVertexNormals(mesh);
        RenderObject object;
        std::vector<RenderVertex> verts;
        verts.reserve(mesh.vertices.size());
        for (size_t i = 0; i < mesh.vertices.size(); ++i)
            verts.emplace_back(vkMath::Vec3(mesh.vertices[i]), vkMath::Vec3(normals[i]), color);
        std::vector<RenderObject::Index> indices;
        indices.reserve(mesh.faces.size() * 3);
        for (const Eigen::Vector3i &f : mesh.faces) {
            indices.push_back(static_cast<uint32_t>(f[0]));
            indices.push_back(static_cast<uint32_t>(f[1]));
            indices.push_back(static_cast<uint32_t>(f[2]));
        }
        object.SetGeometry(std::move(verts), std::move(indices));
        return object;
    }

    RenderObject LoadMeshObject(const std::string &path) {
        util::TriMesh mesh;
        if (EndsWithLower(path, ".obj")) {
            if (!LoadObjMesh(path, mesh)) throw std::runtime_error("cannot load OBJ: " + path);
        } else {
            util::LoadPlyMesh(path, mesh); // throws on failure
        }
        if (mesh.vertices.empty() || mesh.faces.empty())
            throw std::runtime_error("mesh has no geometry: " + path);
        CenterAndScale(mesh);
        return MeshToObject(mesh, {1.0f, 1.0f, 1.0f});
    }
```

- [ ] **Step 2: Use the CLI argument in `main`**

Change `int main()` to `int main(int argc, char **argv)` and replace the sphere-construction lines:
```cpp
        std::vector<RenderObject> objects;
        if (argc > 1)
            objects.push_back(LoadMeshObject(argv[1]));
        else
            objects.push_back(MakeSphere(48, 64, 1.0f, {1.0f, 1.0f, 1.0f}));
        objects.push_back(MakeFloor(-1.1f, 4.0f, {1.0f, 1.0f, 1.0f}));
```

- [ ] **Step 3: Build**

Run:
```bash
cmake --build build --target dlp_simulator 2>&1 | tail -15
```
Expected: builds clean.

- [ ] **Step 4: Run with a mesh and with no arg (visual acceptance)**

Run (a chair/ground-truth PLY exists under `scanData/`; any triangle-mesh PLY/OBJ works):
```bash
./build/example2/dlp_simulator scanData/ground_truth.ply    # or any .ply/.obj mesh
./build/example2/dlp_simulator                                # no arg → built-in sphere
```
Expected: the loaded mesh appears centered and unit-scaled with the fringe projected onto it and a shadow on the floor; the no-arg run still shows the sphere.

- [ ] **Step 5: Commit**

```bash
git add example2/dlp_simulator.cpp
git commit -m "feat(dlp): load and center PLY/OBJ meshes via CLI arg"
```

---

### Task 6: Interactive controls — ImGui panel + keys + trackball

**Files:**
- Modify: `example2/dlp_simulator.cpp`, `example2/CMakeLists.txt`

**Interfaces:**
- Consumes: `DlpProjectorPass::{SetPattern,SetProjector}` (Task 4); `Engine::StructuredLight::{PatternType,ColorCodeMode,PatternCount,ColorCodeMetadata}` (Tasks 1–2); `ImGuiPass` (`example2/ImGuiPass.h`); `Camera` trackball API.
- Produces: the full interactive viewer (final Step-1 deliverable).

- [ ] **Step 1: Add ImGui to the CMake target**

In `example2/CMakeLists.txt`, update the `dlp_simulator` target to include `ImGuiPass.cpp` and link `imgui`:
```cmake
    add_executable(dlp_simulator dlp_simulator.cpp DlpProjectorPass.cpp ImGuiPass.cpp)
    target_link_libraries(dlp_simulator PRIVATE Engine::Render Engine::Core Engine::Compute imgui)
    add_compiled_shaders(dlp_simulator DLP_SHADER_DIR
            dlp_depth.vert dlp_scene.vert dlp_scene.frag)
```

- [ ] **Step 2: Add trackball + keyboard controls in `main`**

Add includes:
```cpp
#include "ImGuiPass.h"
#include "Engine/Render/GlfwWindow.h"

#include <algorithm>
```
Replace the `pass->SetModel(0, vkMath::RotationY(0.4f));` line and the minimal key handler with trackball + step/type keys. Add this after `graph.AddPass(...)` and the `View` wiring, and BEFORE `app.Run()`:
```cpp
        // Shared UI state driving both the ImGui panel and the key handlers.
        namespace SL = Engine::StructuredLight;
        struct UiState {
            SL::PatternParams params;
            int step = 0;
            float fovDeg = 45.0f;
            float baselineDeg = 15.0f;
        } ui;
        ui.params = params;

        auto applyPattern = [&]() {
            const int count = SL::PatternCount(ui.params);
            ui.step = count > 0 ? ((ui.step % count) + count) % count : 0;
            pass->SetPattern(ui.params, ui.step);
        };
        auto applyProjector = [&]() {
            pass->SetProjector(ui.fovDeg * kPi / 180.0f, ui.baselineDeg * kPi / 180.0f);
        };
        applyPattern();
        applyProjector();

        // Trackball (mirrors ShadowMap.cpp).
        Engine::Render::MouseListenerGroup trackball(app.GetWindow().Mouse());
        trackball.Add(Engine::Render::MouseEventType::Drag, [&](Engine::Render::MouseEvent &e) {
            if (e.button != Engine::Render::MouseButton::Left) return;
            const VkExtent2D size = app.GetWindow().FramebufferSize();
            if (size.width == 0 || size.height == 0) return;
            if (!camera.IsTrackballDragging()) {
                camera.BeginTrackballDrag(e.x, e.y, int(size.width), int(size.height));
                return;
            }
            camera.DragTrackball(e.x, e.y, int(size.width), int(size.height));
            e.handled = true;
        });
        trackball.Add(Engine::Render::MouseEventType::ButtonUp, [&](Engine::Render::MouseEvent &e) {
            if (e.button == Engine::Render::MouseButton::Left) camera.EndTrackballDrag();
        });
        trackball.Add(Engine::Render::MouseEventType::Scroll, [&](Engine::Render::MouseEvent &e) {
            camera.SetDistance(std::clamp(
                    camera.GetDistance() * std::exp(float(-e.scrollY) * 0.08f), 1.5f, 12.0f));
            e.handled = true;
        });

        Engine::Render::KeyListenerGroup keys(app.GetWindow().Keys());
        keys.Add(Engine::Render::KeyEventType::Press, [&](Engine::Render::KeyEvent &e) {
            using K = Engine::Render::KeyCode;
            if (e.keyCode == K::Escape) { app.GetWindow().RequestClose(); return; }
            if (e.keyCode == K::Right) { ui.step += 1; applyPattern(); }        // next step
            else if (e.keyCode == K::Left) { ui.step -= 1; applyPattern(); }     // prev step
            else if (e.keyCode == K::P) {                                        // cycle type
                const int next = (static_cast<int>(ui.params.type) + 1) % 5;
                ui.params.type = static_cast<SL::PatternType>(next);
                ui.step = 0;
                applyPattern();
            }
        });
```
Key mapping uses only confirmed `KeyCode` enumerators (`src/Engine/Render/KeyInput.h`): `Escape`, `Left`, `Right`, `P` — there are no bracket keys in the enum, so step navigation is bound to the Left/Right arrows.

- [ ] **Step 3: Add the ImGui panel pass (LAST in the graph)**

Immediately after adding `DlpProjectorPass` to the graph, add the ImGui pass and its UI callback. Requires the GLFW window handle:
```cpp
        auto &glfwWindow = static_cast<Engine::Render::GlfwWindow &>(app.GetWindow());
        auto imguiOwned = std::make_unique<ImGuiPass>(
                app.GetContext(), glfwWindow.Handle(),
                app.GetSwapChain().Format(), app.GetSwapChain().ImageCount());
        ImGuiPass *imgui = imguiOwned.get();
        graph.AddPass(std::move(imguiOwned)); // MUST be added after DlpProjectorPass

        imgui->SetUi([&]() {
            ImGui::Begin("DLP Simulator");

            const char *types[] = {"Phase Shift", "Gray Code", "Binary",
                                   "Color Stripes", "RGB Phase"};
            int typeIndex = static_cast<int>(ui.params.type);
            if (ImGui::Combo("Pattern", &typeIndex, types, 5)) {
                ui.params.type = static_cast<SL::PatternType>(typeIndex);
                ui.step = 0;
                applyPattern();
            }

            bool dirty = false;
            switch (ui.params.type) {
                case SL::PatternType::PhaseShift:
                    dirty |= ImGui::SliderInt("Steps (N)", &ui.params.steps, 3, 16);
                    dirty |= ImGui::SliderFloat("Freq", &ui.params.freq, 1.0f, 64.0f);
                    break;
                case SL::PatternType::GrayCode:
                    dirty |= ImGui::SliderInt("Bits", &ui.params.bits, 1, 8);
                    break;
                case SL::PatternType::Binary:
                    dirty |= ImGui::SliderInt("Patterns", &ui.params.binaryPatterns, 1, 10);
                    break;
                case SL::PatternType::ColorCoded: {
                    const char *modes[] = {"De Bruijn", "Hamming", "Self-equalizing"};
                    int modeIndex = static_cast<int>(ui.params.colorMode);
                    if (ImGui::Combo("Color mode", &modeIndex, modes, 3)) {
                        ui.params.colorMode = static_cast<SL::ColorCodeMode>(modeIndex);
                        dirty = true;
                    }
                    if (ui.params.colorMode != SL::ColorCodeMode::Hamming)
                        dirty |= ImGui::SliderInt("Colors (k)", &ui.params.colorK, 3, 6);
                    dirty |= ImGui::SliderInt("Window (n)", &ui.params.colorN, 3, 6);
                    dirty |= ImGui::Checkbox("Precompensate", &ui.params.precompensate);
                    const SL::ColorCodeInfo info = SL::ColorCodeMetadata(ui.params);
                    ImGui::Text("logical %d / projected %d", info.logicalStripes,
                                info.projectedStripes);
                    break;
                }
                case SL::PatternType::ColorPhase:
                    dirty |= ImGui::SliderFloat("RGB Phase Freq", &ui.params.colorPhaseFreq,
                                                1.0f, 256.0f);
                    dirty |= ImGui::Checkbox("Precompensate", &ui.params.precompensate);
                    break;
            }
            if (dirty) applyPattern();

            const int count = SL::PatternCount(ui.params);
            ImGui::Separator();
            if (ImGui::Button("< Prev")) { ui.step -= 1; applyPattern(); }
            ImGui::SameLine();
            ImGui::Text("Step %d / %d", ui.step + 1, count);
            ImGui::SameLine();
            if (ImGui::Button("Next >")) { ui.step += 1; applyPattern(); }

            ImGui::Separator();
            bool projDirty = false;
            projDirty |= ImGui::SliderFloat("Projector FOV", &ui.fovDeg, 10.0f, 120.0f);
            projDirty |= ImGui::SliderFloat("Baseline angle", &ui.baselineDeg, -60.0f, 60.0f);
            if (projDirty) applyProjector();

            ImGui::End();
        });
```
Remove the earlier minimal `KeyListenerGroup keys(...)` block from Task 4's `main` (the one that only handled `Escape`) — the Step-2 handler above replaces it. Keep only one `keys` group.

- [ ] **Step 4: Build**

Run:
```bash
cmake --build build --target dlp_simulator 2>&1 | tail -20
```
Expected: builds clean. If the ImGui pass or `GlfwWindow`/`ImageCount` signatures differ, cross-check against `example2/object_scan_viewer.cpp` (it wires `ImGuiPass` the same way) and adjust.

- [ ] **Step 5: Run and exercise every control (visual acceptance)**

Run:
```bash
./build/example2/dlp_simulator                              # sphere
./build/example2/dlp_simulator scanData/ground_truth.ply    # a real mesh
```
Verify: mouse-drag orbits; scroll zooms; the ImGui panel switches pattern type; per-type sliders change the fringe live; `< Prev`/`Next >` and the Left/Right arrow keys step through phase/gray/binary; `P` cycles types; color-stripe modes render colored stripes and show logical/projected counts; changing projector FOV/baseline visibly changes the fringe parallax; occluded regions stay dark on the floor.

- [ ] **Step 6: Commit**

```bash
git add example2/dlp_simulator.cpp example2/CMakeLists.txt
git commit -m "feat(dlp): interactive controls — ImGui panel, trackball, step/type keys"
```

---

## Self-Review

**Spec coverage:**
- Pattern library (all types + precompensation + metadata) → Tasks 1–2. ✓
- Projective-texture + projector shadow-map pass → Tasks 3–4. ✓
- CPU→GPU texture upload helper (staging + one-shot) → Task 4 (`UploadPattern`). ✓
- Mesh load (PLY/OBJ) + center/scale + CLI → Task 5. ✓
- White Lambertian surface, ambient floor 0.08 → Task 3 frag. ✓
- Interactive viewer: orbit camera, ImGui panel mirroring Blender, `[`/`]`/`P` keys, projector FOV/baseline → Task 6. ✓
- Unit tests for pattern invariants → Tasks 1–2. ✓ Visual acceptance → Tasks 4–6. ✓
- Out-of-scope items (PNG export, calibration JSON, StructuredLightFrameSource) → intentionally excluded. ✓

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to". `KeyCode`, `ClearOptions.color`, `ImageDescriptor::Color2D`, `SwapChain`, and `View` signatures were confirmed against the headers while writing this plan (keys bound to Left/Right/P/Escape — no bracket keys exist). One remaining integration note (ImGui/`GlfwWindow`/`ImageCount` wiring in Task 6 Step 4) points at `object_scan_viewer.cpp`, which wires `ImGuiPass` identically — a real cross-check, not deferred work.

**Type consistency:** `DlpPush`/shader `PushConstants` block identical (model/viewProj/projVP/projPos) across Tasks 3–4. `PatternParams`/`PatternImage`/`PatternCount`/`ColorCodeMetadata`/`ColorCodeInfo` names identical across Tasks 1, 2, 4, 6. `DlpProjectorPass` ctor + `SetPattern(params,step)`/`SetProjector(fovY,baseline)`/`SetModel(index,model)` consistent between Tasks 4–6. Pattern texture format `VK_FORMAT_R8G8B8A8_UNORM` and RGBA8 (4-channel) representation consistent between library and pass.
