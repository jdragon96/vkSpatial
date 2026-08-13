# TSDF Volume 인터페이스 + Memory 축 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** TSDF 볼륨을 이름 문자열로 갈아끼우며 점유율·메모리를 비교할 수 있게 한다 (`flat` / `tile` / `submap`).

**Architecture:** `TSDF::Volume`이 교체 가능한 표면, `TSDF::MemoryStrategy`가 공간 조직 축이다. 세 전략은 기존 `AdvancedTSDF` / `TiledAdvancedTSDF` / `SubmapAdvancedTSDF`를 **상속하지 않고 소유·위임**한다. `ComposedVolume`이 전략을 물고 `Volume`을 구현하며, 이후 계획에서 Integrate/Extract 축이 같은 자리에 추가된다.

**Tech Stack:** C++17, Vulkan(Engine::Core/Compute), Eigen, GoogleTest.

## Global Constraints

- 스펙: [`docs/superpowers/specs/2026-08-13-tsdf-backend-strategy-design.md`](../specs/2026-08-13-tsdf-backend-strategy-design.md)
- 네임스페이스는 **`TSDF`** (`Engine::TSDF` 아님). `src/TSDF/`는 `src/Engine/`의 형제인 최상위 도메인.
- **함수 이름은 동사 하나.** `TSDF` 네임스페이스의 새 API는 `Build` / `Reset` / `Configure` / `Record` / `Download` / `Integrate` / `Stats` / `Name` / `Device` 처럼 한 단어로 읽힌다. `SetIntegrationOptions`, `RecordIntegrate`, `DownloadEntries` 같은 복합어를 새로 만들지 않는다.
  - **예외**: `Engine::Spatial` 기존 클래스에 추가하는 접근자는 이웃(`FilledCount()`, `TileCount()`)의 표기를 따른다 — 한 헤더 안에서 두 규칙이 섞이는 것이 더 나쁘다.
- 변수 이름에 약어 금지 — `cameraPosition`, `insertFailureCount`처럼 전부 풀어 쓴다.
- 기존 `Engine::Spatial` 클래스는 **추가만** 한다. 기존 시그니처·동작을 바꾸는 수정 금지.
- 빌드/테스트: `cmake --build build-rel --target vkspatial_tests -j8` → `./build-rel/test/vkspatial_tests`
- CMake 파일을 만들거나 고친 태스크는 반드시 `cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release`를 먼저 다시 돌린다 (소스 목록이 GLOB이라 재구성 없이는 새 파일이 안 잡힌다).

## 이 계획의 범위

스펙 §8의 **2~4단계**를 담는다. 1단계(`ComputePipeline` 탐색 경로/includer/캐시 키)는 커널을 옮기는 5~7단계에서만 필요하므로 **다음 계획으로 미룬다** — 이 계획은 커널을 하나도 옮기지 않는다.

완료 시점의 검증 가능한 결과물: 같은 스캔을 `flat`/`tile`/`submap`으로 각각 적분하고 `VolumeStats`를 비교하는 테스트. 이것이 스펙 §1에서 최대 리스크로 지목한 **"메모리가 load-limited인가 count-limited인가"** 를 판정하는 도구다.

## 파일 구조

| 파일 | 책임 |
|---|---|
| `src/TSDF/Volume.h` (수정) | 교체 가능 인터페이스. `Device()` 추가 |
| `src/TSDF/Volume.cpp` (신규) | `Volume::Integrate` 정의 (배치 래퍼) |
| `src/TSDF/VolumeRegistry.cpp` (신규) | `Names()`, `Default()` |
| `src/TSDF/CMakeLists.txt` (신규) | `TSDF` 정적 라이브러리 |
| `src/TSDF/Memory/MemoryStrategy.h` (신규) | Memory 축 인터페이스 |
| `src/TSDF/Memory/FlatStrategy.{h,cpp}` (신규) | 단일 512³ 창 — `AdvancedTSDF` 위임 |
| `src/TSDF/Memory/TileStrategy.{h,cpp}` (신규) | 지연 타일링 — `TiledAdvancedTSDF` 위임 |
| `src/TSDF/Memory/SubmapStrategy.{h,cpp}` (신규) | 2단계 밀도 적응 — `SubmapAdvancedTSDF` 위임 |
| `src/TSDF/ComposedVolume.{h,cpp}` (신규) | 전략을 물고 `Volume`을 구현 |
| `src/Engine/Spatial/AdvancedTSDF.h` (수정) | `HashCapacity()` 접근자 추가 |
| `src/Engine/Spatial/TiledDirectionalTSDF.h` (수정) | `SlotCapacity()` 접근자 추가 |
| `src/Engine/Spatial/SubmapAdvancedTSDF.h` (수정) | `FilledCount()`, `SlotCapacity()` 추가 |
| `test/test_tsdf_volume.cpp` (신규) | 전 태스크의 테스트 |

---

### Task 1: 용량·점유 접근자 (기존 클래스에 추가만)

전략이 `VolumeStats`를 채우려면 슬롯 용량을 읽을 수 있어야 한다. `AdvancedTSDF::m_hashCapacity`는
private이고 성장하면 2배가 되므로, 전략이 Build 때 넘긴 초기값을 기억하는 것으로는 **리해시 후
통계가 틀린다.** 그래서 접근자를 추가한다.

**Files:**
- Modify: `src/Engine/Spatial/AdvancedTSDF.h:135`
- Modify: `src/Engine/Spatial/TiledDirectionalTSDF.h:141-147`
- Modify: `src/Engine/Spatial/SubmapAdvancedTSDF.h:149-150`
- Test: `test/test_tsdf_volume.cpp`

**Interfaces:**
- Consumes: 없음
- Produces: `AdvancedTSDF::HashCapacity() -> uint32_t`, `TiledDirectionalTSDF<Backend>::SlotCapacity() -> uint64_t`, `SubmapAdvancedTSDF::FilledCount() -> uint32_t`, `SubmapAdvancedTSDF::SlotCapacity() -> uint64_t`

- [ ] **Step 1: 실패하는 테스트 작성**

`test/test_tsdf_volume.cpp` 를 새로 만든다:

```cpp
#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"
#include "Engine/Spatial/AdvancedTSDF.h"
#include "Engine/Spatial/SubmapAdvancedTSDF.h"
#include "Engine/Spatial/TiledAdvancedTSDF.h"

#include <gtest/gtest.h>

#include <vector>

using Eigen::Vector3f;

namespace {

    // +Z를 향하는 평면 패치. span 폭을 (2*half+1)^2 점으로 채운다.
    void MakePlane(std::vector<Vector3f> &points, std::vector<Vector3f> &normals,
                   float span, int half) {
        points.clear();
        normals.clear();
        const float step = span / float(2 * half);
        for (int i = -half; i <= half; ++i)
            for (int j = -half; j <= half; ++j) {
                points.emplace_back(float(i) * step, float(j) * step, 0.0f);
                normals.emplace_back(0.0f, 0.0f, 1.0f);
            }
    }

} // namespace

TEST(TsdfAccessors, AdvancedReportsHashCapacity) {
    Engine::Core::Context context;
    Engine::Spatial::AdvancedTSDF tsdf;
    tsdf.Build(context, 0.05f, 0.15f, 1u << 16, 1u << 15);
    EXPECT_EQ(tsdf.HashCapacity(), 1u << 16);
}

TEST(TsdfAccessors, TiledSlotCapacityScalesWithTiles) {
    Engine::Core::Context context;
    Engine::Spatial::TiledAdvancedTSDF tsdf;
    tsdf.Build(context, 0.05f, 0.15f, 1u << 16, 1u << 15);
    EXPECT_EQ(tsdf.SlotCapacity(), 0u) << "타일이 아직 없으면 용량도 0";

    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, 0.6f, 8);
    tsdf.Integrate(points, normals, Vector3f(0.0f, 0.0f, 1.0f));

    EXPECT_GE(tsdf.TileCount(), 1u);
    EXPECT_EQ(tsdf.SlotCapacity(), uint64_t(tsdf.TileCount()) * (1u << 16));
}

TEST(TsdfAccessors, SubmapSumsBaseAndDetail) {
    Engine::Core::Context context;
    Engine::Spatial::SubmapAdvancedTSDF tsdf;
    tsdf.Build(context, 0.05f, 0.15f, 32, 4.0f, 1u << 16, 1u << 15);

    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, 0.6f, 8);
    tsdf.Integrate(points, normals, Vector3f(0.0f, 0.0f, 1.0f));

    const uint64_t expected =
            uint64_t(tsdf.BaseTileCount() + tsdf.DetailTileCount()) * (1u << 16);
    EXPECT_EQ(tsdf.SlotCapacity(), expected);
    EXPECT_GT(tsdf.FilledCount(), 0u);
}
```

- [ ] **Step 2: 실패 확인**

```bash
cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release
cmake --build build-rel --target vkspatial_tests -j8
```
Expected: 컴파일 실패 — `no member named 'HashCapacity' in 'Engine::Spatial::AdvancedTSDF'`

- [ ] **Step 3: 접근자 구현**

`src/Engine/Spatial/AdvancedTSDF.h`, `uint32_t FilledCount() const;` 바로 아래:

```cpp
        // Current slot count. Doubles on every growHash, so a caller that cached the Build-time
        // capacity would report a stale load factor -- read it here instead.
        uint32_t HashCapacity() const { return m_hashCapacity; }
```

`src/Engine/Spatial/TiledDirectionalTSDF.h`, `uint32_t TileCount() const` 바로 위:

```cpp
        // Total slots across every live tile. Mirrors FilledCount(); the two together give the
        // load factor. Instantiated lazily, so a Backend without HashCapacity() only fails if
        // this is actually called on that instantiation.
        uint64_t SlotCapacity() const {
            uint64_t total = 0;
            for (const auto &kv: m_tiles) total += kv.second->HashCapacity();
            return total;
        }
```

`src/Engine/Spatial/SubmapAdvancedTSDF.h`, `BaseTileCount()` 바로 위:

```cpp
        // Base + detail combined. Both levels are real storage, so a memory comparison must see
        // the sum rather than either level alone.
        uint32_t FilledCount() const { return m_base.FilledCount() + m_detail.FilledCount(); }

        uint64_t SlotCapacity() const { return m_base.SlotCapacity() + m_detail.SlotCapacity(); }
```

- [ ] **Step 4: 통과 확인**

```bash
cmake --build build-rel --target vkspatial_tests -j8 && \
./build-rel/test/vkspatial_tests --gtest_filter='TsdfAccessors.*'
```
Expected: 3 tests PASS

- [ ] **Step 5: 커밋**

```bash
git add src/Engine/Spatial/AdvancedTSDF.h src/Engine/Spatial/TiledDirectionalTSDF.h \
        src/Engine/Spatial/SubmapAdvancedTSDF.h test/test_tsdf_volume.cpp
git commit -m "feat(spatial): expose slot capacity + submap fill count for load-factor stats"
```

---

### Task 2: `TSDF` 라이브러리 타깃 + 레지스트리 골격

**Files:**
- Modify: `src/TSDF/Volume.h` (`Device()` 추가)
- Create: `src/TSDF/Volume.cpp`
- Create: `src/TSDF/VolumeRegistry.cpp`
- Create: `src/TSDF/CMakeLists.txt`
- Modify: `src/CMakeLists.txt:13`
- Modify: `test/CMakeLists.txt:10-20`
- Test: `test/test_tsdf_volume.cpp`

**Interfaces:**
- Consumes: 없음
- Produces: `TSDF::Volume` (기존 헤더) + `protected: virtual Engine::Core::Context *Device() const = 0`, `TSDF::VolumeRegistry::Names() -> std::vector<std::string>`, `TSDF::VolumeRegistry::Default() -> VolumeRegistry`

`Volume`의 순수 가상 메서드는 이미 헤더에 있다: `Build`, `Reset`, `Configure`, `Record`,
`Download`, `Stats`, `Name`. 이 태스크는 여기에 `Device()`만 더한다.

- [ ] **Step 1: 실패하는 테스트 작성**

`test/test_tsdf_volume.cpp` 상단 include에 추가:

```cpp
#include "TSDF/Volume.h"
```

파일 끝에 추가:

```cpp
TEST(TsdfRegistry, UnknownNameReturnsNull) {
    const TSDF::VolumeRegistry registry = TSDF::VolumeRegistry::Default();
    EXPECT_FALSE(registry.Has("no-such-volume"));
    EXPECT_EQ(registry.Create("no-such-volume"), nullptr);
}

TEST(TsdfRegistry, NamesAreSorted) {
    TSDF::VolumeRegistry registry;
    registry.Register("zulu", [] { return std::unique_ptr<TSDF::Volume>(); });
    registry.Register("alpha", [] { return std::unique_ptr<TSDF::Volume>(); });
    const std::vector<std::string> names = registry.Names();
    ASSERT_EQ(names.size(), 2u);
    EXPECT_EQ(names[0], "alpha");
    EXPECT_EQ(names[1], "zulu");
}
```

- [ ] **Step 2: 실패 확인**

```bash
cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release
cmake --build build-rel --target vkspatial_tests -j8
```
Expected: `fatal error: 'TSDF/Volume.h' file not found` 또는 `undefined symbol: TSDF::VolumeRegistry::Default()`

- [ ] **Step 3: `Volume.h`에 디바이스 접근자 추가**

`src/TSDF/Volume.h`의 `protected:` 블록을 다음으로 교체한다:

```cpp
    protected:
        Volume() = default;

        // The non-virtual Integrate needs a device to open a CommandBatch on, but the base class
        // owns no state -- the concrete strategy holds the context and hands it back here.
        // Null before Build.
        virtual Engine::Core::Context *Device() const = 0;
```

- [ ] **Step 4: `Volume.cpp` 작성**

```cpp
#include "TSDF/Volume.h"

namespace TSDF {

    void Volume::Integrate(const std::vector<Eigen::Vector3f> &points,
                           const std::vector<Eigen::Vector3f> &normals,
                           const Eigen::Vector3f &cameraPosition) {
        if (points.empty()) return;
        Engine::Core::Context *device = Device();
        if (device == nullptr) return; // Build has not run -- nothing to record onto
        Engine::Compute::CommandBatch batch(*device);
        Record(points, normals, cameraPosition, batch);
        batch.Submit();
    }

} // namespace TSDF
```

- [ ] **Step 5: `VolumeRegistry.cpp` 작성**

```cpp
#include "TSDF/Volume.h"

#include <algorithm>

namespace TSDF {

    std::vector<std::string> VolumeRegistry::Names() const {
        std::vector<std::string> names;
        names.reserve(m_factories.size());
        for (const auto &entry: m_factories) names.push_back(entry.first);
        std::sort(names.begin(), names.end()); // unordered_map order is not reproducible
        return names;
    }

    // Task 6에서 flat/tile/submap이 여기에 등록된다.
    VolumeRegistry VolumeRegistry::Default() {
        VolumeRegistry registry;
        return registry;
    }

} // namespace TSDF
```

- [ ] **Step 6: `src/TSDF/CMakeLists.txt` 작성**

```cmake
# TSDF domain library. A sibling of Engine/ (not a submodule of it): src/TSDF/, src/Engine/,
# and future src/Registration/ sit at the same level, and namespaces mirror those paths.
file(GLOB_RECURSE TSDF_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_library(TSDF STATIC ${TSDF_SOURCES})
add_library(TSDF::TSDF ALIAS TSDF)

target_link_libraries(TSDF
        PUBLIC Engine::Core Engine::Compute Engine::Spatial Vulkan::Vulkan)

target_include_directories(TSDF
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            /opt/homebrew/opt/eigen/include/eigen3
        PRIVATE
            "${VULKAN_SDK}/include")
```

- [ ] **Step 7: 상위 CMake 연결**

`src/CMakeLists.txt`의 `add_subdirectory(Engine)` 다음 줄에 추가:

```cmake
add_subdirectory(TSDF)
```

`test/CMakeLists.txt`의 `target_link_libraries(vkspatial_tests PRIVATE ...)` 목록에서
`Engine::Pipeline` 다음 줄에 추가:

```cmake
        TSDF::TSDF
```

- [ ] **Step 8: 통과 확인**

```bash
cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release && \
cmake --build build-rel --target vkspatial_tests -j8 && \
./build-rel/test/vkspatial_tests --gtest_filter='TsdfRegistry.*'
```
Expected: 2 tests PASS

- [ ] **Step 9: 커밋**

```bash
git add src/TSDF/Volume.h src/TSDF/Volume.cpp src/TSDF/VolumeRegistry.cpp \
        src/TSDF/CMakeLists.txt src/CMakeLists.txt test/CMakeLists.txt test/test_tsdf_volume.cpp
git commit -m "feat(tsdf): add TSDF library target with Volume interface and registry"
```

---

### Task 3: `MemoryStrategy` 인터페이스 + `FlatStrategy`

**Files:**
- Create: `src/TSDF/Memory/MemoryStrategy.h`
- Create: `src/TSDF/Memory/FlatStrategy.h`
- Create: `src/TSDF/Memory/FlatStrategy.cpp`
- Test: `test/test_tsdf_volume.cpp`

**Interfaces:**
- Consumes: `TSDF::VolumeParams`, `TSDF::VolumeStats`, `TSDF::IntegrationOptions` (Task 2), `AdvancedTSDF::HashCapacity()` (Task 1)
- Produces: `TSDF::MemoryStrategy` (순수 가상: `Build`, `Reset`, `Configure`, `Record`, `Download`, `Stats`, `Name`, `Device`), `TSDF::FlatStrategy`, `TSDF::kBytesPerHashSlot`

- [ ] **Step 1: 실패하는 테스트 작성**

include 추가:

```cpp
#include "TSDF/Memory/FlatStrategy.h"
```

파일 끝에 추가:

```cpp
TEST(TsdfFlatStrategy, StatsTrackFillAndCapacity) {
    Engine::Core::Context context;
    TSDF::FlatStrategy strategy;
    TSDF::VolumeParams params;
    params.voxelSize = 0.05f;
    params.truncation = 0.15f;
    params.hashCapacity = 1u << 16;
    strategy.Build(context, params);

    EXPECT_STREQ(strategy.Name(), "flat");
    EXPECT_EQ(strategy.Stats().occupiedEntryCount, 0u);
    EXPECT_EQ(strategy.Stats().slotCapacity, 1u << 16);
    EXPECT_EQ(strategy.Stats().tableCount, 1u);

    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, 0.6f, 8);
    Engine::Compute::CommandBatch batch(context);
    strategy.Record(points, normals, Vector3f(0.0f, 0.0f, 1.0f), batch);
    batch.Submit();

    const TSDF::VolumeStats stats = strategy.Stats();
    EXPECT_GT(stats.occupiedEntryCount, 0u);
    EXPECT_GT(stats.LoadFactor(), 0.0);
    EXPECT_LT(stats.LoadFactor(), 1.0);
    EXPECT_EQ(stats.deviceMemoryBytes, stats.slotCapacity * TSDF::kBytesPerHashSlot);

    std::vector<Engine::Spatial::AdvancedEntry> entries;
    strategy.Download(entries);
    EXPECT_EQ(entries.size(), stats.occupiedEntryCount);
}

TEST(TsdfFlatStrategy, StatsAreZeroBeforeBuild) {
    TSDF::FlatStrategy strategy;
    const TSDF::VolumeStats stats = strategy.Stats();
    EXPECT_EQ(stats.slotCapacity, 0u);
    EXPECT_EQ(stats.LoadFactor(), 0.0);
    EXPECT_EQ(strategy.Device(), nullptr);
}
```

- [ ] **Step 2: 실패 확인**

```bash
cmake --build build-rel --target vkspatial_tests -j8
```
Expected: `fatal error: 'TSDF/Memory/FlatStrategy.h' file not found`

- [ ] **Step 3: `MemoryStrategy.h` 작성**

```cpp
#pragma once

#include "TSDF/Volume.h"

namespace TSDF {

    // Bytes each hash slot costs on the device: the 24-byte entry plus the parallel int32
    // first-fill stamp. Memory comparisons are only meaningful if every strategy charges the
    // same rate.
    inline constexpr uint64_t kBytesPerHashSlot =
            sizeof(Engine::Spatial::AdvDirEntry) + sizeof(int32_t);

    /// *********************************************
    /// Memory axis
    /// *********************************************

    // How the volume is organised in space: one fixed window, lazily created tiles, or
    // density-adaptive submaps. This axis is pure C++ -- tiling changes which buffers get bound
    // and what goes in the origin push constant, never the kernel source. That is why it is a
    // virtual-function seam while the integrate/extract axes are shader seams.
    //
    // The operations mirror Volume one for one; ComposedVolume forwards straight through. They
    // diverge once the integrate/extract axes land and this interface keeps only storage.
    class MemoryStrategy {
    public:
        virtual ~MemoryStrategy() = default;

        MemoryStrategy(const MemoryStrategy &) = delete;
        MemoryStrategy &operator=(const MemoryStrategy &) = delete;

        virtual void Build(Engine::Core::Context &context, const VolumeParams &params) = 0;
        virtual void Reset() = 0;
        virtual void Configure(const IntegrationOptions &options) = 0;

        virtual void Record(const std::vector<Eigen::Vector3f> &points,
                            const std::vector<Eigen::Vector3f> &normals,
                            const Eigen::Vector3f &cameraPosition,
                            Engine::Compute::CommandBatch &batch) = 0;

        virtual void Download(std::vector<Engine::Spatial::AdvancedEntry> &out) const = 0;

        virtual VolumeStats Stats() const = 0;
        virtual const char *Name() const = 0;

        // Device this strategy was built on; null before Build. ComposedVolume forwards it so
        // the base Volume::Integrate can open a CommandBatch. Public here (protected on Volume)
        // because the composing class is not a subclass and still has to read it.
        virtual Engine::Core::Context *Device() const = 0;

    protected:
        MemoryStrategy() = default;
    };

} // namespace TSDF
```

- [ ] **Step 4: `FlatStrategy.h` 작성**

```cpp
#pragma once

#include "Engine/Spatial/AdvancedTSDF.h"
#include "TSDF/Memory/MemoryStrategy.h"

namespace TSDF {

    // One fixed 512^3 voxel window. The simplest strategy and the A/B baseline: whatever a tiled
    // or submap layout gains, it gains relative to this. Scenes larger than one window are clipped
    // by the kernel's bounds check rather than growing, which is exactly what makes it a baseline.
    class FlatStrategy final : public MemoryStrategy {
    public:
        void Build(Engine::Core::Context &context, const VolumeParams &params) override;
        void Reset() override;
        void Configure(const IntegrationOptions &options) override;

        void Record(const std::vector<Eigen::Vector3f> &points,
                    const std::vector<Eigen::Vector3f> &normals,
                    const Eigen::Vector3f &cameraPosition,
                    Engine::Compute::CommandBatch &batch) override;

        void Download(std::vector<Engine::Spatial::AdvancedEntry> &out) const override;

        VolumeStats Stats() const override;
        const char *Name() const override { return "flat"; }
        Engine::Core::Context *Device() const override { return m_context; }

    private:
        Engine::Core::Context *m_context = nullptr;
        Engine::Spatial::AdvancedTSDF m_tsdf;
    };

} // namespace TSDF
```

- [ ] **Step 5: `FlatStrategy.cpp` 작성**

```cpp
#include "TSDF/Memory/FlatStrategy.h"

namespace TSDF {

    void FlatStrategy::Build(Engine::Core::Context &context, const VolumeParams &params) {
        m_context = &context;
        m_tsdf.Build(context, params.voxelSize, params.truncation, params.hashCapacity,
                     params.maxPointsPerFrame, params.windowMinCorner);
    }

    void FlatStrategy::Reset() {
        if (m_context == nullptr) return;
        m_tsdf.Reset();
    }

    void FlatStrategy::Configure(const IntegrationOptions &options) {
        m_tsdf.SetIntegrationQuality(options.quality);
        m_tsdf.SetPointToPlane(options.pointToPlane);
        m_tsdf.SetConfidenceWeight(options.confidenceWeight);
        m_tsdf.SetHermitePosition(options.hermitePosition);
        m_tsdf.SetCurrentFrame(options.currentFrame);
    }

    void FlatStrategy::Record(const std::vector<Eigen::Vector3f> &points,
                              const std::vector<Eigen::Vector3f> &normals,
                              const Eigen::Vector3f &cameraPosition,
                              Engine::Compute::CommandBatch &batch) {
        if (m_context == nullptr) return;
        // GPU form: grows the upload buffers to the whole frame instead of clamping, so a
        // comparison run never silently drops points on a large frame.
        m_tsdf.RecordIntegrateGPU(points, normals, cameraPosition, batch);
    }

    void FlatStrategy::Download(std::vector<Engine::Spatial::AdvancedEntry> &out) const {
        if (m_context == nullptr) {
            out.clear();
            return;
        }
        out = m_tsdf.DownloadEntries();
    }

    VolumeStats FlatStrategy::Stats() const {
        VolumeStats stats;
        if (m_context == nullptr) return stats; // FilledCount would dereference an unbuilt buffer
        stats.occupiedEntryCount = m_tsdf.FilledCount();
        stats.slotCapacity = m_tsdf.HashCapacity();
        stats.deviceMemoryBytes = stats.slotCapacity * kBytesPerHashSlot;
        stats.tableCount = 1;
        // insertFailureCount and growCount stay 0 until the hash axis lands in the next plan.
        return stats;
    }

} // namespace TSDF
```

- [ ] **Step 6: 통과 확인**

```bash
cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release && \
cmake --build build-rel --target vkspatial_tests -j8 && \
./build-rel/test/vkspatial_tests --gtest_filter='TsdfFlatStrategy.*'
```
Expected: 2 tests PASS

- [ ] **Step 7: 커밋**

```bash
git add src/TSDF/Memory/MemoryStrategy.h src/TSDF/Memory/FlatStrategy.h \
        src/TSDF/Memory/FlatStrategy.cpp test/test_tsdf_volume.cpp
git commit -m "feat(tsdf): add MemoryStrategy seam with FlatStrategy baseline"
```

---

### Task 4: `TileStrategy`

**Files:**
- Create: `src/TSDF/Memory/TileStrategy.h`
- Create: `src/TSDF/Memory/TileStrategy.cpp`
- Test: `test/test_tsdf_volume.cpp`

**Interfaces:**
- Consumes: `TSDF::MemoryStrategy`, `TSDF::kBytesPerHashSlot` (Task 3), `TiledDirectionalTSDF::SlotCapacity()` (Task 1)
- Produces: `TSDF::TileStrategy`

- [ ] **Step 1: 실패하는 테스트 작성**

include 추가:

```cpp
#include "TSDF/Memory/TileStrategy.h"
```

파일 끝에 추가:

```cpp
TEST(TsdfTileStrategy, TableCountFollowsTiles) {
    Engine::Core::Context context;
    TSDF::TileStrategy strategy;
    TSDF::VolumeParams params;
    params.voxelSize = 0.05f;
    params.truncation = 0.15f;
    params.hashCapacity = 1u << 16;
    strategy.Build(context, params);

    EXPECT_STREQ(strategy.Name(), "tile");
    EXPECT_EQ(strategy.Stats().tableCount, 0u) << "타일은 스캔이 닿을 때 생긴다";

    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, 0.6f, 8);
    Engine::Compute::CommandBatch batch(context);
    strategy.Record(points, normals, Vector3f(0.0f, 0.0f, 1.0f), batch);
    batch.Submit();

    const TSDF::VolumeStats stats = strategy.Stats();
    EXPECT_GE(stats.tableCount, 1u);
    EXPECT_EQ(stats.slotCapacity, uint64_t(stats.tableCount) * (1u << 16));
    EXPECT_GT(stats.occupiedEntryCount, 0u);
    EXPECT_EQ(stats.deviceMemoryBytes, stats.slotCapacity * TSDF::kBytesPerHashSlot);

    std::vector<Engine::Spatial::AdvancedEntry> entries;
    strategy.Download(entries);
    EXPECT_GT(entries.size(), 0u);
}
```

- [ ] **Step 2: 실패 확인**

```bash
cmake --build build-rel --target vkspatial_tests -j8
```
Expected: `fatal error: 'TSDF/Memory/TileStrategy.h' file not found`

- [ ] **Step 3: `TileStrategy.h` 작성**

```cpp
#pragma once

#include "Engine/Spatial/TiledAdvancedTSDF.h"
#include "TSDF/Memory/MemoryStrategy.h"

namespace TSDF {

    // Lazily created 448^3-core tiles, so the scene is not capped by one 512^3 window. Every tile
    // owns its own hash of VolumeParams::hashCapacity slots, which is why tableCount matters:
    // a scene spread over many sparse tiles is memory-bound by tile COUNT, not by load factor.
    class TileStrategy final : public MemoryStrategy {
    public:
        void Build(Engine::Core::Context &context, const VolumeParams &params) override;
        void Reset() override;
        void Configure(const IntegrationOptions &options) override;

        void Record(const std::vector<Eigen::Vector3f> &points,
                    const std::vector<Eigen::Vector3f> &normals,
                    const Eigen::Vector3f &cameraPosition,
                    Engine::Compute::CommandBatch &batch) override;

        void Download(std::vector<Engine::Spatial::AdvancedEntry> &out) const override;

        VolumeStats Stats() const override;
        const char *Name() const override { return "tile"; }
        Engine::Core::Context *Device() const override { return m_context; }

    private:
        Engine::Core::Context *m_context = nullptr;
        Engine::Spatial::TiledAdvancedTSDF m_tsdf;
    };

} // namespace TSDF
```

- [ ] **Step 4: `TileStrategy.cpp` 작성**

```cpp
#include "TSDF/Memory/TileStrategy.h"

namespace TSDF {

    void TileStrategy::Build(Engine::Core::Context &context, const VolumeParams &params) {
        m_context = &context;
        // windowMinCorner is deliberately unused: a tiled layout derives each tile's origin from
        // the tile grid, so there is no single window corner to place.
        m_tsdf.Build(context, params.voxelSize, params.truncation, params.hashCapacity,
                     params.maxPointsPerFrame);
    }

    void TileStrategy::Reset() {
        if (m_context == nullptr) return;
        m_tsdf.Reset();
    }

    void TileStrategy::Configure(const IntegrationOptions &options) {
        m_tsdf.SetIntegrationQuality(options.quality);
        m_tsdf.SetPointToPlane(options.pointToPlane);
        m_tsdf.SetConfidenceWeight(options.confidenceWeight);
        m_tsdf.SetHermitePosition(options.hermitePosition);
        m_tsdf.SetCurrentFrame(options.currentFrame);
    }

    void TileStrategy::Record(const std::vector<Eigen::Vector3f> &points,
                              const std::vector<Eigen::Vector3f> &normals,
                              const Eigen::Vector3f &cameraPosition,
                              Engine::Compute::CommandBatch &batch) {
        if (m_context == nullptr) return;
        // The 4-argument form routes points to tiles and records every tile's dispatch into the
        // one batch -- that fusion is the whole reason Record is the interface primitive.
        m_tsdf.Integrate(points, normals, cameraPosition, batch);
    }

    void TileStrategy::Download(std::vector<Engine::Spatial::AdvancedEntry> &out) const {
        if (m_context == nullptr) {
            out.clear();
            return;
        }
        m_tsdf.DownloadEntries(out);
    }

    VolumeStats TileStrategy::Stats() const {
        VolumeStats stats;
        if (m_context == nullptr) return stats;
        stats.occupiedEntryCount = m_tsdf.FilledCount();
        stats.slotCapacity = m_tsdf.SlotCapacity();
        stats.deviceMemoryBytes = stats.slotCapacity * kBytesPerHashSlot;
        stats.tableCount = m_tsdf.TileCount();
        return stats;
    }

} // namespace TSDF
```

- [ ] **Step 5: 통과 확인**

```bash
cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release && \
cmake --build build-rel --target vkspatial_tests -j8 && \
./build-rel/test/vkspatial_tests --gtest_filter='TsdfTileStrategy.*'
```
Expected: 1 test PASS

- [ ] **Step 6: 커밋**

```bash
git add src/TSDF/Memory/TileStrategy.h src/TSDF/Memory/TileStrategy.cpp test/test_tsdf_volume.cpp
git commit -m "feat(tsdf): add TileStrategy delegating to TiledAdvancedTSDF"
```

---

### Task 5: `SubmapStrategy`

⚠️ **알려진 계약 이탈**: `SubmapAdvancedTSDF::Integrate`는 내부에서 자체 `CommandBatch`를 만들어
바로 제출한다. 배치 지연 형태가 없으므로 `SubmapStrategy::Record`는 **넘겨받은 batch에 아무것도
기록하지 않고 즉시 제출한다.** 결과는 정확하지만 다른 볼륨과 한 submit으로 묶이지 않는다.
`SubmapAdvancedTSDF`에 배치 오버로드를 추가하는 것은 "기존 클래스는 추가만" 원칙에는 맞지만
이 계획의 범위 밖이라 후속으로 남긴다.

**Files:**
- Create: `src/TSDF/Memory/SubmapStrategy.h`
- Create: `src/TSDF/Memory/SubmapStrategy.cpp`
- Test: `test/test_tsdf_volume.cpp`

**Interfaces:**
- Consumes: `TSDF::MemoryStrategy`, `TSDF::kBytesPerHashSlot` (Task 3), `SubmapAdvancedTSDF::FilledCount()` / `SlotCapacity()` (Task 1)
- Produces: `TSDF::SubmapStrategy`

- [ ] **Step 1: 실패하는 테스트 작성**

include 추가:

```cpp
#include "TSDF/Memory/SubmapStrategy.h"
```

파일 끝에 추가:

```cpp
TEST(TsdfSubmapStrategy, IntegratesDespiteSelfSubmittingBackend) {
    Engine::Core::Context context;
    TSDF::SubmapStrategy strategy;
    TSDF::VolumeParams params;
    params.voxelSize = 0.05f;
    params.truncation = 0.15f;
    params.hashCapacity = 1u << 16;
    strategy.Build(context, params);

    EXPECT_STREQ(strategy.Name(), "submap");

    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, 0.6f, 8);
    // 일부러 Submit하지 않는다: 이 전략은 batch에 아무것도 기록하지 않으므로 제출할 것이
    // 없고, 빈 batch 제출이 안전한지는 이 태스크가 검증할 대상이 아니다.
    Engine::Compute::CommandBatch batch(context);
    strategy.Record(points, normals, Vector3f(0.0f, 0.0f, 1.0f), batch);

    const TSDF::VolumeStats stats = strategy.Stats();
    EXPECT_GT(stats.occupiedEntryCount, 0u) << "batch를 제출하지 않아도 적분은 끝나 있어야 한다";
    EXPECT_GE(stats.tableCount, 1u);
    EXPECT_EQ(stats.deviceMemoryBytes, stats.slotCapacity * TSDF::kBytesPerHashSlot);
}
```

- [ ] **Step 2: 실패 확인**

```bash
cmake --build build-rel --target vkspatial_tests -j8
```
Expected: `fatal error: 'TSDF/Memory/SubmapStrategy.h' file not found`

- [ ] **Step 3: `SubmapStrategy.h` 작성**

```cpp
#pragma once

#include "Engine/Spatial/SubmapAdvancedTSDF.h"
#include "TSDF/Memory/MemoryStrategy.h"

namespace TSDF {

    // Two levels of tiled storage: a base grid plus half-voxel detail submaps over blocks the scan
    // covers densely. Costs a second full tile hierarchy, so it is the strategy where tableCount
    // and slotCapacity diverge most from occupancy -- exactly the case the A/B run is for.
    //
    // Contract deviation: the wrapped SubmapAdvancedTSDF::Integrate opens and submits its own
    // CommandBatch, so Record puts NOTHING into the caller's batch and completes the work
    // immediately. Correct, but not fused with other volumes in one submit.
    class SubmapStrategy final : public MemoryStrategy {
    public:
        void Build(Engine::Core::Context &context, const VolumeParams &params) override;
        void Reset() override;
        void Configure(const IntegrationOptions &options) override;

        void Record(const std::vector<Eigen::Vector3f> &points,
                    const std::vector<Eigen::Vector3f> &normals,
                    const Eigen::Vector3f &cameraPosition,
                    Engine::Compute::CommandBatch &batch) override;

        void Download(std::vector<Engine::Spatial::AdvancedEntry> &out) const override;

        VolumeStats Stats() const override;
        const char *Name() const override { return "submap"; }
        Engine::Core::Context *Device() const override { return m_context; }

    private:
        static constexpr int kBlockVoxels = 32;
        static constexpr float kDetailPointsPerVoxel = 4.0f;

        Engine::Core::Context *m_context = nullptr;
        Engine::Spatial::SubmapAdvancedTSDF m_tsdf;
    };

} // namespace TSDF
```

- [ ] **Step 4: `SubmapStrategy.cpp` 작성**

```cpp
#include "TSDF/Memory/SubmapStrategy.h"

namespace TSDF {

    void SubmapStrategy::Build(Engine::Core::Context &context, const VolumeParams &params) {
        m_context = &context;
        m_tsdf.Build(context, params.voxelSize, params.truncation, kBlockVoxels,
                     kDetailPointsPerVoxel, params.hashCapacity, params.maxPointsPerFrame);
    }

    void SubmapStrategy::Reset() {
        if (m_context == nullptr) return;
        m_tsdf.Reset();
    }

    void SubmapStrategy::Configure(const IntegrationOptions &options) {
        // Forward every option, exactly as the sibling strategies do -- SubmapAdvancedTSDF fans
        // each setter out to both its base and detail levels. A strategy that quietly dropped
        // sweep-relevant options would make an A/B run lie: the same parameter set would mean
        // something different for `submap` than for `flat`/`tile`.
        m_tsdf.SetIntegrationQuality(options.quality);
        m_tsdf.SetPointToPlane(options.pointToPlane);
        m_tsdf.SetConfidenceWeight(options.confidenceWeight);
        m_tsdf.SetHermitePosition(options.hermitePosition);
        m_tsdf.SetCurrentFrame(options.currentFrame);
    }

    void SubmapStrategy::Record(const std::vector<Eigen::Vector3f> &points,
                                const std::vector<Eigen::Vector3f> &normals,
                                const Eigen::Vector3f &cameraPosition,
                                Engine::Compute::CommandBatch & /*batch*/) {
        if (m_context == nullptr) return;
        // Deviation documented in the header: this call opens and submits its own batch.
        m_tsdf.Integrate(points, normals, cameraPosition);
    }

    void SubmapStrategy::Download(std::vector<Engine::Spatial::AdvancedEntry> &out) const {
        if (m_context == nullptr) {
            out.clear();
            return;
        }
        m_tsdf.DownloadEntries(out);
    }

    VolumeStats SubmapStrategy::Stats() const {
        VolumeStats stats;
        if (m_context == nullptr) return stats;
        stats.occupiedEntryCount = m_tsdf.FilledCount();
        stats.slotCapacity = m_tsdf.SlotCapacity();
        stats.deviceMemoryBytes = stats.slotCapacity * kBytesPerHashSlot;
        stats.tableCount = m_tsdf.BaseTileCount() + m_tsdf.DetailTileCount();
        return stats;
    }

} // namespace TSDF
```

- [ ] **Step 5: 통과 확인**

```bash
cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release && \
cmake --build build-rel --target vkspatial_tests -j8 && \
./build-rel/test/vkspatial_tests --gtest_filter='TsdfSubmapStrategy.*'
```
Expected: 1 test PASS

- [ ] **Step 6: 커밋**

```bash
git add src/TSDF/Memory/SubmapStrategy.h src/TSDF/Memory/SubmapStrategy.cpp test/test_tsdf_volume.cpp
git commit -m "feat(tsdf): add SubmapStrategy with documented non-deferring record"
```

---

### Task 6: `ComposedVolume` + 레지스트리 등록 + 이름 스위칭 비교

**Files:**
- Create: `src/TSDF/ComposedVolume.h`
- Create: `src/TSDF/ComposedVolume.cpp`
- Modify: `src/TSDF/VolumeRegistry.cpp` (`Default()` 채우기)
- Test: `test/test_tsdf_volume.cpp`

**Interfaces:**
- Consumes: `TSDF::MemoryStrategy` (Task 3), `TSDF::FlatStrategy` / `TileStrategy` / `SubmapStrategy` (Task 3~5)
- Produces: `TSDF::ComposedVolume(std::unique_ptr<MemoryStrategy>)`, 레지스트리 이름 `"flat"`, `"tile"`, `"submap"`

- [ ] **Step 1: 실패하는 테스트 작성**

include 추가:

```cpp
#include "TSDF/ComposedVolume.h"
```

파일 끝에 추가:

```cpp
TEST(TsdfRegistry, DefaultRegistersAllMemoryStrategies) {
    const TSDF::VolumeRegistry registry = TSDF::VolumeRegistry::Default();
    EXPECT_EQ(registry.Names(), (std::vector<std::string>{"flat", "submap", "tile"}));
}

// 이름만 바꿔가며 같은 스캔을 적분하고 통계를 비교한다. 이것이 이 계획의 최종 산출물 --
// 메모리가 점유율(load factor)에 묶여 있는지 타일 개수에 묶여 있는지 판정하는 도구.
TEST(TsdfVolumeSwitching, EveryStrategyIntegratesTheSameScan) {
    const TSDF::VolumeRegistry registry = TSDF::VolumeRegistry::Default();

    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, 0.6f, 8);

    for (const std::string &name: registry.Names()) {
        Engine::Core::Context context;
        std::unique_ptr<TSDF::Volume> volume = registry.Create(name);
        ASSERT_NE(volume, nullptr) << name;
        EXPECT_EQ(std::string(volume->Name()), name);

        TSDF::VolumeParams params;
        params.voxelSize = 0.05f;
        params.truncation = 0.15f;
        params.hashCapacity = 1u << 16;
        volume->Build(context, params);
        volume->Integrate(points, normals, Vector3f(0.0f, 0.0f, 1.0f));

        const TSDF::VolumeStats stats = volume->Stats();
        EXPECT_GT(stats.occupiedEntryCount, 0u) << name;
        EXPECT_GT(stats.slotCapacity, 0u) << name;
        EXPECT_GE(stats.tableCount, 1u) << name;
        EXPECT_EQ(stats.insertFailureCount, 0u) << name << ": 복셀이 조용히 드롭되면 안 된다";

        std::vector<Engine::Spatial::AdvancedEntry> entries;
        volume->Download(entries);
        EXPECT_GT(entries.size(), 0u) << name;
    }
}

TEST(TsdfVolumeSwitching, ResetEmptiesTheVolume) {
    const TSDF::VolumeRegistry registry = TSDF::VolumeRegistry::Default();
    Engine::Core::Context context;
    std::unique_ptr<TSDF::Volume> volume = registry.Create("flat");
    ASSERT_NE(volume, nullptr);

    TSDF::VolumeParams params;
    params.voxelSize = 0.05f;
    params.truncation = 0.15f;
    params.hashCapacity = 1u << 16;
    volume->Build(context, params);

    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, 0.6f, 8);
    volume->Integrate(points, normals, Vector3f(0.0f, 0.0f, 1.0f));
    ASSERT_GT(volume->Stats().occupiedEntryCount, 0u);

    volume->Reset();
    EXPECT_EQ(volume->Stats().occupiedEntryCount, 0u);
}
```

- [ ] **Step 2: 실패 확인**

```bash
cmake --build build-rel --target vkspatial_tests -j8
```
Expected: `fatal error: 'TSDF/ComposedVolume.h' file not found`

- [ ] **Step 3: `ComposedVolume.h` 작성**

```cpp
#pragma once

#include "TSDF/Memory/MemoryStrategy.h"

#include <memory>

namespace TSDF {

    // A Volume assembled from swappable strategies. Today it holds only the Memory axis and
    // forwards to it; the Integrate and Extract axes drop in here without touching the memory
    // strategies, which is the reason this pass-through exists now rather than later.
    class ComposedVolume final : public Volume {
    public:
        explicit ComposedVolume(std::unique_ptr<MemoryStrategy> memory)
            : m_memory(std::move(memory)) {}

        void Build(Engine::Core::Context &context, const VolumeParams &params) override {
            m_memory->Build(context, params);
        }

        void Reset() override { m_memory->Reset(); }

        void Configure(const IntegrationOptions &options) override {
            m_memory->Configure(options);
        }

        void Record(const std::vector<Eigen::Vector3f> &points,
                    const std::vector<Eigen::Vector3f> &normals,
                    const Eigen::Vector3f &cameraPosition,
                    Engine::Compute::CommandBatch &batch) override {
            m_memory->Record(points, normals, cameraPosition, batch);
        }

        void Download(std::vector<Engine::Spatial::AdvancedEntry> &out) const override {
            m_memory->Download(out);
        }

        VolumeStats Stats() const override { return m_memory->Stats(); }

        // The registered name is the memory strategy's name while it is the only axis.
        const char *Name() const override { return m_memory->Name(); }

    protected:
        Engine::Core::Context *Device() const override { return m_memory->Device(); }

    private:
        std::unique_ptr<MemoryStrategy> m_memory;
    };

} // namespace TSDF
```

- [ ] **Step 4: `ComposedVolume.cpp` 작성**

헤더가 전부 인라인이라 이 번역 단위는 헤더가 독립적으로 컴파일되는지만 보증한다.

```cpp
#include "TSDF/ComposedVolume.h"

// ComposedVolume is header-only; this translation unit exists so the header is compiled on its
// own and a missing include here fails the build instead of a distant consumer.
```

- [ ] **Step 5: `VolumeRegistry.cpp`의 `Default()` 채우기**

`#include "TSDF/Volume.h"` 아래에 include를 추가한다:

```cpp
#include "TSDF/ComposedVolume.h"
#include "TSDF/Memory/FlatStrategy.h"
#include "TSDF/Memory/SubmapStrategy.h"
#include "TSDF/Memory/TileStrategy.h"
```

`Default()`를 다음으로 교체한다:

```cpp
    VolumeRegistry VolumeRegistry::Default() {
        VolumeRegistry registry;
        registry.Register("flat", [] {
            return std::make_unique<ComposedVolume>(std::make_unique<FlatStrategy>());
        });
        registry.Register("tile", [] {
            return std::make_unique<ComposedVolume>(std::make_unique<TileStrategy>());
        });
        registry.Register("submap", [] {
            return std::make_unique<ComposedVolume>(std::make_unique<SubmapStrategy>());
        });
        return registry;
    }
```

- [ ] **Step 6: 통과 확인**

```bash
cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release && \
cmake --build build-rel --target vkspatial_tests -j8 && \
./build-rel/test/vkspatial_tests --gtest_filter='Tsdf*'
```
Expected: 이 계획의 테스트 11개 전부 PASS

- [ ] **Step 7: 회귀 확인 — 저장소 전체 테스트**

```bash
./build-rel/test/vkspatial_tests
```
Expected: 기존 테스트가 이 계획 이전과 동일하게 통과. 실패가 있으면 **기준선과 비교**한다
(`git stash`로 되돌려 같은 명령을 돌려본 뒤 새 실패인지 원래 실패인지 판별).

- [ ] **Step 8: 커밋**

```bash
git add src/TSDF/ComposedVolume.h src/TSDF/ComposedVolume.cpp \
        src/TSDF/VolumeRegistry.cpp test/test_tsdf_volume.cpp
git commit -m "feat(tsdf): compose volumes from memory strategies and register flat/tile/submap"
```

---

## 후속 계획으로 넘기는 것

| 항목 | 이유 |
|---|---|
| `ComputePipeline` 탐색 경로 / 2단 includer / 캐시 키 | 커널을 옮기는 시점에만 필요 (스펙 §5) |
| Integrate·Extract 축 분리 + 커널 이동 | 스펙 §8 5~6단계 |
| `Memory/Hash` 전략 주입 | 스펙 §8 7단계 |
| `insertFailureCount` 계측 + 버킷 해시 | 스펙 §8 8단계 — 이 계획은 필드를 0으로 채워둔다 |
| `SubmapAdvancedTSDF` 배치 오버로드 | Task 5의 계약 이탈 해소 |

---

## 실행 결과 (2026-08-14 완료)

7개 커밋 `f8b63c5..8423573`. 테스트 261 → **275 passed / 1 skipped / 0 failed** (신규 14개, 회귀 0).

### 계획서가 틀렸던 곳 두 군데

실행 중 리뷰가 잡아낸 **계획 자체의 결함**이다. 둘 다 같은 원인 — 계획 단계에서 여러 파일을
한 번에 grep했을 때 **한 파일의 매치만 돌아온 것**을 검증 없이 옮겨 적었다. 이 저장소에서
계획을 쓸 때는 파일별로 나눠 확인할 것.

1. **Task 5 `Configure`** — "`SubmapAdvancedTSDF`는 세터를 2개만 노출한다"고 적었으나 실제로는
   6개(`SubmapAdvancedTSDF.h:47,51,55,59,64,66`). 그대로 구현했으면 `pointToPlane` /
   `confidenceWeight` / `hermitePosition`을 바꾸는 스윕이 `submap`에만 조용히 안 먹었다.
2. **Task 4/5 `Record`** (Critical) — 타일드/서브맵에 clamp되는 `Integrate`를 지정했다.
   타일당 `maxPointsPerFrame`(기본 32768)에서 **말없이 잘리는데** `flat`은 버퍼를 키운다.
   메모리 비교가 타일링에 유리한 쪽으로 편향된다. 289점짜리 픽스처로는 보이지도 않는다.
   → `TiledDirectionalTSDF::RecordIntegrateGPU`(신규, 추가만) + `SubmapAdvancedTSDF::IntegrateGPU`로 교체.

### 측정 도구가 내놓은 첫 숫자

289점 평면, 테이블당 65536 슬롯:

| 전략 | occupied | slots | load factor | tables |
|---|---|---|---|---|
| flat | 1,045 | 65,536 | 0.0160 | 1 |
| tile | 5,566 | 524,288 | 0.0106 | 8 |
| submap | 6,838 | 524,288 | 0.0130 | 8 |

**이 숫자로 "load-limited냐 count-limited냐"를 결론지으면 안 된다.** 픽스처가 289점이라
세 전략 모두 load factor가 0.01~0.02로 바닥이고, `slotCapacity`가 `tableCount`에 정확히
비례하는 건 타일당 고정 용량 할당의 정의일 뿐이다. 실제 답은 진짜 스캔을 돌려야 나온다.

### 다음 계획으로 넘어간 것

- **하네스 글루**: 테스트 말고는 아무것도 `TSDF::TSDF`를 링크하지 않는다. `tsdf_folder_eval --tsdf <name>`
  같은 진입점이 있어야 실제 스캔으로 위 질문에 답할 수 있다. **이게 최우선.**
- `ComputePipeline` 탐색 경로 / 2단 includer / 캐시 키 (스펙 §5)
- Integrate·Extract 축 분리 + 커널 이동 (스펙 §8 5~6단계)
- `Memory/Hash` 전략 주입 + `insertFailureCount` 계측 → 버킷 해시 A/B (7~8단계)
- `SubmapAdvancedTSDF` 배치 오버로드 (Task 5 계약 이탈 해소)
- **Integrate 축 착수 전 확인할 것**: 커널을 갈아끼우려면 메모리 전략이 해시 버퍼와
  push constant의 메모리 절반을 노출해야 하는데(스펙 §6.2), 지금은 전부 `AdvancedTSDF` 내부에
  private이다. 현재 이음매에서는 안 보이는 결합이고, `MemoryStrategy` 인터페이스를 바꾸게 만들
  가장 유력한 후보다.
