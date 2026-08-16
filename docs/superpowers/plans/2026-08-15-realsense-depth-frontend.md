# RealSense depth 프론트엔드 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 재구성 파이프라인이 실제 depth 센서(Intel RealSense D435)에서 프레임을 받고, 그 캡처를 녹화·재생할 수 있게 한다.

**Architecture:** `IDepthProvider` 구현 두 개(장치 / 녹화 재생)와 데코레이터 하나(녹화)를 추가한다. `BackprojectDepth`에 깊이 불연속 검사를 넣는다. 파이프라인 구조와 스테이지는 손대지 않는다.

**Tech Stack:** librealsense2 2.58.3 (Homebrew, `/opt/homebrew`), Eigen, gtest

**Spec:** [`docs/superpowers/specs/2026-08-15-realsense-depth-frontend-design.md`](../specs/2026-08-15-realsense-depth-frontend-design.md)

## Global Constraints

- **하드웨어 없이 빌드되고 테스트가 돌아야 한다.** librealsense를 못 찾으면 그 provider만 빠지고 나머지는 그대로 빌드된다. 자동 테스트는 장치를 요구하지 않는다.
- **`Frame`의 계약은 `pts.size() == nrm.size()`.** 파이프라인 전체가 가정한다. 법선을 못 만들면 점도 버린다.
- 축약어 금지: `maxDepthJump`이지 `maxDJ`가 아니다.
- 실패는 `throw std::runtime_error("<Class>::<Method>: ...")`. `assert` 금지 — 빌드가 `-O3 -DNDEBUG`다.
- 빌드: `cmake -S . -B build-rel && cmake --build build-rel -j8 --target vkspatial_tests`, 실행 `./build-rel/test/vkspatial_tests`. **기준선 204 passed / 1 skipped / 0 failed.**
- `src/` 아래 파일을 추가하면 `cmake -S . -B build-rel`를 반드시 다시 돌린다(GLOB).
- 스테이징은 경로 지정. 작업 트리에 사용자의 미커밋 파일이 있으면 건드리지 않는다.

---

## File Structure

| 파일 | 책임 |
|---|---|
| `src/Pipeline/Reconstruction/DepthCameraFrameSource.h` (수정) | `BackprojectDepth`에 불연속 임계값 |
| `src/Pipeline/Reconstruction/DepthRecording.h` / `.cpp` (신규) | `DepthRecorder`(데코레이터) + `RecordedDepthProvider` + 형식 상수 |
| `src/Pipeline/Reconstruction/RealSenseDepthProvider.h` / `.cpp` (신규) | 장치. librealsense 있을 때만 컴파일 |
| `src/Pipeline/CMakeLists.txt` (수정) | librealsense 선택적 탐색 |
| `test/test_depthFrontend.cpp` (신규) | 불연속·법선·왕복 |
| `example2/depth_capture.cpp` (신규) | 캡처/재생 도구 |

---

### Task 1: 깊이 불연속 검사

**Files:**
- Modify: `src/Pipeline/Reconstruction/DepthCameraFrameSource.h`
- Test: `test/test_depthFrontend.cpp` (신규)

**Interfaces:**
- Consumes: 기존 `CameraIntrinsics`, `DepthFrame`, `Frame`
- Produces: `struct DepthFilterOptions { float relativeDepthJump = 0.02f; float minimumDepthJump = 0.005f; };`
  그리고 `Frame BackprojectDepth(const DepthFrame &, const CameraIntrinsics &, const DepthFilterOptions & = {});`
  — 기본 인자라 기존 호출부는 그대로 컴파일된다.

- [ ] **Step 1: 실패하는 테스트 작성**

`test/test_depthFrontend.cpp`:

```cpp
#include "Pipeline/Reconstruction/DepthCameraFrameSource.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using Pipeline::BackprojectDepth;
using Pipeline::CameraIntrinsics;
using Pipeline::DepthFilterOptions;
using Pipeline::DepthFrame;

namespace {
    CameraIntrinsics MakeIntrinsics(int width, int height) {
        CameraIntrinsics k;
        k.width = width;
        k.height = height;
        k.fx = k.fy = 400.0f;
        k.cx = float(width) * 0.5f;
        k.cy = float(height) * 0.5f;
        return k;
    }
} // namespace

// A depth step is two different surfaces, one behind the other. Differencing across it produces a
// normal that belongs to neither -- the "flying pixel" a stereo sensor generates at every object
// boundary. No normal may span the step.
TEST(DepthFrontend, DepthStepProducesNoNormalAcrossIt) {
    const CameraIntrinsics k = MakeIntrinsics(64, 64);
    DepthFrame frame;
    frame.depth.assign(std::size_t(k.width) * k.height, 0.0f);
    for (int v = 0; v < k.height; ++v)
        for (int u = 0; u < k.width; ++u)
            frame.depth[std::size_t(v) * k.width + u] = u < 32 ? 1.0f : 2.0f;

    const Pipeline::Frame out = BackprojectDepth(frame, k, DepthFilterOptions{});

    ASSERT_GT(out.pts.size(), 0u) << "the two flat halves must still produce normals";
    for (std::size_t i = 0; i < out.pts.size(); ++i) {
        // Both halves face the camera, so every surviving normal must be ~(0,0,-1). A normal built
        // across the step tilts away from that by a large angle.
        EXPECT_GT(-out.nrm[i].z(), 0.9f)
                << "point " << i << " at z=" << out.pts[i].z() << " has a normal spanning the step";
    }
}

// The guard must not kill ordinary surfaces: a plane facing the camera keeps every pixel.
TEST(DepthFrontend, FlatPlaneKeepsEveryInteriorPixel) {
    const CameraIntrinsics k = MakeIntrinsics(32, 32);
    DepthFrame frame;
    frame.depth.assign(std::size_t(k.width) * k.height, 1.5f);

    const Pipeline::Frame out = BackprojectDepth(frame, k, DepthFilterOptions{});

    // BackprojectDepth walks [0,W-1) x [0,H-1) -- it needs the u+1 and v+1 neighbours.
    EXPECT_EQ(out.pts.size(), std::size_t(k.width - 1) * (k.height - 1));
    for (const Eigen::Vector3f &n: out.nrm) EXPECT_NEAR(n.z(), -1.0f, 1e-3f);
}

// A slanted plane is a real surface with a per-pixel depth change. The guard is relative to depth,
// so it must survive -- a fixed threshold would cut it.
TEST(DepthFrontend, SlantedPlaneSurvivesTheGuard) {
    const CameraIntrinsics k = MakeIntrinsics(64, 64);
    DepthFrame frame;
    frame.depth.assign(std::size_t(k.width) * k.height, 0.0f);
    for (int v = 0; v < k.height; ++v)
        for (int u = 0; u < k.width; ++u) // 1.0 m rising to 1.5 m across the image
            frame.depth[std::size_t(v) * k.width + u] = 1.0f + 0.5f * float(u) / float(k.width);

    const Pipeline::Frame out = BackprojectDepth(frame, k, DepthFilterOptions{});
    EXPECT_EQ(out.pts.size(), std::size_t(k.width - 1) * (k.height - 1))
            << "a slanted surface must not be mistaken for a discontinuity";
}
```

- [ ] **Step 2: 실패 확인**

```bash
cmake -S . -B build-rel && cmake --build build-rel -j8 --target vkspatial_tests
./build-rel/test/vkspatial_tests --gtest_filter='DepthFrontend.*'
```
Expected: 컴파일 실패(`DepthFilterOptions` 없음). 옵션 구조체만 먼저 추가하고 필터는 넣지 말 것 — 그러면 `DepthStepProducesNoNormalAcrossIt`가 계단을 가로지르는 법선 때문에 FAIL해야 한다. **그 FAIL을 눈으로 확인한 뒤** Step 3으로 갈 것.

- [ ] **Step 3: 필터 구현**

`DepthCameraFrameSource.h`의 `BackprojectDepth` 법선 루프에 넣는다:

```cpp
struct DepthFilterOptions {
    // Reject a neighbour whose depth differs by more than max(minimumDepthJump,
    // relativeDepthJump * z). Relative because stereo depth error grows as z^2/(f*baseline): a
    // fixed threshold over-rejects near the camera and under-rejects far from it.
    float relativeDepthJump = 0.02f;  // 2 % of range
    float minimumDepthJump = 0.005f;  // 5 mm floor, for the near field
};

inline Frame BackprojectDepth(const DepthFrame &d, const CameraIntrinsics &k,
                              const DepthFilterOptions &filter = {}) {
    ...
    for (int v = 0; v + 1 < H; ++v)
        for (int u = 0; u + 1 < W; ++u) {
            const std::size_t i = std::size_t(v) * W + u;
            if (!valid[i] || !valid[i + 1] || !valid[i + W]) continue;

            // A step in depth is two surfaces, not one: differencing across it yields a normal
            // belonging to neither. The point goes with the normal -- Frame's contract is
            // pts.size() == nrm.size(), and the whole pipeline assumes it.
            const float z = grid[i].z();
            const float maxJump = std::max(filter.minimumDepthJump, filter.relativeDepthJump * z);
            if (std::abs(grid[i + 1].z() - z) > maxJump) continue;
            if (std::abs(grid[i + W].z() - z) > maxJump) continue;

            Eigen::Vector3f n = (grid[i + 1] - grid[i]).cross(grid[i + W] - grid[i]);
            ...
        }
}
```

`<algorithm>`과 `<cmath>` include를 확인할 것.

- [ ] **Step 4: 통과 확인 후 커밋**

```bash
cmake --build build-rel -j8 --target vkspatial_tests && ./build-rel/test/vkspatial_tests
```
Expected: `DepthFrontend.*` 3개 PASS, 전체 207 passed / 1 skipped / 0 failed.

```bash
git add src/Pipeline/Reconstruction/DepthCameraFrameSource.h test/test_depthFrontend.cpp
git commit -m "feat(pipeline): reject normals built across a depth discontinuity"
```

---

### Task 2: 녹화와 재생

**Files:**
- Create: `src/Pipeline/Reconstruction/DepthRecording.h`, `.cpp`
- Test: `test/test_depthFrontend.cpp` (추가)

**Interfaces:**
- Consumes: Task 1의 `IDepthProvider`, `CameraIntrinsics`, `DepthFrame`
- Produces:
```cpp
namespace Pipeline {
    // <dir>/intrinsics.txt : "fx fy cx cy width height"
    // <dir>/depth_%04d.bin : float32 * width*height, row-major, metres
    class DepthRecorder : public IDepthProvider {
    public:
        DepthRecorder(std::unique_ptr<IDepthProvider> device, std::string directory);
        const CameraIntrinsics &Intrinsics() const override;
        bool Grab(DepthFrame &out) override;   // 원본에서 받아 디스크에 쓰고 그대로 돌려준다
        int RecordedFrameCount() const;
    };

    class RecordedDepthProvider : public IDepthProvider {
    public:
        explicit RecordedDepthProvider(std::string directory);  // 없거나 깨졌으면 throw
        const CameraIntrinsics &Intrinsics() const override;
        bool Grab(DepthFrame &out) override;   // 마지막 프레임 뒤에는 false
        int FrameCount() const;
    };
}
```

- [ ] **Step 1: 실패하는 테스트 작성**

```cpp
#include "Pipeline/Reconstruction/DepthRecording.h"

#include <filesystem>

namespace {
    // A provider that hands out a fixed number of synthetic frames -- stands in for the device so
    // the round-trip test needs no hardware.
    class FakeDepthProvider : public Pipeline::IDepthProvider {
    public:
        FakeDepthProvider(int frames, Pipeline::CameraIntrinsics k) : m_left(frames), m_k(k) {}
        const Pipeline::CameraIntrinsics &Intrinsics() const override { return m_k; }
        bool Grab(Pipeline::DepthFrame &out) override {
            if (m_left-- <= 0) return false;
            out.depth.assign(std::size_t(m_k.width) * m_k.height, 0.0f);
            for (std::size_t i = 0; i < out.depth.size(); ++i)
                out.depth[i] = 1.0f + 0.001f * float(i % 97) + 0.01f * float(m_left);
            return true;
        }
    private:
        int m_left;
        Pipeline::CameraIntrinsics m_k;
    };
} // namespace

// Recording stores the RAW depth, before back-projection, so replay runs the same normal
// estimation the device path does. A lossy round trip would silently change every reconstruction
// made from a recording.
TEST(DepthFrontend, RecordingRoundTripsBitExact) {
    const std::filesystem::path dir =
            std::filesystem::temp_directory_path() / "vkbvh_depth_roundtrip";
    std::filesystem::remove_all(dir);

    const Pipeline::CameraIntrinsics k = MakeIntrinsics(16, 12);
    std::vector<Pipeline::DepthFrame> written;
    {
        Pipeline::DepthRecorder recorder(std::make_unique<FakeDepthProvider>(3, k), dir.string());
        Pipeline::DepthFrame frame;
        while (recorder.Grab(frame)) written.push_back(frame);
        EXPECT_EQ(recorder.RecordedFrameCount(), 3);
    }

    Pipeline::RecordedDepthProvider replay(dir.string());
    EXPECT_EQ(replay.FrameCount(), 3);
    EXPECT_EQ(replay.Intrinsics().width, k.width);
    EXPECT_FLOAT_EQ(replay.Intrinsics().fx, k.fx);

    std::size_t read = 0;
    Pipeline::DepthFrame frame;
    while (replay.Grab(frame)) {
        ASSERT_LT(read, written.size());
        EXPECT_EQ(frame.depth, written[read].depth) << "frame " << read << " changed on round trip";
        ++read;
    }
    EXPECT_EQ(read, written.size());
    std::filesystem::remove_all(dir);
}

TEST(DepthFrontend, RecordedProviderRejectsAMissingDirectory) {
    EXPECT_THROW(Pipeline::RecordedDepthProvider("/no/such/recording"), std::runtime_error);
}
```

- [ ] **Step 2: 실패 확인** — `DepthRecording.h` 없음으로 컴파일 실패.

- [ ] **Step 3: 구현**

`DepthRecorder::Grab`는 원본 `Grab`를 호출하고, 첫 프레임에서 디렉터리를 만들고 `intrinsics.txt`를 쓴 뒤, 프레임마다 `depth_%04d.bin`에 `depth.data()`를 그대로 `write`한다. `RecordedDepthProvider`는 생성자에서 `intrinsics.txt`를 읽고 `depth_*.bin`을 정렬해 목록을 만든다. 파일 크기가 `width*height*4`와 다르면 throw — 조용히 잘린 녹화가 재구성을 망치는 것보다 낫다.

- [ ] **Step 4: 통과 확인 후 커밋**

```bash
git add src/Pipeline/Reconstruction/DepthRecording.h src/Pipeline/Reconstruction/DepthRecording.cpp test/test_depthFrontend.cpp
git commit -m "feat(pipeline): record and replay raw depth frames"
```

---

### Task 3: RealSense provider와 캡처 도구

**Files:**
- Create: `src/Pipeline/Reconstruction/RealSenseDepthProvider.h`, `.cpp`
- Modify: `src/Pipeline/CMakeLists.txt`
- Create: `example2/depth_capture.cpp`
- Modify: `example2/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 1·2의 전부
- Produces: `class RealSenseDepthProvider : public IDepthProvider` — 생성자에서 `rs2::pipeline`을 시작하고, 장치가 없으면 `std::runtime_error`.

- [ ] **Step 1: CMake 선택적 탐색**

`src/Pipeline/CMakeLists.txt`:

```cmake
# Optional: the RealSense provider compiles only where librealsense2 is installed, so the build
# works unchanged on a machine without the SDK (or without the camera).
find_package(realsense2 QUIET)
if (realsense2_FOUND)
    target_link_libraries(EnginePipeline PUBLIC realsense2::realsense2)
    target_compile_definitions(EnginePipeline PUBLIC VKBVH_HAS_REALSENSE)
    message(STATUS "librealsense2 found -- RealSenseDepthProvider enabled")
else ()
    message(STATUS "librealsense2 not found -- RealSenseDepthProvider disabled")
endif ()
```

GLOB이 `.cpp`를 전부 걷으므로, `RealSenseDepthProvider.cpp`는 파일 전체를 `#ifdef VKBVH_HAS_REALSENSE`로 감싼다.

- [ ] **Step 2: provider 구현**

```cpp
// D435 defaults: 848x480 depth @ 30 fps. The depth scale comes from the device -- it varies by
// model and firmware, so hard-coding 0.001 silently mis-scales the whole reconstruction.
RealSenseDepthProvider::RealSenseDepthProvider(int width, int height, int fps) {
    rs2::config config;
    config.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);
    rs2::pipeline_profile profile = m_pipeline.start(config);   // throws rs2::error if no device
    m_depthScale = profile.get_device().first<rs2::depth_sensor>().get_depth_scale();

    const rs2_intrinsics intrinsics =
            profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();
    m_intrinsics = {intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy,
                    intrinsics.width, intrinsics.height};
}

bool RealSenseDepthProvider::Grab(DepthFrame &out) {
    rs2::frameset frames;
    if (!m_pipeline.try_wait_for_frames(&frames, 1000)) return false;
    const rs2::depth_frame depth = frames.get_depth_frame();
    const uint16_t *raw = static_cast<const uint16_t *>(depth.get_data());
    const std::size_t count = std::size_t(m_intrinsics.width) * m_intrinsics.height;
    out.depth.resize(count);
    for (std::size_t i = 0; i < count; ++i) out.depth[i] = float(raw[i]) * m_depthScale;
    return true;
}
```

생성자에서 `rs2::error`를 잡아 `std::runtime_error("RealSenseDepthProvider: ...")`로 다시 던질 것 — 호출자가 rs2 예외 타입을 알 필요가 없다.

- [ ] **Step 3: 캡처 도구**

`example2/depth_capture.cpp`:

```
depth_capture --record <dir> [--frames N] [--width 848] [--height 480] [--fps 30]
depth_capture --replay <dir>            녹화를 읽어 프레임/점/법선 수를 출력
```

`--replay`는 하드웨어 없이 동작해야 한다 — 녹화 형식과 역투영을 눈으로 확인하는 경로다. `--record`는 `VKBVH_HAS_REALSENSE`가 없으면 사유를 출력하고 종료.

- [ ] **Step 4: 검증**

```bash
cmake -S . -B build-rel && cmake --build build-rel -j8
./build-rel/test/vkspatial_tests            # 209 passed 예상
./build-rel/example2/depth_capture --replay <task2가 만든 디렉터리>
```

장치가 붙어 있으면 `--record`도 수동으로 확인. 없으면 그 사실을 보고서에 적을 것 — 하드웨어 경로는 **검증되지 않았다고 명시**한다.

- [ ] **Step 5: 커밋**

```bash
git add src/Pipeline/Reconstruction/RealSenseDepthProvider.h src/Pipeline/Reconstruction/RealSenseDepthProvider.cpp \
        src/Pipeline/CMakeLists.txt example2/depth_capture.cpp example2/CMakeLists.txt
git commit -m "feat(pipeline): RealSense D435 depth provider and capture tool"
```

---

## 후속으로 넘기는 것

| 항목 | 이유 |
|---|---|
| `ReconstructionThread`에 DepthCamera 배선 | 지금은 provider 주입 경로가 없다. 별도 작업 |
| 컬러/텍스처 | `VoxelAttribute`가 빈 구조체 |
| PoseGraph/LoopClosure 연결 | 다음 스펙 |
| TUM RGB-D 로더 (ATE 측정) | 다음 스펙 |
