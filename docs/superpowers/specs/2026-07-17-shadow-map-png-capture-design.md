# Shadow Map PNG 캡처 기능 설계

## 배경 / 목적

`example/realtime_shadow.cpp`는 이미 Enter 키로 최종 화면(스왑체인 컬러 이미지)을 캡처해 `.ppm`으로 저장하는 기능을 갖고 있다 (`RealtimeShadowPass::RequestCapture`/`SaveCapturedImage`). 여기에 더해, 빛의 시점에서 렌더링되는 셰도우맵 자체(`m_shadowTarget`, `D32_SFLOAT` depth 이미지)를 `1`번 키로 PNG 이미지로 내보내, 셰도우맵이 실제로 어떻게 그려지고 있는지 육안으로 확인할 수 있게 한다.

기존 `vkRender::Capture::CaptureToPNG`는 8비트 컬러 포맷(`B8G8R8A8`/`R8G8B8A8`)만 지원하므로, `D32_SFLOAT` depth 이미지에는 그대로 쓸 수 없다. depth 값을 grayscale로 변환하는 새 경로가 필요하다.

## 범위

- `example/realtime_shadow.cpp`의 `RealtimeShadowPass`에 셰도우맵 캡처 요청/저장 기능 추가
- `vkRender::Capture`에 grayscale PNG 저장용 재사용 가능한 헬퍼(`WriteGrayscalePNG`) 추가
- `vkRender::Capture::TimestampedPath`에 파일명 프리픽스 파라미터 추가 (기본값으로 기존 동작 유지)
- `1`번 키(`GLFW_KEY_1`) 입력 리스너 연결

### 범위 밖

- 다른 예제에 동일 기능 적용 (`realtime_shadow.cpp`만 대상)
- 셰도우맵 캡처를 스왑체인 캡처처럼 매 프레임 크기가 바뀔 수 있는 것으로 다루는 것 — 셰도우맵은 `kShadowMapSize`(2048) 고정이므로 스왑체인 캡처(`EnsureCaptureBuffer`)처럼 크기 재확인 로직 불필요
- depth 값을 실제 월드 공간 거리(미터 등)로 환산해 보여주는 것 — 이번 스펙은 육안 확인용 정규화된 grayscale만 다룬다

## 핵심 제약: `Image`의 레이아웃 추적과 raw 캡처 헬퍼의 불일치

`vkRender::Image`(`m_shadowTarget`)는 자신의 현재 레이아웃을 `m_currentLayout`으로 내부 추적하며, `RealtimeShadowPass`는 이미 이 인스턴스 메서드 `TransitionLayout(cmd, newLayout, ...)`만으로 셰도우맵의 레이아웃을 관리한다 (`realtime_shadow.cpp:549-554`, `:581-586`).

기존 `Capture::RecordImageToBufferCopy`(`Capture.h:34-41`)는 raw `VkImage` 핸들을 받아 자체적으로(추적되지 않는) 레이아웃 전환을 두 번 기록한다. 이걸 `m_shadowTarget`에 그대로 쓰면, 그 함수가 실제 GPU 이미지를 원하는 최종 레이아웃으로 전환해도 `Image` 객체 내부의 `m_currentLayout` 필드는 갱신되지 않는다. 그러면 다음 프레임에 `m_shadowTarget->TransitionLayout(...)`을 다시 호출할 때 "이미 그 레이아웃이니 no-op" 같은 잘못된 판단을 하거나, 잘못된 `oldLayout`을 가정한 배리어를 기록해 검증 레이어 에러나 미정의 동작으로 이어질 수 있다.

따라서 셰도우맵 캡처는 `Capture::RecordImageToBufferCopy`를 재사용하지 않고, `m_shadowTarget`의 인스턴스 `TransitionLayout` 호출로 감싼 별도의 복사 경로(`RecordShadowMapCaptureCopy`)를 새로 만든다.

## 아키텍처

### 1. 트리거 & 프레임 내 데이터 흐름

`main()`에 `1`번 키 리스너를 추가한다 (기존 Enter 키 리스너와 동일한 패턴, `realtime_shadow.cpp:776-783` 참고):

```cpp
keyInput->AddListener(vkRender::KeyEventType::Press,
                      [realtimeShadow](vkRender::KeyEvent &event) {
                          if (event.keyCode == GLFW_KEY_1) {
                              realtimeShadow->RequestShadowMapCapture();
                              std::cout << "[realtime_shadow] shadow map capture requested\n";
                          }
                      });
```

`RecordShadowPass()`의 마지막 부분(`realtime_shadow.cpp:581-586`, 셰도우맵을 `SHADER_READ_ONLY_OPTIMAL`로 전환하는 지점)을 분기한다:

- **캡처 요청 없음** (대부분의 프레임): 기존 코드 그대로 — `DEPTH_ATTACHMENT_OPTIMAL` → `SHADER_READ_ONLY_OPTIMAL` 한 번에 전환.
- **캡처 요청 있음**: `RecordShadowMapCaptureCopy(cmd)` 호출:
  1. `m_shadowTarget->TransitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, LATE_FRAGMENT_TESTS, TRANSFER, DEPTH_STENCIL_ATTACHMENT_WRITE, TRANSFER_READ)` — 인스턴스 메서드라 `m_currentLayout` 정상 갱신.
  2. `vkCmdCopyImageToBuffer(cmd, m_shadowTarget->Handle(), TRANSFER_SRC_OPTIMAL, m_shadowCaptureBuffer, 1, &region)` — `region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT`, extent `{kShadowMapSize, kShadowMapSize, 1}`. Raw 호출이라 레이아웃 추적에 영향 없음(레이아웃을 바꾸지 않는 명령이므로 문제 없음).
  3. 버퍼 배리어 (`TRANSFER_WRITE` → `HOST_READ`), `Capture::RecordImageToBufferCopy` 내부의 동일 패턴을 그대로 인라인으로 기록.
  4. `m_shadowTarget->TransitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, TRANSFER, FRAGMENT_SHADER, TRANSFER_READ, SHADER_READ)` — 다시 인스턴스 메서드로 정상 전환. 이후 씬 패스가 정상적으로 셰도우맵을 샘플링한다.
- 복사가 끝나면 `m_shadowCaptureRequested = false; m_shadowCaptureReady = true;`.

메인 루프에서는 기존 컬러 캡처와 동일한 자리(`renderer->EndFrame();` 직후)에 다음을 추가한다:

```cpp
if (realtimeShadow->HasShadowMapCaptureToSave()) {
    vkDeviceWaitIdle(context.device);
    const std::string filename = realtimeShadow->SaveCapturedShadowMap();
    std::cout << "[realtime_shadow] saved shadow map " << filename << "\n";
}
```

### 2. `RealtimeShadowPass`에 추가되는 상태/메서드

```cpp
// public
void RequestShadowMapCapture();               // m_shadowCaptureRequested = true
bool HasShadowMapCaptureToSave() const;        // m_shadowCaptureReady
std::string SaveCapturedShadowMap();           // 아래 3번 참고, 저장된 경로 반환

// private
void CreateShadowCaptureBuffer();              // CreateShadowResources()에서 함께 호출, 크기 고정(kShadowMapSize^2 * 4 bytes)
void DestroyShadowCaptureBuffer();              // 소멸자에서 함께 호출
void RecordShadowMapCaptureCopy(VkCommandBuffer cmd);  // 위 1번의 4단계 구현

// private members
VkBuffer m_shadowCaptureBuffer = VK_NULL_HANDLE;
VkDeviceMemory m_shadowCaptureMemory = VK_NULL_HANDLE;
bool m_shadowCaptureRequested = false;
bool m_shadowCaptureReady = false;
```

`m_shadowCaptureBuffer`/`Memory`는 셰도우맵 크기가 고정(`kShadowMapSize x kShadowMapSize`)이므로 컬러 캡처 버퍼(`EnsureCaptureBuffer`, 스왑체인 리사이즈에 따라 재생성)와 달리 `CreateShadowResources()` 시점에 한 번만 만들고 소멸자까지 재사용한다. 메모리 타입 선택은 기존 `FindMemoryType(...)` 멤버 헬퍼(`realtime_shadow.cpp:534-544`)를 그대로 재사용한다.

`CreateShadowResources()`에 `descriptor.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;`를 추가해야 한다 (현재는 `VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT`만 있음, `Image.cpp:8-17`의 `Depth2D` 기본값 + 기존 코드의 `SAMPLED_BIT` 추가분).

### 3. `SaveCapturedShadowMap()` — depth → grayscale PNG

1. `m_shadowCaptureMemory`를 매핑해 `kShadowMapSize * kShadowMapSize`개의 `float` (D32_SFLOAT raw 값)을 읽는다.
2. CPU에서 min/max를 스캔한다.
3. 각 픽셀을 `normalized = (depth - min) / (max - min)`으로 정규화한 뒤, **가까운 표면일수록 밝게** 보이도록 반전해 `uint8_t value = static_cast<uint8_t>((1.0f - normalized) * 255.0f)`로 변환한다. (`max == min`인 극단적 경우, 즉 셰도우맵에 아무것도 안 찍힌 프레임은 전부 동일 값으로 채워지는 것을 허용 — 별도 예외 처리 불필요.)
4. `vkRender::Capture::WriteGrayscalePNG(path, kShadowMapSize, kShadowMapSize, grayscalePixels)` 호출.
5. `path`는 `vkRender::Capture::TimestampedPath("screenshots", "shadow_map")` (아래 4번 참고)로 생성.
6. `m_shadowCaptureReady = false;`로 리셋 후 `path` 반환.

### 4. `vkRender::Capture`에 추가되는 API

```cpp
// Capture.h
static void WriteGrayscalePNG(const std::string &path,
                              uint32_t width,
                              uint32_t height,
                              const std::vector<uint8_t> &grayscalePixels);

static std::string TimestampedPath(const std::string &directory = "screenshots",
                                   const std::string &prefix = "screenshot");
```

- `WriteGrayscalePNG`는 `stbi_write_png(path.c_str(), width, height, /*channels=*/1, pixels.data(), width)`를 감싼 얇은 래퍼. 실패 시 기존 관례대로 `std::runtime_error`. `Capture.cpp`에 이미 `stb_image_write.h`가 include되어 있으므로 새 include 불필요.
- `TimestampedPath`에 `prefix` 파라미터를 추가하되 기본값을 `"screenshot"`으로 유지해 기존 호출부(`cube_render.cpp` 등)는 변경 없이 그대로 동작한다. 파일명 포맷은 `"<prefix>_YYYYMMDD_HHMMSS.png"`로, 기존 로직에서 하드코딩된 `"screenshot_"` 리터럴만 파라미터로 대체.

이 두 API는 셰도우맵 전용이 아니라 `Capture` 클래스의 범용 유틸리티로 추가하는 것이므로, 향후 다른 depth/1채널 디버그 덤프에도 재사용 가능하다.

## 파일 변경 목록

| 파일 | 변경 |
|---|---|
| `src/vkRender/Capture.h` | `WriteGrayscalePNG` 선언 추가, `TimestampedPath`에 `prefix` 파라미터 추가 |
| `src/vkRender/Capture.cpp` | `WriteGrayscalePNG` 구현, `TimestampedPath` 파일명 조립 로직에 `prefix` 반영 |
| `example/realtime_shadow.cpp` | `RealtimeShadowPass`에 셰도우맵 캡처 상태/메서드 추가, `CreateShadowResources`에 `TRANSFER_SRC_BIT` 추가, `RecordShadowPass` 분기, `main()`에 `1`번 키 리스너 및 저장 트리거 추가 |

CMake 변경 불필요 (`src/CMakeLists.txt`가 `vkRender/*.cpp`를 `GLOB_RECURSE`로 자동 수집, `example/CMakeLists.txt`의 `realtime_shadow` 타겟은 기존 파일 수정이라 타겟 정의 변경 없음).

## 에러 처리

기존 코드베이스 관례(`std::runtime_error` + `throw`)를 그대로 따른다. `WriteGrayscalePNG`는 `stbi_write_png` 실패 시 예외를 던지고, `main()`의 기존 `try/catch`가 처리한다.

## 테스트/검증 방법

Vulkan 리소스가 필요해 유닛 테스트로 격리하기 어려우므로 실제 실행으로 검증한다.

1. `realtime_shadow` 빌드 및 실행.
2. `1`번 키를 눌러 `screenshots/shadow_map_*.png` 파일이 생성되는지 확인.
3. 생성된 PNG를 열어 큐브/바닥의 그림자 윤곽이 grayscale로 잘 드러나는지(가까운 표면이 밝게) 육안 확인.
4. `1`번과 Enter를 같은 세션에서 번갈아 눌러도 서로 간섭 없이(각자 다른 파일로) 정상 저장되는지 확인.
5. 셰도우맵 갱신이 매 프레임 일어나므로, 카메라/큐브 애니메이션이 진행 중인 임의의 시점에 캡처해도 크래시 없이 저장되는지 확인.
