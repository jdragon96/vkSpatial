# Enter 키 스크린샷 캡처 기능 설계

## 배경 / 목적

`example/cube_render.cpp`에서 Enter 키를 누르면 현재 화면을 PNG로 저장하고 싶다. 요구사항은 두 가지다.

1. 키 입력 이벤트가 `vkRender` 레벨에서 리스너 구조로 흘러오도록 한다 (예제 코드에서 GLFW 콜백을 직접 처리하지 않고, 엔진이 제공하는 이벤트 객체를 구독하는 형태).
2. 현재 스왑체인 화면을 PNG 파일로 저장하는 기능을 추가한다.

`vkRender`에는 이미 `MouseInput`([MouseInput.h](../../../src/vkRender/MouseInput.h))이라는 리스너 기반 이벤트 컴포넌트가 존재하고, `Engine::CreateMouseInput()`으로 생성한다. 키보드 입력에 대한 동일한 컴포넌트는 아직 없다. 스크린샷 저장을 위한 PNG 인코더나 이미지 리드백 유틸리티도 현재 코드베이스에 없다.

## 범위

- `vkRender::KeyInput` 컴포넌트 신규 추가 (키보드 이벤트 리스너 구조)
- `vkRender::Screenshot` 유틸리티 신규 추가 (Vulkan 이미지 → PNG 파일)
- `example/cube_render.cpp`에 위 두 컴포넌트를 연결해 Enter 키 캡처 동작 구현
- PNG 인코딩을 위해 `stb_image_write.h`를 `lib/stb/`에 벤더링

### 범위 밖

- 다른 예제(`bvh_path_tracer.cpp` 등)에 키 입력/스크린샷 적용 — 이번 스펙에서는 `cube_render.cpp`만 대상으로 한다.
- 논블로킹/더블 버퍼링 캡처, 여러 프레임 연속 캡처(GIF 등), JPG/다른 포맷 지원.
- 오프스크린 렌더 타깃 캡처 (스왑체인 이미지 캡처만 다룬다). `Screenshot::CaptureToPNG`의 시그니처는 임의의 `VkImage`를 받으므로 향후 재사용 가능하지만, 이번 구현·테스트 대상은 스왑체인 이미지로 한정한다.

## 아키텍처

### 1. `vkRender::KeyInput` (신규: `src/vkRender/KeyInput.h` / `.cpp`)

`MouseInput`과 동일한 리스너/디스패치 패턴을 따른다.

```cpp
enum class KeyEventType : uint32_t { Any = 0, Press, Release, Repeat };

struct KeyEvent {
    KeyEventType type = KeyEventType::Press;
    int keyCode = 0;              // GLFW_KEY_* 값 그대로 전달 (vkRender는 GLFW 헤더에 의존하지 않지만
                                   // 값 규약은 기존 MouseModifierBits가 GLFW_MOD_*와 일치하는 것과 동일한 관례를 따름)
    uint32_t modifiers = 0;        // MouseModifierBits 비트값 재사용 (Shift/Control/Alt/Super)
    double timestampSeconds = 0.0;
    bool handled = false;
};

class KeyInput {
public:
    using ListenerId = uint64_t;
    using Callback = std::function<void(KeyEvent &)>;

    ListenerId AddListener(KeyEventType type, Callback callback, int priority = 0);
    bool RemoveListener(ListenerId id);
    void ClearListeners();

    void OnKey(int keyCode, KeyEventType type, uint32_t modifiers = 0, double timestampSeconds = 0.0);

    bool IsKeyDown(int keyCode) const;

private:
    std::unordered_map<int, bool> m_keyStates;   // 키 코드 공간이 넓고 sparse하므로 배열 대신 map 사용
    // ... Listener 저장/디스패치는 MouseInput과 동일한 구조
};

class KeyListenerGroup { /* MouseListenerGroup과 동일한 RAII 그룹 */ };
```

- `Engine::CreateKeyInput() const` 추가 (`Engine::CreateMouseInput()`과 동일 패턴, `Engine.h`/`Engine.cpp`에 추가).
- 디스패치 규칙(우선순위 정렬, `handled` 시 중단)은 `MouseInput::Dispatch`와 동일하게 구현한다.

### 2. `vkRender::Screenshot` (신규: `src/vkRender/Screenshot.h` / `.cpp`)

`Renderer`/`SwapChain` 내부에 의존하지 않는 독립 유틸리티. 임의의 `VkImage` + 포맷 + extent를 받아 PNG로 저장한다.

```cpp
class Screenshot {
public:
    // image는 호출 시점에 currentLayout 상태여야 한다 (예: VK_IMAGE_LAYOUT_PRESENT_SRC_KHR).
    // image는 VK_IMAGE_USAGE_TRANSFER_SRC_BIT로 생성되어 있어야 한다.
    // 내부에서 자체 커맨드 풀/버퍼를 생성해 동기적으로(blocking) 캡처하고,
    // 완료 후 image의 레이아웃을 currentLayout으로 복원한다.
    static void CaptureToPNG(vkCommon::VkContext *context,
                             VkImage image,
                             VkFormat format,
                             VkExtent2D extent,
                             VkImageLayout currentLayout,
                             const std::string &path);

    // "<directory>/screenshot_YYYYMMDD_HHMMSS.png" 형태의 경로를 만들고,
    // <directory>가 없으면 생성한다.
    static std::string TimestampedPath(const std::string &directory = "screenshots");
};
```

**`CaptureToPNG` 동작 순서** (Sascha Willems Vulkan 예제의 표준 present-후-캡처 패턴을 따름):

1. `vkQueueWaitIdle(context->graphicsQueue)` — 직전에 제출된 프레임(및 present)이 완전히 끝났음을 보장.
2. 지원 포맷 확인: `VK_FORMAT_B8G8R8A8_UNORM/SRGB`, `VK_FORMAT_R8G8B8A8_UNORM/SRGB`만 지원. 그 외 포맷이면 `std::runtime_error`.
3. 호스트 가시 스테이징 버퍼(`width * height * 4` 바이트) 생성.
4. 임시 커맨드 풀 + 1회용 커맨드 버퍼 생성 후:
   - 이미지 배리어: `currentLayout` → `VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL`
   - `vkCmdCopyImageToBuffer` (전체 extent, `VK_IMAGE_ASPECT_COLOR_BIT`)
   - 이미지 배리어: `VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL` → `currentLayout` (원래 상태로 복원)
5. 커맨드 버퍼 제출 후 `vkQueueWaitIdle`로 완료 대기.
6. 스테이징 메모리 매핑 → CPU 버퍼로 복사. 포맷이 BGRA 계열이면 R/B 채널을 스왑.
7. `stbi_write_png(path.c_str(), width, height, 4, pixels.data(), width * 4)` 호출, 실패 시 `std::runtime_error`.
8. 스테이징 버퍼/메모리, 커맨드 버퍼/풀 정리.

**PNG 인코더**: `stb_image_write.h`(public domain, single header)를 `lib/stb/`에 벤더링한다. `Screenshot.cpp`에서만 `#define STB_IMAGE_WRITE_IMPLEMENTATION` 후 include한다. `${CMAKE_SOURCE_DIR}/lib`가 이미 `vkRender` 타깃의 `PUBLIC` include 경로에 포함되어 있으므로(`src/CMakeLists.txt:61-64`) 별도 CMake 수정 없이 `#include "stb/stb_image_write.h"`로 사용 가능하다.

**`TimestampedPath`**: `<filesystem>`으로 디렉토리 생성, `<chrono>`/`<ctime>`으로 `screenshot_YYYYMMDD_HHMMSS.png` 형식의 파일명을 만든다.

### 3. `example/cube_render.cpp` 연결

- `vkRender::SwapChainDescriptor`에 `VK_IMAGE_USAGE_TRANSFER_SRC_BIT` 추가:
  ```cpp
  swapChainDescriptor.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  ```
  (표면이 이 usage를 지원하지 않으면 `SwapChain::Create`가 이미 명확한 예외를 던진다 — `SwapChain.cpp:123-124`.)
- `auto keyInput = engine->CreateKeyInput();` 생성.
- `glfwSetWindowUserPointer(window, keyInput.get());` 후 `glfwSetKeyCallback`으로 `KeyInput::OnKey` 연결 (`bvh_path_tracer.cpp`가 `MouseInput`을 연결하는 방식과 동일 패턴).
- Enter 키(`GLFW_KEY_ENTER`) `Press` 이벤트 리스너 등록 → `bool captureRequested` 플래그를 세팅.
- 메인 루프에서 `renderer->EndFrame();` 직후, 플래그가 서 있으면:
  ```cpp
  if (captureRequested) {
      captureRequested = false;
      vkRender::Screenshot::CaptureToPNG(
          &context,
          swapChain->Image(renderer->CurrentImageIndex()),
          swapChain->Format(),
          swapChain->Extent(),
          VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
          vkRender::Screenshot::TimestampedPath());
  }
  ```
  (`Renderer::CurrentImageIndex()`는 `EndFrame()` 이후에도 마지막 사용 인덱스를 그대로 반환하므로 이 시점에 호출 가능 — `Renderer.h:35`, `Renderer.cpp`의 `EndFrame()`은 `m_imageIndex`를 리셋하지 않음.)

## 파일 변경 목록

| 파일 | 변경 |
|---|---|
| `src/vkRender/KeyInput.h` (신규) | `KeyEvent`, `KeyEventType`, `KeyInput`, `KeyListenerGroup` 정의 |
| `src/vkRender/KeyInput.cpp` (신규) | `KeyInput`/`KeyListenerGroup` 구현 |
| `src/vkRender/Screenshot.h` (신규) | `Screenshot` 클래스 정의 |
| `src/vkRender/Screenshot.cpp` (신규) | 캡처/저장 구현, `stb_image_write.h` include |
| `lib/stb/stb_image_write.h` (신규, 벤더링) | 외부 라이브러리 |
| `src/vkRender/Engine.h` / `.cpp` | `CreateKeyInput()` 추가 |
| `src/vkRender/vkRender.h` | `KeyInput.h`, `Screenshot.h` 포함 (마스터 헤더에 등록되어 있는 경우) |
| `example/cube_render.cpp` | 키 콜백 wiring, Enter 캡처 로직, `imageUsage`에 `TRANSFER_SRC_BIT` 추가 |

`src/CMakeLists.txt`는 `file(GLOB_RECURSE VKRENDER_SOURCES ...)`로 `vkRender/*.cpp`를 자동 수집하므로 신규 `.cpp` 파일에 대한 CMake 수정은 필요 없다.

## 에러 처리

- 기존 코드베이스 관례(`std::runtime_error` + `throw`)를 그대로 따른다.
- `Screenshot::CaptureToPNG`는 지원하지 않는 포맷, 스테이징 버퍼 생성 실패, PNG 쓰기 실패 시 예외를 던진다.
- 캡처 실패가 렌더 루프 자체를 중단시키는 것이 맞는지에 대한 특별 처리는 하지 않는다 — 기존 `main()`의 `try/catch`가 이미 모든 예외를 잡아 정리 후 종료하므로 별도 처리 불필요.

## 테스트/검증 방법

자동화된 유닛 테스트보다는 실제 실행으로 검증한다 (Vulkan 리소스 캡처는 GPU 컨텍스트가 필요해 단위 테스트로 격리하기 어려움).

1. `cube_render` 빌드 및 실행.
2. Enter 키를 눌러 `screenshots/screenshot_*.png` 파일이 생성되는지 확인.
3. 생성된 PNG를 열어 현재 화면(회전하는 큐브)과 색상이 일치하는지(BGRA 스왑 정상 여부) 육안 확인.
4. 창 리사이즈 직후 Enter를 눌러도 크래시 없이 새 해상도로 캡처되는지 확인.
5. 연속으로 여러 번 Enter를 눌러 파일명이 겹치지 않는지 확인 (초 단위 타임스탬프이므로 1초 이내 연타 시 덮어쓰기 가능 — 알려진 제한사항으로 문서화, 이번 스펙에서는 해결하지 않음).
