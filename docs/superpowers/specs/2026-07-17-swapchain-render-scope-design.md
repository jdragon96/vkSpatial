# SwapchainColorDepthScope 설계 (Execute() 보일러플레이트 정리)

## 배경 / 목적

`example/cube_render.cpp`의 `CubePass::Execute()`가 스왑체인 컬러 이미지 + 뎁스 이미지를 다루는 저수준 Vulkan 배리어 코드와 실제 그리기 로직이 뒤섞여 있어 읽기 어렵다. 조사해보니 이 보일러플레이트(컬러 배리어 `UNDEFINED→COLOR_ATTACHMENT_OPTIMAL`, 뎁스 배리어 `→DEPTH_ATTACHMENT_OPTIMAL`, `RenderingScope` 열기, 그리기 후 컬러 배리어 `→PRESENT_SRC_KHR`로 복귀)가 `example/realtime_shadow.cpp`의 `RealtimeShadowPass::RecordScenePass`에도 스테이지/액세스 플래그까지 동일하게 중복되어 있다 — `vkRender`가 이 공통 패턴을 위한 헬퍼를 제공하지 않아서 각 예제가 직접 손으로 짜고 있다.

조사 중 `CubePass::Execute()`에서 별개의 잠재 버그도 발견했다: `RenderingScope`를 컬러/뎁스 배리어보다 **먼저** 열고 있다(`cube_render.cpp:71` 배리어보다 앞섬). Vulkan dynamic rendering에서 `vkCmdBeginRendering`은 attachment의 `imageLayout` 필드가 선언한 레이아웃으로 이미지가 이미 전환되어 있을 것을 기대하는데, 지금 코드는 그 전환을 렌더링 스코프가 열린 *이후에* 수행한다. `RealtimeShadowPass::RecordScenePass`는 이미 올바른 순서(배리어 먼저, 스코프는 그다음)를 쓰고 있다. 새 헬퍼는 이 올바른 순서를 강제하는 형태로 만들어, 이 부수적 버그도 같이 고친다.

## 범위

- `vkRender::SwapchainColorDepthScope`를 `src/vkRender/Rendering.h`/`.cpp`에 추가 (기존 `RenderingScope`/`RenderingDescriptor`와 같은 파일 — 서로 밀접하게 연관된 개념이라 함께 둔다).
- `example/cube_render.cpp`의 `CubePass::Execute()`를 이 새 타입을 쓰도록 리팩터링.

### 범위 밖

- `example/realtime_shadow.cpp`는 건드리지 않는다. `RecordScenePass`는 캡처 요청 시 `rendering.End()`를 명시적으로 호출한 뒤 추가 커맨드(버퍼 카피)를 기록하고, `depthAttachment.storeOp`를 `DONT_CARE`로 오버라이드하는 등 CubePass와 다른 분기가 있어 이번 범위에서 안전하게 통합하기 어렵다. 이 파일로의 적용은 별도 작업으로 남긴다.
- 조사 중 발견한, `RealtimeShadowPass`의 캡처 경로가 `PRESENT_SRC_KHR`로 최종 전환하지 않는 것으로 보이는 버그는 이번 작업과 무관하므로 고치지 않는다 (사용자에게 별도로 알림).
- 기존 `RenderingScope`/`RenderingDescriptor`의 동작이나 시그니처는 변경하지 않는다 — 새 타입은 그 위에 얹히는 상위 계층이다.

## 아키텍처

### `vkRender::SwapchainColorDepthScope` (신규)

```cpp
// Rendering.h에 추가
class SwapchainColorDepthScope {
public:
    // swapImage를 UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL로, depthTarget을
    // -> DEPTH_ATTACHMENT_OPTIMAL로 전환한 뒤(이 순서가 중요 — RenderingScope를
    // 열기 전에 완료되어야 한다), RenderingDescriptor::ColorDepth(...)로
    // RenderingScope를 연다.
    SwapchainColorDepthScope(VkCommandBuffer commandBuffer,
                             VkImage swapImage,
                             VkImageView swapImageView,
                             Image &depthTarget,
                             VkExtent2D extent,
                             const ClearOptions &clear);

    // RenderingScope를 닫고(vkCmdEndRendering), swapImage를
    // COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR로 되돌린다.
    ~SwapchainColorDepthScope();

    SwapchainColorDepthScope(const SwapchainColorDepthScope &) = delete;
    SwapchainColorDepthScope &operator=(const SwapchainColorDepthScope &) = delete;

private:
    static RenderingDescriptor BeginTransitionsAndBuildDescriptor(
            VkCommandBuffer commandBuffer,
            VkImage swapImage,
            VkImageView swapImageView,
            Image &depthTarget,
            VkExtent2D extent,
            const ClearOptions &clear);

    VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
    VkImage m_swapImage = VK_NULL_HANDLE;
    RenderingScope m_scope;   // 초기화 리스트에서 BeginTransitionsAndBuildDescriptor(...)의 반환값으로 구성 — 아래 "멤버 초기화 순서 문제" 참고
};
```

- `depthTarget`은 `vkRender::Image&`이므로 인스턴스 `TransitionLayout(...)`(레이아웃 추적됨)을 쓰고, `swapImage`는 스왑체인의 raw `VkImage`라 정적 `Image::TransitionLayout(cmd, image, ...)` 오버로드를 쓴다 — 지금 `CubePass`/`RealtimeShadowPass`가 이미 쓰고 있는 것과 동일한 두 API를 그대로 재사용한다.
- 배리어 파라미터(파이프라인 스테이지/액세스 마스크)는 `CubePass::Execute()`(cube_render.cpp:73-88, 92-100)에 있는 값을 그대로 옮긴다 — `RealtimeShadowPass::RecordScenePass`의 해당 배리어와 이미 동일한 값이므로 새로 발명할 게 없다.
- **멤버 초기화 순서 문제와 해결**: `m_scope`는 선언 순서상 `SwapchainColorDepthScope`의 생성자 *본문*이 실행되기 전에 구성된다. 그런데 `RenderingScope`의 생성자는 `vkCmdBeginRendering`을 호출하므로, 컬러/뎁스 배리어가 그보다 먼저 기록되어 있어야 한다. 생성자 본문에서 배리어를 기록하면 이미 늦다. 해결: private `static` 헬퍼 `BeginTransitionsAndBuildDescriptor(...)`를 만들어 그 안에서 배리어 두 개를 기록한 뒤 `RenderingDescriptor`를 반환하게 하고, `m_scope`를 초기화 리스트에서 `m_scope(commandBuffer, BeginTransitionsAndBuildDescriptor(...))`처럼 그 헬퍼의 반환값으로 직접 구성한다. 함수 인자는 그 인자를 사용하는 멤버가 구성되기 전에 평가되므로, 배리어 기록 → `RenderingDescriptor` 반환 → 그 값으로 `m_scope` 생성자 호출(`vkCmdBeginRendering`) 순서가 보장된다. `RenderingScope`는 복사도 이동도 안 되는 타입이지만(복사 생성자가 `= delete`로 선언되어 있어 암시적 이동 생성자도 만들어지지 않음), 이 패턴은 `RenderingScope` 자체를 함수에서 반환하지 않고 초기화 리스트에서 직접 생성하므로 문제없다.

### `CubePass::Execute()` 리팩터링

```cpp
void Execute(vkRender::RenderContext &renderContext) override {
    if (!renderContext.swapChain)
        throw std::runtime_error("CubePass requires an active swapchain");

    const VkExtent2D extent = renderContext.swapChain->Extent();
    EnsureDepthResources(extent);
    EnsurePipeline(renderContext.swapChain->Format());

    VkCommandBuffer cmd = renderContext.commandBuffer;
    VkImage swapImage = renderContext.swapChain->Image(renderContext.imageIndex);
    const vkRender::ClearOptions clear =
            renderContext.view ? renderContext.view->GetClearOptions() : vkRender::ClearOptions{};

    {
        vkRender::SwapchainColorDepthScope scope(
                cmd, swapImage, renderContext.swapChain->ImageView(renderContext.imageIndex),
                *m_depthTarget, extent, clear);
        DrawObjects(cmd, extent);
    }
}
```

`Execute()`가 "리소스 준비 → 스코프 열고 그리기 → (자동으로 정리)"만 남아 보일러플레이트와 실제 로직이 분리된다.

## 파일 변경 목록

| 파일 | 변경 |
|---|---|
| `src/vkRender/Rendering.h` | `SwapchainColorDepthScope` 클래스 선언 추가 |
| `src/vkRender/Rendering.cpp` | `SwapchainColorDepthScope` 구현 추가 |
| `example/cube_render.cpp` | `CubePass::Execute()`를 새 타입 사용하도록 리팩터링 (동작 변화 없음) |

CMake 변경 불필요 — 기존 파일 수정이라 `vkRender` 타겟의 `GLOB_RECURSE`에 이미 포함됨.

## 테스트/검증 방법

이 프로젝트의 기존 관례(Vulkan 캡처/렌더링은 실제 GPU+창 컨텍스트가 필요해 유닛 테스트로 격리하기 어려움)를 따라 실행으로 검증한다.

1. `cube_render` 빌드 및 실행 — 리팩터링 전과 시각적으로 동일하게(회전하는 큐브, 배경색, 뎁스 테스트 정상) 렌더링되는지 확인.
2. Enter 키로 스크린샷을 캡처해 `screenshots/*.png`가 정상 생성되는지 확인 (스왑체인 이미지가 캡처 시점에 올바르게 `PRESENT_SRC_KHR` 상태인지 간접 검증).
3. 창 리사이즈 후에도 크래시 없이 정상 렌더링되는지 확인 (`EnsureDepthResources`/`EnsurePipeline`의 재생성 경로는 이번 변경과 무관하지만 회귀 확인 차원).
4. (있다면) 검증 레이어를 켠 빌드로 실행해, 레이아웃 전환 순서가 고쳐지면서 기존에 있었을 수 있는 `vkCmdBeginRendering` 관련 검증 경고가 사라지는지 확인 — 이 프로젝트는 기본적으로 검증 레이어를 켜지 않으므로 필수는 아니고, 가능하면 확인.
