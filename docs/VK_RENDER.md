# vkRender 구조

`vkRender`는 `vkSpatial`과 같은 깊이에 있는 렌더링 계층이다. 목적은 Vulkan 객체를 앱 코드 밖으로 밀어내고, Google Filament처럼 `Engine`을 중심으로 필요한 렌더링 객체를 만들어 쓰는 구조를 제공하는 것이다.

## 계층

```text
Application / GLFW / platform
    |
    |  window, surface, mouse/keyboard callback
    v
vkCommon::VkContext
    |
    |  VkInstance, VkDevice, queue, command pool, surface
    v
vkRender::Engine
    |
    +-- SwapChain
    +-- Renderer
    +-- Scene
    +-- View
    +-- Camera
    +-- RenderGraph
    +-- MouseInput
```

`vkRender`는 windowing 라이브러리를 직접 소유하지 않는다. GLFW, SDL, Win32 같은 플랫폼 계층은 앱이 담당하고, `vkRender`는 이미 만들어진 `vkCommon::VkContext`와 이벤트 값을 받아 동작한다.

## 주요 객체

| 객체 | 역할 |
| --- | --- |
| `Engine` | `vkRender`의 진입점이다. `Renderer`, `SwapChain`, `Scene`, `View`, `Camera`, `RenderGraph`, `MouseInput`을 만든다. |
| `SwapChain` | present 가능한 swapchain 이미지와 image view를 관리한다. 창 크기 변경 시 `Recreate()`로 다시 만든다. |
| `Renderer` | frame begin/end, acquire/present, sync object, command buffer 제출을 담당한다. 현재는 GPU buffer를 swapchain 이미지로 복사하는 경로가 구현되어 있다. |
| `Scene` | 렌더 대상 entity 목록을 담는 컨테이너다. 이후 mesh/material/transform registry와 연결될 자리다. |
| `View` | 어떤 `Scene`을 어떤 `Camera`와 viewport/render graph로 그릴지 묶는다. |
| `Camera` | view/projection matrix와 near/far/fov 같은 카메라 상태를 담는다. |
| `RenderGraph` | pass를 순서대로 실행하는 scaffold다. 이후 shadow/depth/opaque/post-process pass 선언 구조로 확장한다. |
| `MouseInput` | 플랫폼 마우스 이벤트를 `vkRender` 공통 이벤트로 변환해 listener에 전달한다. |
| `GraphicsPipeline` | vertex/fragment shader, vertex input, color/depth target, push constant를 선언형 descriptor로 받아 `VkPipeline`을 만든다. |
| `Image` | `VkImage`, `VkDeviceMemory`, `VkImageView`를 하나의 RAII 객체로 묶어 color/depth/shadow target lifetime을 관리한다. |
| `RenderingScope` | dynamic rendering begin/end, color/depth attachment clear, viewport/scissor 설정을 한 번에 처리한다. |

## Graphics Pipeline 추상화

`GraphicsPipelineDescriptor`는 raster pipeline 생성에 필요한 값을 선언형으로 모은다.
예제 코드는 `VertexShader()`, `FragmentShader()`, `ColorTarget()`,
`DepthTarget()`, `PushConstant()`처럼 의도를 적고, `GraphicsPipeline::Build()`가
`VkPipelineShaderStageCreateInfo`, fixed-function state, `VkPipelineLayout`,
`VkGraphicsPipelineCreateInfo` 조립을 담당한다.

자세한 Build 순서와 각 Vulkan 구조체의 이론적 역할은
[`GRAPHICS_PIPELINE_BUILD.md`](GRAPHICS_PIPELINE_BUILD.md)에 정리했다.

## Image 추상화

Vulkan image resource는 보통 `VkImage`, `VkDeviceMemory`, `VkImageView`가 한 세트로
움직인다. `vkRender::ImageDescriptor`는 image 크기, format, usage, aspect mask를
선언하고, `vkRender::Image`는 image 생성, memory 할당/바인딩, image view 생성,
파괴 순서를 RAII로 관리한다.

```cpp
auto depthTarget = std::make_unique<vkRender::Image>(
        &context,
        vkRender::ImageDescriptor::Depth2D(extent, VK_FORMAT_D32_SFLOAT));

VkImage image = depthTarget->Handle();
VkImageView view = depthTarget->View();
```

## Dynamic Rendering 추상화

`RenderingDescriptor`는 render area, color/depth attachment, viewport, scissor를
묶는다. `RenderingScope`는 생성자에서 `vkCmdBeginRendering`,
`vkCmdSetViewport`, `vkCmdSetScissor`를 기록하고, `End()` 또는 소멸자에서
`vkCmdEndRendering`을 기록한다.

```cpp
const vkRender::ClearOptions clear = view->GetClearOptions();
vkRender::RenderingScope rendering(
        cmd,
        vkRender::RenderingDescriptor::ColorDepth(
                extent,
                swapChain.ImageView(imageIndex),
                depthTarget.View(),
                clear));

DrawScene(cmd);
rendering.End();
```

## 기본 렌더링 흐름

현재 구현의 첫 사용 경로는 `vkSpatial` compute/ray tracing 결과를 swapchain에 present하는 방식이다.

```cpp
auto engine = vkRender::Engine::Create(&context);
auto swapChain = engine->CreateSwapChain({width, height});
auto renderer = engine->CreateRenderer();

while (running) {
    if (!renderer->BeginFrame(*swapChain)) {
        vkDeviceWaitIdle(context.device);
        swapChain->Recreate(width, height);
        renderer->ClearSwapChainRecreateFlag();
        continue;
    }

    // vkSpatial 또는 다른 compute pass가 outputBuffer에 RGBA8 결과를 기록한다.
    renderer->CopyBufferToSwapChain(outputBuffer);
    renderer->EndFrame();

    if (renderer->NeedsSwapChainRecreate()) {
        vkDeviceWaitIdle(context.device);
        swapChain->Recreate(width, height);
        renderer->ClearSwapChainRecreateFlag();
    }
}
```

이 흐름에서는 CPU로 이미지를 다시 가져오지 않는다. `vkSpatial::vkWideBVH::TraceRays()`나 path tracing pass가 GPU buffer에 결과를 쓰고, `Renderer`가 그 buffer를 swapchain 이미지로 복사한다.

## 마우스 이벤트 구조

`MouseInput`은 GLFW 같은 플랫폼 이벤트를 직접 의존하지 않는 중간 계층이다. 앱은 플랫폼 콜백에서 좌표, 버튼, modifier, scroll 값을 `MouseInput`에 넣고, 실제 동작은 listener로 등록한다.

지원하는 이벤트 타입은 다음과 같다.

| 이벤트 | 의미 |
| --- | --- |
| `Move` | 커서 이동 |
| `ButtonDown` | 버튼 누름 |
| `ButtonUp` | 버튼 뗌 |
| `DragBegin` | 버튼이 눌린 상태에서 첫 이동 발생 |
| `Drag` | 버튼이 눌린 상태의 이동 |
| `DragEnd` | drag 중 버튼을 뗌 |
| `Scroll` | wheel 또는 trackpad scroll |
| `Enter` | cursor enter |
| `Leave` | cursor leave |
| `Any` | 모든 이벤트를 받는 listener용 타입 |

`MouseEvent`에는 현재 좌표, 이전 좌표, delta, scroll delta, drag 시작 좌표, 눌린 버튼 mask, modifier mask, timestamp가 들어 있다. listener는 이벤트를 처리한 뒤 `event.handled = true`로 설정해 낮은 우선순위 listener로 전파되는 것을 막을 수 있다.

## 플랫폼 콜백 연결

GLFW를 쓰는 앱에서는 window user pointer에 `MouseInput`을 저장하고, GLFW callback에서 공통 이벤트로 넘긴다.

```cpp
auto mouseInput = engine->CreateMouseInput();
glfwSetWindowUserPointer(window, mouseInput.get());

glfwSetCursorPosCallback(window, [](GLFWwindow *window, double x, double y) {
    auto *mouse = static_cast<vkRender::MouseInput *>(glfwGetWindowUserPointer(window));
    if (!mouse)
        return;

    mouse->OnMouseMove(x, y, QueryMouseModifiers(window), glfwGetTime());
});

glfwSetMouseButtonCallback(window, [](GLFWwindow *window, int button, int action, int mods) {
    auto *mouse = static_cast<vkRender::MouseInput *>(glfwGetWindowUserPointer(window));
    if (!mouse)
        return;

    double x = 0.0;
    double y = 0.0;
    glfwGetCursorPos(window, &x, &y);
    mouse->OnButton(ToMouseButton(button), action == GLFW_PRESS, x, y, ToMouseModifiers(mods), glfwGetTime());
});
```

이 방식이면 `vkRender`는 GLFW 헤더를 포함하지 않고, 앱이 SDL이나 native window로 바뀌어도 `OnMouseMove`, `OnButton`, `OnScroll` 호출부만 바꾸면 된다.

## 여러 마우스 동작 등록

마우스 동작이 많아지면 `MouseListenerGroup`으로 관련 listener를 묶는다. 예를 들어 같은 `MouseInput` 위에 trackball camera, picking, UI hover를 독립적으로 등록할 수 있다.

```cpp
TrackballCamera trackball;
bool cameraDirty = true;

vkRender::MouseListenerGroup trackballBindings(*mouseInput);

trackballBindings.Add(
        vkRender::MouseEventType::Drag,
        [&](vkRender::MouseEvent &event) {
            if (event.button != vkRender::MouseButton::Left)
                return;

            if (trackball.Drag(event, framebufferWidth, framebufferHeight))
                cameraDirty = true;
            event.handled = true;
        },
        10);

trackballBindings.Add(
        vkRender::MouseEventType::Scroll,
        [&](vkRender::MouseEvent &event) {
            if (trackball.Scroll(event))
                cameraDirty = true;
            event.handled = true;
        },
        10);
```

priority가 높은 listener가 먼저 호출된다. 예를 들어 UI 조작은 priority `100`, camera 조작은 priority `10`, debug logging은 priority `-100`처럼 둘 수 있다. UI가 이벤트를 처리하면 `handled`를 켜고, 카메라나 picking은 그 이벤트를 받지 않게 만들 수 있다.

`MouseListenerGroup`은 소멸될 때 자신이 등록한 listener를 자동으로 해제한다. 따라서 임시 tool, editor mode, debug overlay처럼 켜고 끄는 입력 모드를 만들기 쉽다.

## 예제

- `example/bvh_path_tracer.cpp`: `MouseInput`과 `MouseListenerGroup`을 사용해 왼쪽 드래그 trackball 회전, scroll zoom을 처리한다.
- `example/realtime_shadow.cpp`: shadow map 기반 실시간 그림자 렌더링 예제다.
- `example/bvh_shadow_room.cpp`: `vkBVH` ray query로 회색 방과 구체의 ray traced shadow를 계산한다.
