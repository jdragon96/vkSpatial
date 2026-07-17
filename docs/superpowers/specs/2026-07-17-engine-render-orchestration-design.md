# Engine::Render Orchestration Layer (Renderer/RenderGraph/View/Scene/Camera/Input) Design

## Goal

`Engine::Render` currently has only the low-level pieces (`SwapChain`, `GraphicsPipeline`, `RenderingDescriptor`/`RenderingScope`), and `example2/cube_render.cpp` hand-rolls the entire frame loop (acquire, resize detection, command buffer recording, submit, present, sync objects) directly in `main()`. This was a deliberate scope cut when `Engine::Render` was first built.

This design adds the orchestration layer that was deferred: a `Renderer` that owns the frame loop end-to-end (including automatic swapchain/depth resize handling), a `RenderGraph`/`RenderPass` pair for composing draw logic into ordered passes, a `View`/`Scene`/`Camera` trio for binding what to render and from where, and a `MouseInput`/`KeyInput` event-callback system for input handling. On top of that, a `Window`/`Application` layer hides the windowing backend (GLFW today, selected via an enum, chosen so another backend can be added later without changing call sites) and the `while` loop itself — `example2/cube_render.cpp`'s `main()` ends up with no Vulkan and no GLFW visible at all, just `Application` setup, `View`/`RenderGraph` wiring, input listener registration, and one `Run()` call.

All of this models Filament's `Engine`/`Renderer`/`View`/`Scene`/`Camera` split, scoped down to what this project needs today — no automatic resource-dependency/barrier graph, no multi-view compositing, no material/shader-graph system, no multi-window support.

## Prior Art

`src/vkRender/` already has a `Renderer`, `RenderGraph`/`RenderPass`, `View`, `Scene`, `Camera`, `MouseInput`, `KeyInput` — built for the old `vkCommon::VkContext`-based stack. This design ports them into `Engine::Render`, adapted to `Engine::Core::Context&` (matching the `SwapChain`/`GraphicsPipeline` precedent), with four deliberate departures from a byte-for-byte port:

1. **`Camera` switches from a hand-rolled `Mat4`/`Vec3` to `vkMath::Mat4` (Eigen-backed, `utilities/Math.h`)** — reusing `vkMath::Perspective`/`Orthographic`/`LookAt` instead of reimplementing the same math, and matching the type `example2/cube_render.cpp` already uses for its MVP push constant. This is the first thing in `Engine::Render` to depend on Eigen; see the CMake section below.
2. **`Renderer` fully automates resize handling.** The old `vkRender::Renderer` exposed `NeedsSwapChainRecreate()`/`ClearSwapChainRecreateFlag()` and made the app call `swapChain.Recreate()` itself. The new `Renderer::BeginFrame(width, height)` takes the app's current framebuffer size every frame and internally recreates the swapchain and its own depth image when the size changes or `AcquireNextImage`/`Present` report `OUT_OF_DATE`/`SUBOPTIMAL` — the app never calls `Recreate()` directly.
3. **`Renderer` owns a default depth `Image`** sized to the swapchain, recreated alongside it. The old `Renderer` had no depth concept at all (its only demonstrated path was `CopyBufferToSwapChain`, not a real graphics-pipeline draw).
4. **A new `Window`/`Application` layer has no equivalent in `vkRender` at all** — every old example (`example/cube_render.cpp`, `example/realtime_shadow.cpp`, etc.) calls GLFW directly in its own `main()`. This design adds `Window` (a `WindowBackend`-selected abstraction, GLFW being the only implementation today) and `Application` (owning `Window`+`Context`+`SwapChain`+`Renderer`+`View`, with `Run()` absorbing the `while` loop itself) so that `example2`'s `main()` has neither Vulkan nor GLFW visible.

Everything else (`Scene` as a flat entity-ID container, `View` binding `Scene`+`Camera`+`RenderGraph`+`Viewport`+`ClearOptions`, `RenderGraph`/`RenderPass`'s ordered-execution model, `MouseInput`/`KeyInput`'s `AddListener(type, callback, priority)` + `ListenerGroup` RAII pattern) ports with no behavioral change beyond the namespace and `Context&` adjustments already established for `SwapChain`/`GraphicsPipeline`.

## Architecture

```text
Engine::Render
    |
    +-- Scene            entity 목록 컨테이너 (vkRender::Scene 그대로 포트)
    +-- Camera           vkMath::Mat4 기반 view/projection (vkMath::Perspective/Orthographic/LookAt 사용)
    +-- View             Scene* + Camera* + RenderGraph* + Viewport + ClearOptions 묶음, Context 비의존
    +-- RenderPass        (abstract) Execute(RenderContext&)
    +-- RenderGraph       RenderPass 목록을 순서대로 Execute (자동 배리어/리소스 추적 없음)
    +-- Renderer          Context&, SwapChain&를 받아 프레임 동기화 객체 + 기본 depth Image를 내부 소유
    +-- MouseInput / KeyInput / MouseListenerGroup / KeyListenerGroup
    +-- Window           (abstract) WindowBackend enum으로 선택되는 창 백엔드. GlfwWindow가 유일한 구현체.
    |                    MouseInput/KeyInput을 내부 소유하고 플랫폼 콜백을 여기로 연결한다.
    +-- Application       Window + Context + SwapChain + Renderer + View를 전부 소유. Run()이 while
                         루프 전체(+ 예외 안전 wrapping)를 흡수한다.
```

`Renderer`는 `RenderGraph`를 직접 실행하지 않고 `View`를 거친다 — `View::SetRenderGraph()`로 그래프를 연결하면 앱은 매 프레임 `renderer.Render(view)`만 부르면 된다. `Application`은 이 `renderer.Render(view)` 호출조차 `Run()` 안으로 감춘다 — 앱은 `Application::GetView()`로 얻은 `View`를 한 번 구성해두기만 하면 된다.

## Components

### Scene

```cpp
using Entity = uint32_t;

class Scene {
public:
    Entity CreateEntity();
    void AddEntity(Entity entity);
    void Remove(Entity entity);
    void Clear();
    bool Contains(Entity entity) const;
    const std::vector<Entity> &Entities() const;
};
```

직접 포트, 로직 변경 없음. `Entity`는 여전히 opaque `uint32_t` — mesh/material/transform registry와의 연결은 이번 범위 밖이다 (미래 작업).

### Camera

```cpp
class Camera {
public:
    enum class Projection { Perspective, Orthographic };

    void SetPerspective(float fovYRadians, float aspect, float nearPlane, float farPlane);
    void SetOrthographic(float left, float right, float bottom, float top, float nearPlane, float farPlane);
    void LookAt(Eigen::Vector3f eye, Eigen::Vector3f target, Eigen::Vector3f up = {0, 1, 0});

    Projection GetProjectionType() const;
    float GetNearPlane() const;
    float GetFarPlane() const;
    Eigen::Vector3f GetEye() const;
    Eigen::Vector3f GetTarget() const;
    const vkMath::Mat4 &GetViewMatrix() const;
    const vkMath::Mat4 &GetProjectionMatrix() const;
};
```

`SetPerspective`/`SetOrthographic`/`LookAt`의 구현은 `vkMath::Perspective`/`vkMath::Orthographic`/`vkMath::LookAt`(이미 `utilities/Math.h`에 존재)를 그대로 호출한다 — 행렬 조립 코드를 새로 쓰지 않는다. 헤더 하나로 충분하며 (`Camera.h`, `.cpp` 없음), 옛 `vkRender::Camera`와 동일하게 header-only로 유지한다.

### View

```cpp
class View {
public:
    void SetScene(Scene *scene);
    void SetCamera(Camera *camera);
    void SetRenderGraph(RenderGraph *renderGraph);
    void SetViewport(Viewport viewport);
    void SetClearOptions(ClearOptions clearOptions);

    Scene *GetScene() const;
    Camera *GetCamera() const;
    RenderGraph *GetRenderGraph() const;
    Viewport GetViewport() const;
    ClearOptions GetClearOptions() const;
};
```

직접 포트, `Engine::Core::Context`에 비의존. 이번 범위에서는 `View` 하나만 있으면 충분하다 — 여러 `View`를 한 프레임에 합성하는 것(멀티 패스 컴포지팅)은 범위 밖.

### RenderPass / RenderGraph

```cpp
struct RenderContext {
    Engine::Core::Context *context = nullptr;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    Engine::Render::SwapChain *swapChain = nullptr;
    Engine::Core::Image *depthImage = nullptr;  // Renderer가 소유, 매 프레임 유효한 포인터
    View *view = nullptr;
    uint32_t imageIndex = 0;
    FrameInfo frame;
};

class RenderPass {
public:
    virtual ~RenderPass() = default;
    virtual const char *Name() const = 0;
    virtual void Execute(RenderContext &context) = 0;
};

class RenderGraph {
public:
    RenderGraph &AddPass(std::unique_ptr<RenderPass> pass);
    void Clear();
    void Execute(RenderContext &context) const;
    bool Empty() const;
    size_t PassCount() const;
};
```

`vkRender::RenderGraph`에서 유일하게 달라지는 부분은 `RenderContext`에 `depthImage` 필드가 추가된 것 — 옛 버전은 depth 개념이 없었다. 나머지(순서대로 `Execute` 호출, 자동 리소스 의존성 추적 없음)는 그대로.

**레이아웃 전이 책임 경계**: `Renderer::Render()`가 그래프를 실행하기 *전에* 기본 컬러(swapchain image)와 기본 깊이(`depthImage`) 두 개만 `ATTACHMENT_OPTIMAL`로 전이해준다. 각 `RenderPass::Execute()`는 이 두 타겟으로 `RenderingDescriptor::ColorDepth()` + `RenderingScope`를 열어 그리면 된다. 별도의 프라이빗 렌더 타겟(예: 그림자맵)이 필요한 pass는 그 이미지의 전이를 자신이 직접 책임진다 — `Renderer`는 기본 타겟 두 개 이상은 건드리지 않는다. 자동 배리어 계산은 이번 범위에 넣지 않는다.

### Renderer

```cpp
class Renderer {
public:
    explicit Renderer(Engine::Core::Context &context,
                       Engine::Render::SwapChain &swapChain,
                       VkFormat depthFormat = VK_FORMAT_D32_SFLOAT);
    ~Renderer();  // vkDeviceWaitIdle() 후 프레임 동기화 객체 + depth Image 정리

    // 앱이 매 프레임 알고 있는 현재 프레임버퍼 크기를 그대로 넘긴다. swapchain extent와
    // 다르면 SwapChain::Recreate() + depth Image 재생성을 자동 처리한 뒤 false를 반환해
    // 그 프레임을 건너뛴다. AcquireNextImage가 OUT_OF_DATE를 반환해도 동일하게 처리.
    bool BeginFrame(uint32_t width, uint32_t height);

    // graph->Empty()면 즉시 반환. 아니면 컬러/깊이를 ATTACHMENT_OPTIMAL로 전이한 뒤
    // RenderContext를 만들어 view.GetRenderGraph()->Execute(context)를 호출한다.
    void Render(View &view);

    // 컬러 이미지를 PRESENT_SRC로 전이, submit, present. OUT_OF_DATE/SUBOPTIMAL이면
    // 자동 recreate (다음 BeginFrame이 다시 시도).
    void EndFrame();
};
```

내부적으로 세마포어 2개(`imageAvailable`/`renderFinished`), 펜스 1개, 전용 command pool/buffer를 소유한다 (옛 `vkRender::Renderer`와 동일한 자원 구성). `BeginFrame`에서 `vkWaitForFences` → (크기 비교, 다르면 recreate) → `AcquireNextImage` 순서.

**`Renderer` 단독 사용 시 앱의 메인 루프 형태** (아래 `Window`/`Application`이 없다고 가정한 경우의 계약을 보여주기 위함 — 실제 `example2`는 이 루프를 직접 쓰지 않고, 뒤에 나오는 `Application::Run()`이 이걸 대신 호출한다):

```cpp
while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    glfwGetFramebufferSize(window, &w, &h);

    if (!renderer.BeginFrame(w, h)) continue;
    renderer.Render(view);
    renderer.EndFrame();
}
```

**예외 안전성**: `Renderer`의 소멸자는 `vkDeviceWaitIdle`을 호출한 뒤 자원을 정리한다 (옛 `DestroyFrameObjects()`와 동일). 다만 `BeginFrame`/`Render`/`EndFrame` 사이(즉 `vkQueueSubmit` 이후 `EndFrame` 호출 전)에 예외가 발생하면 커맨드버퍼가 begin된 채로 `Renderer`가 파괴될 수 있다 — 이 wrapping은 이제 아래 `Application::Run()`이 떠맡는다 (`example2`가 더 이상 while 루프를 직접 쓰지 않으므로).

### MouseInput / KeyInput

`vkRender::MouseInput`/`KeyInput`/`MouseListenerGroup`/`KeyListenerGroup`을 거의 동일한 API로 포트한다 (`AddListener(EventType, callback, priority)`, `RemoveListener(id)`, `ListenerGroup`이 소멸 시 자신의 리스너를 자동 해제). 이 두 클래스는 Vulkan 헤더에 의존하지 않으므로 네임스페이스만 바꿔 그대로 옮긴다. 유일한 차이는 `KeyEvent.keyCode`의 타입이다 — 아래 `Window` 섹션에서 다루듯 플랫폼 원시 `int`가 아니라 `KeyCode`(작은 공용 enum)로 바뀐다. 창(resize/close) 이벤트는 이 시스템에 통합하지 않는다 — resize는 `Renderer::BeginFrame`이, close는 아래 `Window`/`Application`이 각자 처리한다.

### Window

지금까지는 GLFW 호출(`glfwInit`, `glfwCreateWindow`, `glfwGetRequiredInstanceExtensions`, `glfwCreateWindowSurface`, `glfwPollEvents`, `glfwGetFramebufferSize`, `glfwWindowShouldClose`, `glfwDestroyWindow`, `glfwTerminate`)이 `example2/cube_render.cpp`의 `main()`에 그대로 노출되어 있었다. `Window`는 이걸 백엔드 추상화 뒤로 감춘다.

```cpp
enum class WindowBackend { GLFW };

// MouseButton과 같은 스타일: 흔한 키만 이름 붙이고 나머지는 Unknown으로 받는다
enum class KeyCode {
    Unknown, Escape, Space, Enter, Tab, Backspace,
    A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
    Left, Right, Up, Down,
    LeftShift, LeftControl, LeftAlt, LeftSuper,
};

struct WindowDescriptor {
    uint32_t width = 1280;
    uint32_t height = 720;
    std::string title = "Engine::Render";
};

class Window {
public:
    static std::unique_ptr<Window> Create(WindowBackend backend, const WindowDescriptor &descriptor);
    virtual ~Window() = default;

    virtual bool ShouldClose() const = 0;
    virtual void RequestClose() = 0;
    virtual void PollEvents() = 0;
    virtual VkExtent2D FramebufferSize() const = 0;

    // Application이 Context를 만들 때 내부적으로만 호출한다 — 앱 코드는 직접 쓰지 않는다.
    virtual std::vector<const char *> RequiredInstanceExtensions() const = 0;
    virtual VkSurfaceKHR CreateSurface(VkInstance instance) const = 0;

    MouseInput &Mouse() { return m_mouseInput; }
    KeyInput &Keys() { return m_keyInput; }

protected:
    MouseInput m_mouseInput;
    KeyInput m_keyInput;
};
```

`GlfwWindow : public Window`가 실제 GLFW 호출을 전부 감춘다. GLFW 콜백(`glfwSetCursorPosCallback` 등)은 `glfwSetWindowUserPointer`로 저장해둔 `GlfwWindow*`를 통해 `m_mouseInput`/`m_keyInput`으로 이벤트를 전달하는 방식으로 연결한다. 이 변환 로직(GLFW 키코드 → `KeyCode`, GLFW 마우스 버튼 → `MouseButton`)도 `GlfwWindow.cpp` 안에 있다. **`example2/cube_render.cpp`는 `<GLFW/glfw3.h>`를 더 이상 include하지 않는다.**

`Window::Create(WindowBackend backend, ...)`는 `backend` 값에 따라 분기해 해당 구현체를 만든다 — 지금은 `GLFW` 하나뿐이지만, 나중에 다른 백엔드를 추가할 때 호출부(`Application`)를 바꿀 필요가 없다.

### Application

```cpp
struct ApplicationDescriptor {
    WindowBackend backend = WindowBackend::GLFW;
    WindowDescriptor window;
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
};

class Application {
public:
    explicit Application(const ApplicationDescriptor &descriptor = {});
    ~Application();

    Engine::Core::Context &GetContext();
    Engine::Render::SwapChain &GetSwapChain();
    Engine::Render::Renderer &GetRenderer();
    Window &GetWindow();
    View &GetView();   // Scene/Camera/RenderGraph는 여기다 구성해서 넣는다

    void Run();   // ShouldClose()까지 내부 반복. 예외 발생 시 vkDeviceWaitIdle 후 rethrow.
};
```

생성자 안에서 `Window::Create(descriptor.backend, descriptor.window)` → `window->RequiredInstanceExtensions()`/`window->CreateSurface(instance)`를 넘겨 `Engine::Core::Context(true, ...)` 생성 → `SwapChain` → `Renderer(context, swapChain, descriptor.depthFormat)` 순으로 만든다 (지금 `example2`의 `main()` 앞부분이 하던 일 그대로, 위치만 옮겨진 것). `ApplicationDescriptor::depthFormat`는 그대로 `Renderer`의 생성자 인자로 전달된다.

`Run()`이 지금까지 `example2`가 직접 쓰던 while 루프 전체와 그 바깥의 try/catch까지 흡수한다:

```cpp
void Application::Run() {
    try {
        while (!m_window->ShouldClose()) {
            m_window->PollEvents();
            const VkExtent2D size = m_window->FramebufferSize();
            if (size.width == 0 || size.height == 0) continue;  // 최소화 상태

            if (!m_renderer->BeginFrame(size.width, size.height)) continue;
            m_renderer->Render(m_view);
            m_renderer->EndFrame();
        }
    } catch (...) {
        vkDeviceWaitIdle(m_context->device);
        throw;
    }
    vkDeviceWaitIdle(m_context->device);
}
```

`Run()`을 멈추는 방법은 오직 이벤트 리스너뿐이다 — 별도의 per-frame 콜백은 없다. 앱은 `KeyListenerGroup`에서 `window.RequestClose()`를 부르거나(예: Escape 키), OS의 창 닫기 버튼(GLFW가 이미 내부적으로 `glfwSetWindowShouldClose`를 통해 `ShouldClose()`에 반영)으로 종료한다.

## example2/cube_render.cpp 마이그레이션

새 API를 검증하기 위해 `example2/cube_render.cpp`를 다시 작성한다. `main()`은 Vulkan 핸들을 직접 다루지 않고, 큐브를 그리는 로직은 `CubePass : public Engine::Render::RenderPass`로 옮긴다.

```cpp
class CubePass : public Engine::Render::RenderPass {
public:
    CubePass(Engine::Core::Context &context, VkFormat colorFormat);

    const char *Name() const override { return "CubePass"; }
    void Execute(Engine::Render::RenderContext &ctx) override;

private:
    Engine::Core::Buffer m_vertexBuffer;
    Engine::Core::Buffer m_indexBuffer;
    Engine::Render::GraphicsPipeline m_pipeline;
    uint32_t m_indexCount = 0;
};
```

`Execute()` 안에서 `RenderingDescriptor::ColorDepth(ctx.swapChain->Extent(), ctx.swapChain->ImageView(ctx.imageIndex), ctx.depthImage->View(), clear)` + `RenderingScope`를 열고, 파이프라인 바인딩과 `vkCmdDrawIndexed`를 호출한다. MVP 행렬은 `ctx.view->GetCamera()`에서 얻은 `GetViewMatrix()`/`GetProjectionMatrix()`와 회전 애니메이션을 곱해 push constant로 넘긴다.

`main()`의 새 모습 — Vulkan도 GLFW도 보이지 않는다:

```cpp
int main() {
    try {
        Engine::Render::ApplicationDescriptor descriptor;
        descriptor.window = {900, 700, "Engine::Render Cube"};
        Engine::Render::Application app(descriptor);

        Engine::Render::Scene scene;
        Engine::Render::Camera camera;
        const VkExtent2D extent = app.GetSwapChain().Extent();
        const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
        camera.SetPerspective(60.0f * kPi / 180.0f, aspect, 0.1f, 100.0f);
        camera.LookAt({0, 0, 4.5f}, {0, 0, 0});

        Engine::Render::RenderGraph graph;
        graph.AddPass(std::make_unique<CubePass>(app.GetContext(), app.GetSwapChain().Format()));

        app.GetView().SetScene(&scene);
        app.GetView().SetCamera(&camera);
        app.GetView().SetRenderGraph(&graph);

        Engine::Render::MouseListenerGroup trackball(app.GetWindow().Mouse());
        trackball.Add(Engine::Render::MouseEventType::Drag, [&](Engine::Render::MouseEvent &e) {
            if (e.button == Engine::Render::MouseButton::Left) {
                // e.deltaX/deltaY로 camera.LookAt()을 다시 계산해 orbit한다
            }
        });

        Engine::Render::KeyListenerGroup keys(app.GetWindow().Keys());
        keys.Add(Engine::Render::KeyEventType::Press, [&](Engine::Render::KeyEvent &e) {
            if (e.keyCode == Engine::Render::KeyCode::Escape)
                app.GetWindow().RequestClose();
        });

        app.Run();
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}
```

`main()`은 이제 `Application`/`Scene`/`Camera`/`RenderGraph`/`View`/`MouseListenerGroup`/`KeyListenerGroup`만 다룬다 — `VkInstance`, `GLFWwindow*`, 세마포어, 커맨드버퍼 어느 것도 나타나지 않는다. trackball 드래그로 카메라를 조작하게 만들어 `MouseInput` 통합을, Escape 키 바인딩으로 `KeyInput`/`KeyCode` 통합을 실제로 검증한다.

## CMake 변경

`src/Engine/CMakeLists.txt`의 `EngineRender` 타겟에 Eigen include 경로를 `PUBLIC`으로 추가한다:

```cmake
target_include_directories(EngineRender
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
            /opt/homebrew/opt/eigen/include/eigen3
        PRIVATE
            "${VULKAN_SDK}/include")
```

`example2/CMakeLists.txt`가 지금 직접 추가하고 있는 Eigen include(`target_include_directories(cube_render2 PRIVATE /opt/homebrew/opt/eigen/include/eigen3)`)는 `Engine::Render`를 통해 전이되므로 제거한다 (중복 제거).

`GlfwWindow.cpp`가 `Engine::Render` 내부로 들어오므로, `EngineRender` 타겟이 `glfw`를 링크해야 한다. GLFW 헤더는 `Engine::Render`의 공개 헤더(`Window.h` 등)에 노출되지 않으므로 `PRIVATE`으로 링크한다:

```cmake
target_link_libraries(EngineRender
        PUBLIC Engine::Core Vulkan::Vulkan
        PRIVATE glfw)
```

`example2/CMakeLists.txt`가 지금 직접 링크하는 `glfw`도 더 이상 필요 없다 — `example2/cube_render.cpp`가 GLFW 헤더를 include하지 않으므로.

## Out of Scope

- 자동 리소스 의존성 추적/배리어 계산 (frame graph의 "진짜" 기능) — pass 순서 실행만 지원, 배리어는 여전히 수동.
- 여러 `View`를 한 프레임에 합성 (예: 그림자 뷰 + 메인 뷰 + UI 뷰) — `View` 하나만 지원.
- Material/Mesh/Transform 컴포넌트 시스템 — `Scene`은 여전히 opaque entity ID 목록일 뿐이다.
- `MouseInput`/`KeyInput` 외의 입력(게임패드, 터치) — 범위 밖.
- 창(resize/close) 이벤트를 `MouseInput`/`KeyInput`과 통합한 단일 이벤트 시스템 — resize는 `Renderer`가, close는 `Window`/`Application`이 각자 처리.
- `WindowBackend`는 `GLFW` 하나만 구현한다 — enum 자체는 확장 가능한 형태로 두되 SDL/native Win32 백엔드 구현은 범위 밖.
- `KeyCode`는 자주 쓰는 키만 다룬다 (문자/숫자/방향키/일부 modifier/Escape/Space/Enter/Tab/Backspace) — 매핑 안 되는 키는 `Unknown`으로 들어오고 원시 코드는 유실된다. 필요해지면 나중에 추가.
- `Application`은 창 하나, `View` 하나만 지원한다 — 멀티 윈도우는 범위 밖.
- `Application::Run()`의 per-frame 콜백은 지원하지 않는다 — 종료는 오직 `Window::RequestClose()`(이벤트 리스너에서 호출) 또는 OS 창 닫기로만 가능하다.

## Testing Strategy

- 기존 `test/test_engineCore.cpp`와 같은 패턴으로 `test/test_engineRender.cpp`(신규)에 `Scene`/`Camera`/`View`/`RenderGraph`/`MouseInput`/`KeyInput` 단위 테스트 추가 — 이들은 GPU 디스패치나 실제 창이 필요 없는 순수 로직이므로 빠르게 검증 가능하다 (`Camera::SetPerspective` 후 `GetProjectionMatrix()` 값 확인, `RenderGraph::AddPass`/`Execute` 순서 확인, `MouseInput::AddListener` + 이벤트 발생 시 콜백 호출 확인 등).
- `Renderer`의 자동 resize/recreate 로직과 `Window`/`Application`(실제 GLFW 창 생성이 필요)은 유닛 테스트로 검증하기 어렵다 — `example2/cube_render2`를 빌드해 창을 리사이즈하며 시각적으로 확인하는 것이 이번에도 주된 검증 수단이다.
- `example2/cube_render2`를 빌드/실행해 큐브가 그려지고, trackball 마우스 드래그로 회전하며, Escape 키로 종료되고, 리사이즈 후에도 정상 동작하고, 종료 시 깨끗하게 닫히는지 확인한다.
- 전체 회귀 스위트(`ctest`)가 여전히 50/51(기존 `WideBVHTest.RadiusMatchesCpuReference` 실패만 남고) 통과하는지 확인한다.
