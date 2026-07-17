# Engine::Render Orchestration Layer (Renderer/RenderGraph/View/Scene/Camera/Input) Design

## Goal

`Engine::Render` currently has only the low-level pieces (`SwapChain`, `GraphicsPipeline`, `RenderingDescriptor`/`RenderingScope`), and `example2/cube_render.cpp` hand-rolls the entire frame loop (acquire, resize detection, command buffer recording, submit, present, sync objects) directly in `main()`. This was a deliberate scope cut when `Engine::Render` was first built.

This design adds the orchestration layer that was deferred: a `Renderer` that owns the frame loop end-to-end (including automatic swapchain/depth resize handling), a `RenderGraph`/`RenderPass` pair for composing draw logic into ordered passes, a `View`/`Scene`/`Camera` trio for binding what to render and from where, and a `MouseInput`/`KeyInput` event-callback system for input handling. `example2/cube_render.cpp` is rewritten against this new API to prove it end-to-end, replacing its ~286-line hand-rolled `main()` with a thin loop shell plus one `RenderPass` implementation.

All of this models Filament's `Renderer`/`View`/`Scene`/`Camera` split, scoped down to what this project needs today — no automatic resource-dependency/barrier graph, no multi-view compositing, no material/shader-graph system.

## Prior Art

`src/vkRender/` already has a `Renderer`, `RenderGraph`/`RenderPass`, `View`, `Scene`, `Camera`, `MouseInput`, `KeyInput` — built for the old `vkCommon::VkContext`-based stack. This design ports them into `Engine::Render`, adapted to `Engine::Core::Context&` (matching the `SwapChain`/`GraphicsPipeline` precedent), with three deliberate departures from a byte-for-byte port:

1. **`Camera` switches from a hand-rolled `Mat4`/`Vec3` to `vkMath::Mat4` (Eigen-backed, `utilities/Math.h`)** — reusing `vkMath::Perspective`/`Orthographic`/`LookAt` instead of reimplementing the same math, and matching the type `example2/cube_render.cpp` already uses for its MVP push constant. This is the first thing in `Engine::Render` to depend on Eigen; see the CMake section below.
2. **`Renderer` fully automates resize handling.** The old `vkRender::Renderer` exposed `NeedsSwapChainRecreate()`/`ClearSwapChainRecreateFlag()` and made the app call `swapChain.Recreate()` itself. The new `Renderer::BeginFrame(width, height)` takes the app's current framebuffer size every frame and internally recreates the swapchain and its own depth image when the size changes or `AcquireNextImage`/`Present` report `OUT_OF_DATE`/`SUBOPTIMAL` — the app never calls `Recreate()` directly.
3. **`Renderer` owns a default depth `Image`** sized to the swapchain, recreated alongside it. The old `Renderer` had no depth concept at all (its only demonstrated path was `CopyBufferToSwapChain`, not a real graphics-pipeline draw).

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
```

`Renderer`는 `RenderGraph`를 직접 실행하지 않고 `View`를 거친다 — `View::SetRenderGraph()`로 그래프를 연결하면 앱은 매 프레임 `renderer.Render(view)`만 부르면 된다.

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

**앱의 메인 루프:**

```cpp
while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    glfwGetFramebufferSize(window, &w, &h);

    if (!renderer.BeginFrame(w, h)) continue;
    renderer.Render(view);
    renderer.EndFrame();
}
```

**예외 안전성**: `Renderer`의 소멸자는 `vkDeviceWaitIdle`을 호출한 뒤 자원을 정리한다 (옛 `DestroyFrameObjects()`와 동일). 다만 `BeginFrame`/`Render`/`EndFrame` 사이(즉 `vkQueueSubmit` 이후 `EndFrame` 호출 전)에 예외가 발생하면 커맨드버퍼가 begin된 채로 `Renderer`가 파괴될 수 있다 — `example2`의 메인 루프는 지금처럼 루프 전체를 `try { ... } catch (...) { vkDeviceWaitIdle(context.device); throw; } catch`로 감싸 유지한다 (`Context`가 아니라 `Renderer`가 자원을 갖고 있어도 동일한 원칙이 적용된다).

### MouseInput / KeyInput

`vkRender::MouseInput`/`KeyInput`/`MouseListenerGroup`/`KeyListenerGroup`을 100% 동일한 API로 포트한다 (`AddListener(EventType, callback, priority)`, `RemoveListener(id)`, `ListenerGroup`이 소멸 시 자신의 리스너를 자동 해제). 이 두 클래스는 Vulkan이나 GLFW 헤더에 의존하지 않으므로 네임스페이스만 바꿔 그대로 옮긴다. 창(resize/close) 이벤트는 이 시스템에 통합하지 않는다 — resize는 `Renderer::BeginFrame`이 이미 자동 처리하고, close는 앱이 `glfwWindowShouldClose`로 직접 확인한다.

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

`main()`의 새 모습(뼈대):

```cpp
Engine::Core::Context context(true, instanceExtensions, surfaceFactory);
Engine::Render::SwapChain swapChain(context, swapChainDescriptor);
Engine::Render::Renderer renderer(context, swapChain);

Engine::Render::Scene scene;
Engine::Render::Camera camera;
camera.SetPerspective(60.0f * kPi / 180.0f, aspect, 0.1f, 100.0f);
camera.LookAt({0, 0, 4.5f}, {0, 0, 0});

Engine::Render::RenderGraph graph;
graph.AddPass(std::make_unique<CubePass>(context, swapChain.Format()));

Engine::Render::View view;
view.SetScene(&scene);
view.SetCamera(&camera);
view.SetRenderGraph(&graph);

Engine::Render::MouseInput mouseInput;
glfwSetWindowUserPointer(window, &mouseInput);
// GLFW 콜백에서 mouseInput.OnMouseMove/OnButton/OnScroll로 전달
// MouseListenerGroup으로 trackball 회전 바인딩 (example/bvh_path_tracer.cpp 패턴 참고)

try {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        glfwGetFramebufferSize(window, &w, &h);
        if (!renderer.BeginFrame(w, h)) continue;
        renderer.Render(view);
        renderer.EndFrame();
    }
} catch (...) {
    vkDeviceWaitIdle(context.device);
    throw;
}
```

애니메이션은 고정 회전 대신 (또는 함께) trackball 마우스 드래그로 카메라를 조작하도록 만들어 `MouseInput` 통합을 실제로 검증한다.

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

## Out of Scope

- 자동 리소스 의존성 추적/배리어 계산 (frame graph의 "진짜" 기능) — pass 순서 실행만 지원, 배리어는 여전히 수동.
- 여러 `View`를 한 프레임에 합성 (예: 그림자 뷰 + 메인 뷰 + UI 뷰) — `View` 하나만 지원.
- Material/Mesh/Transform 컴포넌트 시스템 — `Scene`은 여전히 opaque entity ID 목록일 뿐이다.
- `MouseInput`/`KeyInput` 외의 입력(게임패드, 터치) — 범위 밖.
- 창(resize/close) 이벤트를 `MouseInput`/`KeyInput`과 통합한 단일 이벤트 시스템 — resize는 `Renderer`가, close는 앱이 각자 처리.

## Testing Strategy

- 기존 `test/test_engineCore.cpp`와 같은 패턴으로 `test/test_engineRender.cpp`(신규)에 `Scene`/`Camera`/`View`/`RenderGraph` 단위 테스트 추가 — 이들은 GPU 디스패치가 필요 없는 순수 로직이므로 빠르게 검증 가능하다 (`Camera::SetPerspective` 후 `GetProjectionMatrix()` 값 확인, `RenderGraph::AddPass`/`Execute` 순서 확인 등).
- `Renderer`의 자동 resize/recreate 로직은 유닛 테스트로 검증하기 어렵다 (실제 GLFW 창 크기 변경 필요) — `example2/cube_render2`를 빌드해 창을 리사이즈하며 시각적으로 확인하는 것이 이번에도 주된 검증 수단이다.
- `example2/cube_render2`를 빌드/실행해 큐브가 그려지고, trackball 마우스 드래그로 회전하며, 리사이즈 후에도 정상 동작하고, 종료 시 깨끗하게 닫히는지 확인한다.
- 전체 회귀 스위트(`ctest`)가 여전히 50/51(기존 `WideBVHTest.RadiusMatchesCpuReference` 실패만 남고) 통과하는지 확인한다.
