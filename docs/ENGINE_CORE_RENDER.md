# Engine::Core / Engine::Render 사용법

`Engine::Core`와 `Engine::Render`는 이 저장소의 Vulkan 기반 계층이다. Filament를 참고해 두 축으로 설계했다.

1. **GPU 알고리즘 개발용**: 창(window) 없이 compute만 돌리는 헤드리스 워크로드 (BVH 빌드, 이미지 처리 등).
2. **랜더링 엔진용**: swapchain, graphics pipeline, dynamic rendering을 갖춘 실시간 렌더링.

두 축은 `Engine::Core`가 공통 기반이고, `Engine::Render`는 그 위에 선택적으로 얹는 계층이다.

## 계층

```text
Application (GLFW 또는 헤드리스)
    |
    v
Engine::Core::Context
    |   VkInstance, VkPhysicalDevice, VkDevice, compute/graphics queue,
    |   command pool, VmaAllocator — vk-bootstrap + VMA 기반, RAII
    |
    +-- Engine::Core::Buffer           (VMA 디바이스 버퍼, Upload/Download)
    +-- Engine::Core::Image            (VMA 이미지 + view, layout transition)
    +-- Engine::Core::ComputePipeline  (헤드리스 compute dispatch)
    +-- Engine::Core::OneShotCommands  (SubmitOneShot, 위 세 개가 내부적으로 사용)
    |
    |  Context(enablePresent=true)일 때만 유효
    v
    +-- Engine::Render::SwapChain
    +-- Engine::Render::GraphicsPipeline
    +-- Engine::Render::Rendering (RenderingDescriptor / RenderingScope)
```

## 설계 원칙

- **`Context`는 RAII다.** 생성자에서 vk-bootstrap으로 instance/device를 만들고 VMA allocator까지 준비하며, 소멸자가 역순으로 전부 정리한다. 별도 `shutdown()`을 호출할 필요가 없다.
- **자원을 소유하는 모든 객체는 `Context&`를 참조로 받는다.** 포인터도 아니고 `VkDevice`를 따로 저장하지도 않는다 — `Buffer`, `Image`, `ComputePipeline`, `SwapChain`, `GraphicsPipeline` 모두 자신을 만든 `Context`보다 오래 살아남으면 안 된다.
- **`Context` 생성 모드가 두 가지다.**
  - `Context()` 또는 `Context(false)` — headless/compute 전용. 창도, surface도, graphics queue도 만들지 않는다. `graphicsQueue`/`graphicsCmdPool`/`surface`는 `VK_NULL_HANDLE`로 남는다. GPU 알고리즘 개발은 이 모드로 충분하다.
  - `Context(true, instanceExtensions, surfaceFactory)` — presentation 지원. `graphicsQueue`/`graphicsFamily`/`graphicsCmdPool`/`surface`까지 채운다. `Engine::Render`를 쓰려면 이 모드가 필요하다.
- **`Engine::Render`는 `Engine::Core`에만 의존한다.**

## Engine::Core — GPU 알고리즘 개발용 (헤드리스)

### Context 생성

```cpp
Engine::Core::Context ctx;  // enablePresent=false, 창/surface 불필요
```

`ctx.device`, `ctx.computeQueue`, `ctx.cmdPool`, `ctx.allocator`가 바로 유효한 핸들이다.

### Buffer — GPU 메모리 할당과 Upload/Download

```cpp
Engine::Core::Buffer buffer(ctx, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
buffer.Allocate(kCount * sizeof(float));
buffer.Upload(source.data(), kCount * sizeof(float));

std::vector<float> result(kCount);
buffer.Download(result.data(), kCount * sizeof(float));
```

`Upload`/`Download`는 내부에서 host-visible staging buffer를 만들고 `SubmitOneShot`으로 한 번 복사한 뒤 블로킹으로 끝낸다. 실패하면 `std::runtime_error`를 던진다 (예: `Allocate`한 크기보다 큰 바이트 수를 요청한 경우).

### ComputePipeline — 컴퓨트 셰이더 dispatch

`ComputePipeline`은 기존 `vkComputeBase`를 대체한다. fluent 체이닝으로 바인딩과 push constant, dispatch까지 한 번에 표현한다.

```cpp
struct PushConstants { uint32_t count; };

Engine::Core::ComputePipeline pipeline(ctx);
pipeline.Build(shaderSource, Engine::Core::ShaderInput::GlslSrc)
        .Bind(0, inputBuffer)
        .Bind(1, outputBuffer)
        .Args(PushConstants{N})
        .DispatchElements(N);
```

- `Build(source, ShaderInput::GlslSrc)`는 GLSL 소스 문자열을 런타임에 `shaderc`로 컴파일한다. `Build(path)`는 이미 컴파일된 `.spv` 파일을 읽는다.
- `local_size_x/y/z`는 SPIRV-Reflect로 셰이더에서 직접 읽어오므로 dispatch 시점에 grid 크기를 수동으로 나눌 필요가 없다 — `DispatchElements(N)`이 알아서 `ceil(N / local_size_x)`로 그룹 수를 계산한다.
- `Bind(binding, Buffer&)`처럼 `Engine::Core::Buffer`를 직접 넘기는 오버로드와, `Bind(binding, VkBuffer, VkDeviceSize)`처럼 raw handle을 넘기는 오버로드가 둘 다 있다.
- `Dispatch()`가 끝나면 이미 GPU 작업이 완료된 상태다 (내부적으로 `SubmitOneShot`이 블로킹). `Sync()`는 옛 API와의 호출부 호환을 위해 남겨둔 no-op다.

전체 동작하는 예제는 `test/test_engineCore.cpp`의 `ComputePipelineTest.DispatchSimpleShaderProducesExpectedOutput`을 참고한다 — `0`부터 `10000`까지 합을 GPU atomic add로 구해 CPU에서 검증한다.

### Image — compute 결과를 담는 이미지가 필요할 때

```cpp
Engine::Core::Image image(ctx, Engine::Core::ImageDescriptor::Color2D(
        {width, height}, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT));
```

`ImageDescriptor::Depth2D(extent, format = VK_FORMAT_D32_SFLOAT)`와 `Color2D(extent, format, usage)` 두 팩토리가 흔한 경우를 커버한다. `Create()`를 다시 호출하면 기존 이미지/뷰를 먼저 `Destroy()`하고 새로 만들기 때문에, resize 같은 상황에서 별도로 파괴할 필요 없이 그냥 다시 `Create()`를 부르면 된다.

### SubmitOneShot — 직접 커맨드를 기록해야 할 때

`Buffer`/`ComputePipeline`이 내부적으로 쓰는 헬퍼지만, 그 둘로 표현 안 되는 임시 GPU 작업(예: `vkCmdFillBuffer`, 커스텀 배리어)에는 직접 써도 된다.

```cpp
Engine::Core::SubmitOneShot(ctx, Engine::Core::QueueRole::Compute, [&](VkCommandBuffer cmd) {
    vkCmdFillBuffer(cmd, buffer, 0, 256, 0x2A2A2A2Au);
});
```

`QueueRole::Graphics`는 `Context`가 `enablePresent=true`로 만들어졌을 때만 쓸 수 있다. 아니면 예외를 던진다.

## Engine::Render — 랜더링 엔진용 (창 필요)

### Context 생성 (presentation 활성화)

GLFW를 쓴다면 인스턴스 확장 목록과 surface 생성 콜백을 넘긴다.

```cpp
uint32_t glfwExtensionCount = 0;
const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
std::vector<const char *> instanceExtensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

Engine::Core::Context context(
        true, instanceExtensions,
        [&](VkInstance instance) {
            VkSurfaceKHR surface = VK_NULL_HANDLE;
            if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
                throw std::runtime_error("failed to create window surface");
            return surface;
        });
```

`surfaceFactory` 콜백은 `Context` 생성자 내부, instance가 만들어진 직후에 호출된다. 이후 `context.graphicsQueue`/`context.graphicsCmdPool`/`context.surface`가 유효해진다.

### SwapChain

```cpp
Engine::Render::SwapChainDescriptor descriptor{};
descriptor.width = framebufferWidth;
descriptor.height = framebufferHeight;
Engine::Render::SwapChain swapChain(context, descriptor);
```

- `AcquireNextImage(signalSemaphore, signalFence, &imageIndex)`가 `VK_ERROR_OUT_OF_DATE_KHR`를 반환하면 `Recreate(width, height)`를 부르고 그 프레임은 건너뛴다.
- `Present(imageIndex, waitSemaphore)`가 `VK_ERROR_OUT_OF_DATE_KHR`나 `VK_SUBOPTIMAL_KHR`를 반환해도 마찬가지로 `Recreate()`한다.
- `Recreate()` 전후로 depth image처럼 swapchain extent에 맞춰야 하는 리소스가 있다면 `Recreate()` 직후 다시 만들어줘야 한다 (아래 프레임 루프 예제 참고).

### GraphicsPipeline

`GraphicsPipelineDescriptor`는 선언형 빌더다. `GraphicsPipeline::Build()`가 shader module, vertex input, rasterization, depth-stencil, color blend, dynamic state, `VkPipelineLayout`, dynamic-rendering 기반 `VkGraphicsPipelineCreateInfo`까지 전부 조립한다.

```cpp
struct Vertex { float position[3]; float color[3]; };
struct PushConstants { vkMath::Mat4 mvp; };

Engine::Render::GraphicsPipelineDescriptor descriptor;
descriptor.VertexShader(shaderDir + "/cube.vert.spv")
          .FragmentShader(shaderDir + "/cube.frag.spv")
          .VertexBinding<Vertex>()
          .VertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position))
          .VertexAttribute(1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color))
          .ColorTarget(swapChain.Format())
          .DepthTarget(VK_FORMAT_D32_SFLOAT)
          .PushConstant<PushConstants>(VK_SHADER_STAGE_VERTEX_BIT);

Engine::Render::GraphicsPipeline pipeline(context);
pipeline.Build(descriptor);
```

- `VertexBinding<VertexT>()`와 `PushConstant<PushT>(stageFlags)`는 `sizeof(VertexT)`/`sizeof(PushT)`를 자동으로 채워주는 템플릿 오버로드다. 크기를 직접 넘기고 싶으면 non-template 오버로드를 쓴다.
- `dynamicStates` 기본값이 `VIEWPORT`/`SCISSOR`이므로, 매 프레임 `RenderingDescriptor`가 만드는 viewport/scissor 값이 그대로 반영된다 — resize 후에도 별도 처리 없이 정확하다.
- 셰이더는 `.spv`만 읽는다 (`ComputePipeline`과 달리 런타임 GLSL 컴파일을 지원하지 않는다). CMake의 `add_custom_command` + `glslc`로 빌드 시점에 미리 컴파일해둬야 한다 — `example2/CMakeLists.txt` 참고.
- `pipeline.Bind(cmd)`로 파이프라인을 바인딩하고, `pipeline.PushConstants(cmd, stageFlags, value)`로 push constant를 올린다.

### RenderingDescriptor / RenderingScope — dynamic rendering

이 프로젝트는 render pass/framebuffer 없이 Vulkan 1.3 dynamic rendering(`vkCmdBeginRendering`/`vkCmdEndRendering`)만 쓴다.

```cpp
auto renderingDescriptor = Engine::Render::RenderingDescriptor::ColorDepth(
        swapChain.Extent(), swapChain.ImageView(imageIndex), depthImage.View(), clear);

{
    Engine::Render::RenderingScope scope(cmd, renderingDescriptor);
    pipeline.Bind(cmd);
    // vkCmdBindVertexBuffers, vkCmdDrawIndexed 등
}  // scope 소멸자가 vkCmdEndRendering을 기록한다
```

`RenderingScope`는 생성자에서 `vkCmdBeginRendering`과 (`setViewport`/`setScissor`가 켜져 있으면) `vkCmdSetViewport`/`vkCmdSetScissor`까지 기록하고, `End()` 또는 소멸자에서 `vkCmdEndRendering`을 기록한다. **`RenderingScope`를 열기 전에 color/depth 이미지를 각각 `COLOR_ATTACHMENT_OPTIMAL`/`DEPTH_ATTACHMENT_OPTIMAL`로 전이해둬야 한다** — dynamic rendering은 render pass처럼 자동으로 레이아웃을 바꿔주지 않는다.

## 전체 프레임 루프 예제

`Engine::Core`와 `Engine::Render`를 함께 쓰는 가장 완전한 예제는 `example2/cube_render.cpp`다 (회전하는 큐브를 그린다). 뼈대만 요약하면:

```cpp
Engine::Core::Context context(true, instanceExtensions, surfaceFactory);
Engine::Render::SwapChain swapChain(context, swapChainDescriptor);

Engine::Core::Buffer vertexBuffer(context, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
vertexBuffer.Allocate(vertexBytes);
vertexBuffer.Upload(vertices.data(), vertexBytes, Engine::Core::QueueRole::Graphics);

Engine::Core::Image depthImage(context);
depthImage.Create(Engine::Core::ImageDescriptor::Depth2D(swapChain.Extent()));

Engine::Render::GraphicsPipeline pipeline(context);
pipeline.Build(pipelineDescriptor);

// semaphore/fence/command buffer는 vkCreateSemaphore 등으로 직접 만든다 —
// Engine::Render는 아직 프레임 동기화 객체를 감싸주지 않는다 (Renderer가 없다).

try {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        // ... resize 감지, vkWaitForFences, AcquireNextImage ...
        // ... vkBeginCommandBuffer, layout transition, RenderingScope, draw ...
        // ... vkQueueSubmit, Present ...
    }
} catch (...) {
    vkDeviceWaitIdle(context.device);  // 아래 "예외 안전성" 참고
    throw;
}
```

전체 코드와 resize/OUT_OF_DATE 처리, 종료 시퀀스는 `example2/cube_render.cpp`를 그대로 읽는 게 가장 정확하다.

## CMake에서 링크하기

```cmake
# 헤드리스 GPU 알고리즘만 필요할 때
target_link_libraries(my_target PRIVATE Engine::Core)

# 랜더링까지 필요할 때
target_link_libraries(my_target PRIVATE Engine::Render Engine::Core glfw)
```

`Engine::Render`는 `Engine::Core`를 `PUBLIC`으로 링크하므로 `Engine::Render`만 링크해도 `Engine::Core` 헤더/심볼이 전이적으로 딸려온다 — 위처럼 둘 다 명시해도 되고 `Engine::Render`만 링크해도 된다. 둘 다 직접 쓰는 코드에서는 명시하는 쪽이 의도를 더 분명히 드러낸다.

**Eigen을 쓴다면 (`utilities/Math.h`, `utilities/SimpleResource.h`) include 경로를 타겟에 직접 추가해야 한다** — `Engine::Core`/`Engine::Render` 어느 쪽도 Eigen 의존성을 갖고 있지 않다.

```cmake
target_include_directories(my_target PRIVATE /opt/homebrew/opt/eigen/include/eigen3)
```

## 예외 안전성 주의사항

`Context::~Context()`는 `vkDeviceWaitIdle()`을 호출하지 않는다. GPU에 작업이 in-flight인 상태로 `Context`가 소멸되면 정의되지 않은 동작이다. 따라서 **`vkQueueSubmit` 이후 지점을 포함하는 코드 블록은 예외가 발생해도 `Context` 소멸자에 도달하기 전에 반드시 `vkDeviceWaitIdle`을 호출하도록 감싸야 한다**:

```cpp
try {
    while (!glfwWindowShouldClose(window)) {
        // ... vkQueueSubmit을 포함하는 프레임 루프 ...
    }
} catch (...) {
    vkDeviceWaitIdle(context.device);
    throw;
}
```

정상 종료 경로도 마찬가지로, 루프를 빠져나온 직후 (fence/semaphore를 파괴하거나 RAII 객체가 소멸되기 전에) `vkDeviceWaitIdle`을 한 번 호출해야 한다. `example2/cube_render.cpp`가 이 패턴의 참고 구현이다.

`Buffer`와 `ComputePipeline`은 복사는 물론 이동도 금지되어 있다 (`Image`는 이동 가능). `std::vector` 같은 컨테이너에 담아야 한다면 `std::unique_ptr<Buffer>`처럼 감싸서 넣는다.

## 예제 / 참고

- [`example2/cube_render.cpp`](../example2/cube_render.cpp) — `Engine::Core` + `Engine::Render`를 함께 쓰는 첫 실사용 예제. 회전하는 큐브를 그린다.
- [`test/test_engineCore.cpp`](../test/test_engineCore.cpp) — `Engine::Core`(Context/Buffer/Image/ComputePipeline)만 쓰는 헤드리스 사용 예제 겸 단위 테스트.
