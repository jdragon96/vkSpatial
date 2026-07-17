# Engine/Render 최소 이관 + example2 큐브 렌더링 설계

## 배경 / 목적

`Engine/Core`(vk-bootstrap + VMA 기반)는 완성되어 `main`에 병합되었지만, 렌더링에 필요한 게 아무것도 없다 — 스왑체인도, 그래픽스 파이프라인도, 렌더 패스를 기록할 방법도 없다. 이번 스펙은 `src/Engine/Render/`에 큐브 하나를 화면에 그리는 데 필요한 **최소한의 저수준 조각**만 만들고, `example2/`에 그걸 쓰는 가장 단순한 예제를 추가한다.

`RenderGraph`/`RenderPass`/`Scene`/`View`/`Camera`/`Renderer`/`Capture`/`KeyInput` 같은 상위 오케스트레이션 레이어는 의도적으로 이번 범위에서 제외한다 — 프레임 루프는 `example2/cube_render.cpp`의 `main()`에 직접 쓴다. 이렇게 하는 이유: 지금 필요한 건 "큐브 하나가 화면에 뜨는 것"이지 렌더링 엔진 전체가 아니고, 저수준 조각(`SwapChain`/`GraphicsPipeline`/`Rendering`)은 이미 `vkRender`에 검증된 형태로 존재해 이관 리스크가 낮은 반면, `Renderer`/`RenderGraph` 같은 오케스트레이션 레이어는 설계 여지가 커서 나중에 실제 필요(여러 렌더 패스, 여러 오브젝트 관리 등)가 생겼을 때 제대로 설계하는 게 낫다.

## 범위

- `src/Engine/Render/RenderTypes.h`, `RenderAttachments.h`, `Rendering.h`/`.cpp`, `SwapChain.h`/`.cpp`, `GraphicsPipeline.h`/`.cpp` 신규 추가.
- `src/Engine/CMakeLists.txt`에 `EngineRender` 타겟 추가 (`Engine::Core`에 `PUBLIC` 의존).
- `example2/` 폴더 신규 생성, `cube_render.cpp` 하나 추가 — 회전하는 큐브를 그리는 최소 예제. 셰이더는 기존 `example/cube.vert`/`cube.frag`를 그대로 재사용(복제하지 않음). 창 리사이즈 시 스왑체인 재생성은 포함한다(크래시 방지에 필요한 최소 기능으로 간주 — 완전히 빼면 창 크기 조절만으로 바로 크래시하는 예제가 되어 "동작하는 예제"라는 목표에 못 미친다).

### 범위 밖

- `vkRender::Engine`/`Renderer`/`RenderGraph`/`RenderPass`/`Scene`/`View`/`Camera`/`Capture`/`KeyInput`/`MouseInput`의 `Engine::Render` 이관 — 전부 다음 단계로 미룬다.
- 기존 `vkCommon`/`vkSpatial`/`vkRender`/`example`/`test`는 이 스펙에서 **한 글자도 바꾸지 않는다**.
- 스크린샷 캡처, 키 입력 처리 등 `example/cube_render.cpp`에 있던 부가 기능 — `example2`에서는 뺀다 (단순함이 목적).

## 아키텍처

### 1. `RenderTypes.h`, `RenderAttachments.h`, `Rendering.h`/`.cpp` — 순수 네임스페이스 이관

세 파일 다 `vkCommon`/`vkRender::VkContext` 등 어떤 컨텍스트 타입도 참조하지 않는 순수 Vulkan 구조체/RAII 스코프라, `namespace vkRender` → `namespace Engine::Render`만 바꾸고 로직은 전혀 손대지 않는다.

- `RenderTypes.h`: `Extent`, `Viewport`, `ClearOptions`, `FrameInfo`, `SwapChainDescriptor`.
- `RenderAttachments.h`: `ColorAttachment`, `DepthAttachment` (빌더 패턴, `VkRenderingAttachmentInfo` 조립).
- `Rendering.h`/`.cpp`: `RenderingDescriptor`(빌더), `RenderingScope`(RAII, `vkCmdBeginRendering`/`vkCmdEndRendering`).

### 2. `SwapChain` — `Context&` 기반으로 교체

지금 `vkRender::SwapChain`은 `vkCommon::VkContext*`를 받아 `->surface`, `->graphicsQueue`, `->physDevice`, `->device`를 읽는다. `Engine::Core::Context`는 필드 이름이 동일하되 `physDevice` → `physicalDevice`로 바뀌었다(`Engine::Core::Context` 설계 당시 결정). 포팅 시:
- 생성자를 `explicit SwapChain(Engine::Core::Context &context, const SwapChainDescriptor &descriptor = {})`로 변경(포인터 대신 참조 — `Engine::Core`의 다른 클래스들과 일관).
- 멤버를 `Engine::Core::Context *m_context`로 유지(참조로 받아 포인터로 저장 — 재대입 가능한 내부 상태 유지 필요 없으면 참조 멤버도 가능하지만, 기존 `vkRender::SwapChain`이 포인터 멤버였던 것과 동일하게 유지해 최소 변경).
- 모든 `m_context->physDevice` → `m_context->physicalDevice`로 치환. 나머지(`surface`, `graphicsQueue`, `device`)는 이름 동일이라 그대로.
- 에러 메시지 중 `"SwapChain requires a valid VkContext"` → `"SwapChain requires a valid Context"`로 텍스트만 갱신(타입 이름이 바뀌었으므로).

### 3. `GraphicsPipeline` — `Context&`로 통일

지금 `vkRender::GraphicsPipeline`은 특이하게 `VkDevice` 하나만 받는다(다른 `vkRender` 클래스들과 달리 `VkContext*`도 안 받음). `Engine::Core`의 나머지 클래스(`Buffer`, `Image`, `ComputePipeline`)가 전부 `Context&`를 받는 것과 일관되게 맞춘다:
- 생성자를 `explicit GraphicsPipeline(Engine::Core::Context &context)`로 변경, 내부에서 `m_device`는 그대로 `VkDevice` 타입 멤버로 유지하되 생성자에서 `m_device = context.device;`로 초기화(멤버 자체를 `Context&`로 바꾸는 게 아니라, 지금처럼 `VkDevice` 멤버 하나만 필요하므로 생성자 인자만 바꾸는 최소 변경).
- `GraphicsPipelineDescriptor`(순수 데이터/빌더 구조체, `Context` 불필요)는 완전히 그대로 이관.
- 셰이더 로딩은 `LoadSPIRV()`(파일에서 이미 컴파일된 `.spv` 읽기)만 하지 GLSL 컴파일은 안 한다 — `shaderc`/`spirv-reflect` 의존성 불필요, `ComputePipeline`과 다른 점.

### 4. `example2/cube_render.cpp` — 최소 프레임 루프

기존 `example/cube_render.cpp`를 새 API로 옮기되, `CubePass`(`RenderPass` 상속) 클래스 없이 `main()` 하나에 다 쓴다. 흐름:

```cpp
// 1. GLFW 윈도우 생성 (기존과 동일)
// 2. Engine::Core::Context context(true, instanceExtensions, surfaceFactory);  — RAII, init()/shutdown() 없음
// 3. Engine::Render::SwapChain swapChain(context, descriptor);
// 4. Engine::Core::Buffer vertexBuffer(context, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
//    vertexBuffer.Allocate(...); vertexBuffer.Upload(..., Engine::Core::QueueRole::Graphics);
//    (인덱스 버퍼도 동일)
// 5. Engine::Core::Image depthTarget(context, Engine::Core::ImageDescriptor::Depth2D(extent));
// 6. Engine::Render::GraphicsPipeline pipeline(context); pipeline.Build(descriptor);
// 7. 프레임 동기화 primitive 직접 생성: imageAvailable/renderFinished 세마포어, inFlight 펜스
//    (vkRender::Renderer가 내부적으로 하던 것을 여기서는 main()에 직접 기록)
// 8. 메인 루프:
//    - glfwPollEvents(), 리사이즈 감지 시 swapChain.Recreate(...) + depthTarget 재생성
//    - vkWaitForFences(inFlight) → vkAcquireNextImageKHR
//    - vkResetCommandBuffer, vkBeginCommandBuffer
//    - 컬러 이미지 UNDEFINED→COLOR_ATTACHMENT_OPTIMAL, 뎁스 이미지 →DEPTH_ATTACHMENT_OPTIMAL 배리어
//      (Engine::Render::RenderingScope 열기 전에 — Engine::Core::Image::TransitionLayout 재사용)
//    - RenderingDescriptor::ColorDepth(...) 로 RenderingScope 열기
//    - pipeline.Bind(cmd); vkCmdBindVertexBuffers/IndexBuffer; pipeline.PushConstants(mvp); vkCmdDrawIndexed
//    - RenderingScope 닫기, 컬러 이미지 →PRESENT_SRC_KHR 배리어
//    - vkEndCommandBuffer, vkQueueSubmit(세마포어+펜스), swapChain.Present(...)
// 9. vkDeviceWaitIdle, 정리 — Context/SwapChain/Buffer/Image는 소멸자가 알아서 정리(RAII),
//    직접 만든 세마포어/펜스만 수동으로 vkDestroy*
```

정점/인덱스 버퍼 업로드는 `QueueRole::Graphics`를 써야 한다(그래픽스 커맨드풀 필요) — `Context`가 `enablePresent=true`로 만들어졌으니 `context.graphicsCmdPool`이 이미 있다.

`RenderingScope`를 여는 시점(배리어 이후)은 이전에 발견했던 `CubePass::Execute()`의 순서 버그(스코프를 배리어보다 먼저 여는 것)를 반복하지 않도록, `RealtimeShadowPass`가 이미 쓰던 올바른 순서(배리어 → 스코프)를 따른다.

## 파일 변경 목록

| 파일 | 변경 |
|---|---|
| `src/Engine/Render/RenderTypes.h` (신규) | 순수 이관 |
| `src/Engine/Render/RenderAttachments.h` (신규) | 순수 이관 |
| `src/Engine/Render/Rendering.h`/`.cpp` (신규) | 순수 이관 |
| `src/Engine/Render/SwapChain.h`/`.cpp` (신규) | `Context&` 기반으로 이관 |
| `src/Engine/Render/GraphicsPipeline.h`/`.cpp` (신규) | `Context&` 기반으로 이관 |
| `src/Engine/CMakeLists.txt` | `EngineRender` 타겟 추가 |
| `example2/CMakeLists.txt` (신규) | `cube_render2` 타겟, 기존 `example/cube.vert`/`cube.frag` 재사용 컴파일 |
| `example2/cube_render.cpp` (신규) | 최소 프레임 루프로 큐브 렌더링 |
| `CMakeLists.txt` (루트) | `add_subdirectory(example2)` 추가 |

`vkCommon`/`vkSpatial`/`vkRender`/`example`/`test`는 변경 없음.

## CMake 통합

`src/Engine/CMakeLists.txt`에 기존 `EngineCore` 블록 뒤에 추가:
```cmake
file(GLOB_RECURSE ENGINE_RENDER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Render/*.cpp")

add_library(EngineRender STATIC ${ENGINE_RENDER_SOURCES})
add_library(Engine::Render ALIAS EngineRender)

target_link_libraries(EngineRender
        PUBLIC Engine::Core Vulkan::Vulkan)

target_include_directories(EngineRender
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        PRIVATE
            "${VULKAN_SDK}/include")
```
`shaderc`/`spirv-reflect`/`VKBVH_SHADER_DIR` 불필요 — `GraphicsPipeline`은 이미 컴파일된 `.spv`만 읽는다.

`example2/CMakeLists.txt`는 기존 `example/CMakeLists.txt`의 `cube_render` 타겟이 쓰는 `glslc` 커스텀 커맨드 패턴을 그대로 따르되, 셰이더 소스 경로를 `${CMAKE_SOURCE_DIR}/example/cube.vert`/`cube.frag`(기존 파일)로 지정해 복제 없이 재사용한다. 타겟명은 `cube_render2`(기존 `cube_render`와 이름 충돌 방지).

## 테스트/검증 방법

이 프로젝트의 기존 관례(Vulkan 렌더링은 실제 GPU+창 컨텍스트 필요, 실행으로 검증)를 따른다.

1. `cube_render2` 빌드 및 실행 — 창이 뜨고 회전하는 큐브가 배경과 함께 정상 렌더링되는지 확인 (기존 `example/cube_render`와 시각적으로 동일해야 함).
2. 창 리사이즈 시 크래시 없이 새 해상도로 계속 렌더링되는지 확인.
3. 창을 닫아 정상 종료(교착 상태나 크래시 없이) 확인.
4. 기존 `ctest` 스위트는 이 스펙과 무관하지만, 회귀 확인 차원에서 50/51(알려진 사전 실패 하나 제외) 그대로인지 확인.
