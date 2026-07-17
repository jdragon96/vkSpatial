# Engine/Core 설계 (vkCommon을 대체할 새 기반 레이어)

## 배경 / 목적

프로젝트 구조를 `src/Engine`으로 완전히 새로 잡기로 했다 (기존 `vkCommon`/`vkSpatial`/`vkRender`를 최종적으로 대체). 다만 한 번에 갈아엎지 않고, 항상 빌드되는 상태를 유지하면서 **기반 레이어 → 알고리즘 레이어 → 렌더링 레이어 → 소비자 마이그레이션 → 옛 코드 삭제** 순으로 단계적으로 진행한다. 이 스펙은 그 첫 단계, `Engine/Core`(기존 `vkCommon` 대체)를 다룬다.

`Engine/Core`는 두 가지 완전히 다른 소비자를 모두 지원해야 한다:
- **GPU 알고리즘 개발** (`vkSpatial`류): 창/스왑체인 없이 컨텍스트 → 버퍼 업로드 → 컴퓨트 디스패치 → 결과 회수만 필요.
- **렌더링 엔진** (`vkRender`류): 위에 더해 이미지, 프레젠트, 프레임 루프가 필요.

Filament의 Engine/Renderer/View 분리 같은 렌더링 전용 패턴은 두 번째 소비자에만 해당하므로, `Engine/Core` 자체는 렌더링을 전혀 모르는 순수 GPU 기반 레이어로 설계한다 — `vkSpatial`이 `vkRender` 없이도 동작하는 지금의 레이어링 방향을 그대로 유지한다.

이미 완료/설계된 두 작업이 이 스펙에 흡수된다:
- vk-bootstrap 기반 디바이스/인스턴스 초기화 (기존 `vkCommon::VkContext`, 이미 `main`에 병합됨) → `Engine::Core::Context`로 이관.
- VMA 기반 GPU 메모리 설계 (앞서 논의) → `Engine::Core::Buffer`/`Image`로 확장.

조사 중 발견한 "1회성 커맨드버퍼 생성→제출→대기" 중복 패턴(`vkGPUMemory::submitCopy`, `Capture.cpp`의 4개 헬퍼, `realtime_shadow.cpp`/`cube_render.cpp`의 인라인 업로드 풀)도 이번에 `OneShotCommands`로 통합한다.

## 범위

- `src/Engine/Core/`에 5개 신규 타입 추가: `Context`, `Buffer`, `Image`, `ComputePipeline`, `OneShotCommands`.
- VMA를 `lib/vma/`에 벤더링 (단일 헤더).
- 새 CMake 타겟 `EngineCore` 추가.
- `Engine::Core`의 새 동작을 검증하는 GTest 테스트 추가.

### 범위 밖

- **기존 `vkCommon`/`vkSpatial`/`vkRender`/모든 example/test는 이 스펙에서 변경하지 않는다.** `Engine/Core`는 완성되어도 아직 아무도 쓰지 않는, 독립적으로 빌드되는 새 라이브러리로 남는다. 기존 코드를 `Engine/Core`로 옮기는 마이그레이션은 `Engine/Spatial`, `Engine/Render`가 만들어진 뒤 별도 스펙으로 진행한다.
- `Engine/Spatial`(BVH 등), `Engine/Render`(렌더링 엔진) — 각각 별도 후속 스펙.
- `vkComputeBase`의 fluent 빌더 API(`.Build().Bind().Args().Dispatch()`) 자체의 재설계 — 이미 좋은 설계라 판단, `ComputePipeline`으로 옮기면서 생성자만 단순화한다.
- 검증 레이어/디버그 메신저 — 이번 스펙 밖 (기존 관례와 동일하게 켜지 않음).

## 아키텍처

### 1. `Engine::Core::Context`

기존 `vkCommon::VkContext`(vk-bootstrap 기반, 이미 검증됨)를 이관하되, **명시적 `init()`/`shutdown()`을 진짜 RAII로 전환**한다 — 지금은 호출자가 `shutdown()`을 깜빡할 수 있는 구조인데, 생성자/소멸자로 바꾸면 그 버그 클래스 자체가 사라진다.

```cpp
namespace Engine::Core {

    class Context {
    public:
        VkInstance instance = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;

        VkQueue computeQueue = VK_NULL_HANDLE;
        uint32_t computeFamily = 0;
        VkCommandPool cmdPool = VK_NULL_HANDLE;

        // enablePresent=true일 때만 채워짐
        VkQueue graphicsQueue = VK_NULL_HANDLE;
        uint32_t graphicsFamily = 0;
        VkCommandPool graphicsCmdPool = VK_NULL_HANDLE;   // 신규 — 아래 참고
        VkSurfaceKHR surface = VK_NULL_HANDLE;

        VmaAllocator allocator = VK_NULL_HANDLE;           // 신규 — VMA 통합

        Context(bool enablePresent = false,
               const std::vector<const char *> &extraInstanceExtensions = {},
               const std::function<VkSurfaceKHR(VkInstance)> &surfaceFactory = nullptr);
        ~Context();

        Context(const Context &) = delete;
        Context &operator=(const Context &) = delete;

    private:
        bool m_presentEnabled = false;
    };

} // namespace Engine::Core
```

- 내부 구현은 기존 `vkContext.cpp`(vk-bootstrap 기반, `main`에 이미 병합됨)를 그대로 가져온다 — `vkb::InstanceBuilder`→`vkb::PhysicalDeviceSelector`→`vkb::DeviceBuilder` 체인, present 큐를 `vkb::QueueType::present`로 얻고 `VK_QUEUE_GRAPHICS_BIT` 여부를 명시적으로 검증하는 로직(최근 리뷰에서 고친 것)까지 그대로 이식한다.
- 소멸자 마지막에 생성 시점의 역순으로 정리: `graphicsCmdPool`(있으면) → `cmdPool` → `vmaDestroyAllocator(allocator)`(디바이스 파괴 **전**) → `device` → `surface` → `instance`.
- **`graphicsCmdPool` 신규 이유**: 지금 `realtime_shadow.cpp`/`cube_render.cpp`의 `CreateGeometry()`, `Capture::CaptureToPNG`가 호출될 때마다 매번 새 임시 커맨드풀을 만들었다 지운다. `Context`가 `enablePresent`일 때 이 풀을 한 번만 만들어 갖고 있으면, `OneShotCommands`(아래)가 매번 재사용할 수 있다 — 성능과 코드 중복을 동시에 개선.
- `VmaAllocator` 생성은 `device`가 만들어진 직후, `VmaAllocatorCreateInfo{instance, physicalDevice, device, vulkanApiVersion=VK_API_VERSION_1_3}`로 `vmaCreateAllocator(...)` 호출.

### 2. `Engine::Core::OneShotCommands`

```cpp
namespace Engine::Core {

    enum class QueueRole { Compute, Graphics };

    // `role`에 맞는 큐/커맨드풀(Context가 소유)로 1회성 커맨드버퍼를 할당해 `record`를
    // 기록시키고, 제출 후 완료까지 블로킹 대기한다. 실패 시 std::runtime_error.
    void SubmitOneShot(Context &context,
                       QueueRole role,
                       const std::function<void(VkCommandBuffer)> &record);

} // namespace Engine::Core
```

- `QueueRole::Compute` → `context.computeQueue` + `context.cmdPool`. `QueueRole::Graphics` → `context.graphicsQueue` + `context.graphicsCmdPool` (이 role을 쓰려면 `Context`가 `enablePresent=true`로 만들어져 있어야 하며, 아니면 `std::runtime_error`).
- 내부 동작은 지금 `vkGPUMemory::submitCopy`/`Capture.cpp`의 `AllocateCommandBuffer`+`BeginOneTimeCommands`+`EndSubmitAndWait` 패턴과 동일 (1개 커맨드버퍼 할당 → `VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT`로 begin → `record(cmd)` 호출 → end → submit → `vkQueueWaitIdle` → 커맨드버퍼 해제). 다른 점은 커맨드풀을 매번 새로 만들지 않고 `Context`가 미리 가진 걸 재사용한다는 것.

### 3. `Engine::Core::Buffer` (vkGPUMemory 대체)

```cpp
namespace Engine::Core {

    class Buffer {
    public:
        Buffer(Context &context, VkBufferUsageFlags usage = 0);
        ~Buffer();

        Buffer(const Buffer &) = delete;
        Buffer &operator=(const Buffer &) = delete;

        void Allocate(uint32_t bytes);
        void Upload(const void *data, uint32_t bytes, QueueRole role = QueueRole::Compute);
        void Download(void *data, uint32_t bytes, QueueRole role = QueueRole::Compute);

        VkBuffer Handle() const { return m_buffer; }
        uint32_t Size() const { return m_size; }

    private:
        Context &m_context;
        VkBufferUsageFlags m_extraUsage;
        VkBuffer m_buffer = VK_NULL_HANDLE;
        VmaAllocation m_allocation = VK_NULL_HANDLE;
        uint32_t m_size = 0;
    };

} // namespace Engine::Core
```

- `Allocate(bytes)`: `vmaCreateBuffer`로 `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | m_extraUsage` 버퍼를 `VmaAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO`로 생성 (device-local로 귀결, 기존 `vkGPUMemory::Allocate`와 동일한 최종 배치). 이미 할당되어 있으면 먼저 `vmaDestroyBuffer`.
- `Upload`/`Download`: 임시 스테이징 버퍼를 `VmaAllocationCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT`(업로드) / `..._RANDOM_BIT | ..._MAPPED_BIT`(다운로드)로 만들면 `VmaAllocationInfo::pMappedData`에서 바로 포인터를 얻는다 — **`vkScopedMemory`에 대응하는 타입을 따로 만들 필요가 없다** (VMA의 persistent-mapped 플래그가 그 역할을 대신함). `memcpy` 후 `OneShotCommands::SubmitOneShot(m_context, role, [&](cmd){ vkCmdCopyBuffer(...); })`로 GPU 복사, 끝나면 `vmaDestroyBuffer`로 스테이징 정리.
- 실패 시 전부 `std::runtime_error("Buffer: ...")` — 기존 `vkGPUMemory`만 유일하게 `bool`을 반환하던 불일치를 해소하고 코드베이스의 지배적 관례(`std::runtime_error`)로 통일.
- **`vkScopedMemory`는 `Engine::Core`로 이관하지 않는다** — 리포 전체에서 `vkGPUMemory.cpp` 자기 자신 말고 아무도 안 쓰는 걸 이미 확인했고, VMA의 매핑 방식(`VMA_ALLOCATION_CREATE_MAPPED_BIT`)이 그 필요를 대체한다.

### 4. `Engine::Core::Image` (vkRender::Image에서 이관)

지금 `vkRender::Image`가 이미 잘 설계되어 있는 부분(레이아웃 추적, `TransitionLayout`, `ImageDescriptor` 빌더 팩토리)은 그대로 유지하고, 메모리 할당 내부만 VMA로 교체한다.

```cpp
namespace Engine::Core {

    struct ImageDescriptor {
        // 기존 vkRender::ImageDescriptor와 동일한 필드 구성 유지
        // (width/height/depth/mipLevels/arrayLayers/format/usage/aspectMask/
        //  imageType/viewType/samples/tiling/initialLayout/sharingMode/createView)
        // 단, memoryProperties(VkMemoryPropertyFlags) 필드는 제거 — VMA가
        // VMA_MEMORY_USAGE_AUTO로 알아서 고르므로 더 이상 호출자가 지정할 필요 없음.

        static ImageDescriptor Depth2D(VkExtent2D extent, VkFormat format = VK_FORMAT_D32_SFLOAT);
        static ImageDescriptor Color2D(VkExtent2D extent, VkFormat format,
                                       VkImageUsageFlags usage =
                                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                               VK_IMAGE_USAGE_SAMPLED_BIT);
    };

    class Image {
    public:
        explicit Image(Context &context);
        Image(Context &context, const ImageDescriptor &descriptor);
        ~Image();

        Image(const Image &) = delete;
        Image &operator=(const Image &) = delete;
        Image(Image &&rhs) noexcept;
        Image &operator=(Image &&rhs) noexcept;

        Image &Create(const ImageDescriptor &descriptor);
        void Destroy();

        // Handle()/Memory()/View()/Format()/AspectMask()/CurrentLayout()/Extent()/
        // Extent2D()/Descriptor()/Valid()/HasView()/Matches(...)/TransitionLayout(...)
        // — 기존 vkRender::Image와 동일하게 유지 (아래 참고)

    private:
        Context *m_context = nullptr;
        VkImage m_image = VK_NULL_HANDLE;
        VmaAllocation m_allocation = VK_NULL_HANDLE;   // VkDeviceMemory m_memory 대체
        VkImageView m_view = VK_NULL_HANDLE;
        ImageDescriptor m_descriptor{};
        VkImageLayout m_currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    };

} // namespace Engine::Core
```

- `Memory()` 접근자(기존에 `VkDeviceMemory`를 반환하던 것)는 제거한다 — 조사 결과 이 접근자를 실제로 쓰는 곳이 없었고(전부 `Handle()`/`View()`/`Format()` 등만 사용), VMA에서 개별 이미지의 raw `VkDeviceMemory`를 노출하는 게 자연스럽지 않다(여러 이미지가 하나의 디바이스 메모리 블록을 서브 할당해 공유할 수 있음). 필요해지면 `VmaAllocationInfo::deviceMemory`로 나중에 추가 가능.
- `CreateImageObject()`/`AllocateAndBindMemory()`(2단계로 나뉘어 있던 것) 대신 `vmaCreateImage(allocator, &imageCreateInfo, &allocInfo, &m_image, &m_allocation, nullptr)` 한 번으로 통합 — `vkCreateImage`+`vkAllocateMemory`+`vkBindImageMemory` 3단계가 원자적 1단계가 된다.
- `Image::FindMemoryType()` 전체 삭제 — VMA가 대신함. `TransitionLayout(...)` 두 오버로드(인스턴스용/정적 raw-handle용)는 메모리 할당과 무관하므로 그대로 유지.

### 5. `Engine::Core::ComputePipeline` (vkComputeBase 대체)

```cpp
namespace Engine::Core {

    class ComputePipeline {
    public:
        explicit ComputePipeline(Context &context);
        ~ComputePipeline();

        ComputePipeline(const ComputePipeline &) = delete;
        ComputePipeline &operator=(const ComputePipeline &) = delete;

        // AddInclude/Build/Bind/Args/Dispatch/DispatchElements/Sync — 기존
        // vkComputeBase와 동일한 fluent 시그니처 유지 (이미 좋은 설계).

    private:
        Context &m_context;
        // ... 기존 vkComputeBase의 나머지 멤버(파이프라인/디스크립터셋 캐싱 등) 동일
    };

} // namespace Engine::Core
```

- 생성자만 `(VkDevice, VkPhysicalDevice, VkQueue, VkCommandPool)` 4개 raw 인자 대신 `Context&` 하나로 단순화 — 내부에서 `m_context.device`/`.physicalDevice`/`.computeQueue`/`.cmdPool`을 그대로 쓴다.
- `Dispatch()`의 내부 제출도 `OneShotCommands::SubmitOneShot(m_context, QueueRole::Compute, [&](cmd){ ...bind+dispatch... })`로 통일한다 — 지금은 `vkComputeBase`가 자체적으로 커맨드버퍼 할당/제출/대기를 따로 구현하고 있는데, 이걸 `Buffer`의 업로드/다운로드와 같은 통로로 합친다. `Sync()`는 기존처럼 유지하되(호출부 호환), 이제 명확히 no-op에 가까운 이유(Dispatch가 이미 블로킹)를 문서화한다.

## 파일 변경 목록

| 파일 | 변경 |
|---|---|
| `lib/vma/vk_mem_alloc.h` (신규) | VMA 단일 헤더 벤더링 |
| `src/Engine/Core/Context.h`/`.cpp` (신규) | vk-bootstrap 기반, `VmaAllocator`+`graphicsCmdPool` 추가 |
| `src/Engine/Core/OneShotCommands.h`/`.cpp` (신규) | 1회성 커맨드버퍼 통합 헬퍼 |
| `src/Engine/Core/Buffer.h`/`.cpp` (신규) | VMA 기반 버퍼, 예외 기반 에러 처리 |
| `src/Engine/Core/Image.h`/`.cpp` (신규) | VMA 기반 이미지, 레이아웃 추적은 기존 유지 |
| `src/Engine/Core/ComputePipeline.h`/`.cpp` (신규) | `Context&` 기반으로 단순화된 vkComputeBase |
| `src/Engine/CMakeLists.txt` (신규) | `EngineCore` 정적 라이브러리 타겟 |
| `src/CMakeLists.txt` | `add_subdirectory(Engine)` 추가 |
| `test/test_engineCore.cpp` (신규) | `Context`/`Buffer`/`Image`/`ComputePipeline` 동작 검증 |
| `test/CMakeLists.txt` | `EngineCore::EngineCore` 링크 추가 |

`vkCommon`, `vkSpatial`, `vkRender`, 기존 example/test는 이 스펙에서 **한 글자도 바꾸지 않는다**.

## CMake 통합

`lib/vma/`는 단일 헤더라 별도 라이브러리 타겟이 필요 없다 — `src/Engine/Core/Buffer.cpp`(또는 별도의 얇은 `.cpp` 하나)에서 `#define VMA_IMPLEMENTATION` 후 `#include "vma/vk_mem_alloc.h"`로 구현을 컴파일한다 (`Capture.cpp`가 `stb_image_write.h`를 쓰는 것과 동일 패턴). `lib`가 이미 `PUBLIC` include 경로에 있는 관례(`src/CMakeLists.txt`)를 그대로 따른다.

`src/Engine/CMakeLists.txt`:
```cmake
file(GLOB_RECURSE ENGINE_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Core/*.cpp")
add_library(EngineCore STATIC ${ENGINE_CORE_SOURCES})
add_library(Engine::Core ALIAS EngineCore)

target_link_libraries(EngineCore
        PUBLIC  Vulkan::Vulkan
        PRIVATE ${SHADERC_LIB} spirv-reflect vk-bootstrap::vk-bootstrap)

target_include_directories(EngineCore
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib>
        PRIVATE
            "${VULKAN_SDK}/include")

target_compile_definitions(EngineCore PRIVATE
        VKBVH_SHADER_DIR=\"${VKBVH_SHADER_DIR}\")
```
`vk-bootstrap::vk-bootstrap`는 `PRIVATE`다 — 기존 `vkCommon`의 `VkContext` 마이그레이션과 동일하게, `vkb::` 타입은 `Context.cpp` 내부에서만 쓰고 `Context.h`에는 전혀 노출하지 않는다 (`Context.h`는 순수 Vulkan 핸들 타입만 포함). 이미 검증된 패턴을 그대로 재사용한다.

`src/Engine`은 `src/` 하위의 새 디렉토리이므로, `src/CMakeLists.txt`에 `add_subdirectory(Engine)`을 파일 맨 앞(`vkCommon`/`vkSpatial`/`vkRender` 타겟 정의보다 먼저)에 추가한다. `EngineCore`는 이 스펙 범위에서 다른 어떤 타겟과도 링크되지 않으므로 위치 자체는 형제 디렉토리들과 순서상 무관하지만, 앞쪽에 두는 게 읽기 자연스럽다. 루트 `CMakeLists.txt`는 변경하지 않는다 (`add_subdirectory(src)` 호출 하나가 이미 `src/CMakeLists.txt`를 통해 `Engine`까지 끌어옴).

## 테스트/검증 방법

`Engine/Core`는 아직 아무 소비자가 없는 새 라이브러리이므로, 기존 프로젝트 관례(실제 실행으로 검증)와 별개로 **새 동작을 처음부터 GTest로 직접 검증**한다 — 특성화 테스트가 아니라 신규 기능 테스트다.

`test/test_engineCore.cpp`에 추가할 테스트 (모두 실제 GPU 컨텍스트 필요, `VkComputeTest`류 기존 픽스처와 동일하게 헤드리스로 동작):

1. `ContextTest.ComputeOnlyConstructionProducesValidHandles` — `Context ctx;`(기본 `enablePresent=false`) 생성 후 `instance`/`physicalDevice`/`device`/`computeQueue`/`cmdPool`/`allocator`가 전부 non-null인지, `graphicsQueue`/`graphicsCmdPool`/`surface`는 null인지 확인.
2. `BufferTest.AllocateUploadDownloadRoundTrip` — `Buffer`에 N개의 `float`을 `Upload`한 뒤 `Download`해서 원본과 일치하는지 확인 (`vkGPUMemory`의 기존 유닛 테스트가 없었으므로, 이번에 처음으로 이 경로를 직접 테스트하게 됨).
3. `ImageTest.CreateDepth2DProducesValidHandles` — `ImageDescriptor::Depth2D(...)`로 `Image`를 만들고 `Valid()`/`HasView()`/`Format()`이 기대값과 일치하는지 확인.
4. `ComputePipelineTest.DispatchSimpleShaderProducesExpectedOutput` — 기존 `test/test_vkCompute.cpp`의 `SumZeroToTenThousand`과 동등한 검증을 `ComputePipeline`으로 재현 (같은 셰이더 재사용 가능하면 재사용).

실행: `cmake --build build --parallel && ctest --test-dir build --output-on-failure -R "ContextTest|BufferTest|ImageTest|ComputePipelineTest"`. 기존 44/45 테스트(알려진 사전 실패 하나 제외)에 영향 없이 그대로 통과해야 한다.
