# vk-bootstrap 통합 설계 (`VkContext` 인프라 교체)

## 배경 / 목적

이 프로젝트의 진짜 가치는 `vkSpatial`의 GPU LBVH/Wide BVH 빌드 알고리즘에 있다. 그 주변을 감싸는 Vulkan 인스턴스/디바이스 초기화 보일러플레이트(`vkCommon::VkContext`)는 직접 구현되어 있는데, 이 부분은 이미 검증된 오픈소스 라이브러리로 안전하게 대체할 수 있는 영역이다.

전면 재작성 대신 **인프라 레이어만 점진적으로 교체**하는 방향을 택했다 (기존 BVH/렌더링 자산을 보존하면서, 각 교체 단계가 독립적으로 빌드·검증 가능하도록). 이번 스펙은 그 첫 단계로, `VkContext`의 인스턴스/물리 디바이스 선택/논리 디바이스 생성 내부 구현을 [vk-bootstrap](https://github.com/charles-lunarg/vk-bootstrap)으로 교체한다. VMA(GPU 메모리 할당자) 도입은 별도의 후속 스펙으로 다룬다.

## 범위

- `src/vkCommon/vkContext.cpp`의 `createInstance()`/`pickPhysicalDevice()`/`createDevice()` 내부 구현을 vk-bootstrap 기반으로 교체
- `vk-bootstrap`을 CMake `FetchContent`로 통합, `vkCommon` 타겟에 `PRIVATE` 링크
- 기존 예제/테스트가 재컴파일만으로 그대로 동작하는지 검증

### 범위 밖

- **`vkContext.h`의 공개 인터페이스 변경** — 필드(`instance`, `physDevice`, `device`, `computeQueue`, `computeFamily`, `cmdPool`, `graphicsQueue`, `graphicsFamily`, `surface`)와 `init()`/`shutdown()` 시그니처는 그대로 유지한다. 이게 이 스펙의 핵심 제약이다 — `vkSpatial`/`vkRender`/모든 example이 이 필드들을 직접 읽으므로, 인터페이스가 바뀌면 이번 스펙의 "안전하게 격리된 교체"라는 전제가 깨진다.
- `createCommandPool()` — vk-bootstrap은 커맨드풀을 다루지 않으므로 변경 없음.
- 검증 레이어/디버그 메신저 활성화 — 지금 코드가 켜지 않고 있으므로 이번 스펙도 켜지 않는다. 필요해지면 별도 스펙.
- VMA 도입, `vkGPUMemory`/`Image`/`Capture`의 메모리 관리 리팩터링 — 다음 단계로 분리.

## 아키텍처

### 1. `createInstance()` → `vkb::InstanceBuilder`

```cpp
vkb::InstanceBuilder builder;
builder.set_app_name("vkbvh")
       .require_api_version(1, 3, 0);

for (const char *ext : extraInstanceExtensions)
    if (enablePresent) builder.enable_extension(ext);

#ifdef __APPLE__
// macOS/MoltenVK 포터빌리티 열거 확장 — vk-bootstrap이 자동으로 처리하는지
// 설치된 버전 기준으로 실제 빌드해서 확인한다. 자동 처리가 안 되면
// builder.enable_extension_if_present(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)를
// 명시적으로 추가한다 (구현 시점 확인 사항, 아래 리스크 참고).
#endif

auto instRet = builder.build();
if (!instRet)
    throw std::runtime_error("VkContext: failed to create instance: " + instRet.error().message());
vkb::Instance vkbInstance = instRet.value();
instance = vkbInstance.instance;   // 기존 필드에 raw 핸들만 저장
```

### 2. Surface 생성 — 변경 없음

기존과 동일하게 `surfaceFactory(instance)` 콜백을 그대로 호출한다 (vk-bootstrap은 창 시스템 종속적인 surface 생성을 의도적으로 담당하지 않음 — 지금 설계와 이미 일치).

### 3. `pickPhysicalDevice()` + `createDevice()` → `vkb::PhysicalDeviceSelector` + `vkb::DeviceBuilder`

```cpp
vkb::PhysicalDeviceSelector selector(vkbInstance, surface);  // surface는 enablePresent일 때만 유효한 값
selector.set_minimum_version(1, 3)
        .require_present(enablePresent);

if (enablePresent) {
    selector.add_required_extension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    VkPhysicalDeviceVulkan13Features features13{};
    features13.dynamicRendering = VK_TRUE;
    selector.set_required_features_13(features13);
}
// disable_portability_subset()를 호출하지 않음 — vk-bootstrap이 기본으로
// VK_KHR_portability_subset을 사용 가능하면 자동 활성화 (기존 #ifdef __APPLE__ 수동 처리 대체)

auto physRet = selector.select();
if (!physRet)
    throw std::runtime_error("VkContext: failed to select physical device: " + physRet.error().message());
vkb::PhysicalDevice vkbPhysDevice = physRet.value();
physDevice = vkbPhysDevice.physical_device;

vkb::DeviceBuilder deviceBuilder(vkbPhysDevice);
auto devRet = deviceBuilder.build();
if (!devRet)
    throw std::runtime_error("VkContext: failed to create device: " + devRet.error().message());
vkb::Device vkbDevice = devRet.value();
device = vkbDevice.device;

auto computeQueueRet = vkbDevice.get_queue(vkb::QueueType::compute);
auto computeFamilyRet = vkbDevice.get_queue_index(vkb::QueueType::compute);
if (!computeQueueRet || !computeFamilyRet)
    throw std::runtime_error("VkContext: no compute queue available");
computeQueue = computeQueueRet.value();
computeFamily = computeFamilyRet.value();

if (enablePresent) {
    // "graphics"가 아니라 "present" 큐 타입을 쓴다 — 기존 코드의 실제 요구사항이
    // "그래픽스 큐이면서 이 surface에 present 가능한 큐"이기 때문에 의미가 더 정확하다.
    auto presentQueueRet = vkbDevice.get_queue(vkb::QueueType::present);
    auto presentFamilyRet = vkbDevice.get_queue_index(vkb::QueueType::present);
    if (!presentQueueRet || !presentFamilyRet)
        throw std::runtime_error("VkContext: no present-capable graphics queue available");
    graphicsQueue = presentQueueRet.value();
    graphicsFamily = presentFamilyRet.value();
}
```

### 4. `createCommandPool()` — 변경 없음

지금 코드 그대로 `vkCreateCommandPool(device, ..., computeFamily, ...)`.

### 5. `shutdown()` — 변경 없음

`vkb::Instance`/`vkb::Device`/`vkb::PhysicalDevice`는 RAII 객체가 아니라 값 구조체이므로(자체 소멸자가 핸들을 자동 파괴하지 않음), raw 핸들만 뽑아 쓰고 지금처럼 `vkDestroyDevice`/`vkDestroySurfaceKHR`/`vkDestroyInstance`를 직접 호출해도 완전히 동일하게 동작한다. `vkb::Instance`/`vkb::Device` 값 자체는 `init()` 내부 지역 변수로만 존재하다 스코프를 벗어나며 사라진다 — 새 멤버 변수를 추가할 필요가 없다.

## CMake 변경

`CMakeLists.txt` (루트):

```cmake
include(FetchContent)
FetchContent_Declare(
    fetch_vk_bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap
    GIT_TAG        <구현 시점의 최신 안정 릴리스 태그로 고정>
)
FetchContent_MakeAvailable(fetch_vk_bootstrap)
```

`src/CMakeLists.txt`의 `vkCommon` 타겟:

```cmake
target_link_libraries(vkCommon
        PUBLIC  Vulkan::Vulkan
        PRIVATE ${SHADERC_LIB} spirv-reflect vk-bootstrap::vk-bootstrap)
```

`PRIVATE`로 링크하는 게 핵심이다 — `vkContext.h`가 vk-bootstrap 타입을 전혀 노출하지 않으므로, `vkSpatial`/`vkRender`/example 어느 타겟도 vk-bootstrap을 알 필요가 없다. `vk-bootstrap`은 소스가 작아(헤더/cpp 몇 개) 기존 `lib/`에 벤더링된 imgui/stb/spirv-reflect와 성격이 비슷하지만, GitHub에서 특정 태그로 직접 받아오는 `FetchContent` 방식이 vk-bootstrap 자체 문서가 권장하는 표준 통합 방법이라 이를 따른다.

## 파일 변경 목록

| 파일 | 변경 |
|---|---|
| `CMakeLists.txt` (루트) | `FetchContent`로 vk-bootstrap 선언/획득 |
| `src/CMakeLists.txt` | `vkCommon` 타겟에 `vk-bootstrap::vk-bootstrap` `PRIVATE` 링크 추가 |
| `src/vkCommon/vkContext.cpp` | `createInstance()`/`pickPhysicalDevice()`/`createDevice()` 내부를 vk-bootstrap 호출로 교체. `createCommandPool()`/`shutdown()`은 변경 없음 |
| `src/vkCommon/vkContext.h` | **변경 없음** (이 스펙의 핵심 검증 대상) |

## 에러 처리

vk-bootstrap의 각 빌더는 `Result<T>`를 반환하며 `operator bool()`로 성공 여부를 확인한다. 실패 시 `result.error().message()`로 사람이 읽을 수 있는 에러 메시지를 얻어, 기존 코드베이스 관례대로 `std::runtime_error`로 감싸 던진다 (기존 `VkContext`가 이미 이 관례를 쓰고 있어 자연스럽게 이어진다).

## 리스크 / 구현 시 확인 사항

1. **macOS 포터빌리티 열거** — `VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME` 인스턴스 확장과 `VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR` 플래그를 vk-bootstrap이 자동으로 처리하는지, 아니면 수동으로 `enable_extension_if_present(...)`를 호출해야 하는지 실제 설치된 vk-bootstrap 버전으로 빌드해보며 확인해야 한다. 이 사용자의 개발 환경이 macOS(Darwin)이므로 이번 마이그레이션에서 가장 먼저, 가장 확실하게 검증해야 할 항목이다.
2. **디바이스 선택 변경** — 기존 코드는 `vkEnumeratePhysicalDevices`가 반환하는 첫 번째 디바이스를 무조건 선택했다(스코어링 없음). vk-bootstrap의 기본 `.select()`는 discrete GPU를 우선시하는 스코어링을 적용한다 — 멀티 GPU 환경에서 다른 디바이스가 선택될 수 있다. 이는 의도된 개선으로 간주하되, 검증 단계에서 실제로 이전과 동일한(또는 더 적합한) 디바이스가 선택되는지 로그로 확인한다.
3. **큐 생성 범위 확장** — `vkb::DeviceBuilder().build()`는 기본적으로 물리 디바이스가 노출하는 모든 큐 패밀리에서 큐를 하나씩 만든다. 지금 코드는 필요한 큐(compute, 그리고 필요 시 present)만 명시적으로 만든다. 여분의 큐가 생겨도 사용하지 않으면 문제되지 않지만, 만약 예상치 못한 동작이 발견되면 `vkb::DeviceBuilder::custom_queue_setup(...)`으로 큐 생성 범위를 좁힐 수 있다.

## 테스트/검증 방법

1. `CMakeLists.txt`/`src/CMakeLists.txt` 변경 후 전체 프로젝트를 처음부터 다시 빌드 (`vk-bootstrap` FetchContent 다운로드 포함).
2. 콘솔 창을 여는 example들(`cube_render`, `realtime_shadow`, `bvh_shadow_room`, `bvh_path_tracer`, `native_raytrace` 등)을 각각 실행해 창 생성/렌더링이 이전과 동일하게 동작하는지 확인 — 특히 셰도우맵/레이트레이싱 예제는 `enablePresent=true` 경로(그래픽스+present 큐)를 검증하는 데 적합하다.
3. `enablePresent=false` 경로(순수 컴퓨트, BVH 빌드/쿼리 관련 example)도 최소 하나 실행해 compute-only 초기화가 정상 동작하는지 확인.
4. `test/` 디렉토리의 전체 테스트 스위트(`ctest`)를 실행해 회귀가 없는지 확인.
5. 콘솔에 출력되는 `[VkContext] Device: ...` 로그로 어떤 GPU가 선택됐는지 확인 (리스크 2번 항목 검증).
