# Enter 키 스크린샷 캡처 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `cube_render.cpp`에서 Enter 키를 누르면 현재 화면을 PNG로 저장한다.

**Architecture:** `vkRender`에 `MouseInput`과 동일한 패턴의 `KeyInput` 리스너 컴포넌트를 추가하고, `Renderer`/`SwapChain` 내부에 의존하지 않는 독립 유틸리티 `Screenshot::CaptureToPNG(VkImage, ...)`를 추가한다. 예제 코드는 GLFW 키 콜백을 `KeyInput`에 연결하고, Enter 키 리스너에서 캡처 플래그를 세팅해 메인 루프의 `EndFrame()` 직후 `Screenshot::CaptureToPNG`를 호출한다.

**Tech Stack:** C++17, Vulkan, GLFW, `stb_image_write.h` (신규 벤더링, public domain 단일 헤더 PNG 인코더).

## Global Constraints

- 기존 코드 스타일을 따른다: 실패 시 `std::runtime_error` throw (에러 코드/optional 리턴 금지), 4-space 들여쓰기, `PascalCase` 메서드명.
- 스펙(`docs/superpowers/specs/2026-07-13-key-input-screenshot-design.md`)에서 검증 방법을 "자동화된 유닛 테스트보다는 실제 실행으로 검증"으로 명시적으로 승인받았다. `vkRender`에는 현재 GTest가 전혀 연결되어 있지 않고(`test/CMakeLists.txt`는 `vkSpatial`만 링크), `MouseInput` 같은 기존 동종 컴포넌트에도 유닛 테스트가 없다. 이 계획은 그 결정을 따르며 **새 GTest 테스트를 추가하지 않는다** — 각 태스크의 "테스트" 단계는 빌드 성공 + 수동 실행 확인이다.
- `${CMAKE_SOURCE_DIR}/lib`는 이미 `vkRender` 타깃의 `PUBLIC` include 경로에 포함되어 있다 (`src/CMakeLists.txt:61-64`). `src/CMakeLists.txt`는 `file(GLOB_RECURSE VKRENDER_SOURCES "vkRender/*.cpp")`로 소스를 자동 수집하므로, 새 `.cpp` 파일 추가 시 CMakeLists 수정은 필요 없다.
- 빌드 명령: `cmake -S . -B build && cmake --build build --parallel` (전체) 또는 `cmake --build build --target vkRender cube_render` (부분, `README.md:22-23` 참고).

---

### Task 1: `vkRender::KeyInput` 컴포넌트 추가

**Files:**
- Create: `src/vkRender/KeyInput.h`
- Create: `src/vkRender/KeyInput.cpp`
- Modify: `src/vkRender/Engine.h`
- Modify: `src/vkRender/Engine.cpp`
- Modify: `src/vkRender/vkRender.h`

**Interfaces:**
- Produces: `vkRender::KeyEventType{Any,Press,Release,Repeat}`, `vkRender::KeyEvent{type, keyCode(int), modifiers(uint32_t), timestampSeconds(double), handled(bool)}`, `vkRender::KeyInput{AddListener(KeyEventType, Callback, int priority=0) -> ListenerId, RemoveListener(ListenerId) -> bool, ClearListeners(), OnKey(int keyCode, KeyEventType, uint32_t modifiers=0, double timestampSeconds=0.0), IsKeyDown(int keyCode) const -> bool}`, `vkRender::KeyListenerGroup`, `vkRender::Engine::CreateKeyInput() const -> std::unique_ptr<KeyInput>`.
- Consumes: 없음 (이 태스크는 독립적인 신규 컴포넌트이며 Vulkan 리소스에 의존하지 않는다).

- [ ] **Step 1: `KeyInput.h` 작성**

`src/vkRender/KeyInput.h`:

```cpp
#pragma once

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace vkRender {

    enum class KeyEventType : uint32_t {
        Any = 0,
        Press,
        Release,
        Repeat,
    };

    struct KeyEvent {
        KeyEventType type = KeyEventType::Press;
        int keyCode = 0;
        uint32_t modifiers = 0;
        double timestampSeconds = 0.0;
        bool handled = false;
    };

    class KeyInput {
    public:
        using ListenerId = uint64_t;
        using Callback = std::function<void(KeyEvent &)>;

        ListenerId AddListener(KeyEventType type,
                               Callback callback,
                               int priority = 0);
        bool RemoveListener(ListenerId id);
        void ClearListeners();

        void OnKey(int keyCode,
                   KeyEventType type,
                   uint32_t modifiers = 0,
                   double timestampSeconds = 0.0);

        bool IsKeyDown(int keyCode) const;

    private:
        struct Listener {
            ListenerId id = 0;
            KeyEventType type = KeyEventType::Any;
            int priority = 0;
            Callback callback;
        };

        std::unordered_map<ListenerId, Listener> m_listeners;
        std::unordered_map<int, bool> m_keyStates;
        ListenerId m_nextListenerId = 1;

        void Dispatch(KeyEvent event);
    };

    class KeyListenerGroup {
    public:
        KeyListenerGroup() = default;
        explicit KeyListenerGroup(KeyInput &input);
        ~KeyListenerGroup();

        KeyListenerGroup(const KeyListenerGroup &) = delete;
        KeyListenerGroup &operator=(const KeyListenerGroup &) = delete;

        KeyListenerGroup(KeyListenerGroup &&other) noexcept;
        KeyListenerGroup &operator=(KeyListenerGroup &&other) noexcept;

        KeyInput::ListenerId Add(KeyEventType type,
                                 KeyInput::Callback callback,
                                 int priority = 0);
        void Clear();
        bool Empty() const { return m_listenerIds.empty(); }

    private:
        KeyInput *m_input = nullptr;
        std::vector<KeyInput::ListenerId> m_listenerIds;
    };

} // namespace vkRender
```

- [ ] **Step 2: `KeyInput.cpp` 작성**

`src/vkRender/KeyInput.cpp`:

```cpp
#include "vkRender/KeyInput.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace vkRender {

    KeyInput::ListenerId KeyInput::AddListener(KeyEventType type,
                                               Callback callback,
                                               int priority) {
        if (!callback)
            throw std::runtime_error("KeyInput::AddListener received empty callback");

        const ListenerId id = m_nextListenerId++;
        m_listeners.emplace(id, Listener{id, type, priority, std::move(callback)});
        return id;
    }

    bool KeyInput::RemoveListener(ListenerId id) {
        return m_listeners.erase(id) != 0;
    }

    void KeyInput::ClearListeners() {
        m_listeners.clear();
    }

    void KeyInput::OnKey(int keyCode,
                         KeyEventType type,
                         uint32_t modifiers,
                         double timestampSeconds) {
        if (type == KeyEventType::Press)
            m_keyStates[keyCode] = true;
        else if (type == KeyEventType::Release)
            m_keyStates[keyCode] = false;

        KeyEvent event{};
        event.type = type;
        event.keyCode = keyCode;
        event.modifiers = modifiers;
        event.timestampSeconds = timestampSeconds;

        Dispatch(event);
    }

    bool KeyInput::IsKeyDown(int keyCode) const {
        const auto it = m_keyStates.find(keyCode);
        return it != m_keyStates.end() && it->second;
    }

    void KeyInput::Dispatch(KeyEvent event) {
        std::vector<Listener> listeners;
        listeners.reserve(m_listeners.size());
        for (const auto &entry: m_listeners) {
            const Listener &listener = entry.second;
            if (listener.type == KeyEventType::Any || listener.type == event.type)
                listeners.push_back(listener);
        }

        std::sort(listeners.begin(), listeners.end(),
                  [](const Listener &a, const Listener &b) {
                      if (a.priority != b.priority)
                          return a.priority > b.priority;
                      return a.id < b.id;
                  });

        for (const Listener &listener: listeners) {
            listener.callback(event);
            if (event.handled)
                break;
        }
    }

    KeyListenerGroup::KeyListenerGroup(KeyInput &input)
        : m_input(&input) {
    }

    KeyListenerGroup::~KeyListenerGroup() {
        Clear();
    }

    KeyListenerGroup::KeyListenerGroup(KeyListenerGroup &&other) noexcept
        : m_input(other.m_input),
          m_listenerIds(std::move(other.m_listenerIds)) {
        other.m_input = nullptr;
    }

    KeyListenerGroup &KeyListenerGroup::operator=(KeyListenerGroup &&other) noexcept {
        if (this == &other)
            return *this;

        Clear();
        m_input = other.m_input;
        m_listenerIds = std::move(other.m_listenerIds);
        other.m_input = nullptr;
        return *this;
    }

    KeyInput::ListenerId KeyListenerGroup::Add(KeyEventType type,
                                               KeyInput::Callback callback,
                                               int priority) {
        if (!m_input)
            throw std::runtime_error("KeyListenerGroup requires a KeyInput");

        KeyInput::ListenerId id = m_input->AddListener(type, std::move(callback), priority);
        m_listenerIds.push_back(id);
        return id;
    }

    void KeyListenerGroup::Clear() {
        if (m_input) {
            for (KeyInput::ListenerId id: m_listenerIds)
                m_input->RemoveListener(id);
        }
        m_listenerIds.clear();
    }

} // namespace vkRender
```

- [ ] **Step 3: `Engine`에 `CreateKeyInput()` 추가**

`src/vkRender/Engine.h`의 `#include "vkRender/MouseInput.h"` 아래에 추가:

```cpp
#include "vkRender/KeyInput.h"
```

같은 파일의 `std::unique_ptr<MouseInput> CreateMouseInput() const;` 바로 아래에 추가:

```cpp
        std::unique_ptr<KeyInput> CreateKeyInput() const;
```

`src/vkRender/Engine.cpp`에 다음 블록이 있다 (37-39번째 줄):

```cpp
    std::unique_ptr<MouseInput> Engine::CreateMouseInput() const {
        return std::make_unique<MouseInput>();
    }

} // namespace vkRender
```

`CreateMouseInput` 정의와 `} // namespace vkRender` 사이에 추가:

```cpp
    std::unique_ptr<KeyInput> Engine::CreateKeyInput() const {
        return std::make_unique<KeyInput>();
    }
```

- [ ] **Step 4: `vkRender.h` 마스터 헤더에 등록**

`src/vkRender/vkRender.h`의 `#include "vkRender/Image.h"` 다음 줄, `#include "vkRender/MouseInput.h"` 앞에 추가 (알파벳 순서 유지):

```cpp
#include "vkRender/KeyInput.h"
```

- [ ] **Step 5: 빌드 확인**

Run: `cmake --build build --target vkRender`
Expected: 에러 없이 `vkRender` 정적 라이브러리가 빌드된다 (KeyInput.cpp가 `file(GLOB_RECURSE ...)`에 의해 자동으로 소스 목록에 포함됨).

- [ ] **Step 6: 커밋**

```bash
git add src/vkRender/KeyInput.h src/vkRender/KeyInput.cpp src/vkRender/Engine.h src/vkRender/Engine.cpp src/vkRender/vkRender.h
git commit -m "Add vkRender::KeyInput keyboard event component"
```

---

### Task 2: `vkRender::Screenshot` 유틸리티 추가 (PNG 캡처)

**Files:**
- Create: `lib/stb/stb_image_write.h` (벤더링)
- Create: `src/vkRender/Screenshot.h`
- Create: `src/vkRender/Screenshot.cpp`
- Modify: `src/vkRender/vkRender.h`

**Interfaces:**
- Consumes: `vkCommon::VkContext{device, physDevice, graphicsQueue, graphicsFamily}` (필드는 `src/vkCommon/vkContext.h`에 이미 정의됨).
- Produces: `vkRender::Screenshot::CaptureToPNG(vkCommon::VkContext *context, VkImage image, VkFormat format, VkExtent2D extent, VkImageLayout currentLayout, const std::string &path)`, `vkRender::Screenshot::TimestampedPath(const std::string &directory = "screenshots") -> std::string`.

- [ ] **Step 1: `stb_image_write.h` 벤더링**

```bash
mkdir -p lib/stb
curl -fsSL -o lib/stb/stb_image_write.h https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
```

Expected: `lib/stb/stb_image_write.h` 파일이 생성되고, 첫 줄이 `/* stb_image_write - v1.`로 시작하는 라이선스 주석임을 확인한다.

Run: `head -5 lib/stb/stb_image_write.h`
Expected: stb 라이선스/버전 헤더 출력 (파일이 비어있거나 HTML 에러 페이지가 아님을 확인).

- [ ] **Step 2: `Screenshot.h` 작성**

`src/vkRender/Screenshot.h`:

```cpp
#pragma once

#include "vkCommon/vkContext.h"

#include <string>
#include <vulkan/vulkan.h>

namespace vkRender {

    class Screenshot {
    public:
        // Copies `image` (currently in `currentLayout`, e.g. VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
        // to an RGBA PNG file at `path`. Blocks until the GPU copy completes and restores
        // `currentLayout` on `image` afterwards. `image` must have been created with
        // VK_IMAGE_USAGE_TRANSFER_SRC_BIT. Supports VK_FORMAT_{B,R}8G8R8A8_{UNORM,SRGB} only.
        static void CaptureToPNG(vkCommon::VkContext *context,
                                 VkImage image,
                                 VkFormat format,
                                 VkExtent2D extent,
                                 VkImageLayout currentLayout,
                                 const std::string &path);

        // Builds "<directory>/screenshot_YYYYMMDD_HHMMSS.png", creating <directory> if needed.
        static std::string TimestampedPath(const std::string &directory = "screenshots");
    };

} // namespace vkRender
```

- [ ] **Step 3: `Screenshot.cpp` 작성**

`src/vkRender/Screenshot.cpp`:

```cpp
#include "vkRender/Screenshot.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <chrono>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vkRender {
    namespace {

        bool RequiresRedBlueSwap(VkFormat format) {
            return format == VK_FORMAT_B8G8R8A8_UNORM ||
                   format == VK_FORMAT_B8G8R8A8_SRGB;
        }

        bool IsSupportedFormat(VkFormat format) {
            return format == VK_FORMAT_B8G8R8A8_UNORM ||
                   format == VK_FORMAT_B8G8R8A8_SRGB ||
                   format == VK_FORMAT_R8G8B8A8_UNORM ||
                   format == VK_FORMAT_R8G8B8A8_SRGB;
        }

        uint32_t FindMemoryType(VkPhysicalDevice physicalDevice,
                                uint32_t typeFilter,
                                VkMemoryPropertyFlags properties) {
            VkPhysicalDeviceMemoryProperties memProperties;
            vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
            for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
                if ((typeFilter & (1u << i)) &&
                    (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
                    return i;
            }
            throw std::runtime_error("Screenshot: no suitable memory type for staging buffer");
        }

        void TransitionImageLayout(VkCommandBuffer cmd,
                                   VkImage image,
                                   VkImageLayout oldLayout,
                                   VkImageLayout newLayout) {
            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = oldLayout;
            barrier.newLayout = newLayout;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image;
            barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

            vkCmdPipelineBarrier(cmd,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                0, 0, nullptr, 0, nullptr, 1, &barrier);
        }

    } // namespace

    void Screenshot::CaptureToPNG(vkCommon::VkContext *context,
                                  VkImage image,
                                  VkFormat format,
                                  VkExtent2D extent,
                                  VkImageLayout currentLayout,
                                  const std::string &path) {
        if (!context)
            throw std::runtime_error("Screenshot::CaptureToPNG requires a valid VkContext");
        if (!IsSupportedFormat(format))
            throw std::runtime_error("Screenshot::CaptureToPNG received an unsupported swapchain format");

        vkQueueWaitIdle(context->graphicsQueue);

        const VkDeviceSize bufferSize =
                static_cast<VkDeviceSize>(extent.width) * extent.height * 4u;

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = bufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        if (vkCreateBuffer(context->device, &bufferInfo, nullptr, &stagingBuffer) != VK_SUCCESS)
            throw std::runtime_error("Screenshot: failed to create staging buffer");

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(context->device, stagingBuffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = FindMemoryType(
                context->physDevice, memRequirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
        if (vkAllocateMemory(context->device, &allocInfo, nullptr, &stagingMemory) != VK_SUCCESS) {
            vkDestroyBuffer(context->device, stagingBuffer, nullptr);
            throw std::runtime_error("Screenshot: failed to allocate staging memory");
        }
        vkBindBufferMemory(context->device, stagingBuffer, stagingMemory, 0);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolInfo.queueFamilyIndex = context->graphicsFamily;

        VkCommandPool commandPool = VK_NULL_HANDLE;
        if (vkCreateCommandPool(context->device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            vkFreeMemory(context->device, stagingMemory, nullptr);
            vkDestroyBuffer(context->device, stagingBuffer, nullptr);
            throw std::runtime_error("Screenshot: failed to create command pool");
        }

        VkCommandBufferAllocateInfo cmdAllocInfo{};
        cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAllocInfo.commandPool = commandPool;
        cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAllocInfo.commandBufferCount = 1;

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(context->device, &cmdAllocInfo, &cmd);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        TransitionImageLayout(cmd, image, currentLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               stagingBuffer, 1, &region);

        TransitionImageLayout(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, currentLayout);

        vkEndCommandBuffer(cmd);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        vkQueueSubmit(context->graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(context->graphicsQueue);

        std::vector<uint8_t> pixels(static_cast<size_t>(bufferSize));
        void *mapped = nullptr;
        vkMapMemory(context->device, stagingMemory, 0, bufferSize, 0, &mapped);
        std::memcpy(pixels.data(), mapped, static_cast<size_t>(bufferSize));
        vkUnmapMemory(context->device, stagingMemory);

        if (RequiresRedBlueSwap(format)) {
            for (size_t i = 0; i + 2 < pixels.size(); i += 4)
                std::swap(pixels[i], pixels[i + 2]);
        }

        vkFreeCommandBuffers(context->device, commandPool, 1, &cmd);
        vkDestroyCommandPool(context->device, commandPool, nullptr);
        vkFreeMemory(context->device, stagingMemory, nullptr);
        vkDestroyBuffer(context->device, stagingBuffer, nullptr);

        const int stride = static_cast<int>(extent.width) * 4;
        if (!stbi_write_png(path.c_str(),
                            static_cast<int>(extent.width),
                            static_cast<int>(extent.height),
                            4, pixels.data(), stride))
            throw std::runtime_error("Screenshot: failed to write PNG file: " + path);
    }

    std::string Screenshot::TimestampedPath(const std::string &directory) {
        std::filesystem::create_directories(directory);

        const auto now = std::chrono::system_clock::now();
        const std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
        std::tm localTime{};
#if defined(_WIN32)
        localtime_s(&localTime, &nowTime);
#else
        localtime_r(&nowTime, &localTime);
#endif

        std::ostringstream oss;
        oss << "screenshot_"
            << std::put_time(&localTime, "%Y%m%d_%H%M%S")
            << ".png";

        std::filesystem::path result = std::filesystem::path(directory) / oss.str();
        return result.string();
    }

} // namespace vkRender
```

- [ ] **Step 4: `vkRender.h` 마스터 헤더에 등록**

`src/vkRender/vkRender.h`의 `#include "vkRender/Scene.h"` 다음 줄, `#include "vkRender/SwapChain.h"` 앞에 추가 (알파벳 순서 유지):

```cpp
#include "vkRender/Screenshot.h"
```

- [ ] **Step 5: 빌드 확인**

Run: `cmake --build build --target vkRender`
Expected: 에러 없이 빌드 성공. (`stb_image_write.h`가 헤더 전용이라 별도 링크 설정 불필요, `lib/`이 이미 include 경로에 있음.)

- [ ] **Step 6: 커밋**

```bash
git add lib/stb/stb_image_write.h src/vkRender/Screenshot.h src/vkRender/Screenshot.cpp src/vkRender/vkRender.h
git commit -m "Add vkRender::Screenshot PNG capture utility"
```

---

### Task 3: `cube_render.cpp`에 Enter 키 캡처 연결

**Files:**
- Modify: `example/cube_render.cpp`

**Interfaces:**
- Consumes: `vkRender::Engine::CreateKeyInput()`, `vkRender::KeyInput::AddListener/OnKey`, `vkRender::KeyEventType::Press`, `vkRender::Screenshot::CaptureToPNG`, `vkRender::Screenshot::TimestampedPath`, `vkRender::Renderer::CurrentImageIndex()` (기존, `Renderer.h:35`), `vkRender::SwapChain::Image()/Format()/Extent()` (기존).
- Produces: 없음 (최종 통합 태스크).

- [ ] **Step 1: 스왑체인에 `TRANSFER_SRC_BIT` usage 추가**

`example/cube_render.cpp`에서 (현재 267번째 줄 부근) 다음 블록:

```cpp
        vkRender::SwapChainDescriptor swapChainDescriptor{};
        swapChainDescriptor.width = static_cast<uint32_t>(framebufferWidth);
        swapChainDescriptor.height = static_cast<uint32_t>(framebufferHeight);
        auto swapChain = engine->CreateSwapChain(swapChainDescriptor);
```

다음과 같이 수정:

```cpp
        vkRender::SwapChainDescriptor swapChainDescriptor{};
        swapChainDescriptor.width = static_cast<uint32_t>(framebufferWidth);
        swapChainDescriptor.height = static_cast<uint32_t>(framebufferHeight);
        swapChainDescriptor.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        auto swapChain = engine->CreateSwapChain(swapChainDescriptor);
```

- [ ] **Step 2: `KeyInput` 생성 및 GLFW 콜백 연결**

같은 파일에서 (현재) 다음 블록:

```cpp
        auto renderer = engine->CreateRenderer();
        auto scene = engine->CreateScene();
        auto camera = engine->CreateCamera();
        auto view = engine->CreateView();
        auto graph = engine->CreateRenderGraph();
```

다음과 같이 수정 (`keyInput` 생성 추가):

```cpp
        auto renderer = engine->CreateRenderer();
        auto scene = engine->CreateScene();
        auto camera = engine->CreateCamera();
        auto view = engine->CreateView();
        auto graph = engine->CreateRenderGraph();
        auto keyInput = engine->CreateKeyInput();
```

파일 익명 네임스페이스(`namespace { ... }`) 안, `CubePass` 클래스 정의 아래에 GLFW 키 콜백에서 쓸 헬퍼 함수를 추가:

```cpp
    vkRender::KeyInput *WindowKeyInput(GLFWwindow *window) {
        return static_cast<vkRender::KeyInput *>(glfwGetWindowUserPointer(window));
    }
```

`main()`에서 `keyInput` 생성 직후, `glfwSetWindowUserPointer`와 `glfwSetKeyCallback`을 등록:

```cpp
        glfwSetWindowUserPointer(window, keyInput.get());
        glfwSetKeyCallback(window, [](GLFWwindow *callbackWindow, int key, int, int action, int mods) {
            vkRender::KeyInput *keys = WindowKeyInput(callbackWindow);
            if (!keys)
                return;

            vkRender::KeyEventType type;
            if (action == GLFW_PRESS)
                type = vkRender::KeyEventType::Press;
            else if (action == GLFW_RELEASE)
                type = vkRender::KeyEventType::Release;
            else
                type = vkRender::KeyEventType::Repeat;

            keys->OnKey(key, type, static_cast<uint32_t>(mods), glfwGetTime());
        });
```

- [ ] **Step 3: Enter 키 리스너 및 캡처 플래그**

`glfwSetKeyCallback` 등록 다음 줄에 캡처 요청 플래그와 리스너를 추가:

```cpp
        bool captureRequested = false;
        keyInput->AddListener(vkRender::KeyEventType::Press,
                              [&captureRequested](vkRender::KeyEvent &event) {
                                  if (event.keyCode == GLFW_KEY_ENTER)
                                      captureRequested = true;
                              });
```

- [ ] **Step 4: 메인 루프에서 캡처 수행**

현재 메인 루프의 다음 블록:

```cpp
            renderer->Render(*view);
            renderer->EndFrame();

            if (renderer->NeedsSwapChainRecreate()) {
```

다음과 같이 수정 (`EndFrame()` 직후 캡처 삽입):

```cpp
            renderer->Render(*view);
            renderer->EndFrame();

            if (captureRequested) {
                captureRequested = false;
                vkRender::Screenshot::CaptureToPNG(
                        &context,
                        swapChain->Image(renderer->CurrentImageIndex()),
                        swapChain->Format(),
                        swapChain->Extent(),
                        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                        vkRender::Screenshot::TimestampedPath());
                std::cout << "cube_render: screenshot saved\n";
            }

            if (renderer->NeedsSwapChainRecreate()) {
```

- [ ] **Step 5: 빌드**

Run: `cmake --build build --target vkRender cube_render`
Expected: 에러 없이 빌드 성공.

- [ ] **Step 6: 수동 실행 검증**

Run: `./build/example/cube_render` (실제 바이너리 경로는 빌드 출력에서 확인)

수행할 확인:
1. 창이 뜨고 회전하는 큐브가 정상적으로 렌더링되는지 확인.
2. Enter 키를 누른 뒤 터미널에 `cube_render: screenshot saved`가 출력되는지 확인.
3. 실행 파일 기준 상대 경로 `screenshots/` 폴더에 `screenshot_YYYYMMDD_HHMMSS.png` 파일이 생성됐는지 확인 (`ls screenshots/`).
4. 생성된 PNG를 이미지 뷰어로 열어 화면에 보이던 큐브 색상과 일치하는지(RGB/BGR 스왑 정상 여부) 육안 확인.
5. 창 크기를 리사이즈한 직후 Enter를 눌러도 크래시 없이 새 해상도로 캡처되는지 확인.
6. Enter를 연속으로 여러 번 눌러 크래시나 검증 에러 없이 여러 PNG 파일이 쌓이는지 확인.

Expected: 위 6가지 모두 정상 동작, 크래시/예외 없음.

- [ ] **Step 7: 커밋**

```bash
git add example/cube_render.cpp
git commit -m "Wire Enter-key screenshot capture into cube_render example"
```

---

## Self-Review Notes

- **스펙 커버리지:** KeyInput 컴포넌트(스펙 §1) → Task 1. Screenshot 유틸리티(스펙 §2, stb 벤더링 포함) → Task 2. cube_render.cpp 연결(스펙 §3) → Task 3. 스펙의 "테스트/검증 방법" 6개 항목 중 1~4, 6번은 Task 3 Step 6에 반영. 5번(리사이즈 직후 캡처)도 Task 3 Step 6-5에 반영.
- **타입 일관성:** `KeyEventType`, `KeyEvent`, `KeyInput::Callback`, `Screenshot::CaptureToPNG`/`TimestampedPath` 시그니처가 Task 1~3에서 동일하게 유지됨을 확인함.
- **플레이스홀더 스캔:** 모든 스텝에 실제 코드/명령어가 포함되어 있고 "TODO"/"나중에" 등의 표현 없음을 확인함.
