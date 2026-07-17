# realtime_shadow Vulkan Structure Pipeline

대상 코드:

- [`example/realtime_shadow.cpp`](../example/realtime_shadow.cpp)
- [`example/realtime_shadow_depth.vert`](../example/realtime_shadow_depth.vert)
- [`example/realtime_shadow_scene.vert`](../example/realtime_shadow_scene.vert)
- [`example/realtime_shadow_scene.frag`](../example/realtime_shadow_scene.frag)

이 문서는 `realtime_shadow.cpp`를 Vulkan 구조체 기준으로 정리한다. 핵심은 두 개의 graphics pipeline이다.

- `m_shadowPipeline`: 조명 시점 depth-only shadow map 생성
- `m_scenePipeline`: 카메라 시점 color + depth 렌더링, shadow map sampling

## 1. Vulkan 구조체 전체 관계

```mermaid
flowchart TD
    App["RealtimeShadowExample"] --> Geo["CreateGeometry()"]
    App --> Desc["CreateDescriptorResources()"]
    App --> Images["CreateShadowResources() / EnsureSceneDepthResources()"]
    App --> Pipes["EnsurePipelines(colorFormat)"]

    Geo --> VB["VkBufferCreateInfo\nvertex buffer"]
    Geo --> IB["VkBufferCreateInfo\nindex buffer"]
    Geo --> UP["VkCommandPoolCreateInfo\nuploadPool"]

    Images --> ShadowImage["VkImageCreateInfo\nD32 shadow image"]
    ShadowImage --> ShadowView["VkImageViewCreateInfo\nshadow depth view"]
    ShadowView --> ShadowSampler["VkSamplerCreateInfo\nshadow sampler"]
    Images --> SceneDepth["VkImageCreateInfo\nD32 scene depth image"]
    SceneDepth --> SceneDepthView["VkImageViewCreateInfo\nscene depth view"]

    Desc --> DSLB["VkDescriptorSetLayoutBinding\nbinding 0: combined image sampler"]
    DSLB --> DSL["VkDescriptorSetLayoutCreateInfo"]
    DSL --> Pool["VkDescriptorPoolCreateInfo"]
    Pool --> SetAlloc["VkDescriptorSetAllocateInfo"]
    SetAlloc --> Write["VkWriteDescriptorSet\nshadow sampler + view"]

    Pipes --> Layout["VkPipelineLayoutCreateInfo"]
    DSL --> Layout
    Push["VkPushConstantRange\nmodel/viewProj/lightViewProj/lightDir"] --> Layout
    Layout --> ShadowPipe["VkGraphicsPipelineCreateInfo\nshadow pipeline"]
    Layout --> ScenePipe["VkGraphicsPipelineCreateInfo\nscene pipeline"]
```

## 2. Swapchain 구조

`main()`은 `SwapChainDescriptor`에 color attachment와 transfer usage를 갖는 swapchain을 만든다.

```cpp
vkRender::SwapChainDescriptor descriptor{};
descriptor.width = framebufferWidth;
descriptor.height = framebufferHeight;
descriptor.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
auto swapChain = engine->CreateSwapChain(descriptor);
```

`vkRender::SwapChain` 내부에서는 이 값이 `VkSwapchainCreateInfoKHR::imageUsage`로 들어간다.

```mermaid
flowchart LR
    SCD["SwapChainDescriptor"] --> SCI["VkSwapchainCreateInfoKHR"]
    SCD --> W["width / height"]
    SCD --> F["preferredFormat: B8G8R8A8_UNORM"]
    SCD --> U["imageUsage"]
    U --> U1["VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT"]
    U --> U2["VK_IMAGE_USAGE_TRANSFER_DST_BIT"]
    U --> U3["VK_IMAGE_USAGE_TRANSFER_SRC_BIT"]
    SCI --> Images["VkImage[] swapchain images"]
    Images --> Views["VkImageView[] swapchain image views"]
```

`TRANSFER_SRC`는 Enter 캡처에서 swapchain color image를 readback buffer로 복사하기 위해 필요하다.

## 3. 공통 Pipeline Layout

shadow pipeline과 scene pipeline은 같은 `VkPipelineLayout`을 공유한다.

```mermaid
flowchart TD
    PC["VkPushConstantRange"] --> PL["VkPipelineLayoutCreateInfo"]
    DSL["VkDescriptorSetLayout"] --> PL
    PL --> VKPL["vkCreatePipelineLayout()"]
    VKPL --> Layout["m_pipelineLayout"]

    PC --> PC1["stageFlags = VERTEX | FRAGMENT"]
    PC --> PC2["offset = 0"]
    PC --> PC3["size = sizeof(PushConstants)"]

    DSL --> D0["set 0, binding 0"]
    D0 --> D1["VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER"]
    D0 --> D2["stageFlags = FRAGMENT"]
```

`PushConstants` 구조는 draw마다 갱신된다.

```cpp
struct PushConstants {
    Mat4 model;
    Mat4 viewProj;
    Mat4 lightViewProj;
    float lightDir[4];
};
```

shadow pass에서는 `model`, `lightViewProj`, `lightDir`가 중요하고, scene pass에서는 `model`, `viewProj`, `lightViewProj`, `lightDir` 전체를 사용한다.

## 4. 공통 VkGraphicsPipelineCreateInfo 베이스

`MakeBasePipelineInfo()`는 shadow/scene pipeline이 공유하는 고정 기능 상태를 만든다.

```mermaid
flowchart TD
    Base["MakeBasePipelineInfo()"] --> GPCI["VkGraphicsPipelineCreateInfo"]

    Base --> VI["VkPipelineVertexInputStateCreateInfo"]
    VI --> Bind["VkVertexInputBindingDescription\nbinding=0, stride=sizeof(Vertex)"]
    VI --> A0["location 0: position, R32G32B32_SFLOAT"]
    VI --> A1["location 1: normal, R32G32B32_SFLOAT"]
    VI --> A2["location 2: color, R32G32B32_SFLOAT"]

    Base --> IA["VkPipelineInputAssemblyStateCreateInfo\ntopology=TRIANGLE_LIST"]
    Base --> VP["VkPipelineViewportStateCreateInfo\nviewportCount=1, scissorCount=1"]
    Base --> MS["VkPipelineMultisampleStateCreateInfo\nsamples=1"]
    Base --> DS["VkPipelineDynamicStateCreateInfo\nVIEWPORT, SCISSOR"]

    VI --> GPCI
    IA --> GPCI
    VP --> GPCI
    MS --> GPCI
    DS --> GPCI
```

공통 vertex 구조는 다음과 같다.

```cpp
struct Vertex {
    float position[3]; // location 0
    float normal[3];   // location 1
    float color[3];    // location 2
};
```

## 5. Shadow Pipeline 구조

`CreateShadowPipeline()`은 depth-only pipeline이다. fragment shader가 없고, color attachment도 없다.

```mermaid
flowchart TD
    S["CreateShadowPipeline()"] --> SM["CreateShaderModule(realtime_shadow_depth.vert.spv)"]
    SM --> Stage["VkPipelineShaderStageCreateInfo\nstage=VERTEX"]
    S --> Base["MakeBasePipelineInfo()"]
    S --> Rast["VkPipelineRasterizationStateCreateInfo"]
    S --> Depth["VkPipelineDepthStencilStateCreateInfo"]
    S --> Blend["VkPipelineColorBlendStateCreateInfo\nempty color blend"]
    S --> Rendering["VkPipelineRenderingCreateInfo\ndepthAttachmentFormat=D32_SFLOAT"]

    Rast --> R0["polygonMode=FILL"]
    Rast --> R1["cullMode=NONE"]
    Rast --> R2["frontFace=COUNTER_CLOCKWISE"]
    Rast --> R3["depthBiasEnable=TRUE"]
    Rast --> R4["constantFactor=1.2"]
    Rast --> R5["slopeFactor=1.8"]

    Depth --> D0["depthTestEnable=TRUE"]
    Depth --> D1["depthWriteEnable=TRUE"]
    Depth --> D2["depthCompareOp=LESS"]

    Stage --> GPCI["VkGraphicsPipelineCreateInfo"]
    Base --> GPCI
    Rast --> GPCI
    Depth --> GPCI
    Blend --> GPCI
    Rendering --> GPCI
    Layout["m_pipelineLayout"] --> GPCI
    GPCI --> Create["vkCreateGraphicsPipelines()"]
    Create --> Pipe["m_shadowPipeline"]
```

shadow pipeline의 `VkGraphicsPipelineCreateInfo` 핵심 값:

| 필드 | 값 |
| --- | --- |
| `stageCount` | `1` |
| `pStages` | depth vertex shader |
| `pNext` | `VkPipelineRenderingCreateInfo` |
| `depthAttachmentFormat` | `VK_FORMAT_D32_SFLOAT` |
| `pRasterizationState` | depth bias enabled |
| `pDepthStencilState` | depth test/write enabled |
| `layout` | `m_pipelineLayout` |

이 pipeline은 `RecordShadowPass()`에서 shadow map depth image에만 기록한다.

## 6. Scene Pipeline 구조

`CreateScenePipeline()`은 swapchain color attachment와 scene depth attachment를 사용하는 일반 렌더링 pipeline이다.

```mermaid
flowchart TD
    S["CreateScenePipeline(colorFormat)"] --> VSM["CreateShaderModule(scene.vert.spv)"]
    S --> FSM["CreateShaderModule(scene.frag.spv)"]
    VSM --> VS["VkPipelineShaderStageCreateInfo\nstage=VERTEX"]
    FSM --> FS["VkPipelineShaderStageCreateInfo\nstage=FRAGMENT"]
    S --> Base["MakeBasePipelineInfo()"]
    S --> Rast["VkPipelineRasterizationStateCreateInfo"]
    S --> Depth["VkPipelineDepthStencilStateCreateInfo"]
    S --> BlendAttach["VkPipelineColorBlendAttachmentState\nRGBA write mask"]
    BlendAttach --> Blend["VkPipelineColorBlendStateCreateInfo\nattachmentCount=1"]
    S --> Rendering["VkPipelineRenderingCreateInfo"]

    Rendering --> C0["colorAttachmentCount=1"]
    Rendering --> C1["pColorAttachmentFormats=&colorFormat"]
    Rendering --> C2["depthAttachmentFormat=D32_SFLOAT"]

    Depth --> D0["depthTestEnable=TRUE"]
    Depth --> D1["depthWriteEnable=TRUE"]
    Depth --> D2["depthCompareOp=LESS"]

    VS --> GPCI["VkGraphicsPipelineCreateInfo"]
    FS --> GPCI
    Base --> GPCI
    Rast --> GPCI
    Depth --> GPCI
    Blend --> GPCI
    Rendering --> GPCI
    Layout["m_pipelineLayout"] --> GPCI
    GPCI --> Create["vkCreateGraphicsPipelines()"]
    Create --> Pipe["m_scenePipeline"]
```

scene pipeline의 `VkGraphicsPipelineCreateInfo` 핵심 값:

| 필드 | 값 |
| --- | --- |
| `stageCount` | `2` |
| `pStages` | vertex shader + fragment shader |
| `pNext` | `VkPipelineRenderingCreateInfo` |
| `colorAttachmentCount` | `1` |
| `pColorAttachmentFormats` | current swapchain format |
| `depthAttachmentFormat` | `VK_FORMAT_D32_SFLOAT` |
| `pColorBlendState` | RGBA write enabled |
| `layout` | `m_pipelineLayout` |

fragment shader는 descriptor set binding 0의 shadow map을 샘플링한다.

## 7. Descriptor 구조

shadow map은 scene fragment shader에서 `sampler2D shadowMap`으로 읽힌다.

```mermaid
flowchart TD
    ShadowImage["m_shadowImage\nD32_SFLOAT"] --> ShadowView["m_shadowView"]
    ShadowView --> ImageInfo["VkDescriptorImageInfo"]
    ShadowSampler["m_shadowSampler"] --> ImageInfo
    ImageInfo --> Write["VkWriteDescriptorSet"]

    Binding["VkDescriptorSetLayoutBinding"] --> LayoutInfo["VkDescriptorSetLayoutCreateInfo"]
    LayoutInfo --> DSL["m_descriptorSetLayout"]
    DSL --> PoolInfo["VkDescriptorPoolCreateInfo"]
    PoolInfo --> Pool["m_descriptorPool"]
    Pool --> Alloc["VkDescriptorSetAllocateInfo"]
    Alloc --> Set["m_descriptorSet"]
    Write --> Set
    Set --> Frag["realtime_shadow_scene.frag\nlayout(set=0,binding=0) sampler2D shadowMap"]
```

구조체 값:

| 구조체 | 주요 값 |
| --- | --- |
| `VkDescriptorSetLayoutBinding` | `binding=0`, `descriptorType=COMBINED_IMAGE_SAMPLER`, `stageFlags=FRAGMENT` |
| `VkDescriptorImageInfo` | `sampler=m_shadowSampler`, `imageView=m_shadowView`, `imageLayout=SHADER_READ_ONLY_OPTIMAL` |
| `VkWriteDescriptorSet` | descriptor set binding 0에 shadow map 연결 |

## 8. Dynamic Rendering Attachment 구조

이 예제는 render pass object를 만들지 않고 dynamic rendering을 사용한다. 따라서 pipeline 생성 시 `VkPipelineRenderingCreateInfo`로 attachment format을 알려주고, command buffer 기록 시 `VkRenderingInfo`와 `VkRenderingAttachmentInfo`를 사용한다.

```mermaid
flowchart TD
    subgraph ShadowDynamicRendering["Shadow Pass Dynamic Rendering"]
        SA["VkRenderingAttachmentInfo\nimageView=m_shadowView\nlayout=DEPTH_ATTACHMENT\nloadOp=CLEAR\nstoreOp=STORE"] --> SI["VkRenderingInfo\npDepthAttachment=&SA\nrenderArea=2048x2048"]
        SI --> SB["vkCmdBeginRendering()"]
        SB --> SD["vkCmdDrawIndexed()"]
        SD --> SE["vkCmdEndRendering()"]
    end

    subgraph SceneDynamicRendering["Scene Pass Dynamic Rendering"]
        CA["VkRenderingAttachmentInfo\nimageView=swapChain.ImageView\nlayout=COLOR_ATTACHMENT\nloadOp=CLEAR\nstoreOp=STORE"] --> RI["VkRenderingInfo"]
        DA["VkRenderingAttachmentInfo\nimageView=m_sceneDepthView\nlayout=DEPTH_ATTACHMENT\nloadOp=CLEAR\nstoreOp=DONT_CARE"] --> RI
        RI --> RB["vkCmdBeginRendering()"]
        RB --> RD["vkCmdDrawIndexed()"]
        RD --> RE["vkCmdEndRendering()"]
    end
```

## 9. Frame 실행 시 Pipeline 바인딩 구조

pipeline 생성 구조와 실제 command buffer 실행 구조는 다음처럼 연결된다.

```mermaid
sequenceDiagram
    participant Cmd as VkCommandBuffer
    participant Shadow as m_shadowPipeline
    participant Scene as m_scenePipeline
    participant Desc as m_descriptorSet
    participant Swap as Swapchain Image
    participant Cap as Capture Buffer

    Cmd->>Cmd: vkCmdPipelineBarrier(shadow image -> DEPTH_ATTACHMENT)
    Cmd->>Cmd: vkCmdBeginRendering(shadow depth)
    Cmd->>Shadow: vkCmdBindPipeline()
    Cmd->>Cmd: vkCmdPushConstants(lightViewProj)
    Cmd->>Cmd: vkCmdDrawIndexed(plane)
    Cmd->>Cmd: vkCmdPushConstants(lightViewProj)
    Cmd->>Cmd: vkCmdDrawIndexed(cube)
    Cmd->>Cmd: vkCmdEndRendering()
    Cmd->>Cmd: vkCmdPipelineBarrier(shadow image -> SHADER_READ_ONLY)

    Cmd->>Swap: vkCmdPipelineBarrier(swap image -> COLOR_ATTACHMENT)
    Cmd->>Cmd: vkCmdBeginRendering(scene color + depth)
    Cmd->>Scene: vkCmdBindPipeline()
    Cmd->>Desc: vkCmdBindDescriptorSets(shadow sampler)
    Cmd->>Cmd: vkCmdPushConstants(viewProj + lightViewProj)
    Cmd->>Cmd: vkCmdDrawIndexed(plane)
    Cmd->>Cmd: vkCmdPushConstants(viewProj + lightViewProj)
    Cmd->>Cmd: vkCmdDrawIndexed(cube)
    Cmd->>Cmd: vkCmdEndRendering()

    alt normal present
        Cmd->>Swap: vkCmdPipelineBarrier(COLOR_ATTACHMENT -> PRESENT_SRC)
    else Enter capture
        Cmd->>Swap: vkCmdPipelineBarrier(COLOR_ATTACHMENT -> TRANSFER_SRC)
        Cmd->>Cap: vkCmdCopyImageToBuffer()
        Cmd->>Cap: vkCmdPipelineBarrier(TRANSFER_WRITE -> HOST_READ)
        Cmd->>Swap: vkCmdPipelineBarrier(TRANSFER_SRC -> PRESENT_SRC)
    end
```

## 10. Pipeline별 구조체 비교

| 항목 | Shadow Pipeline | Scene Pipeline |
| --- | --- | --- |
| 생성 함수 | `CreateShadowPipeline()` | `CreateScenePipeline()` |
| shader stages | vertex 1개 | vertex + fragment |
| color attachment | 없음 | swapchain format 1개 |
| depth attachment | `VK_FORMAT_D32_SFLOAT` | `VK_FORMAT_D32_SFLOAT` |
| descriptor 사용 | layout에는 포함, pass에서는 bind하지 않음 | shadow sampler descriptor bind |
| depth bias | 켬 | 끔 |
| 출력 | shadow depth image | swapchain color image + scene depth image |
| 주 목적 | light-space closest depth 저장 | camera-space shading + shadow comparison |

## 11. 구조체 생성 순서 요약

```mermaid
flowchart TD
    A["VkBufferCreateInfo\nvertex/index"] --> B["VkImageCreateInfo\nshadow depth"]
    B --> C["VkImageViewCreateInfo\nshadow view"]
    C --> D["VkSamplerCreateInfo\nshadow sampler"]
    D --> E["VkDescriptorSetLayoutCreateInfo"]
    E --> F["VkDescriptorPoolCreateInfo"]
    F --> G["VkDescriptorSetAllocateInfo"]
    G --> H["VkWriteDescriptorSet"]
    H --> I["VkPipelineLayoutCreateInfo"]
    I --> J["VkGraphicsPipelineCreateInfo\nshadow"]
    I --> K["VkGraphicsPipelineCreateInfo\nscene"]
    J --> L["VkRenderingInfo\nshadow pass runtime"]
    K --> M["VkRenderingInfo\nscene pass runtime"]
```

이 구조에서 `VkGraphicsPipelineCreateInfo`는 정적 pipeline 상태를 정의하고, `VkRenderingInfo`는 매 프레임 실제 attachment image view를 command buffer에 연결한다.

