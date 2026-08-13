# Vulkan Pipeline Stage Flags

`VkPipelineStageFlagBits`는 GPU 명령이 실행되는 큰 단계를 가리키는 synchronization scope다.
즉, "이 리소스 접근이 파이프라인의 어느 시점에서 일어나는가"를 Vulkan에게 알려서
barrier, semaphore wait, event, dependency가 정확한 단계 사이에서 동작하게 한다.

`VkPipelineStageFlagBits`는 개별 bit 값이고, `VkPipelineStageFlags`는 여러 bit를 OR로 묶는 mask 타입이다.

```cpp
VkPipelineStageFlags stages =
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
```

```cpp
vkCmdPipelineBarrier(
        cmd,
        srcStageMask,  // 이전 작업 중 어디까지 기다릴지
        dstStageMask,  // 이후 작업 중 어디부터 막아둘지
        ...);
```

pipeline stage는 메모리 접근 종류가 아니다. 실제 읽기/쓰기는 `VkAccessFlags`가 설명한다.
보통 stage와 access mask는 쌍으로 생각해야 한다.

예:

```cpp
// depth attachment에 쓰기를 끝낸 뒤 fragment shader에서 shadow map으로 읽는다.
srcStage = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
srcAccess = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
dstAccess = VK_ACCESS_SHADER_READ_BIT;
```

## 전체 흐름

대략적인 graphics pipeline 흐름은 아래와 같다.

```text
DRAW_INDIRECT
    |
VERTEX_INPUT
    |
VERTEX_SHADER
    |
TESSELLATION_CONTROL_SHADER
    |
TESSELLATION_EVALUATION_SHADER
    |
GEOMETRY_SHADER
    |
RASTERIZATION
    |
EARLY_FRAGMENT_TESTS
    |
FRAGMENT_SHADER
    |
LATE_FRAGMENT_TESTS
    |
COLOR_ATTACHMENT_OUTPUT
```

Compute, transfer, ray tracing, acceleration structure build 등은 graphics pipeline과 별도의 작업 도메인이다.

```text
COMPUTE_SHADER
TRANSFER
ACCELERATION_STRUCTURE_BUILD
RAY_TRACING_SHADER
HOST
```

`TOP_OF_PIPE`와 `BOTTOM_OF_PIPE`는 실제 작업 단계라기보다 dependency를 표현하기 위한 경계에 가깝다.

## 자주 쓰는 Stage

| Stage                     | 의미                                            | 흔한 access mask                              |
| ------------------------- | ----------------------------------------------- | --------------------------------------------- |
| `TOP_OF_PIPE`             | 명령이 아직 어떤 실질 작업도 시작하기 전        | 보통 `0`                                      |
| `VERTEX_INPUT`            | vertex/index buffer를 읽고 vertex fetch를 수행  | `VERTEX_ATTRIBUTE_READ`, `INDEX_READ`         |
| `VERTEX_SHADER`           | vertex shader 실행                              | `SHADER_READ`, `SHADER_WRITE`, `UNIFORM_READ` |
| `FRAGMENT_SHADER`         | fragment shader 실행, texture/sampler 읽기 포함 | `SHADER_READ`, `SHADER_WRITE`, `UNIFORM_READ` |
| `EARLY_FRAGMENT_TESTS`    | fragment shader 전 depth/stencil test           | `DEPTH_STENCIL_ATTACHMENT_READ/WRITE`         |
| `LATE_FRAGMENT_TESTS`     | fragment shader 후 depth/stencil test/write     | `DEPTH_STENCIL_ATTACHMENT_READ/WRITE`         |
| `COLOR_ATTACHMENT_OUTPUT` | color attachment blend/write, resolve           | `COLOR_ATTACHMENT_READ/WRITE`                 |
| `COMPUTE_SHADER`          | compute shader dispatch                         | `SHADER_READ/WRITE`, `UNIFORM_READ`           |
| `TRANSFER`                | copy, blit, clear, resolve 같은 transfer 명령   | `TRANSFER_READ/WRITE`                         |
| `HOST`                    | CPU 쪽 host read/write                          | `HOST_READ/WRITE`                             |
| `BOTTOM_OF_PIPE`          | 명령이 모든 실질 pipeline 작업을 끝낸 뒤        | 보통 `0`                                      |
| `ALL_GRAPHICS`            | 모든 graphics pipeline stage                    | 상황별                                        |
| `ALL_COMMANDS`            | 모든 queue command stage                        | 상황별                                        |

## 각 Flag 설명

### `VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT`

명령이 pipeline에 들어가기 전의 경계다. 실제 메모리 접근을 수행하지 않는다.

주로 "이전 작업이 없다"거나 "초기 레이아웃에서 처음 사용하는 이미지" 같은 전환에서 src stage로 쓴다.

```cpp
srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
srcAccess = 0;
```

주의: 이미 어떤 작업이 리소스를 쓰고 있었다면 `TOP_OF_PIPE`을 src로 쓰면 안 된다.
기다려야 할 실제 producer stage를 지정해야 한다.

### `VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT`

`vkCmdDrawIndirect`, `vkCmdDrawIndexedIndirect`, `vkCmdDispatchIndirect` 등이 indirect argument buffer를 읽는 단계다.

예:

```cpp
srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
srcAccess = VK_ACCESS_SHADER_WRITE_BIT;

dstStage = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
dstAccess = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
```

compute shader가 indirect draw command buffer를 만든 뒤 draw indirect가 읽을 때 사용한다.

### `VK_PIPELINE_STAGE_VERTEX_INPUT_BIT`

vertex buffer와 index buffer를 읽는 단계다. shader 실행 전 fixed-function vertex fetch에 해당한다.

흔한 access mask:

- `VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT`
- `VK_ACCESS_INDEX_READ_BIT`

예: transfer로 vertex buffer를 업로드한 뒤 draw에서 읽을 때.

```cpp
srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
srcAccess = VK_ACCESS_TRANSFER_WRITE_BIT;

dstStage = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
dstAccess = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
```

### `VK_PIPELINE_STAGE_VERTEX_SHADER_BIT`

vertex shader 실행 단계다. uniform buffer, storage buffer, sampled image 등을 vertex shader에서 읽거나 쓸 때 여기에 해당한다.

흔한 access mask:

- `VK_ACCESS_UNIFORM_READ_BIT`
- `VK_ACCESS_SHADER_READ_BIT`
- `VK_ACCESS_SHADER_WRITE_BIT`

vertex shader에서 storage buffer를 읽는다면 `VERTEX_INPUT`이 아니라 `VERTEX_SHADER`가 맞다.

### `VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT`

tessellation control shader 실행 단계다. tessellation을 사용하지 않는 pipeline에서는 발생하지 않는다.

patch 단위 데이터를 만들고 tessellation level을 출력하는 shader stage다.

### `VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT`

tessellation evaluation shader 실행 단계다. tessellator가 만든 좌표를 실제 vertex 위치 등으로 평가한다.

tessellation control/evaluation shader에서 buffer나 texture를 접근하면 각 stage에 맞는 shader access mask를 사용한다.

### `VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT`

geometry shader 실행 단계다. primitive를 입력받아 새로운 primitive를 emit한다.

geometry shader에서 buffer/image를 읽거나 쓰는 경우 `SHADER_READ/WRITE` access와 함께 사용한다.

### `VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT`

fragment shader 실행 단계다. texture sampling, shadow map sampling, storage image access 등이 여기에 해당한다.

shadow map을 fragment shader에서 읽는 경우:

```cpp
dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
dstAccess = VK_ACCESS_SHADER_READ_BIT;
```

주의: color attachment write는 fragment shader가 아니라 `COLOR_ATTACHMENT_OUTPUT` stage다.
depth/stencil attachment write도 fragment shader가 아니라 `EARLY_FRAGMENT_TESTS` 또는 `LATE_FRAGMENT_TESTS`다.

### `VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT`

fragment shader 전에 수행될 수 있는 depth/stencil test 단계다.

depth test가 early로 수행되거나 depth attachment가 이 단계에서 읽기/쓰기될 수 있다.
depth attachment를 렌더링 타깃으로 처음 사용할 때 dst stage로 자주 쓴다.

```cpp
dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
dstAccess = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
```

### `VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT`

fragment shader 이후 수행되는 depth/stencil test 및 depth/stencil write 단계다.

shadow pass에서 depth attachment에 쓴 뒤, 이후 fragment shader에서 shadow map으로 샘플링할 때 producer stage로 자주 쓴다.

```cpp
srcStage = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
srcAccess = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
```

depth write가 early/late 중 어디서 일어나는지는 pipeline state와 shader 특성에 영향을 받으므로,
보수적으로는 `EARLY_FRAGMENT_TESTS | LATE_FRAGMENT_TESTS`를 함께 쓰기도 한다.

### `VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT`

color attachment write, blending, multisample resolve, color attachment load/store가 일어나는 단계다.

swapchain image를 color attachment로 사용할 때 대표적으로 사용한다.

```cpp
dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
dstAccess = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
```

fragment shader가 color를 계산하는 단계는 `FRAGMENT_SHADER`, 그 결과가 attachment에 쓰이는 단계는
`COLOR_ATTACHMENT_OUTPUT`이다.

### `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT`

compute shader dispatch가 실행되는 단계다.

storage buffer, storage image, sampled image, uniform buffer를 compute shader에서 접근하면 이 stage를 사용한다.

예: compute shader가 buffer에 쓴 뒤 transfer copy로 읽을 때.

```cpp
srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
srcAccess = VK_ACCESS_SHADER_WRITE_BIT;

dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
dstAccess = VK_ACCESS_TRANSFER_READ_BIT;
```

### `VK_PIPELINE_STAGE_TRANSFER_BIT`

transfer command가 실행되는 단계다.

해당 명령 예:

- `vkCmdCopyBuffer`
- `vkCmdCopyImage`
- `vkCmdCopyBufferToImage`
- `vkCmdCopyImageToBuffer`
- `vkCmdBlitImage`
- `vkCmdResolveImage`
- `vkCmdClearColorImage`
- `vkCmdClearDepthStencilImage`

흔한 access mask:

- `VK_ACCESS_TRANSFER_READ_BIT`
- `VK_ACCESS_TRANSFER_WRITE_BIT`

### `VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT`

명령의 모든 실질 작업이 끝난 뒤의 경계다. 실제 메모리 접근을 수행하지 않는다.

예전 코드에서 "마지막으로 보내기" 느낌으로 많이 보이지만, 메모리 visibility를 만들기에는 부적절한 경우가 많다.
실제 producer/consumer stage를 지정하는 편이 더 정확하다.

present 전환 같은 오래된 예제에서 dst stage로 보일 수 있지만, sync2에서는 `NONE` 또는 더 구체적인 stage를 쓰는 방향이 선호된다.

### `VK_PIPELINE_STAGE_HOST_BIT`

CPU host access 단계다. mapped memory를 CPU가 읽거나 쓸 때 사용한다.

예: GPU transfer가 staging buffer에 쓴 뒤 CPU가 읽을 때.

```cpp
srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
srcAccess = VK_ACCESS_TRANSFER_WRITE_BIT;

dstStage = VK_PIPELINE_STAGE_HOST_BIT;
dstAccess = VK_ACCESS_HOST_READ_BIT;
```

host coherent memory 여부에 따라 `vkMakeVisibleToCPUMemoryRanges`, `vkMakeVisibleToGPUMemoryRanges` 같은 cache 관리가 별도로 필요할 수 있다.

### `VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT`

모든 graphics pipeline stage를 포함하는 넓은 stage mask다.

정확한 stage를 고르기 어렵거나 여러 graphics stage가 동시에 관련될 때 사용할 수 있지만, 불필요하게 넓은 synchronization을 만들 수 있다.
성능과 명확성을 위해 가능하면 구체적인 stage를 쓰는 편이 좋다.

### `VK_PIPELINE_STAGE_ALL_COMMANDS_BIT`

queue에서 실행되는 모든 command stage를 포함하는 가장 넓은 stage mask다.

디버깅, 캡처, 단순화된 utility code에서는 편하지만, 실제 렌더링 루프에서는 과도한 동기화가 되기 쉽다.
최적화가 필요하면 구체적인 stage/access 쌍으로 좁히는 것이 좋다.

### `VK_PIPELINE_STAGE_NONE`

어떤 stage도 지정하지 않는 값이다. Vulkan 1.3 / synchronization2 스타일에서 더 자주 보인다.

예를 들어 이전 작업이 없거나 이후 기다릴 작업이 없는 dependency를 명확히 표현할 때 사용한다.
legacy `vkCmdPipelineBarrier`에서는 `TOP_OF_PIPE`/`BOTTOM_OF_PIPE`가 비슷한 역할로 쓰이는 경우가 많았다.

`VK_PIPELINE_STAGE_NONE_KHR`는 같은 값의 KHR alias다.

### `VK_PIPELINE_STAGE_FLAG_BITS_MAX_ENUM`

실제 pipeline stage가 아니다. Vulkan C enum의 크기를 강제로 32-bit 범위로 맞추기 위한 sentinel 값이다.

barrier나 wait stage에 사용하면 안 된다.

## Extension / Advanced Stages

### `VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT`

transform feedback이 vertex/geometry shader 출력 데이터를 buffer에 기록하는 단계다.

OpenGL의 transform feedback과 비슷한 용도이며, `VK_EXT_transform_feedback` 확장이 필요하다.

### `VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT`

conditional rendering predicate buffer를 읽어 draw/dispatch 실행 여부를 결정하는 단계다.

`VK_EXT_conditional_rendering` 확장이 필요하다.

### `VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR`

ray tracing acceleration structure를 build/update/copy하는 단계다.

BLAS/TLAS 빌드에서 scratch buffer, acceleration structure storage를 접근할 때 사용한다.

흔한 access mask:

- `VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR`
- `VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR`

`VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV`는 같은 의미의 NV alias다.

### `VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR`

ray generation, closest hit, any hit, miss, intersection, callable shader 실행 단계다.

ray tracing shader에서 acceleration structure, storage buffer, sampled image 등을 접근할 때 사용한다.

`VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_NV`는 같은 의미의 NV alias다.

### `VK_PIPELINE_STAGE_FRAGMENT_DENSITY_PROCESS_BIT_EXT`

fragment density map을 처리하는 단계다. variable rate rendering 계열에서 fragment 밀도 정보를 적용하는 데 사용된다.

`VK_EXT_fragment_density_map` 확장이 필요하다.

### `VK_PIPELINE_STAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR`

fragment shading rate attachment를 읽어 fragment shading rate를 결정하는 단계다.

`VK_PIPELINE_STAGE_SHADING_RATE_IMAGE_BIT_NV`는 같은 의미의 NV alias다.

### `VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT`

mesh shading pipeline의 task shader 실행 단계다.

task shader는 mesh shader workgroup을 생성하거나 culling/LOD 같은 전처리를 수행한다.

`VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV`는 같은 의미의 NV alias다.

### `VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT`

mesh shader 실행 단계다.

traditional vertex input + vertex shader + primitive assembly 일부를 대체할 수 있으며,
meshlet 기반 렌더링에서 사용한다.

`VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV`는 같은 의미의 NV alias다.

### `VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_EXT`

device-generated commands 또는 command preprocessing 단계다.

GPU가 draw/dispatch command를 전처리하거나 생성하는 확장 기능과 관련된다.

`VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_NV`는 같은 의미의 NV alias다.

## 현재 프로젝트에서 보이는 예

### Shadow map: depth attachment write 후 fragment shader sampling

`example2/ShadowMap.cpp`의 shadow pass는 depth image에 쓴 뒤, 같은 프레임의 scene pass에서 fragment shader가 shadow map으로 읽는다.

```cpp
m_shadowTarget.TransitionLayout(
        cmd,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT);
```

의미:

- producer: depth test/write 단계가 shadow map depth를 기록했다.
- consumer: fragment shader가 sampler로 shadow map을 읽는다.
- layout: depth attachment optimal에서 shader read-only optimal로 바뀐다.

### Swapchain image: present image를 color attachment로 사용

```cpp
srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
srcAccess = 0;
dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
dstAccess = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
```

처음 color attachment로 쓰기 전에 color output 단계가 시작되지 않도록 막는다.

### Vertex buffer upload 후 draw

```cpp
srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
srcAccess = VK_ACCESS_TRANSFER_WRITE_BIT;
dstStage = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT;
dstAccess = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
```

transfer copy가 vertex buffer에 쓴 데이터가 vertex input 단계에서 보이도록 한다.

### Compute output 후 CPU readback

```cpp
srcStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
srcAccess = VK_ACCESS_SHADER_WRITE_BIT;
dstStage = VK_PIPELINE_STAGE_HOST_BIT;
dstAccess = VK_ACCESS_HOST_READ_BIT;
```

compute shader가 쓴 결과를 CPU가 읽기 전에 보이도록 한다.

## Stage를 고르는 규칙

1. 이전에 리소스를 실제로 접근한 작업을 찾는다.
2. 그 접근이 일어난 pipeline stage를 `srcStageMask`로 잡는다.
3. 그 접근 종류를 `srcAccessMask`로 잡는다.
4. 이후에 리소스를 접근할 작업을 찾는다.
5. 그 접근이 일어날 pipeline stage를 `dstStageMask`로 잡는다.
6. 그 접근 종류를 `dstAccessMask`로 잡는다.

좋은 barrier는 "최대한 좁지만 정확한" stage/access 쌍이다.

나쁜 패턴:

```cpp
srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
```

항상 틀린 것은 아니지만, dependency가 너무 넓어서 GPU 병렬성을 많이 잃을 수 있다.

더 좋은 패턴:

```cpp
srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
srcAccess = VK_ACCESS_TRANSFER_WRITE_BIT;
dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
dstAccess = VK_ACCESS_SHADER_READ_BIT;
```

리소스가 "transfer로 쓰인 뒤 fragment shader에서 sampled image로 읽힌다"는 사실을 정확히 표현한다.

## 주의할 점

- stage mask는 실행 순서를 말한다.
- access mask는 메모리 접근 종류를 말한다.
- image layout은 이미지 사용 방식과 attachment/shader/transfer 최적 레이아웃을 말한다.
- 이 셋은 같이 맞아야 한다.
- `TOP_OF_PIPE`/`BOTTOM_OF_PIPE`는 실제 메모리 접근 단계가 아니므로 access mask와 잘못 조합하기 쉽다.
- `FRAGMENT_SHADER`와 `COLOR_ATTACHMENT_OUTPUT`은 다르다.
- `FRAGMENT_SHADER`와 `EARLY/LATE_FRAGMENT_TESTS`도 다르다.
- sync2(`VkPipelineStageFlagBits2`, `VkAccessFlagBits2`)에서는 `NONE`, `ALL_TRANSFER`, `COPY`, `BLIT` 등 더 명확한 표현이 추가된다.

## 빠른 매핑표

| 하고 싶은 일                                       | src stage/access                                         | dst stage/access                         |
| -------------------------------------------------- | -------------------------------------------------------- | ---------------------------------------- |
| transfer upload -> vertex buffer read              | `TRANSFER` / `TRANSFER_WRITE`                            | `VERTEX_INPUT` / `VERTEX_ATTRIBUTE_READ` |
| transfer upload -> index buffer read               | `TRANSFER` / `TRANSFER_WRITE`                            | `VERTEX_INPUT` / `INDEX_READ`            |
| transfer upload -> texture sampling                | `TRANSFER` / `TRANSFER_WRITE`                            | `FRAGMENT_SHADER` / `SHADER_READ`        |
| depth attachment write -> shadow map sampling      | `LATE_FRAGMENT_TESTS` / `DEPTH_STENCIL_ATTACHMENT_WRITE` | `FRAGMENT_SHADER` / `SHADER_READ`        |
| color attachment write -> transfer copy screenshot | `COLOR_ATTACHMENT_OUTPUT` / `COLOR_ATTACHMENT_WRITE`     | `TRANSFER` / `TRANSFER_READ`             |
| compute write -> compute read                      | `COMPUTE_SHADER` / `SHADER_WRITE`                        | `COMPUTE_SHADER` / `SHADER_READ`         |
| compute write -> transfer copy                     | `COMPUTE_SHADER` / `SHADER_WRITE`                        | `TRANSFER` / `TRANSFER_READ`             |
| transfer write -> CPU read                         | `TRANSFER` / `TRANSFER_WRITE`                            | `HOST` / `HOST_READ`                     |
