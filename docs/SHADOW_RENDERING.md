# 그림자 렌더링 원리

> 기준 예제:
>
> - 실시간 shadow map: [`example/realtime_shadow.cpp`](../example/realtime_shadow.cpp)
> - BVH hard shadow ray: [`example/bvh_shadow_room.cpp`](../example/bvh_shadow_room.cpp)
> - BVH path tracing: [`example/bvh_path_tracer.cpp`](../example/bvh_path_tracer.cpp)

## 1. 결론

그림자를 만드는 방법은 크게 두 계열로 나뉜다.

1. **Raster shadow mapping**
   - 실시간 게임/뷰어에서 가장 흔히 쓰는 방식이다.
   - 조명 시점에서 depth map을 만든 뒤, 카메라 렌더링 때 그 depth와 비교한다.
   - 빠르지만 bias, aliasing, peter-panning, shadow acne 같은 보정 문제가 있다.

2. **Ray tracing shadow**
   - 표면 hit 지점에서 조명 방향으로 ray를 쏴서 중간에 막히는 물체가 있는지 검사한다.
   - BVH가 있으면 정확한 hard shadow를 만들기 쉽다.
   - area light를 여러 번 샘플링하면 soft shadow도 가능하지만 비용이 증가한다.

현재 프로젝트에서는 두 방향 모두 가능하다.

- `realtime_shadow`는 **shadow mapping** 예제다.
- `bvh_shadow_room`은 `vkWideBVH`로 **shadow ray**를 쏘는 예제다.
- `bvh_path_tracer`는 여러 bounce와 누적 샘플을 사용하는 **path tracing** 예제다.

## 2. Shadow Mapping

Shadow mapping의 핵심 질문은 다음과 같다.

```text
카메라에서 보이는 현재 점이 조명에서도 보이는가?
```

조명에서 보이지 않는다면, 그 점과 조명 사이에 다른 물체가 있다는 뜻이므로 그림자다.

### 2.1 1단계: 조명 시점 depth pass

먼저 조명을 카메라처럼 취급한다.

```text
scene geometry
  -> light view-projection
  -> depth-only render
  -> shadow map
```

이 pass에서는 색을 렌더링하지 않는다. 오직 depth만 기록한다.

예제에서는 다음 리소스를 만든다.

```cpp
VK_FORMAT_D32_SFLOAT
VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
VK_IMAGE_USAGE_SAMPLED_BIT
```

즉 shadow map은 처음에는 depth attachment로 쓰이고, 이후 scene pass에서는 texture처럼 샘플링된다.

### 2.2 2단계: 카메라 시점 scene pass

일반 렌더링에서는 각 픽셀의 world position을 다시 light clip space로 변환한다.

```glsl
vec4 lightClip = lightViewProj * worldPosition;
vec3 proj = lightClip.xyz / lightClip.w;
vec2 uv = proj.xy * 0.5 + 0.5;
float currentDepth = proj.z;
```

그 다음 shadow map에서 조명 기준으로 가장 가까운 depth를 읽는다.

```glsl
float closestDepth = texture(shadowMap, uv).r;
```

비교는 다음과 같다.

```text
currentDepth <= closestDepth  -> 조명에 보임, lit
currentDepth >  closestDepth  -> 다른 물체 뒤에 있음, shadow
```

실제 셰이더에서는 self-shadowing 오차를 줄이기 위해 bias를 뺀다.

```glsl
currentDepth - bias <= closestDepth
```

## 3. Shadow Acne와 Bias

Shadow map은 finite resolution depth texture다.
카메라에서 계산한 depth와 shadow map에 저장된 depth가 같은 표면에서도 완전히 일치하지 않을 수 있다.

이때 표면이 자기 자신을 가리는 것처럼 보이는 줄무늬가 생긴다.
이를 **shadow acne**라고 한다.

해결에는 bias가 필요하다.

```glsl
float ndotl = max(dot(normal, -lightDir), 0.0);
float bias = max(0.0025 * (1.0 - ndotl), 0.0007);
```

기울어진 표면일수록 depth 오차가 커지므로 normal과 light direction을 이용해 bias를 키운다.

주의할 점:

- bias가 너무 작으면 shadow acne가 생긴다.
- bias가 너무 크면 물체와 그림자가 떨어져 보인다.
- 그림자가 물체 밑에서 뜨는 현상을 **peter-panning**이라고 한다.

## 4. PCF

Shadow map을 한 점만 샘플링하면 그림자 경계가 계단처럼 딱딱하게 보인다.

PCF, Percentage-Closer Filtering은 주변 texel 여러 개를 비교한 뒤 평균을 내는 방식이다.

```glsl
float visibility = 0.0;
for (int y = -1; y <= 1; ++y) {
    for (int x = -1; x <= 1; ++x) {
        float closestDepth = texture(shadowMap, uv + vec2(x, y) * texelSize).r;
        visibility += currentDepth - bias <= closestDepth ? 1.0 : 0.28;
    }
}
visibility /= 9.0;
```

이 프로젝트의 `realtime_shadow_scene.frag`는 3x3 PCF를 사용한다.

PCF는 실제 area light soft shadow와는 다르다.
경계를 보기 좋게 부드럽게 만드는 필터링이다.

## 5. Directional Light와 Point/Spot Light

Shadow map은 조명 타입에 따라 projection이 달라진다.

### 5.1 Directional Light

태양처럼 방향만 있고 위치가 중요하지 않은 조명이다.
보통 orthographic projection을 사용한다.

```text
lightView = lookAt(lightPositionAlongDirection, target)
lightProj = orthographic(...)
```

`realtime_shadow` 예제는 이 방식에 가깝다.
조명 위치를 움직이지만, shadow projection 자체는 orthographic이다.

### 5.2 Spot Light

손전등처럼 위치와 방향, cone angle이 있는 조명이다.
perspective projection을 사용한다.

```text
lightView = lookAt(lightPosition, target)
lightProj = perspective(coneAngle, aspect, near, far)
```

### 5.3 Point Light

모든 방향으로 빛을 쏘는 조명이다.
일반 2D shadow map 하나로는 부족하다.
보통 cubemap shadow map을 사용한다.

```text
point light
  -> 6 faces
  -> cube depth map
```

## 6. Ray Traced Shadow

BVH 기반 shadow는 질문이 훨씬 직접적이다.

```text
표면 hit point에서 light 방향으로 ray를 쏜다.
light에 도착하기 전에 교차가 있으면 shadow.
```

hard shadow는 다음처럼 계산한다.

```glsl
vec3 shadowOrigin = hitPoint + normal * 0.001;
bool occluded = traceAny(shadowOrigin, lightDir, lightDistance);
```

장점:

- shadow map resolution 문제가 없다.
- bias 문제는 훨씬 작다.
- 삼각형 geometry 기준으로 정확하다.
- 점광원/area light/복잡한 조명 모델로 확장하기 쉽다.

단점:

- 픽셀마다 추가 BVH traversal이 필요하다.
- soft shadow는 light를 여러 번 샘플링해야 해서 더 비싸다.
- 투명체 그림자, caustics, multi-bounce는 path tracing 영역으로 넘어간다.

현재 `cmd_raytrace_wide.comp`는 primary ray hit 후 shadow ray를 한 번 더 쏘는 구조다.

```text
camera ray
  -> closest hit
  -> shadow ray
  -> Lambert shading
```

## 7. Path Tracing의 그림자

Path tracing에서는 그림자를 별도 효과로 “붙이는” 것이 아니라, 빛의 경로를 샘플링하면서 자연스럽게 생긴다.

```text
camera ray
  -> surface hit
  -> direct light sample
  -> shadow ray
  -> bounce ray
  -> repeat
```

`bvh_path_tracer`는 다음을 수행한다.

- `traceClosest()`로 가장 가까운 삼각형 hit를 찾는다.
- diffuse/metal/dielectric/emissive 재질을 구분한다.
- diffuse 표면에서는 area light를 직접 샘플링한다.
- light sample까지 `traceAny()` shadow ray를 쏜다.
- 반사/굴절 bounce를 이어간다.
- frame별 결과를 accumulation buffer에 누적한다.

Path tracing의 장점:

- 반사와 굴절이 자연스럽다.
- area light soft shadow가 물리적으로 더 그럴듯하다.
- indirect lighting을 표현할 수 있다.

단점:

- 노이즈가 있다.
- 여러 frame 또는 여러 sample이 필요하다.
- 실시간 raster shadow보다 훨씬 비싸다.

## 8. 현재 예제별 역할

| 예제 | 방식 | 목적 |
| --- | --- | --- |
| `realtime_shadow` | shadow map | 실시간 raster 그림자 |
| `bvh_shadow_room` | BVH hard shadow ray | BVH traversal로 직접 가림 검사 |
| `bvh_path_tracer` | BVH path tracing | 반사/굴절/soft shadow/누적 렌더링 |

실시간 앱 또는 에디터 viewport는 보통 `realtime_shadow` 방식이 적합하다.
품질 위주의 오프라인/프리뷰 렌더링은 `bvh_path_tracer` 쪽이 적합하다.

## 9. 구현 체크리스트

실시간 shadow map 구현 시 필요한 리소스:

- shadow depth image
- shadow image view
- shadow sampler
- descriptor set layout
- descriptor set
- depth-only pipeline
- scene pipeline
- light view-projection matrix
- scene view-projection matrix

프레임마다 필요한 순서:

```text
1. shadow image -> DEPTH_ATTACHMENT_OPTIMAL
2. depth-only pass from light
3. shadow image -> SHADER_READ_ONLY_OPTIMAL
4. swapchain image -> COLOR_ATTACHMENT_OPTIMAL
5. scene depth image -> DEPTH_ATTACHMENT_OPTIMAL
6. scene pass samples shadow map
7. swapchain image -> PRESENT_SRC_KHR
```

품질 조절 포인트:

- shadow map 해상도
- light projection 범위
- depth bias
- PCF kernel 크기
- cascade shadow map 적용 여부

## 10. 다음 개선 방향

`realtime_shadow`를 실제 렌더러 계층으로 끌어올리려면 다음을 `vkRender` 쪽 공통 기능으로 옮기는 것이 좋다.

1. `Image`, `ImageView`, `Sampler` RAII 래퍼
2. depth attachment 생성 헬퍼
3. descriptor set allocator
4. graphics pipeline builder
5. shadow pass abstraction
6. light component와 camera component
7. cascade shadow map
8. variance shadow map 또는 EVSM

현재 예제는 원리를 명확히 보여주기 위해 리소스 생성과 pass 기록을 한 파일 안에 둔 상태다.
