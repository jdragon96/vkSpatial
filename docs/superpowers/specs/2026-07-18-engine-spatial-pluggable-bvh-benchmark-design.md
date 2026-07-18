# 교체 가능한 Spatial 가속구조 + 벤치마크 (Engine::Spatial)

**날짜:** 2026-07-18
**대상 모듈:** `Engine::Spatial` (신규), `example2/` (벤치마크 실행파일), `test/` (정확도 회귀)
**상태:** 설계 승인됨 — 구현 계획 작성 예정

## 1. 배경 / 문제

`src/Engine/Spatial/BVH.h`는 현재 **헤더 스텁만** 존재하고 `.cpp` 구현이 없다. 옛 `src/vkSpatial/`에는
서로 같은 쿼리 면(`Build`/`KNN`/`RadiusSearch`/`Length`)을 공유하는 두 가속구조가 이미 있다:

- `vkBVH` — 이진 LBVH (Karras 스타일: Morton → radix sort → hierarchy → bounds). 단순하지만 쿼리마다 파이프라인을 재생성한다.
- `vkWideBVH` — N-wide BVH (PIMPL, 커널 캐시, ray/path tracing 포함).

목표는 이 알고리즘들을 프로젝트의 마이그레이션 방향(`Engine::Core` 위에 신규 구축)에 맞춰
**쉽게 교체하며 벤치마크할 수 있는 구조**로 새로 설계하는 것이다. 옛 `vkSpatial` 코드는 건드리지 않는다.

### 승인된 결정 (브레인스토밍)

1. **1차 범위:** binary LBVH와 wide BVH **둘 다** Engine::Core로 포팅해 처음부터 두 알고리즘을 나란히 벤치마크한다.
2. **공통 교체 면:** 쿼리(`Build`/`KNN`/`RadiusSearch`) + 메트릭만 공통. ray/path tracing은 별도 `RayTraceable` capability로 분리.
3. **측정 지표:** Build 시간, 쿼리 시간, 노드/메모리, 정확도 검증 (4가지 모두).
4. **타이밍:** wall-clock (CPU측, `chrono`). `Dispatch`가 `queueWaitIdle`로 블록하므로 벽시간에 GPU 실행이 온전히 포함됨.
5. **교체 메커니즘:** 런타임 다형(추상 `SpatialIndex` base + 팩토리).
6. **개선 자유도:** "순수 포팅"에 얽매이지 않고 **완전한 개선**을 허용 — 셰이더/알고리즘은 재사용하되 C++는 클린하게 새로 설계한다.

### 목표 / 비목표

**목표**
- 하나의 추상 인터페이스 뒤에서 알고리즘을 교체할 수 있다.
- 새 알고리즘 추가 = 서브클래스 1개 + 팩토리 1줄.
- 두 백엔드를 Build/쿼리/메모리/정확도로 공정하게 비교하는 실행파일.
- 두 백엔드 정확성을 CI에서 보장하는 parametrized gtest.

**비목표 (이번 범위 아님)**
- ray/path tracing 구현(`RayTraceable`는 **선언만**).
- GPU timestamp query 기반 타이밍(후속 hook 여지만 남김).
- 옛 `vkSpatial` 코드의 마이그레이션/삭제.
- 새 가속구조 알고리즘(SAH/PLOC/TRBVH 등) 실제 구현 — 인터페이스가 수용만 하면 됨.

## 2. 아키텍처 개요

```
                 ┌──────────────────────────────┐
                 │  SpatialIndex (추상 base)     │  ← 교체 면
                 │  Build<T> / KNN / RadiusSearch │
                 │  Length / NodeCount /          │
                 │  MemoryBytes / Name            │
                 └───────────────┬──────────────┘
                        ┌─────────┴─────────┐
              ┌─────────▼────────┐  ┌────────▼─────────┐
              │   BinaryLBVH     │  │     WideBVH      │──▶ (후속) RayTraceable
              │  bvh_*.comp +    │  │ bvh_wide_*.comp +│
              │  cmd_knn/radius  │  │ cmd_*_wide       │
              └──────────────────┘  └──────────────────┘
                        ▲                    ▲
                        └──── MakeSpatialIndex(ctx, kind, params) ────┘
                                        │
                        ┌───────────────┴───────────────┐
                        │  bvh_benchmark (example2)      │  벤치마크가 리스트로 균일 순회
                        │  test_spatialIndex (TEST_P)    │  백엔드별 정확도 회귀
                        └───────────────────────────────┘
```

모든 백엔드는 `Engine::Core::Context&`를 받아 `Engine::Core::Buffer`/`ComputePipeline`만 사용한다
(별도 컨텍스트/서비스 로케이터 없음 — 프로젝트 관례 유지).

## 3. 모듈 레이아웃

```
src/Engine/Spatial/
  BVHTypes.h                  (기존) Primitive/MortonCode/MortonConstant/converters — 공유, 변경 없음
  SpatialIndex.h              NEW  추상 base + RayTraceable capability + BVHKind/BVHParams
  BinaryLBVH.h / .cpp         NEW  이진 LBVH 백엔드
  WideBVH.h / .cpp            NEW  wide BVH 백엔드 (build + query만)
  SpatialIndexFactory.h/.cpp  NEW  MakeSpatialIndex(...)
  BVH.h                       삭제 (미참조 스텁 — BinaryLBVH가 대체)

example2/
  bvh_benchmark.cpp           NEW  벤치마크 실행파일
  CMakeLists.txt              수정 (bvh_benchmark 타깃 추가)

test/
  test_spatialIndex.cpp       NEW  parametrized 정확도 gtest (GLOB이라 CMake 수정 불필요)
```

- `src/Engine/CMakeLists.txt`의 `Engine::Spatial`은 `GLOB_RECURSE Spatial/*.cpp`이므로 **라이브러리 CMake 수정 불필요** — 새 `.cpp`만 넣으면 됨.
- `test/CMakeLists.txt`도 `GLOB *.cpp` — 새 테스트 파일 자동 수집.
- `example2/`만 실행파일 타깃 추가가 필요.

## 4. 인터페이스 정의 (`SpatialIndex.h`)

```cpp
#pragma once
#include "Engine/Core/Context.h"
#include "Engine/Spatial/BVHTypes.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace Engine::Spatial {

    // 모든 가속구조가 구현하는 공통 교체 면.
    class SpatialIndex {
    public:
        virtual ~SpatialIndex() = default;

        SpatialIndex(const SpatialIndex &) = delete;
        SpatialIndex &operator=(const SpatialIndex &) = delete;

        // 편의 템플릿(비가상): T→Primitive 변환 후 protected 가상으로 위임.
        // 모든 백엔드에서 idx->Build(points) 편의 API가 동일하게 동작.
        template<typename T>
        void Build(const std::vector<T> &primitives) {
            std::vector<Primitive> prims;
            prims.reserve(primitives.size());
            for (uint32_t i = 0; i < static_cast<uint32_t>(primitives.size()); ++i)
                prims.push_back(PrimitiveConverter<T>::convert(primitives[i], i));
            BuildFromPrimitives(prims);
        }

        virtual std::vector<uint32_t> RadiusSearch(float cx, float cy, float cz, float r) = 0;
        virtual std::vector<uint32_t> KNN(float cx, float cy, float cz, int k) = 0;

        virtual uint32_t Length() const = 0;       // primitive 수
        virtual uint32_t NodeCount() const = 0;    // 구조 노드 수 (binary: 2N-1, wide: 더 적음)
        virtual uint32_t MemoryBytes() const = 0;  // 소유한 GPU 버퍼 총 바이트
        virtual const char *Name() const = 0;      // 벤치마크 라벨 (예: "BinaryLBVH", "WideBVH(leaf=4)")

    protected:
        SpatialIndex() = default;
        virtual void BuildFromPrimitives(const std::vector<Primitive> &prims) = 0;
    };

    // 선택적 capability — ray tracing 가능한 백엔드만 구현.
    // 이번 범위에서는 선언만; WideBVH가 후속에서 이 인터페이스를 추가로 상속하고 구현한다.
    // 벤치마크/호출자는 dynamic_cast<RayTraceable*>(idx)로 지원 여부를 조회한다.
    class RayTraceable {
    public:
        virtual ~RayTraceable() = default;
        // TODO(후속): TraceRays / TracePath 선언. 시그니처는 옛 vkWideBVH 참고.
    };

    // ── 팩토리 선택 ────────────────────────────────────────────────
    enum class BVHKind {
        BinaryLBVH,
        Wide,
    };

    struct BVHParams {
        uint32_t maxLeafPrimitives = 4; // Wide 전용. BinaryLBVH는 무시.
    };

    std::unique_ptr<SpatialIndex>
    MakeSpatialIndex(Engine::Core::Context &ctx, BVHKind kind, const BVHParams &params = {});

} // namespace Engine::Spatial
```

**설계 근거**
- `Build<T>`는 base 비가상 템플릿 → 가상 `BuildFromPrimitives`로 위임. 편의 API 유지 + 각 백엔드는 `Primitive[]`만 처리.
- `NodeCount`/`MemoryBytes`/`Name`을 인터페이스에 포함 → 벤치마크가 백엔드를 몰라도 메모리·라벨 지표 수집 가능.
- 가상 호출 오버헤드는 GPU dispatch + `queueWaitIdle` 대비 무시 가능.

## 5. 백엔드 설계

### 5.1 커널 수명 패턴 (양쪽 공통)

**캐시 패턴**을 채택한다: `Build()`에서 GPU 버퍼와 **쿼리 커널(`ComputePipeline`)까지 한 번 생성**하고,
각 `KNN`/`RadiusSearch` 호출은 push-constant/입력만 갱신(rebind)한다.

- 근거: 옛 `vkBVH`는 쿼리마다 파이프라인을 재생성해 쿼리 시간에 셰이더 컴파일/파이프라인 생성 비용이 섞였다.
  벤치마크가 **순수 순회 비용**을 재려면 커널을 캐시해야 공정하다. (승인된 "완전한 개선" 범위.)
- 참고 관례: `Engine::Spatial::SimpleTSDF`가 이미 이 패턴(Build에서 버퍼+커널 캐시, Integrate에서 rebind)을 사용.

### 5.2 BinaryLBVH (`BinaryLBVH.h/.cpp`)

옛 `vkBVH`의 6단계 빌드를 Engine::Core로 재작성. 셰이더는 기존 것 재사용:
`bvh_mortonCode.comp` → `bvh_radixSort_histogram/prefixScan/reorder.comp` → `bvh_hierarchy.comp` →
`bvh_boundingBox.comp`, 쿼리는 `cmd_knn.comp` / `cmd_radiusSearch.comp`.

- 버퍼: `Engine::Core::Buffer` — prim / morton(ping) / morton(pong) / histogram / node / constructionInfo.
- 빌드 단계는 private 멤버 함수로 분리(allocate/upload/morton/sort/hierarchy/bounds).
- 쿼리 커널 2개(knn, radius)는 Build에서 생성해 캐시. 쿼리 시 push-constant(중심/r/k) + 결과버퍼만 갱신.
- 메트릭: `NodeCount() = 2*N - 1`, `MemoryBytes()` = 소유 버퍼 크기 합, `Name() = "BinaryLBVH"`.
- 2개 미만 primitive는 예외(`std::runtime_error("BinaryLBVH: ...")`) — 프로젝트 관례.

### 5.3 WideBVH (`WideBVH.h/.cpp`)

옛 `vkWideBVH`의 **build + KNN/RadiusSearch만** 포팅. 셰이더 재사용:
`bvh_wide_morton.comp`, `bvh_wide_range*.comp`, `bvh_wide_build*.comp`, `bvh_wide_bounds*.comp`,
쿼리는 `cmd_knn_wide.comp` / `cmd_radiusSearch_wide.comp`.

- 생성자 `explicit WideBVH(Context&, uint32_t maxLeafPrimitives = 4)`. 범위 검증(예: 1..64) 실패 시 예외.
- 캐시 패턴 동일. 내부 순회 버퍼(node/leaf/sortedMorton/primitive) 보유.
- 메트릭: `NodeCount()` = 실제 wide 노드 수(2N-1보다 작음), `MemoryBytes()`, `Name() = "WideBVH(leaf=K)"`.
- **ray/path tracing 미포팅.** 후속에서 `class WideBVH : public SpatialIndex, public RayTraceable`로 확장하고,
  순회 버퍼 접근자와 `cmd_raytrace_wide.comp`/`cmd_pathtrace_wide.comp`를 연결. 이번엔 구현하지 않는다.

### 5.4 팩토리 (`SpatialIndexFactory.cpp`)

```cpp
std::unique_ptr<SpatialIndex>
MakeSpatialIndex(Engine::Core::Context &ctx, BVHKind kind, const BVHParams &p) {
    switch (kind) {
        case BVHKind::BinaryLBVH: return std::make_unique<BinaryLBVH>(ctx);
        case BVHKind::Wide:       return std::make_unique<WideBVH>(ctx, p.maxLeafPrimitives);
    }
    throw std::runtime_error("MakeSpatialIndex: unknown BVHKind");
}
```

새 알고리즘 추가 = `BVHKind` enum 값 1개 + 이 switch에 case 1줄 + 서브클래스 1개.

## 6. 벤치마크 하네스 (`example2/bvh_benchmark.cpp`)

독립 실행파일. Vulkan 컨텍스트를 초기화하고(테스트처럼 실패 시 스킵/에러), 아래를 수행한다:

1. **입력 생성:** N 스윕(예: 1k / 10k / 100k / 1M) 균일 랜덤 포인트(고정 시드로 재현성).
2. **백엔드 리스트:** 팩토리로 `{BinaryLBVH, Wide(leaf=4), Wide(leaf=8)}` 등을 만들어 순회.
3. **지표 측정 (백엔드 × N):**
   - **Build 시간** — warmup 1회 후 R회 반복, 평균/중앙값(ms).
   - **쿼리 시간** — M개 랜덤 쿼리점에 대해 `KNN(k)` / `RadiusSearch(r)`, 쿼리당 평균 µs.
   - **메모리/노드** — `NodeCount()`, `MemoryBytes()`.
   - **정확도** — CPU brute-force 레퍼런스와 비교(KNN 순서 일치 / radius 집합 일치 → match/recall).
4. **출력:** 알고리즘 × 지표 표를 stdout에 정렬 출력. `--csv <path>` 옵션 시 CSV도 기록.

- **`BenchTimer`**(chrono 기반, warmup+repeat+mean/median 유틸)는 example 내부에 로컬 정의(간단·자족).
- 정확도용 CPU 레퍼런스(`cpuKNN`/`cpuRadiusSearch`)는 기존 테스트의 것을 벤치마크 파일에도 로컬 복사(작은 함수, 의존 최소화).
- 파라미터(N 스윕, R, M, k, r)는 코드 상단 상수 또는 간단한 argv 파싱.

## 7. 정확도 회귀 (`test/test_spatialIndex.cpp`)

`TEST_P`로 `BVHKind`(+`BVHParams`)를 파라미터화하고, 팩토리로 백엔드를 생성해 동일한 검사를 실행:

- 단일 primitive, radius=CPU 일치, KNN=CPU 순서 일치, wide는 `NodeCount() < 2N-1` 확인.
- 검증(널 컨텍스트/잘못된 leaf 크기/유효하지 않은 AABB) 케이스.
- Vulkan 미가용 시 `GTEST_SKIP`(기존 테스트 관례).
- `INSTANTIATE_TEST_SUITE_P`에 `BinaryLBVH`, `Wide(4)`, `Wide(8)`를 등록. 미래 백엔드는 값 1개 추가로 커버.

숫자는 실행파일이, 정확성 가드는 gtest가 담당한다.

## 8. 데이터 흐름 (예: KNN 쿼리)

```
사용자 → idx->Build(points)               [Build 시점 1회]
          └ BuildFromPrimitives(prims)
              └ 버퍼 Allocate/Upload + 빌드 커널 dispatch + 쿼리 커널 생성·캐시
사용자 → idx->KNN(cx,cy,cz,k)              [쿼리마다]
          └ 캐시된 knn 커널에 push-constant + 결과버퍼 rebind → Dispatch(동기) → Download → vector<uint32_t>
```

## 9. 에러 처리 관례

- 실패 시 `std::runtime_error("<ClassName>: <메시지>")`.
- GPU 공유 구조체(push-constant 등)는 `static_assert(std::is_standard_layout_v<...>)` + 필요한 `offsetof`/`sizeof` 체크 유지.
- 팩토리는 알 수 없는 `kind`에 예외.

## 10. 리스크 / 미해결

- **wide 빌드 셰이더 std430 레이아웃**: 옛 `vkWideBVH`가 쓰던 push-constant/SSBO 구조체를 Engine 포팅 시 그대로 옮겨야 함
  (셰이더는 재사용하므로 C++ 구조체 바이트 레이아웃이 정확히 일치해야 함). `static_assert`로 방어.
- **셰이더 경로**: 테스트/실행파일은 `VKBVH_SHADER_DIR` 정의를 통해 셰이더를 찾음 — 벤치마크 타깃도 동일 정의 필요.
- **큰 N(1M)에서의 GPU 메모리**: N 스윕 상한은 장비에 따라 조정 가능하도록 상수화.

## 11. 향후 작업 (이번 범위 밖)

- `WideBVH`에 `RayTraceable` 구현(TraceRays/TracePath + wide raytrace/pathtrace 셰이더 연결).
- GPU timestamp query 기반 정밀 타이밍(인터페이스에 hook만 남김).
- 추가 알고리즘 백엔드(SAH/PLOC/TRBVH 등) — `BVHKind` + 서브클래스로 확장.
- 옛 `vkSpatial` 소비자를 `Engine::Spatial`로 마이그레이션.
