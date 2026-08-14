# TSDF 해시 전략 교체 — 설계

> 목적: `AdvancedTSDF`의 **해시 주소 지정 방식**을 실행 중 갈아끼우고, 실제 스캔으로
> 메모리·점유율을 재서 오래된 질문 하나에 답한다 — **선형탐사를 버킷 해시로 바꾸면
> 타일당 메모리가 정말 1.6~1.8배 줄어드는가?**
>
> 작성일: 2026-08-14 · 상태: 리뷰 대기
> 선행: [`2026-08-13-tsdf-backend-strategy-design.md`](2026-08-13-tsdf-backend-strategy-design.md) (3축 구조) ·
> [`HASH_PROBING_AND_LOAD_FACTOR.md`](../../HASH_PROBING_AND_LOAD_FACTOR.md) (이론적 근거) ·
> [`TSDF_VOLUME_STRATEGIES.md`](../../TSDF_VOLUME_STRATEGIES.md) (현재 구조)

---

## 1. 범위

선행 스펙은 축을 셋(Memory / Integrate / Extract) 두었지만, **실제로 쓸어볼 축은 해시 하나**로
좁혔다. 그에 따라:

| 항목 | 결정 |
|---|---|
| 교체 축 | **해시 주소 지정만** — 선형탐사 / 버킷 |
| 대상 백엔드 | **`AdvancedTSDF` 하나.** Tiled·Submap은 이걸 합성하므로 자동으로 따라온다 |
| 나머지 백엔드 | `SimpleTSDF` / `DirectionalTSDF` / `CompactDirectionalTSDF`는 **동결** — 계속 빌드되고 기존 테스트·벤치마크도 그대로 돌지만, 스위칭 대상이 아니다 |
| `Integrate/` `Extract/` | **비워둔다.** 전략이 하나뿐인 축을 지금 만드는 것은 YAGNI |
| 실측 하네스 | **포함.** 이게 없으면 메커니즘만 남고 질문에는 답하지 못한다 |
| SoA 레이아웃 | 후속. 버킷만으로 α≈0.8이 나오는지 먼저 본다 |

`pointToPlane` / `confidenceWeight` / `hermite`는 **이미 push constant 플래그로 스위칭된다.**
이 스펙이 다루는 것은 커널 구조 자체가 달라지는 축뿐이다.

---

## 2. 주입 메커니즘 (검증 완료)

### 2.1 안 되는 것

매크로로 include 경로를 넘기는 방식은 **glslang이 지원하지 않는다.** 실제로 확인했다:

```glsl
#include HASH_STRATEGY_INCLUDE      // -DHASH_STRATEGY_INCLUDE="inc/frag_a.glsl"
```
```
error: '#include' : must be followed by a header name
```

C 전처리기와 달리 glslang의 `#include`는 매크로 전개를 거치지 않는다. 설계가 여기 기대고 있었으므로
먼저 확인한 것이고, 다음 방식으로 대체한다.

### 2.2 되는 것 — 디스패처 파일 + `-D`

include **이름은 리터럴**로 두고, 실제 파일인 디스패처가 매크로로 조각을 고른다.
`glslc`로 양쪽 분기가 실제로 다른 SPIR-V를 내는 것까지 확인했다.

```glsl
// TSDF/Memory/Hash/HashStrategy.glsl — 디스패처 (실제 파일)
#if defined(HASH_BUCKETED)
#include "Bucketed.glsl"
#else
#include "LinearProbe.glsl"
#endif
```

```glsl
// 커널은 이 한 줄만 안다
#include "TSDF/Memory/Hash/HashStrategy.glsl"
```

이 방식의 이점:

- 조각이 **디스크에 실재하는 파일** — 에디터·린터·grep이 그대로 동작한다
- `ComputePipeline`에 **인메모리 includer가 필요 없다.** 매크로 정의 전달과 include 검색 경로면 충분
- 커널 본문은 하나. 적분 수식을 고치면 모든 변형에 한 번에 반영된다

기각한 대안: 변형마다 커널을 통째로 복제하는 방식은 250줄짜리 적분 커널이 변형 수만큼 늘어나고,
수식 수정이 한 군데라도 어긋나면 **A/B 결과가 조용히 거짓말을 한다.**

---

## 3. GLSL 계약

`LinearProbe.glsl`과 `Bucketed.glsl`은 동일한 시그니처를 제공한다. 커널은 저장 구조를 직접
만지지 않는다.

| 심볼 | 계약 |
|---|---|
| `uint findOrInsert(uint key)` | 슬롯 인덱스, 실패 시 `HASH_INSERT_FAILED` |
| `uint findSlot(uint key)` | 조회 전용, 없으면 `HASH_NOT_FOUND` |
| `HASH_LOAD_FACTOR_LIMIT` | 성장 임계값 (선형탐사 0.5, 버킷 0.8) |
| `HASH_NAME` | 계측 출력에 찍을 이름 |

```glsl
uint slot = findOrInsert(key);
if (slot == HASH_INSERT_FAILED) { atomicAdd(g_insertFailureCount, 1u); return; }
```

**성장 임계값이 전략과 한 몸이라는 점이 중요하다.** C++ 쪽에 따로 두면 선형탐사에 0.8을 물려
복셀이 조용히 사라진다. C++ `HashStrategy`가 조각 파일과 임계값을 함께 들고, `maybeGrow`가
그 값을 읽는다.

### 3.1 버킷 변형의 파라미터

`HASH_PROBING_AND_LOAD_FACTOR.md`의 분석에 따라 **버킷 크기 32, 임계값 0.8**로 시작한다.
b=32에서 α=0.9면 버킷 오버플로가 25%까지 올라가 클러스터링이 버킷 단위로 되살아나므로,
0.9는 목표로 잡지 않는다. 실측이 더 허용하면 그때 올린다.

---

## 4. 파일 배치

커널을 **호출하는 CPP 옆으로** 옮긴다. `<CppStem>.<kernel>.comp.glsl` 규칙 —
`AdvancedTSDF.cpp` 하나가 커널 다섯을 부르므로 중간 세그먼트가 필요하다.

```
src/TSDF/
    Memory/
        Hash/
            HashStrategy.h / .cpp      # 이름 → (조각 파일, 매크로, 임계값)
            HashStrategy.glsl          # 디스패처
            LinearProbe.glsl           # 현행 동작
            Bucketed.glsl              # 신규
    Backends/
        AdvancedTSDF.h / .cpp
        AdvancedTSDF.integrate.comp.glsl
        AdvancedTSDF.extract.comp.glsl
        AdvancedTSDF.compact.comp.glsl
        AdvancedTSDF.clear.comp.glsl
        AdvancedTSDF.rehash.comp.glsl
```

동결된 백엔드 셋의 커널은 `src/shader/`에 그대로 둔다 — 건드리지 않기로 한 코드다.

---

## 5. `ComputePipeline` 변경 세 가지

전부 [`ComputePipeline.cpp`](../../../src/Engine/Core/ComputePipeline.cpp) 국소 변경이다.

### 5.1 매크로 정의 전달

`shaderc::CompileOptions::AddMacroDefinition`을 태울 경로가 없다. 빌더에 정의 목록을 받는
형태를 추가한다. 기존 호출부는 정의 없이 그대로 동작한다.

### 5.2 탐색 경로를 목록으로

지금은 커널 경로도 include 경로도 단일 루트다. 커널이 `src/TSDF/Backends/`로 가고 조각이
`src/TSDF/Memory/Hash/`에 있으므로 둘 다 목록이어야 한다.

- 커널 파일: `[VKBVH_SHADER_DIR, src/]` 순서로 탐색 → **기존 27개 호출부는 한 줄도 안 바뀐다**
- `#include`: `[셰이더 자기 폴더, src/shader, src/]`

상대 경로(`../../shader/voxel_common.glsl`)로 때우는 방법은 셰이더를 옮길 때마다 깨져서 기각한다.

### 5.3 캐시 키 ⚠️

`compileFileCached`가 `fullPath`만으로 캐싱한다. **같은 커널에 다른 해시를 물리면 두 번째가
첫 번째의 SPIR-V를 그대로 돌려받는다.** 조용히 잘못된 결과가 나오고 A/B 전체가 무효가 되는
종류의 버그다.

**캐시 키 = 커널 경로 + 정렬된 매크로 정의 목록.**

---

## 6. 계측: `insertFailureCount`

`VolumeStats::insertFailureCount`는 선언만 되어 있고 아무도 증가시키지 않는다. 이 축에서는
**필수**다 — "α를 0.8까지 올려도 안전한가"를 판정하는 유일한 수단이기 때문이다.

- 커널: `findOrInsert` 실패 시 GPU 카운터에 `atomicAdd`
- C++: `AdvancedTSDF`가 그 버퍼를 읽어 노출하고, `FlatStrategy`/`TileStrategy`/`SubmapStrategy`가
  `VolumeStats`에 채운다
- 테스트: 정상 조건에서 **0이어야 한다**. 0이 아니면 그 전략의 임계값이 너무 높은 것이다

`growCount`도 같은 김에 채운다 — 리해시가 몇 번 일어났는지는 메모리 해석에 직접 쓰인다.

---

## 7. 실측 하네스

[`example2/tsdf_folder_eval.cpp`](../../../example2/tsdf_folder_eval.cpp)가 이미 폴더의
`frame_*.ply`를 읽어 적분한다. 다만 `AdvancedTSDF`/`Tiled`/`Submap`을 **직접** 쓰고 있어서
`TSDF::Volume`을 거치지 않는다. 이걸 레지스트리 경유로 바꾸고 플래그를 둘 더한다.

```
tsdf_folder_eval --dir <scan> --voxel 0.01 --tsdf tile --hash bucketed
```

- `--tsdf flat|tile|submap` (기본 `flat`) — 기존 `--submap` 플래그를 대체
- `--hash linear|bucketed` (기본 `linear`)

출력 형식은 아래와 같다. **숫자는 형식을 보이기 위한 예시일 뿐 측정값이 아니다** — 실제 값은
이 작업이 끝나야 나온다:

```
hash      tsdf     occupied     slots    load  tables   tableMB  drops  grows
linear    tile      4812390   9437184   0.510      18    252.0      0      2
bucketed  tile      4812390   5898240   0.816      18    157.5      0      1
```

**이 표의 `tableMB` 두 줄 비율이 이번 작업의 답이다.** 동시에 `tables`가 점유율과 무관하게
크면 메모리가 count-limited라는 뜻이고, 그때는 버킷 해시가 아니라 타일당 초기 용량을 줄이는
것이 올바른 처방이다 — 그 판정도 같은 표에서 나온다.

---

## 8. 테스트 전략

| 무엇을 | 어떻게 |
|---|---|
| 주입이 동작한다 | 같은 커널을 두 전략으로 빌드해 **SPIR-V가 서로 다름**을 확인 (5.3 캐시 키 함정의 회귀 테스트) |
| 선형탐사가 현행과 동일 | 주입 방식으로 바꾼 뒤에도 기존 `TsdfAccessors`/`TsdfFlatStrategy`/`TsdfTileStrategy` 결과가 그대로 |
| 버킷이 옳다 | 같은 스캔 → 추출 결과가 선형탐사와 허용오차 내 일치, `insertFailureCount == 0` |
| 임계값이 전략을 따라간다 | 선형탐사 볼륨의 성장 임계값이 0.5, 버킷이 0.8임을 관측 (리해시 시점의 `slotCapacity` 변화로) |
| 큰 프레임 | 기존 `LargeFrameIsNotTruncatedByMaxPointsPerFrame`가 두 해시 모두에서 통과 |

---

## 9. 위험과 완화

| 위험 | 완화 |
|---|---|
| 캐시 키를 놓쳐 A/B가 조용히 무효 | 5.3 + SPIR-V 상이 테스트를 **첫 태스크**에 배치 |
| 버킷이 α=0.8을 못 버틴다 | `insertFailureCount`가 즉시 드러낸다. 그러면 임계값을 낮추고 절감폭을 다시 보고한다 (가설 기각도 결과다) |
| 실제 장면이 count-limited라 절감이 0 | 같은 표의 `tables` 열이 답한다. 그 경우 결론은 "버킷 해시는 이 문제를 못 푼다"이고, 그것도 유효한 산출물이다 |
| 커널 이동으로 동결 백엔드가 깨짐 | 동결 백엔드의 커널은 옮기지 않는다. 이동은 `AdvancedTSDF`의 5개뿐 |

---

## 10. 명시적 비목표

- `Integrate`/`Extract` 축 — 전략이 하나뿐이라 만들지 않는다
- SoA 레이아웃 — 버킷 결과를 본 뒤 판단
- 동결 백엔드 셋의 정리·삭제
- `Memory/`와 `Backends/`의 계층 중복 해소 — 축이 늘어난 뒤에 판단하는 것이 맞다
