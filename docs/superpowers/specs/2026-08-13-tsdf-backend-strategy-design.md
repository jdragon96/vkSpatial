# TSDF 백엔드 전략 패턴 — 구조 설계

> 목적: TSDF 파이프라인을 **단계별로 갈아끼우며 측정**할 수 있는 구조를 만든다. 직접적인 동기는
> [`HASH_PROBING_AND_LOAD_FACTOR.md`](../../HASH_PROBING_AND_LOAD_FACTOR.md)에서 나온 질문
> — "선형탐사를 버킷 해시로 바꾸면 타일당 메모리를 정말 1.8배 아끼는가" — 를 추측이 아니라
> 숫자로 답하는 것.
>
> 작성일: 2026-08-13 · 상태: 설계 리뷰 대기
> 인터페이스: [`src/TSDF/Volume.h`](../../../src/TSDF/Volume.h)
>
> **네임스페이스 방침**: `src/TSDF/`는 `src/Engine/`의 형제인 최상위 도메인이다. 앞으로
> `src/Registration/` 등이 같은 층에 추가된다. 따라서 네임스페이스는 `Engine::TSDF`가 아니라
> **`TSDF`** 이고, 경로 = 네임스페이스 관례를 그대로 따른다.

---

## 1. 축은 세 개, 그런데 이음매의 종류가 다르다

이게 구조 설계의 출발점이다. Memory / Integrate / Extract로 나누는 것 자체는 맞지만,
**세 축이 같은 방식으로 교체되지 않는다.** 이름만 보고 같은 메커니즘을 적용하면 무너진다.

| 축 | 무엇이 바뀌나 | 이음매 | 교체 방식 |
|---|---|---|---|
| **Memory** | 공간 조직 (Flat / Tile / Submap) | **순수 C++** | 가상 함수 |
| └ Hash | 슬롯 주소 지정 (선형탐사 / 버킷 / SoA) | **GLSL 조각** | `#include` 주입 |
| **Integrate** | 관측 융합 (point-to-plane / projective) | **GLSL 커널** | 커널 교체 |
| **Extract** | 볼륨에서 읽어내기 | **GLSL 커널** | 커널 교체 |

**Memory가 순수 C++인 것은 기존 코드에서 확인된다.** `TiledAdvancedTSDF`는
`TiledDirectionalTSDF<AdvancedTSDF>` 코디네이터이고, 셰이더는 타일별 `AdvancedTSDF`의 것을
그대로 쓴다. 타일링이 바꾸는 것은 어느 타일의 버퍼를 바인딩하고 origin push constant에 무엇을
넣느냐뿐이다. 그래서 `TileStrategy` / `SubmapStrategy`는 GLSL 없이 성립한다.

**Hash를 Memory의 하위 축으로 둔 이유**: 슬롯 주소 지정은 독립된 축이 아니라 메모리 레이아웃의
일부다. `TileStrategy`는 자기가 어떤 해시를 쓰는지 몰라도 되지만, 해시는 반드시 어떤 메모리
전략 안에서 산다. 이 둘을 형제로 두면 백엔드가 곱집합으로 늘어난다.

---

## 2. 폴더 구조

```
src/TSDF/
    Volume.h                                # 교체 가능 인터페이스 (작성 완료)
    VolumeRegistry.cpp                      # Default(), Names(), Volume::Integrate()
    ComposedVolume.h / .cpp                 # 세 축을 조립하는 Volume 구현체
    CMakeLists.txt                          # add_library(TSDF)

    Memory/                                 # C++ 축 — 공간 조직
        MemoryStrategy.h                    # 공통 인터페이스
        FlatStrategy.h / .cpp               # 단일 512^3 창
        TileStrategy.h / .cpp               # 지연 타일링
        SubmapStrategy.h / .cpp             # 2단계 밀도 적응
        Hash/
            HashTable.h / .cpp              # 테이블 버퍼 + 수명 주기
            HashTable.clear.comp.glsl
            HashTable.rehash.comp.glsl
            HashStrategy.h / .cpp           # 어느 조각을 주입할지 + 성장 임계값
            LinearProbe.glsl                # 주입 조각 A
            Bucketed.glsl                   # 주입 조각 B

    Integrate/                              # GLSL 축 — 관측 융합
        IntegrateStrategy.h
        PointToPlaneIntegrator.h / .cpp
        PointToPlaneIntegrator.comp.glsl
        ProjectiveIntegrator.h / .cpp
        ProjectiveIntegrator.comp.glsl

    Extract/                                # GLSL 축 — 읽어내기
        ExtractStrategy.h
        CompactExtractor.h / .cpp           # 점유 슬롯 -> AdvancedEntry
        CompactExtractor.comp.glsl
        OrientedPointExtractor.h / .cpp     # 방향성 점 후보
        OrientedPointExtractor.comp.glsl
```

### 단계별 분할이 이름 규칙 문제를 해결한다

요구사항은 "가속 코드는 호출하는 CPP와 같은 폴더, 같은 이름"이다. 이전 설계에서는
`AdvancedTSDFBackend.cpp` 하나가 커널 다섯 개를 불러서 stem이 겹쳤고,
`<CppStem>.<kernel>.comp.glsl`이라는 군더더기 규칙이 필요했다.

단계별로 쪼개면 그 다섯 커널이 각자의 클래스로 흩어져서 **대부분 클래스 하나 ↔ 커널 하나**가 된다.

| 기존 커널 | 새 위치 | 짝이 되는 CPP |
|---|---|---|
| `advanced_tsdf_integrate` | `Integrate/PointToPlaneIntegrator.comp.glsl` | `PointToPlaneIntegrator.cpp` |
| `advanced_tsdf_extract` | `Extract/OrientedPointExtractor.comp.glsl` | `OrientedPointExtractor.cpp` |
| `advanced_tsdf_compact` | `Extract/CompactExtractor.comp.glsl` | `CompactExtractor.cpp` |
| `advanced_tsdf_clear` | `Memory/Hash/HashTable.clear.comp.glsl` | `HashTable.cpp` |
| `advanced_tsdf_rehash` | `Memory/Hash/HashTable.rehash.comp.glsl` | `HashTable.cpp` |

규칙은 **`<CppStem>.comp.glsl`**, 한 클래스가 커널을 둘 이상 가질 때만
`<CppStem>.<kernel>.comp.glsl`. 실제로 후자가 필요한 곳은 `HashTable`(clear/rehash) 하나뿐이고,
둘 다 테이블 수명 주기라 같은 클래스에 있는 게 맞다.

### 두 종류의 GLSL

| 종류 | 확장자 | 짝 CPP | 위치 |
|---|---|---|---|
| **커널** — `main()`이 있고 `ComputePipeline::Build`가 직접 로드 | `.comp.glsl` | 있음 | 호출 CPP와 같은 폴더 |
| **조각** — `#include` 전용, 진입점 없음 | `.glsl` | 없음 | 역할별 (`Memory/Hash/`, 기존 `shader/voxel_common.glsl`) |

이름 규칙은 커널에만 적용된다. 조각은 짝이 되는 CPP가 애초에 없다.

---

## 3. Extract의 경계 — 기존 `Extraction`과 겹치지 않게

`Engine::Spatial::Extraction`에 `IsoSurfaceExtractor` + `ExtractorRegistry`로 MC 계열 7종이 이미
레지스트리로 붙어 있다. `src/TSDF/Extract/`에 같은 것을 또 만들면 중복이다. 경계를 이렇게 긋는다:

- **TSDF → 데이터** (compact, 방향성 점 후보) = TSDF 관심사 → `src/TSDF/Extract/`
- **데이터 → 메시** (MC / MC33 / DC …) = 기존 `Engine::Spatial::Extraction`에 위임

`advanced_tsdf_extract.comp.glsl`은 전자이므로 옮겨오고, 메시 추출은 얇은 어댑터로 넘긴다.

---

## 4. 조합이 성립하기 위한 GLSL 계약

Integrate / Extract 커널이 저장 구조를 직접 만지면 조합이 깨진다. 커널은 접근자만 부르고,
그 접근자를 Memory/Hash가 주입한다.

```glsl
// Integrate/PointToPlaneIntegrator.comp.glsl
#include "hash_strategy.glsl"   // C++이 주입 — 파일로 존재하지 않는다

uint slot = findOrInsert(key);
if (slot == HASH_INSERT_FAILED) { atomicAdd(g_insertFailureCount, 1u); return; }
accumulate(slot, tsdf, weight, unitNormal);
```

`LinearProbe.glsl`과 `Bucketed.glsl`은 동일한 시그니처를 제공한다.

| 심볼 | 계약 |
|---|---|
| `uint findOrInsert(uint key)` | 슬롯 인덱스, 실패 시 `HASH_INSERT_FAILED` |
| `uint findSlot(uint key)` | 조회 전용, 없으면 `HASH_NOT_FOUND` |
| `void accumulate(uint slot, ...)` | 누산기 원자 갱신 (레이아웃이 SoA면 여기서 흡수) |
| `HASH_LOAD_FACTOR_LIMIT` | 성장 임계값 |

**성장 임계값이 전략과 한 몸이라는 점이 중요하다.** 따로 두면 선형탐사에 0.8을 물려서
복셀이 조용히 사라진다. `HashStrategy`가 주입할 조각과 임계값을 함께 들고, C++ 쪽
`maybeGrow`는 그 값을 읽는다.

---

## 5. `ComputePipeline`에 필요한 변경 세 가지

현재 로더는 이 배치를 지원하지 못한다. 셋 다 [`ComputePipeline.cpp`](../../../src/Engine/Core/ComputePipeline.cpp) 국소 변경이다.

### 5.1 셰이더 탐색 경로를 목록으로

지금은 `VKBVH_SHADER_DIR + "/" + filename` 단일 루트다. `src/`를 두 번째 루트로 추가해 순서대로 찾는다.

- 기존 27개 `Build("...")` 호출부는 **한 줄도 안 바뀐다** (첫 루트에서 그대로 발견됨)
- 새 커널은 `Build("TSDF/Integrate/PointToPlaneIntegrator.comp.glsl")`
- 루트를 `src/`로 통일하는 안은 27곳을 전부 고쳐야 해서 기각

### 5.2 `#include` 탐색도 목록으로

`FilesystemIncluder`가 디렉터리 **하나**만 본다. 커널이 `src/TSDF/`로 가면 공유
`voxel_common.glsl`(= `src/shader/`)을 못 찾는다. 검색 목록을 `[셰이더 자기 폴더, src/shader]`로
바꾼다. 상대 경로(`../../shader/voxel_common.glsl`)로 때우는 방법은 셰이더를 옮길 때마다 깨져서 기각.

### 5.3 인메모리 + 파일 혼합 includer

해시 전략 주입에 필요하다. 지금은 `Build(path)`가 `FilesystemIncluder`만, `Build(source, GlslSrc)`가
`InMemoryIncluder`만 쓴다 — 파일 커널은 `AddInclude()`로 넣은 내용을 **볼 수 없다**.
인메모리를 먼저 찾고 없으면 파일로 폴백하는 합성 includer로 바꾼다.

---

## 6. 실제로 물릴 부분 두 가지

### 6.1 셰이더 순열 캐시 키 ⚠️

`compileFileCached`는 `fullPath`만으로 캐싱한다. 같은 커널에 다른 해시 전략을 주입하면
**두 번째 전략이 첫 번째의 SPIR-V를 그대로 돌려받는다.** 조용히 잘못된 결과가 나오는,
A/B 실험을 통째로 무효화하는 종류의 버그다.

**캐시 키 = 커널 경로 + 주입된 모든 전략 이름.** 축이 늘수록 잘못될 경우의 수도 늘어난다.

### 6.2 push constant 소유권

현재 `IntegratePC`에 메모리 필드(`hashCapacity`, `originX/Y/Z`)와 융합 필드(`pointToPlane`,
`confWeight`)가 **한 블록에 섞여** 있다. Vulkan은 블록이 하나이므로, 두 전략이 각자 자기 필드를
채우려면 레이아웃 규약이 필요하다.

**규약: 메모리 필드가 앞, 융합 필드가 뒤.** 공유 헤더에 두 구조체를 정의하고 커널 쪽
push_constant 블록은 그 순서를 그대로 잇는다. 규약을 문서로만 두면 조합할 때마다 오프셋이
어긋나므로, C++ 쪽에 `static_assert`로 offset을 못박는다.

---

## 7. 기존 코드와의 접합: 상속이 아니라 위임

`FlatStrategy` / `TileStrategy` / `SubmapStrategy`는 기존 `AdvancedTSDF` /
`TiledAdvancedTSDF` / `SubmapAdvancedTSDF`를 **상속하지 않고 소유·위임**한다.

이유: `AdvancedTSDF`는 이미 `TiledAdvancedTSDF`의 템플릿 인자이고, 테스트와 `example2`가 직접
쓴다. 여기에 인터페이스를 상속시키면 템플릿 계약이 바뀌고 기존 사용처가 전부 영향권에 든다.
위임하면 **실험 구조가 순수하게 추가만** 되고, 실험이 실패해도 되돌릴 것이 없다.

비용은 위임 함수 몇 개의 보일러플레이트뿐이다.

---

## 8. 진행 순서

각 단계가 끝날 때마다 빌드가 초록이어야 한다.

| # | 작업 | 검증 |
|---|---|---|
| 1 | `ComputePipeline` 탐색 경로 + 2단 includer + 캐시 키 | 기존 27개 호출부 무변경, 전체 테스트 통과 |
| 2 | `VolumeRegistry.cpp` + `CMakeLists.txt` → `TSDF` 타깃 | 인터페이스가 링크됨 |
| 3 | `MemoryStrategy` + Flat/Tile/Submap 위임 어댑터 | 셰이더 이동 없이 기존과 동일 결과 |
| 4 | `ComposedVolume` 조립자 + 레지스트리 등록 | `--tsdf tile` 등으로 전환 동작 |
| 5 | Integrate 축 분리 + 커널 이동 | **순수 이동 커밋**, 내용 변경 없음 |
| 6 | Extract 축 분리 + 커널 이동 | 위와 같음 |
| 7 | `Memory/Hash` 주입, 선형탐사를 첫 전략으로 | 주입 전후 SPIR-V 동등 |
| 8 | `insertFailureCount` 계측 → 버킷 전략 추가 → A/B | 드롭 0 확인 후 임계값 상향 |

**7단계까지는 동작이 하나도 안 바뀌는 리팩터링이다.** 실제 실험은 8단계에서 시작한다.
1~4단계만 끝나도 Memory 축 A/B(Flat vs Tile vs Submap)는 이미 가능하다.

---

## 9. 결정된 사항

**네임스페이스와 클래스 이름** (2026-08-13 확정)

`src/TSDF/`는 최상위 도메인으로 확정 → 네임스페이스 `TSDF`. 그런데 네임스페이스와 클래스
이름이 같으면 **`using namespace TSDF;`를 쓰는 순간 컴파일이 깨진다**:

```
error: reference to 'TSDF' is ambiguous
note: candidate found by name lookup is 'TSDF'        <- 네임스페이스
note: candidate found by name lookup is 'TSDF::TSDF'  <- 클래스
```

선언 시점에는 멀쩡하고 사용처에서만 터지는 지뢰이며, 이 저장소는 `using namespace`를 33곳에서
쓰고 그 대부분이 테스트 파일 상단의 관례다. 그래서 **클래스를 `Volume`으로** 한다.

- `TSDF::Volume`, `TSDF::VolumeParams`, `TSDF::VolumeStats`, `TSDF::VolumeRegistry`
- `using namespace TSDF;` 이후에는 `Volume`, `TileStrategy`, `MemoryStrategy`로 짧게 쓴다

---

## 10. 열린 질문

1. **`VolumeStats::deviceMemoryBytes` 집계 범위** — 해시 테이블만 셀지, 업로드 버퍼와 compact
   스크래치까지 포함할지. 메모리 비교가 목적이므로 전부 포함하는 쪽이 정직하다.
2. **비-방향성 볼륨** — `SimpleTSDF`를 등록할지. `AdvancedEntry`에 `direction`이 있어 0으로
   채우면 되지만, 비교 대상이 아니면 등록하지 않는 게 YAGNI다.
3. **`AdvancedEntry` 의존** — `Volume.h`가 읽기 구조체 하나 때문에 `Engine/Spatial/AdvancedTSDF.h`
   전체(Buffer, ComputePipeline까지)를 끌어온다. 읽기 구조체를 가벼운 헤더로 분리할지.
