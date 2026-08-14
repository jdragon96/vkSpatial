# TSDF Volume — 교체 가능한 저장 전략

> `src/TSDF/`. TSDF 저장 방식을 **이름 문자열로 갈아끼우며 메모리·점유율을 비교**하기 위한 계층.
> 기존 `Engine::Spatial`의 TSDF 클래스들을 상속하지 않고 **감싸서 위임**하므로, 이 폴더를 통째로
> 지워도 기존 코드는 그대로 돈다.
>
> 설계 문서: [`superpowers/specs/2026-08-13-tsdf-backend-strategy-design.md`](superpowers/specs/2026-08-13-tsdf-backend-strategy-design.md)
> 관련: [`HASH_PROBING_AND_LOAD_FACTOR.md`](HASH_PROBING_AND_LOAD_FACTOR.md) · [`ADVANCED_TSDF.md`](ADVANCED_TSDF.md)

---

## 1. 왜 만들었나

[`HASH_PROBING_AND_LOAD_FACTOR.md`](HASH_PROBING_AND_LOAD_FACTOR.md)에서 나온 질문 하나가 출발점이다.

> 선형탐사 해시를 버킷 해시로 바꾸면 타일당 메모리를 1.8배 아낄 수 있다. 그런데 **그 이득은
> 타일이 꽉 차 있을 때만** 나온다. 실제 장면에서 메모리를 지배하는 게 점유율(load factor)인가,
> 아니면 **타일 개수**인가?

후자라면 해시를 바꿔봐야 한 바이트도 안 줄고, 올바른 처방은 타일당 초기 용량을 줄이는 것이다.
추측으로 결정할 문제가 아니라서, **같은 스캔을 여러 저장 전략에 통과시키고 숫자를 비교하는
도구**를 먼저 만들었다. 이 폴더가 그 도구다.

---

## 2. 구조

### 축이 세 개, 이음매의 종류가 다르다

| 축 | 무엇이 바뀌나 | 이음매 | 상태 |
|---|---|---|---|
| **Memory** | 공간 조직 (flat / tile / submap) | **C++ 가상 함수** | 구현됨 |
| └ Hash | 슬롯 주소 지정 (선형탐사 / 버킷) | GLSL `#include` 주입 | 미착수 |
| **Integrate** | 관측 융합 커널 | GLSL 커널 교체 | 미착수 |
| **Extract** | 읽어내기 커널 | GLSL 커널 교체 | 미착수 |

**Memory 축이 순수 C++인 게 핵심이다.** 타일링은 셰이더를 바꾸지 않는다 — 어느 타일의 버퍼를
바인딩하고 origin push constant에 무엇을 넣느냐만 달라진다. 그래서 가상 함수로 갈린다. 반면
해시 주소 지정은 커널 *안의 함수*라 C++로는 절대 교체할 수 없고, `#include` 주입이 필요하다.
이 둘을 같은 층에 두면 백엔드가 곱집합으로 늘어난다.

### 파일

```
src/TSDF/
    Volume.h              # 교체 가능 인터페이스 + VolumeParams/VolumeStats/IntegrationOptions/VolumeRegistry
    Volume.cpp            # Volume::Integrate (배치 래퍼)
    VolumeRegistry.cpp    # Names(), Default() — flat/tile/submap 등록
    ComposedVolume.h/.cpp # 전략을 물고 Volume을 구현. Integrate/Extract 축이 나중에 여기 붙는다
    Memory/
        MemoryStrategy.h        # Memory 축 인터페이스 + kBytesPerHashSlot
        FlatStrategy.h/.cpp     # AdvancedTSDF 위임
        TileStrategy.h/.cpp     # TiledAdvancedTSDF 위임
        SubmapStrategy.h/.cpp   # SubmapAdvancedTSDF 위임
```

네임스페이스는 `TSDF` — `Engine::TSDF`가 아니다. `src/TSDF/`는 `src/Engine/`의 **형제인 최상위
도메인**이고, 네임스페이스가 경로를 그대로 따른다. 클래스가 `TSDF`가 아니라 `Volume`인 이유는
네임스페이스와 클래스 이름이 같으면 `using namespace TSDF;`를 쓰는 순간 컴파일이 깨지기 때문이다
(이 저장소는 테스트 33곳에서 그 관례를 쓴다).

---

## 3. 쓰는 법

```cpp
#include "TSDF/Volume.h"

const TSDF::VolumeRegistry registry = TSDF::VolumeRegistry::Default();
// registry.Names() == {"flat", "submap", "tile"}

std::unique_ptr<TSDF::Volume> volume = registry.Create("tile");  // 모르는 이름이면 nullptr

TSDF::VolumeParams params;
params.voxelSize = 0.01f;
params.truncation = 0.03f;
params.hashCapacity = 1u << 20;   // 슬롯 수 (타일드는 타일당)
volume->Build(context, params);

TSDF::IntegrationOptions options;
options.pointToPlane = true;
options.currentFrame = frameIndex;
volume->Configure(options);        // ⚠ 첫 적분 전에 (아래 4.2 참조)

volume->Integrate(points, normals, cameraPosition);   // 자체 배치로 즉시 제출
// 또는 여러 볼륨을 한 submit으로 묶고 싶으면:
//   volume->Record(points, normals, cameraPosition, batch);

std::vector<TSDF::AdvancedEntry> entries;
volume->Download(entries);         // 재사용 형태 (프레임 루프용)

const TSDF::VolumeStats stats = volume->Stats();
// stats.occupiedEntryCount / slotCapacity / LoadFactor() / tableCount / deviceMemoryBytes
```

`Record`가 primitive이고 `Integrate`는 그걸 `CommandBatch`로 감싼 편의 함수다. 순서가 반대면
**타일드 전략을 표현할 수 없다** — 수천 타일을 submit 하나로 묶는 게 존재 이유이기 때문이다.

---

## 4. 반드시 알아야 할 함정

### 4.1 `deviceMemoryBytes`는 해시 테이블만 센다

`slotCapacity × 28B`(엔트리 24B + first-fill 스탬프 4B)만 계산한다. **업로드 버퍼와 compact
스크래치는 빠져 있고, 그 크기가 전략마다 극단적으로 다르다** — `TiledAdvancedTSDF`는 첫 다운로드
때 readback 스크래치로 약 335MB를 잡고, `submap`은 그런 인스턴스를 둘 갖는다. 이 숫자는
**테이블 비용끼리만** 비교하는 용도다.

### 4.2 `Configure`는 이미 만들어진 타일에 닿지 않는다

`TiledDirectionalTSDF`의 세터 대부분은 멤버 필드만 쓰고, 그 값은 **타일 생성 시점**에만 타일로
전달된다(`currentFrame`만 예외적으로 기존 타일을 순회한다). 따라서 적분 도중에 옵션을 바꾸면
옛 타일은 옛 설정, 새 타일은 새 설정이 된다. **옵션은 첫 적분 전에 확정할 것.**

반대로 `Build` 전에 `Configure`를 부르는 건 **정상 동작한다** — 감싸는 세터들이 전부 단순 필드
대입이고 `Build`가 그 필드를 초기화하지 않는다. 그래서 `Configure`는 세 전략 모두 의도적으로
null 가드가 없다. 가드를 넣으면 이 순서가 조용히 깨진다.

### 4.3 `Reset` 의미가 전략마다 다르다

`flat`은 테이블을 유지한 채 비우므로 (성장했다면) 커진 용량이 남는다. `tile`/`submap`은 타일 맵을
비우므로 `slotCapacity`/`tableCount`/`deviceMemoryBytes`가 0으로 떨어지고 다음 적분 때 재할당된다.
볼륨 하나를 여러 장면에 재사용하는 스윕은 이 비대칭을 감안해야 한다.

### 4.4 `submap`은 `occupiedEntryCount != entries.size()`

`occupiedEntryCount`는 base + detail의 저장 엔트리를 전부 세지만, `Download`는 dense 블록 안의
base 엔트리를 버린다(detail이 우선). 저장량으로는 정직한 숫자지만 **"표면 복셀 수"로 읽으면 안 된다.**

### 4.5 `submap`의 `Record`는 배치에 아무것도 안 쌓는다

`SubmapAdvancedTSDF`에 배치 지연 형태가 없어서 즉시 자체 제출한다. 결과는 정확하지만 다른 볼륨과
한 submit으로 **융합되지 않는다**. 헤더에 명시돼 있고, 배치 오버로드 추가는 후속 과제다.

### 4.6 `insertFailureCount` / `growCount`는 아직 항상 0

해시 축이 들어오기 전까지 아무도 이 필드를 증가시키지 않는다. **드롭이 없다는 증거로 인용하면 안 된다.**

---

## 5. 세 전략

| | flat | tile | submap |
|---|---|---|---|
| 위임 대상 | `AdvancedTSDF` | `TiledAdvancedTSDF` | `SubmapAdvancedTSDF` |
| 공간 범위 | 고정 512³ 창 1개 | 지연 생성 448³-core 타일 | base 타일 + 절반 복셀 detail 서브맵 |
| 테이블 수 | 1 | 타일 수 | base + detail 타일 수 |
| `windowMinCorner` | 사용 | 무시 (타일 격자가 origin 결정) | 무시 |
| 큰 프레임 | 버퍼 성장 | 버퍼 성장 (라우팅 후, O(점 수)) | 버퍼 성장 (전체 업로드, O(타일×점)) |

세 전략 모두 **점을 버리지 않는다.** 초기 구현에서 `tile`/`submap`이 타일당 32768점에서 조용히
잘렸는데, `flat`은 안 잘려서 메모리 비교가 타일링에 유리한 쪽으로 편향됐다. 리뷰에서 잡혀
`TiledDirectionalTSDF::RecordIntegrateGPU`(신규) / `SubmapAdvancedTSDF::IntegrateGPU`로 교체했고,
`TsdfVolumeSwitching.LargeFrameIsNotTruncatedByMaxPointsPerFrame`가 이를 지킨다.

---

## 6. 현재 측정값과 그 한계

`test/test_tsdf_volume.cpp`의 `EveryStrategyIntegratesTheSameScan`이 매 실행마다 표를 출력한다.
289점 평면, 테이블당 65536 슬롯:

| 전략 | occupied | slots | load factor | tables |
|---|---|---|---|---|
| flat | 1,045 | 65,536 | 0.0160 | 1 |
| tile | 5,566 | 524,288 | 0.0106 | 8 |
| submap | 6,838 | 524,288 | 0.0130 | 8 |

**이 숫자로 원래 질문에 답하면 안 된다.** 픽스처가 289점이라 세 전략 모두 점유율이 0.01대로
바닥이고, `slotCapacity`가 `tableCount`에 정확히 비례하는 건 타일당 고정 용량 할당의 정의일 뿐이다.
`tile`과 `flat`의 occupied 차이(약 5배)도 버그가 아니라 타일 ghost 영역의 실제 중복 저장이다.

**진짜 답은 실제 스캔이 필요하고, 그러려면 하네스가 있어야 한다** — 현재 테스트 말고는 아무것도
`TSDF::TSDF`를 링크하지 않는다.

---

## 7. 다음 단계

1. **하네스 글루** — `tsdf_folder_eval --tsdf <name>` 같은 진입점. 이게 없으면 위 질문에 답할 수 없다. **최우선.**
2. `ComputePipeline` 셰이더 탐색 경로 + 2단 includer + 캐시 키 (설계 문서 §5)
3. Integrate·Extract 축 분리 + 커널을 호출 CPP 옆으로 이동 (§8 5~6단계)
4. `Memory/Hash` 전략 주입 + `insertFailureCount` 계측 → 버킷 해시 A/B (§8 7~8단계)
5. `SubmapAdvancedTSDF` 배치 오버로드 (4.5 해소)

> ⚠️ **3번 착수 전 확인할 것**: 커널을 갈아끼우려면 메모리 전략이 해시 버퍼와 push constant의
> 메모리 절반을 노출해야 하는데(설계 문서 §6.2), 지금은 전부 `AdvancedTSDF` 내부에 private이다.
> 현재 이음매에서는 보이지 않는 결합이고, `MemoryStrategy` 인터페이스를 바꾸게 만들 가장 유력한
> 후보다. 실행 중에 발견하지 말고 미리 스케치할 것.
