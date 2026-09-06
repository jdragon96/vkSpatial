# 알고리즘 작성 규칙

이 저장소에서 알고리즘을 새로 쓰거나 고칠 때 지킬 것. 명령형으로 적었고, 각 항목은 코드를 열어서 검사할 수 있는 형태다.

**읽는 순서**

1. `CLAUDE.md` — 빌드·테스트·타깃 레이어·저장소 전반 규약
2. `.claude/skills/algorithm-module` — **모듈 구조**(파사드 + 전략 인터페이스 + 이름 레지스트리, 폴더 배치, 커널 배치, 새 모듈 만드는 순서). 구조를 잡는 일이면 그쪽이 먼저다
3. **이 문서** — 그 구조 안에서 알고리즘 자체를 어떻게 설계할지

겹치는 내용은 여기 다시 쓰지 않는다. 한 규칙이 두 곳에 있으면 시간이 지나 서로 어긋난다.

---

## 1. 최적화 대상은 TSDF 자료구조다

기준 파일: [`src/TSDF/TSDF.h`](../../src/TSDF/TSDF.h), [`src/TSDF/Backends/TSDFBackend.h`](../../src/TSDF/Backends/TSDFBackend.h), [`src/TSDF/Memory/DataSplitter.h`](../../src/TSDF/Memory/DataSplitter.h).

### 1.1 통화(currency)를 새로 만들지 않는다

- 표면 입력은 `Engine::Core::OrientedPointCloud` — `points`와 `normals`의 **병렬 배열**, 법선은 단위 길이. 길이가 같다는 것은 계약이다.
- 표면 출력은 `TSDFVoxel`의 평탄 배열. 전역 네임스페이스의 평범한 구조체다.
- 설정은 `TSDFBackendConfig`, 통계는 `TSDFBackendStats`.

새 알고리즘이 자기만의 점 표현을 정의하면, 그것을 지나가는 모든 TU가 그 타입을 끌고 다닌다. **경계를 넘는 타입은 이미 있는 것을 쓴다.**

### 1.2 비용 모델의 단위는 "프레임"이 아니라 "창 × 프레임당 점"이다

`TSDF::Integrate`는 점을 통째로 백엔드에 넘기지 않는다. `VoxelKey`(= `floor(position / windowWorld)`)로 **창 단위 라우팅**을 한 뒤 창마다 백엔드를 한 번씩 돌린다(`IntegrateLevel`, `m_perWindowIndex`).

따라서:

- 프레임당 점 $N$을 가정하고 복잡도를 적지 않는다. 실제로 도는 것은 창 $W$개 × 그 창에 속한 점이다.
- 창을 가로지르는 전역 연산(전역 정렬, 전역 축약, 전역 이웃 탐색)은 이 구조와 싸운다. **창 안에서 닫히는 알고리즘을 우선한다.**
- 창 경계를 넘는 이웃이 필요하면 그 사실을 설계에 명시하고, 창 크기(`windowVoxels`)에 대한 의존을 문서화한다.

### 1.3 인덱스를 다루지 점을 복사하지 않는다

`DataSplitter::DividePointOutput`은 점이 아니라 `baseIndex` / `detailIndex`를 낸다. `TSDF::Gather`가 인덱스로 모은다.

**새 알고리즘이 점 부분집합을 고를 때는 인덱스 배열을 내놓는다.** 점을 복사해 돌려주면 프레임마다 수십만 개의 `Vector3f`가 복사된다.

### 1.4 2레벨(base / detail)을 깨뜨리지 않는다

`useSubmap`이면 `hashTSDF`(base)와 `hashTSDFDetail`(detail, 절반 복셀)이 함께 산다. 추출은 detail 우선으로 중복을 제거한다.

- 레벨을 가정하는 알고리즘은 **어느 레벨에서 도는지** 밝힌다.
- detail 해시 용량을 작게 잡지 않는다. 넘치면 구멍이 된다.

### 1.5 모든 스케일은 `voxelSize` / `truncation`에 맞춘다

고정 미터값을 쓰지 않는다. 맵 해상도가 바뀌면 같이 움직여야 한다.

실제 사례: ICP 대응 거리를 고정값으로 두었더니 대응점이 너무 적게 잡히는 **동시에** GPU LocalGrid 셀 수가 폭발했다. 스냅샷에 실린 `voxel`로 스케일링해서 고쳤다.

### 1.6 하드 상한은 전부 카운터로 노출한다

`hashCapacity`, `maxPointPerFrame`, `maxResidentWindow`, 좌표 패킹 범위, 프로빙 예산 — 성장 불가능한 천장에 닿으면 **반드시** 실패 카운터를 올린다.

기존 것: `insertFailureCount`, `growCount`(`TSDFBackendStats`), `blockInsertFailureCount`, `cellInsertFailureCount`(`DividePointOutput`), `WindowLimitRefusalCount()`(`TSDF`).

조용히 잘리면 증상 없이 결과만 망가진다. 이 저장소는 그 실패를 이미 겪었다.

---

## 2. GPU가 최우선이다

기준: [`src/Engine/Core`](../../src/Engine/Core), [`src/Engine/Compute`](../../src/Engine/Compute).

### 2.1 코어 알고리즘은 `Engine` 자료구조로 쓴다

| 필요 | 쓸 것 |
| --- | --- |
| 디바이스·큐 | `Engine::Core::Context` |
| 버퍼 | `Engine::Core::Buffer` |
| 커널 | `Engine::Core::ComputePipeline` |
| 여러 디스패치를 한 번에 제출 | `Engine::Compute::CommandBatch` |
| 업로드 스테이징 | `Engine::Compute::StagingBuffer` |
| 이미지·샘플러·디스크립터 | `Engine::Core::Image` / `Sampler` / `Descriptor` |

Vulkan 핸들을 직접 다루지 않는다. 필요한 것이 없으면 `Engine::Core`에 추가하지, 도메인 폴더에서 raw Vulkan을 쓰지 않는다.

### 2.2 CPU 구현은 오라클로만 존재한다

CPU 경로를 쓰는 이유는 **하나뿐**이다: GPU 결과를 못 박을 읽기 쉬운 레퍼런스.

- 프로덕션 경로가 되면 안 된다.
- 모든 전략을 CPU에도 구현하지 않는다. 기본 전략 하나만 오라클로 두고, 나머지는 GPU끼리 비교한다. 두 벌을 다 구현하면 서로를 못 박을 뿐이다.
- 예: `Pipeline::BackprojectDepth`가 `ValidationMask`의 오라클이다.

### 2.3 배치 규칙

- **한 배치 안에서 같은 `ComputePipeline` 객체를 두 번 `Dispatch`하지 않는다.** 파이프라인은 디스크립터 셋을 하나만 가지므로, 두 번째 바인딩이 첫 번째 디스패치가 아직 참조하는 셋을 덮어쓴다. 파이프라인 객체를 나누거나 배치를 나눈다.
- 의존하는 두 디스패치 사이에는 `Barrier()`를 넣는다.
- 디스패치 그리드는 `GetLocalSize()`를 읽어서 계산한다. 워크그룹 크기를 C++에 하드코딩하지 않는다 — 셰이더가 바뀌면 조용히 어긋난다.

### 2.4 프레임 경로에서 다운로드하지 않는다

리드백은 디버그·테스트·진단 도구 전용이다. 매 프레임 도는 경로에 `MakeVisibleToCPU`가 있으면 설계가 틀린 것이다.

조밀 그리드가 필요한 측정 도구는 `Record*` 메서드를 직접 불러 자기 리드백 버퍼로 받는다(`example2/normal_estimator_eval.cpp`가 그 형태).

### 2.5 std430 미러 규칙

GPU와 공유하는 구조체는 **4바이트 스칼라만** 담는다. `vec3`는 std430에서 16바이트 정렬을 받아 C++ 미러와 크기가 어긋나고, 버퍼가 몇 배 작게 잡혀 커널이 밖으로 쓴다. `vec4`는 안전하고 `vec3`·`mat`·`double`은 아니다.

C++ 쪽에 `static_assert`로 크기와 각 필드 `offsetof`를 못 박는다. 필드를 **끝에 추가**해서 기존 오프셋을 유지한다.

### 2.6 셰이더 안의 전략 축은 `-D` 매크로로 가른다

C++ 가상 함수가 아니라 디스패처 `.glsl` + `ComputePipeline::Define`. glslang은 `#include MACRO`를 확장하지 않으므로 include 이름은 리터럴이어야 한다.

- 참고 구현: `src/TSDF/Hash/HashStrategy.glsl`, `src/Pipeline/Reconstruction/Algorithm/NormalStrategy.glsl`
- 디스패처의 fragment include는 **src 기준 전체 경로**로 쓴다. include 루트는 최상위 컴파일 파일 기준으로 한 번 정해지지, 디스패처 자기 폴더 기준이 아니다.
- 기본 fragment는 `#else`로 둔다. 매크로 전달을 빠뜨려도 기존 동작으로 떨어지지, 컴파일이 깨지지 않는다.
- **컴파일 캐시 키는 경로 + 정렬된 매크로 정의여야 한다.** 경로만으로 키를 잡으면 두 번째 전략이 첫 번째의 SPIR-V를 조용히 받는다. (`ComputePipeline`은 이미 이렇게 되어 있다.)

### 2.7 커널은 부르는 `.cpp`와 같은 깊이에, `kernel_<알고리즘이름>.comp.glsl`로 둔다

**배치.** 컴퓨트 셰이더는 그것을 `Build()`하는 `.cpp`(또는 헤더 온리 모듈이면 그 헤더)와 **같은 폴더**에 둔다. 셰이더 전용 디렉터리로 모으지 않는다 — 알고리즘과 그 커널이 떨어지면 한쪽만 고치게 된다.

```
src/TSDF/Backends/AdvancedTSDF.cpp                        → src/TSDF/Backends/kernel_*.comp.glsl
src/Pipeline/Reconstruction/Algorithm/ValidationMask.h    → 같은 폴더의 kernel_*.comp.glsl
```

**이름.** `kernel_` 접두사 + **알고리즘 이름** + `.comp.glsl`. 알고리즘 이름은 그 커널이 무엇을 하는지이지, 어느 도메인에 속하는지가 아니다.

```
kernel_EstimateNormal.comp.glsl      // 좋음 — 커널이 하는 일
kernel_DepthToCameraSpace.comp.glsl  // 좋음
kernel_tsdf_stuff.comp.glsl          // 나쁨 — 도메인 이름 + 무의미어
```

한 클래스가 여러 단계 커널을 가지면 `kernel_<클래스이름>.<단계>.comp.glsl`을 쓴다. 이때 **단계 이름이 알고리즘 이름 자리**다.

```
kernel_AdvancedTSDF.integrate.comp.glsl
kernel_DenseRegionClassifier.classify.comp.glsl
```

`main()`이 없는 공용 include는 `kernel_` 접두사를 **붙이지 않는다**(`voxel_common.glsl`, `NormalStrategy.glsl`). 디렉터리 목록에서 "디스패치되는 커널"과 "옆에 있는 조각"을 눈으로 가르는 것이 접두사의 목적이다.

> **기존 코드와의 차이.** 이 이름 규칙은 앞으로 만드는 커널에 적용한다. `src/BVH`(`kernel_bvh_*`, `kernel_cmd_*`)와 구형 TSDF 커널(`kernel_directional_tsdf_*`, `kernel_voxel_tsdf_*`, `kernel_extract_mc*`)은 snake_case에 도메인 접두사가 붙은 옛 형태다. **일괄 개명하지 않는다** — 셰이더 경로는 런타임에 해석되므로 개명 누락이 컴파일이 아니라 실행 시점에 터진다. 그 파일들을 어차피 손볼 때 함께 옮긴다.

### 2.8 셰이더 경로 오류는 런타임에 터진다

컴퓨트 셰이더는 `ComputePipeline::Build("경로")`가 런타임에 컴파일한다 — 커널을 추가·이동해도 CMake는 안 건드린다. 대신 **이름을 바꿀 때 `Build("x")` 직접 호출뿐 아니라 래퍼도 전부 찾는다.**

### 2.9 수식을 담은 함수는 수식 먼저, 그 다음 기호 설명

커널 함수(그리고 수식을 담은 C++ 함수)의 헤더 주석은 **산문이 아니라 수식**으로 시작한다. 그 아래에 수식에 등장하는 기호를 하나씩 풀고, 단위는 대괄호로 적는다.

```glsl
/// sigma_z = s * z^2 / (f * B)
///
/// sigma_z : axial depth noise of one sample, one standard deviation [m]
/// s       : subpixel disparity matching error, RMS [px] -- 0.08 with texture, 0.25 without
/// z       : depth [m]
/// f       : focal length [px]
/// B       : stereo baseline [m] -- D435 0.05, D455 0.095
```

산문 서술은 코드를 한 번 더 말하는 데 그치기 쉽다. 수식은 구현을 보기 전에 **무엇을 계산하는가**를 먼저 답하고, 구현이 수식과 어긋났는지도 눈으로 대조된다.

**"왜"는 기호 항목으로 흡수한다.** CLAUDE.md는 주석이 왜를 적으라고 하고 이 규칙은 수식만 적으라고 하는데, 둘은 충돌하지 않는다 — 연산자·지수·상수도 기호로 취급해서 항목을 주면 된다.

```glsl
/// c = c_valid * c_range * c_ir * c_nb
///
/// *       : product, not sum -- the terms are vetoes, not votes. A pixel the VPU rejected is
///           worthless however bright its infrared return, and an average would carry it through
/// z^2     : exact for active stereo, not a fit -- z = f*B/d, so a disparity error e propagates
///           as z^2*e/(f*B)
/// 8       : the full 3x3 minus the centre. A neighbour outside the image is absent rather than
///           rejected, so the image border scores lower by construction
```

`*`가 왜 `+`가 아닌지, 지수가 피팅인지 유도식인지, 분모의 상수가 어디서 왔는지 — 이것들이 나중에 실제로 문제가 되는 지점이고, 기호 항목이 그 자리다.

- 조건부는 갈래를 나란히 적는다: `c_ir = 0 if I >= I_sat`, `= smoothstep(...) otherwise`.
- 집합·조건은 집합 기호로: `c_nb = |N| / 8, N = { p in 3x3(c) \ {c} : z_p > 0 and |z_p - z_c| <= tau }`.
- 함수 본문의 단계 주석도 수식을 가리킨다: `// 4. tau = k * sigma_z(z), then c_nb`.
- 수식이 없는 함수(순수 자료구조 조작, 디스패치)에는 억지로 만들지 않는다.

참조 구현: [`kernel_ValidationScore.comp.glsl`](../../src/Realsense/ValidationMask/kernel_ValidationScore.comp.glsl).

---

## 3. 전략은 벤치마크 가능성을 위한 것이다

전략 패턴을 쓰는 이유는 다형성 자체가 아니라, **어느 쪽이 나은지 측정할 수 있게** 하기 위해서다. 그래서 아래가 따라온다.

### 3.1 새 알고리즘은 새 클래스가 아니라 새 등록 이름이다

구조는 `algorithm-module` 스킬을 따른다. 현재 레지스트리:

| 축 | 이름 함수 / 레지스트리 |
| --- | --- |
| TSDF 백엔드 | `TSDFBackendNames()` |
| 해시 주소법 | `TSDFHashNames()` |
| 데이터 스플리터 | `DataSplitterNames()` |
| BVH 백엔드 | `BVHBackendNames()` |
| 법선 추정기 | `Pipeline::NormalEstimatorNames()` |
| 등가면 추출기 | `Mesh::ExtractorRegistry` / `GpuExtractorRegistry` |
| 트래커 | `src/Pipeline/Registration/TrackerRegistry.cpp` |

모르는 이름은 **던진다**. 기본으로 폴백하지 않는다 — 오타 난 이름이 조용히 기본 전략을 돌리면, 아무도 고르지 않은 구현의 수치를 보고하게 된다.

### 3.2 전략을 고르고 실행하는 진입점은 `Execute()`다

알고리즘 모듈이 "입력 한 벌을 받아 전략을 고르고 전체 단계를 돌리는" 함수를 가진다면, 그 이름은 `Execute()`로 한다. 호출자가 모듈마다 다른 동사를 외우지 않아도 되고, 벤치 하네스가 이름으로 모듈을 바꿔 끼울 수 있다.

```cpp
// 설정에서 전략을 고르고, 그 전략의 커널을 포함한 전체 시퀀스를 기록한다.
void Execute(Engine::Compute::CommandBatch &batch,
             Engine::Core::Buffer &input,
             const Options &options);
```

- 전략 선택은 **`Execute()` 안에서** 설정 문자열로 한다. 호출자가 전략 객체를 만들어 넘기지 않는다.
- 단계별 `Record*` 메서드는 public으로 남긴다. 테스트와 진단 도구가 중간 산출물을 봐야 하고, `Execute()`는 그 조각들을 정해진 순서로 엮은 것이어야 한다 — 별도 구현이 아니라.
- 참조 구현: `ValidationMask::Execute` (`Record*` 6개를 배리어로 엮고, 법선 추정 전략을 이름으로 고른다).

**적용 범위.** 이것은 *한 벌의 입력을 한 번 처리하는 알고리즘 모듈*의 규칙이다. `TSDF`·`BVH` 같은 **상태를 들고 사는 저장소 파사드**에는 적용하지 않는다 — 그쪽은 수명이 다른 여러 연산(`Build` / `Integrate` / `Extract` / `RadiusSearch`)을 가지므로 하나를 `Execute()`로 부를 수 없다. 파사드에 일회성 알고리즘이 들어 있다면 그건 파사드가 아니라 모듈로 빼야 한다는 신호다.

### 3.3 기본값은 기존 동작이다

새 전략을 등록해도 기본 이름은 그대로 둔다. **노브를 노출하는 변경이 동작을 바꾸면 안 된다.** 새 설정 필드의 기본값은 감싸는 구현체의 멤버 기본값을 열어서 그대로 복사한다.

### 3.4 벤치마크 하네스를 함께 낸다

전략을 등록만 하고 비교할 방법을 안 주면 그 전략은 측정되지 않는다.

- `example2/`에 **헤드리스** 도구로 낸다. 창을 띄우는 뷰어는 477프레임 통계를 못 낸다.
- 등록된 모든 이름을 **한 번의 실행에서** 순회해 표를 낸다. 따로 돌린 두 실행을 비교하면 스케줄링·캐시 상태가 섞인다.
- 첫 구성이 셰이더 컴파일 비용을 뒤집어쓰므로 **구성마다 타이밍 없는 워밍업 프레임**을 한 장 돌린다.
- **GPU 구간만 잰다.** 리드백과 CPU 후처리를 포함하면 하네스를 측정하는 것이지 알고리즘을 측정하는 게 아니다.
- 정확도 지표와 비용 지표를 함께 낸다. 그리고 그 지표가 **무엇을 구분하지 못하는지** 적는다.

### 3.5 한 실행에 두 전략을 섞지 않는다

스텐실·표본이 모자랄 때 다른 전략으로 폴백하고 싶어진다. **하지 않는다.** 한 프레임의 결과가 두 알고리즘의 혼합이면 "이 알고리즘이 나은가"를 답할 수 없다. 거부하고 카운터를 올린다.

### 3.6 테스트는 두 종류를 함께 둔다

**레지스트리 합의 테스트** — 등록된 모든 이름이 실제로 빌드되고 돌아가며, 모르는 이름은 던진다. 이때 **전략마다 다를 수밖에 없는 값**을 단언한다. "그럴듯한 결과가 나온다"만 확인하면, 매크로 이름을 틀려서 전부 기본 전략을 돌고 있어도 통과한다.

**전략별 동작 테스트** — 두 전략이 갈리는 지점을 고정한다. 같아야 하는 것과 달라야 하는 것을 둘 다 못 박는다.

### 3.7 판정에 관여하는 단언은 뮤테이션으로 검증한다

단언을 통과시키는 버그를 일부러 넣고 정말 빨개지는지 본다. 실제로 이 저장소에서:

- "두 카운터의 합"으로 점 보존을 확인하던 테스트가 원자적 슬롯 예약 버그를 통과시켰다.
- 정확도만 보던 법선 테스트는 전략이 전부 기본으로 떨어져도 통과했다. 테두리 폭(스텐실의 성질)을 단언에 넣어서 잡았다.

### 3.8 측정 함정

- **파이프라인을 통과하는 A/B는 `FrameHandshake` 없이는 무의미하다.** 블로킹 채널은 프레임 개수만 맞춘다. 맵은 latest-wins `Mailbox`로 전달되므로 프레임 N이 어느 버전의 맵에 정합하는지가 스레드 스케줄링에 달렸다.
- **`submap = true`에서 entry 수는 입력 점 수에 비단조다.** 밀집 블록 latch가 경로 의존이라 dense block 하나 차이가 entry를 8% 넘게 흔든다. 프런트엔드 A/B는 `--no-submap`.
- **`scan_out/`과 `scanData/`는 이미 정합되어 있다.** 트래커 기준선은 `identity`다. `icp`를 걸면 없는 문제를 푼다.

---

## 4. 용어

같은 것을 두 이름으로 부르지 않기 위한 목록. 새 이름을 만들기 전에 여기 있는지 본다.

### 4.1 공간·자료구조

| 용어 | 뜻 |
| --- | --- |
| **voxel** | 복셀. 한 변이 `voxelSize`(m) |
| **truncation** | 절단 거리 $T$. 표면 앞뒤로 값을 기록하는 반경 |
| **band** | 밴드. 표면 주변 $[-T, T]$ 구간. 이 구간만 갱신한다 |
| **window (창)** | TSDF가 점을 라우팅하는 단위 공간. `VoxelKey = floor(position / windowWorld)`, 한 변 `windowVoxels` 복셀. 창마다 백엔드 인스턴스 하나 |
| **base / detail** | 2레벨 submap의 두 층. detail은 복셀과 truncation이 둘 다 절반이고, 밀집 영역에만 생긴다 |
| **block** | `DataSplitter`가 밀집도를 판정하는 단위(`blockVoxels`). 창보다 작다 |
| **dense region** | 밀집 영역. detail 레벨을 받을 자격이 있다고 판정된 블록 |
| **entry / slot** | 해시 테이블에 실제로 채워진 한 칸. `TSDFVoxel` 하나 = (복셀, 방향) 슬롯 하나 |
| **direction** | directional TSDF에서 법선이 배정된 6개 정준 축 중 하나 |
| **tile** | 단일 창 한계를 넘기 위한 지연 공간 타일링 단위 |

### 4.2 해시

| 용어 | 뜻 |
| --- | --- |
| **hash capacity** | 슬롯 총수(`hashCapacity`). 성장 가능하지만 성장 자체가 비용 |
| **linear probe / bucketed** | 주소법 두 전략. `TSDFHashNames()` |
| **probe** | 한 번의 조회가 검사한 슬롯. `probeStats`로 계측 |
| **load factor** | 적재율. 채워진 슬롯 / 용량 |
| **insert failure** | 프로빙 예산 안에 자리를 못 찾은 삽입. 반드시 카운터로 노출 |

### 4.3 융합

| 용어 | 뜻 |
| --- | --- |
| **point-to-plane** | SDF 값을 $\hat n \cdot (\text{복셀중심} - P)$로 계산. 법선에 의존 |
| **projective** | SDF 값을 시선 방향 깊이 차로 계산. 법선을 쓰지 않음 |
| **march** | 밴드를 따라 복셀을 훑는 것. point-to-plane은 $\hat n$을, 카빙은 $\hat r$을 따라 행진 |
| **weight** | 누적 가중치. 값의 신뢰도이자 융합 횟수의 대리 |
| **confidence** | 한 측정이 실을 가중치를 정하는 프로파일. symmetric / behind-dropoff |
| **carving** | 자유공간 카빙. 카메라–표면 구간을 비우는 것. 현재 미구현 |

### 4.4 프런트엔드

| 용어 | 뜻 |
| --- | --- |
| **gate (게이트)** | 측정을 버리는 판정. `[H1]`–`[H6]`(depth), `[F1]`–`[F3]`(융합)로 번호를 매긴다 |
| **jump tolerance** | 같은 표면으로 볼 깊이 차 상한. $\max(\text{minimumDepthJump}, \text{relativeDepthJump} \cdot z)$ |
| **same surface** | 이웃이 유효하고 jump tolerance 안에 있는 것. **정의가 하나여야 한다** |
| **emitted** | 점과 법선을 둘 다 만들어 낸 픽셀. `valid`(측정이 있음)와 다르다 |
| **stencil** | 추정기가 읽는 이웃 집합 |
| **domain** | 스텐실이 이미지 안에 들어가는 픽셀 범위. 정의역 밖은 거부가 아니다 |
| **oracle** | GPU 결과를 못 박는 CPU 레퍼런스 구현 |

### 4.5 정합

| 용어 | 뜻 |
| --- | --- |
| **tracker** | 포즈를 푸는 전략. `identity` / `icp` / `icp-cpu` / `global` |
| **prior** | solve 시작 자세 추정. 직전 포즈 또는 상수속도 외삽 |
| **fitness** | 대응점 수를 프레임 점 수로 나눈 비율 |
| **inlier** | 게이트를 통과한 대응점 |
| **ETrackFailure** | 실패 원인 분류. `NoModel` / `NoLocalTarget` / `TooFewInliers` / `LowOverlap` / `ImplausibleMotion`. **`valid == false`로 뭉뚱그리지 않는다** |

### 4.6 코드 구조

| 용어 | 뜻 |
| --- | --- |
| **facade (파사드)** | 전역 `class TSDF` / `class BVH`. 수명·라우팅·집계만, 알고리즘 없음 |
| **backend / strategy** | 교체 가능한 구현. 순수 가상 인터페이스 뒤 |
| **registry** | 이름 → 구현 매핑. `<Domain>BackendNames()` + `Make<Domain>Backend(name)` |
| **adapter** | `.cpp` 익명 네임스페이스에서 구현체를 인터페이스에 맞추는 껍데기 |
| **boundary type** | 네임스페이스 없는 경계 전용 타입(`TSDFVoxel`, `ModelSnapshot`) |
| **dispatcher .glsl** | 셰이더 안 전략 축을 `-D`로 가르는 파일 |
| **kernel** | `kernel_*.comp.glsl`. `main()`이 있고 런타임에 컴파일된다 |
| **mirror** | GPU 구조체와 짝을 이루는 C++ 구조체. `static_assert`로 못 박는다 |

---

## 5. 체크리스트

새 알고리즘을 내기 전에:

- [ ] 입출력이 기존 경계 타입인가 (`OrientedPointCloud`, `TSDFVoxel`)
- [ ] 창 안에서 닫히는가. 아니면 그 의존을 문서화했는가
- [ ] 점을 복사하지 않고 인덱스로 다루는가
- [ ] 모든 스케일이 `voxelSize` / `truncation`에 묶여 있는가
- [ ] 성장 불가능한 천장마다 실패 카운터가 있는가
- [ ] 코어가 `Engine::Core` / `Engine::Compute` 위에 있는가
- [ ] 전체를 돌리는 진입점 이름이 `Execute()`인가 (저장소 파사드는 제외)
- [ ] 커널이 부르는 `.cpp`와 같은 폴더에 있는가
- [ ] 커널 이름이 `kernel_<알고리즘이름>.comp.glsl`인가. 공용 include에는 접두사가 없는가
- [ ] 수식을 담은 함수가 수식 + 기호 설명(단위 포함)으로 문서화되어 있는가
- [ ] 프레임 경로에 리드백이 없는가
- [ ] GPU 공유 구조체가 4바이트 스칼라만 담고 `static_assert`가 붙었는가
- [ ] 이름으로 등록했고, 모르는 이름은 던지는가
- [ ] 기본값이 기존 동작 그대로인가
- [ ] 헤드리스 벤치 하네스가 있고, 워밍업과 GPU 구간 분리가 되어 있는가
- [ ] 레지스트리 합의 테스트 + 전략별 동작 테스트가 있는가
- [ ] 합의 테스트가 **전략마다 다를 수밖에 없는 값**을 단언하는가
- [ ] 판정 단언을 뮤테이션으로 검증했는가
- [ ] `assert` 없음, `src/`에 로깅 없음, 축약어 없음
