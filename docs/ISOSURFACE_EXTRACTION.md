# Isosurface Extraction 프레임워크 — 7개 전략 해설

> `Engine::Spatial::Extraction`: 등가면(isosurface) 추출을 **교체 가능한 전략(strategy)**
> 으로 다루는 프레임워크. 기존에 하나뿐이던(GPU `voxel_tsdf_mc.comp` + CPU
> `AdaptiveVoxelGrid.cpp`의 15-case Marching Cubes) 추출 경로를, 고전 계보의 7개 알고리즘
> — **mc · mc33 · mtet · emc · dc · dmc · cms** — 을 이름으로 선택 가능한 독립 전략으로
> 재구성했다. `Engine::Pipeline::Registration::Tracker` / `TrackerRegistry` 패턴을 그대로
> 거울상으로 따른다(추상 전략 + `Name()` + 레지스트리 `Register`/`Create`/`Has`/`Default`).

- **관련 코드:** `src/Engine/Spatial/Extraction/`(프레임워크·7개 전략), 각 전략의 헤더 주석에
  전체 알고리즘 설명이 있다 — 이 문서는 7개를 **나란히 비교**하는 것이 목적이라 세부
  유도·증명은 각 `.cpp` 파일 헤더와 `.superpowers/sdd/2026-08-10-isosurface-extraction-
  strategies/task-*-report.md`(구현 기록)를 참조.
- **관련 문서:** [`MARCHING_CUBES_SURVEY.md`](MARCHING_CUBES_SURVEY.md) — 이 7개 전략이
  속한 **문헌 계보**(1987 원조 → 위상 정확성 → 특징 보존/dual → 적응/다중해상도 → 성능 →
  학습기반)와 이 저장소의 다중해상도(submap) 적용 사례. 이 문서는 그 계보 중 "고전
  (classical) 패밀리"를 실제로 **구현**한 결과물의 설명서다.
- **설계 문서:** `docs/superpowers/specs/2026-08-10-isosurface-extraction-strategies-design.md`

---

## 0. 왜 프레임워크인가

추출 알고리즘마다 요구 사항이 다르다 — 어떤 것은 위상적으로 안전해야 하고(비다양체
불가), 어떤 것은 날카로운 특징(모서리/코너)을 보존해야 하고, 어떤 것은 crack 없이
적응/다중해상도에 붙을 수 있어야 한다. 이 저장소는 이제 이 세 가지 요구를 **각각 가장 잘
푸는 전략을 이름으로 골라 쓸 수 있다**:

```cpp
SurfaceMesh mesh = Engine::Spatial::Extraction::ExtractorRegistry::Default()
                        .Create("dc")   // 또는 "mc", "mc33", "mtet", "emc", "dmc", "cms"
                        ->Extract(field, ExtractParams{});
```

7개 모두 **CPU 전용**, 같은 `VoxelField` 입력과 같은 `SurfaceMesh` 출력을 공유하고,
`test/isosurface_test_util.h`의 같은 해석적(analytic) 구(sphere)/박스(box) 필드로
검증된다 — 그래서 "어느 전략이 내 데이터에 맞는가"를 코드 한 줄 교체로 실험할 수 있다.

---

## 1. 프레임워크 구조

```
Engine::Spatial::Extraction
 ├─ VoxelField           입력: signed 스칼라 + (선택) 저장된 gradient, cellSize, occupied 좌표
 ├─ SurfaceMesh          출력: vertices / triangles / normals
 ├─ IsoSurfaceExtractor  추상 전략: Name(), Extract(field, params)
 ├─ ExtractorRegistry    이름 → 팩토리 (Register / Create / Has / Default)
 ├─ MarchingCubesCore    공유 코어: RawTriangle, VertexInterpolate, CandidateBases,
 │                       GenerateRawTriangles, WeldAndComputeNormals, kEdgeCornerPairs,
 │                       VoxelValueMap, IVec3Hash  (mc::CORNER/edgeTable/triTable는
 │                       MarchingCubesTables.h)
 ├─ QuadraticErrorFunction  emc/dc/dmc/cms가 공유하는 QEF 솔버 (특징 정점 배치)
 └─ 전략 7개  MarchingCubesExtractor.cpp, MarchingCubes33Extractor.cpp,
              MarchingTetrahedraExtractor.cpp, ExtendedMarchingCubesExtractor.cpp,
              DualContouringExtractor.cpp, DualMarchingCubesExtractor.cpp,
              CubicalMarchingSquaresExtractor.cpp
```

### `VoxelField` — 입력

정수 격자 좌표(`std::array<int,3>`)로 색인되는 sparse 필드. 코너의 월드 위치는
`coord * CellSize()`(모든 전략이 공유하는 단일 월드 프레임).

```cpp
class VoxelField {
public:
    bool  Sample(const std::array<int,3>& coord, float& outValue) const;
    bool  Gradient(const std::array<int,3>& coord, Eigen::Vector3f& outNormal) const;
    bool  HasGradients() const;
    float CellSize() const;
    const std::vector<std::array<int,3>>& OccupiedCoords() const;
};
// 어댑터: FromImplicit(...)  -- 테스트용 해석적 필드 (exact gradient)
//         FromAdvancedEntries(...)  -- 실제 TSDF(AdvancedEntry)에서 변환
```

`Gradient()`가 `false`를 반환하면(저장된 gradient 없음) `emc`/`dc`/`dmc`/`cms` 모두 같은
central-difference 폴백(`EstimateCornerNormal`, 각 파일에 중복 구현 — 아래 참고)으로
낮춰 잡는다.

### `SurfaceMesh` — 출력

```cpp
struct SurfaceMesh {
    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector3i> triangles;  // 정점 인덱스
    std::vector<Eigen::Vector3f> normals;    // 정점별, area-weighted
};
```

7개 전략 모두 **삼각형**으로 귀결한다 — dual 계열(`dc`/`dmc`)이 자연스럽게 만드는 사각형과
`cms`의 가변 길이 폴리곤 loop은 모두 팬(fan) 삼각분할로 마지막에 삼각형화된다(자세한 내용은
전략별 절 참고).

### `IsoSurfaceExtractor` / `ExtractorRegistry`

```cpp
struct ExtractParams {
    float isoLevel = 0.0f;
    float weldFraction = 0.25f;                // weld 격자 = weldFraction * cellSize
    float featureAngleCosineThreshold = 0.9f;  // emc/dc/dmc/cms: 특징 셀 판정 임계 cos값
};
class IsoSurfaceExtractor {
public:
    virtual const char* Name() const = 0;
    virtual SurfaceMesh Extract(const VoxelField& field, const ExtractParams& params) const = 0;
};
class ExtractorRegistry {
public:
    void Register(const std::string& name, Factory factory);
    std::unique_ptr<IsoSurfaceExtractor> Create(const std::string& name) const;  // 없으면 nullptr
    bool Has(const std::string& name) const;
    static ExtractorRegistry Default();  // mc, mc33, mtet, emc, dc, dmc, cms 등록
};
```

### `MarchingCubesCore` — 공유 코어

`AdaptiveVoxelGrid.cpp`의 anonymous namespace에 있던 원시 함수들을 옮긴 것(rename-only
move, 로직 불변 — `AdaptiveVoxelGrid`의 GPU-bit-exact 테스트가 그 회귀 가드)이다. 7개
전략 전부가 최소한 `VertexInterpolate`(엣지 교점 선형 보간, GLSL `vertInterp`와 동일)와
`WeldAndComputeNormals`(용접 + area-weighted 법선)를 재사용하고, `mc`/`mc33`/`mtet`/
`emc`/`dc`/`dmc`/`cms`는 추가로 `CandidateBases`(8-이웃 스윕으로 후보 큐브 열거)와
`kEdgeCornerPairs`(12개 큐브 엣지 → 코너 쌍, `mc::CORNER` 순서)를 공유한다.

### `QuadraticErrorFunction` — 공유 QEF 솔버

`emc`/`dc`/`dmc`/`cms` 4개 전략이 그대로 재사용하는, 접평면 제약의 최소제곱 교점 솔버:

```cpp
class QuadraticErrorFunction {
public:
    void Add(const Eigen::Vector3f& point, const Eigen::Vector3f& normal);
    Eigen::Vector3f Solve(const Eigen::Vector3f& centroidBias, float regularization = 1e-3f) const;
};
```

`Solve`는 `(AᵀA + regularization·I) x = Aᵀb + regularization·centroidBias`를 `ldlt()`로
푼다 — 정규화 항이 rank-deficient 누적(평평한 셀은 1/3 차원만 구속, 엣지 셀은 2/3만
구속)을 강제로 양정치화해서 항상 풀리게 하고, 구속되지 않은 방향은 `centroidBias`(보통
셀 중심)로 당긴다.

---

## 2. 7개 전략

각 절은 **원리 · primal/dual · 삼각형 단위 · 위상/특징 보장 · 언제 쓰는가 · 알려진 한계**
순서로 통일했다.

### 2.1 `"mc"` — Marching Cubes (원조)

- **파일:** `MarchingCubesExtractor.cpp` (46줄 — 프레임워크 조합만, 로직은 전부 공유 코어)
- **원리:** Lorensen & Cline 1987. 큐브 8코너 부호 → 256(→15 unique)-case 룩업 테이블로
  삼각형 토폴로지 결정, 엣지 교점은 선형 보간.
- **Primal / Dual:** **Primal** — 정점은 항상 큐브 **엣지** 위(격자 자체의 교점).
- **삼각형 단위:** 케이스 테이블이 직접 삼각형을 낸다(가변 개수, 0~4개/큐브).
- **위상/특징 보장:** 없음 — trilinear 보간의 면/내부 모호성을 해소하지 않으므로 이론상
  구멍이 날 수 있다(매끄러운 필드에서는 실전에서 거의 안 걸림). 날카로운 특징은 항상
  선형보간으로 뭉갠다.
- **언제 쓰는가:** 기본값. 가장 빠르고 가장 단순하며 이 저장소의 기존 GPU 경로
  (`voxel_tsdf_mc.comp`)·`AdaptiveVoxelGrid` 다중해상도 추출과 **같은 로직**(공유 코어로
  통합됨) — 매끄러운 TSDF/스캔 데이터, 위상·특징 보장이 굳이 필요 없을 때.
- **한계:** 면/내부 모호성 미해결(→ 2.2), 특징 손실(→ 2.4/2.5/2.6/2.7).
- **테스트:** `test_isosurface_mc.cpp` — 구 정확도(RMSE<cell)/edge-manifold/watertight.

### 2.2 `"mc33"` — Marching Cubes 33

- **파일:** `MarchingCubes33Extractor.cpp` (655줄), 테이블
  `MarchingCubes33Tables.h`(130.7K, Lewiner 참조 구현에서 전사)
- **원리:** Chernyaev(1995)/Lewiner et al.(2003). trilinear 보간이 큐브 안에서 취할 수
  있는 모든 위상 케이스를 33가지로 완전 열거하고, **면 모호성**은 2-D 점근 판정자
  (asymptotic decider, `FaceTest` — bilinear saddle 부호), **내부 모호성**은 trilinear
  saddle의 t-critical 검증(`InteriorTest`)으로 해소한다. 가장 까다로운 케이스 13.5
  ("5-tunnel")까지 전체 구현(실제 매끄러운 격자 샘플링에서는 발생 불가능한 3-D 체커보드
  코너 패턴이 필요해 테스트로 exercise되진 않음, 코드는 완비).
- **Primal / Dual:** **Primal** — `mc`와 동일하게 엣지 교점.
- **삼각형 단위:** 서브케이스별 삼각형 테이블(`tiling*`) — 여전히 직접 삼각형.
- **위상/특징 보장:** **면+내부 모호성 해소** → 이웃 큐브가 같은 면에서 같은 결정을 공유,
  `mc`가 구멍을 낼 수 있는 바로 그 설정에서 watertight를 보장. 특징 보존은 없음(여전히
  선형 보간 정점).
- **언제 쓰는가:** 위상적 정확성(구멍 없음)이 필수인 하류 작업(솔리드 모델링, 3D 프린팅,
  BVH 구축용 워터타이트 메시)이면서 날카로운 특징 보존은 필요 없을 때. `mc` 대비 비용은
  더 큰 케이스 테이블 + 면/내부 테스트뿐 — 여전히 큐브당 O(1) 조회.
- **한계:** 여전히 특징을 뭉갠다(→ emc/dc/dmc/cms). 테이블이 방대해 오타 위험이 커
  provenance 주석 필수(파일 헤더 참고).
- **테스트:** `test_isosurface_mc33.cpp` — 면-모호 고정 큐브(대각 코너 음수)에서
  edge-manifold 확인 + 구 정확도.

### 2.3 `"mtet"` — Marching Tetrahedra

- **파일:** `MarchingTetrahedraExtractor.cpp` (221줄)
- **원리:** Doi & Koide 1991. 각 큐브를 주대각선(코너0–코너6)을 공유하는 6개 사면체로
  분할, 사면체는 코너가 4개뿐이라 내부 보간이 **어핀(affine)** — trilinear이 아니므로
  모호성이 **구조적으로 존재하지 않는다**(표로 해소하는 게 아니라 애초에 발생 불가).
- **Primal / Dual:** **Primal** — 정점은 사면체 엣지 위(큐브 엣지 + 면/공간 대각선).
- **삼각형 단위:** 사면체당 0/1/2개 직접 삼각형(코너 부호가 1:3이면 삼각형 1개, 2:2면
  평면 사각형 → 대각선 분할로 삼각형 2개).
- **위상/특징 보장:** 이 패밀리에서 **가장 강한 위상 보장** — 구 픽스처에서 Euler
  characteristic == 2(위상적 구)까지 직접 검증(`EulerCharacteristic` 테스트). 특징
  보존은 없음(여전히 선형 보간, 오히려 사면체 대각선 절단이 추가로 생겨 더 각져 보일 수
  있음).
- **언제 쓰는가:** "표로 막는" mc33이 아니라 "애초에 모호성이 없는" 위상 보장이 필요할
  때 — 대신 삼각형 수가 `mc` 대비 2~3배(큐브당 최대 6사면체 × 2삼각형=12 vs `mc`
  최대 5), 그리고 weld 허용 오차를 `mc`보다 32배 촘촘히 잡아야 한다(사면체 대각선이
  진짜로 서로 가까운 서로 다른 정점을 낳기 때문 — `kWeldDistanceDivisor`, 파일 헤더에
  이분 탐색으로 검증된 값 기록).
- **한계:** 삼각형 수 증가; 고정된 주대각선(코너0-6) 분할이라 격자 방향에 약간 편향된
  삼각분할(mc의 1985년 특허 회피 목적도 있었던 역사적 배경 — 특허는 현재 만료).
- **테스트:** `test_isosurface_mtet.cpp` — 구 watertight+edge-manifold+Euler==2+정확도.

### 2.4 `"emc"` — Extended Marching Cubes

- **파일:** `ExtendedMarchingCubesExtractor.cpp` (235줄)
- **원리:** Kobbelt, Botsch, Schwanecke, Seidel 2001. `mc`의 케이스/연결성을 그대로 두고,
  **특징 셀**(활성 엣지 교점들의 법선 쌍 중 최소 내적이 `featureAngleCosineThreshold`
  미만 — 즉 어떤 두 교점의 법선이 그 각도보다 더 벌어짐)에서만 `QuadraticErrorFunction`
  으로 접평면 교점을 풀어 특징 정점을 **추가**하고, 그 큐브의 (mc 케이스가 낸) 삼각형
  집합의 **경계**(그 큐브 안에서 등장 횟수 1인 엣지 — 이웃과 공유되거나 필드 경계)만
  특징 정점으로 팬 삼각분할한다. 등장 횟수 2인 엣지(같은 큐브 내부 대각선)는 버린다 —
  그래서 이웃과 공유되는 엣지는 항상 원래 mc와 똑같은 두 끝점/등장 횟수를 유지 →
  cross-cube edge-manifold가 구조적으로 보존된다.
- **Primal / Dual:** **Primal** — `mc`의 엣지 정점 구조 위에 특징 정점을 "얹는" 방식(dc
  처럼 정점을 대체하지 않음).
- **삼각형 단위:** 비특징 셀은 `mc`와 동일한 케이스-테이블 삼각형; 특징 셀은 경계 팬
  삼각형.
- **위상/특징 보장:** 위상 보장은 `mc`와 동급(면/내부 모호성 자체는 미해결 — mc33의
  개선을 흡수하지 않음). **특징 보존은 이 패밀리 최초** — 박스 크리즈 테스트에서 `mc`
  보다 유의하게 크리즈에 가까운 정점 배치 확인.
- **언제 쓰는가:** 데이터에 진짜 날카로운 특징(CAD풍 SDF, 박스/실린더 1차 도형, 실측
  스캔의 물리적 모서리)이 있고, `mc`에서 최소한의 변경(여전히 케이스 테이블 기반, 이해/
  디버깅 쉬움)으로 특징만 보존하고 싶을 때.
- **한계:** `mc`의 면/내부 모호성은 그대로 물려받음(→ 필요하면 `cms`처럼 둘 다 원할 것);
  QEF 수치 안정성은 Tikhonov 정규화에 의존.
- **테스트:** `test_isosurface_emc.cpp` — 박스 크리즈 갭이 `mc`보다 작음(edge-manifold도
  확인).

### 2.5 `"dc"` — Dual Contouring (of Hermite Data)

- **파일:** `DualContouringExtractor.cpp` (299줄)
- **원리:** Ju, Losasso, Schaefer, Warren 2002. **완전한 dual** 재구성 — 셀(큐브)마다
  **정확히 1개**의 정점을, 그 셀의 부호가 바뀌는 모든 엣지의 Hermite 데이터(교점+법선)로
  QEF를 풀어 배치(`emc`와 달리 "표 삼각형 + 특징 정점 추가"가 아니라 애초에 mc 케이스
  테이블 자체를 쓰지 않는다). 부호가 바뀌는 **격자 엣지**마다 그 엣지를 공유하는 4개
  셀의 dual 정점을 사각형으로 연결(2삼각형 분할) — 와인딩은 그 엣지의 부호 방향에서
  동적으로 유도(`EmitOutwardTriangle`, 표로 미리 박아두지 않음).
- **Primal / Dual:** **Dual** — 이 패밀리에서 최초의 진짜 dual 방법. 정점이 엣지가 아닌
  **셀 내부**에 있다.
- **삼각형 단위:** 부호 변화 격자 엣지 1개당 사각형 1개 → 삼각형 2개.
- **위상/특징 보장:** 특징 보존은 **이 패밀리에서 가장 직접적**(정점 자체가 QEF 해이므로
  `emc`처럼 별도 팬 경계 계산이 필요 없음). 반면 위상 보장은 **이 패밀리에서 가장 약함**
  — "체커보드" 모호 면(같은 면의 두 셀이 공유하는 경계 엣지 4개가 동시에 부호 변화)에서
  같은 메시 엣지가 4번 등장하는 **비다양체**가 실제로 재현됨
  (`test_isosurface_dc.cpp`의 `DISABLED_DualContouringAmbiguousFaceIsManifold_
  ManifoldDcFuture` — 의도적으로 현재 FAIL하는 회귀 타깃, 미래의 Manifold DC 구현이
  통과시켜야 함). 매끄러운 구 픽스처에서는 edge-manifold 확인됨 — 설계 문서 Risks
  섹션이 이 한계를 사전에 명시·수용.
- **언제 쓰는가:** 특징 보존이 최우선이고, 병적인 체커보드 설정에서의 드문 비다양체
  엣지를 감수할 수 있을 때(실측/매끄러운 데이터에서는 거의 발생하지 않음). Octree 적응
  격자와 원래 궁합이 좋은 방법(이 저장소 구현은 균일 `VoxelField` 위에서 동작).
- **한계:** 문서화된 비다양체 위험(위 DISABLED 테스트); 활성 셀마다 진짜 QEF solve가
  필요(표 조회보다 무거움).
- **테스트:** `test_isosurface_dc.cpp` — 구 정확도+edge-manifold+outward-normal 부호,
  박스 크리즈 갭, (DISABLED) 체커보드 비다양체 회귀 타깃.

### 2.6 `"dmc"` — Dual Marching Cubes

- **파일:** `DualMarchingCubesExtractor.cpp` (269줄)
- **원리:** Schaefer & Warren 2004/05 ("Primal Contouring of Dual Grids"). `dc`의 셀당
  dual 정점 배치(패스 1 — `dc`의 QEF 구성을 그대로 재사용)를 먼저 만든 다음, 그 dual
  정점들을 코너로 갖는 **진짜 dual 그리드**를 만들고 그 위에서 `mc`가 쓰는 바로 그
  `mc::edgeTable`/`triTable` 케이스 로직을 (패스 2로) 돌린다. Dual 큐브 하나의 8코너는
  8개의 서로 다른 원본(primal) 셀이므로 각자 다른 스칼라가 필요한데, 원 논문 문구를
  글자 그대로 읽으면 퇴화(코너 8개가 전부 같은 값)하므로, 이 구현은 "그 셀 자신의 8코너
  값의 평균"(=셀 중심에서의 trilinear 평가값)을 채택했다(박막(thin-slab) 픽스처로 손
  검증 — 파일 헤더에 다른 후보였던 "셀 자신의 base 코너 값"이 비대칭이라 기각된 이유
  기록).
- **Primal / Dual:** **Dual** — 정점 배치는 `dc`와 동일(셀당 1개, QEF); 다만 그 정점들을
  **primal 방식(케이스 테이블)으로 연결**한다는 점이 `dc`와의 핵심 차이(그래서 이름이
  "Dual Marching Cubes" = dual **그리드**의 primal contouring).
- **삼각형 단위:** dual 큐브당 `mc`와 동일한 케이스 테이블 삼각형(그대로 재사용, dual
  코너에 적용).
- **위상/특징 보장:** `mc`가 crack-free인 것과 같은 이유(공유 코너 = 단일 전역 조회)로
  crack-free — `dc`의 체커보드 비다양체 실패 모드가 여기서는 **재현되지 않는다**(dual
  정점 연결을 `dc`처럼 엣지별 사각형으로 직접 잇는 게 아니라, mc의 검증된 케이스 테이블에
  맡기기 때문). 특징 보존은 `dc`의 QEF 정점을 그대로 상속. **박막/근접 이중 시트를
  분리 재현**하는 것이 `dc`/`mc` 대비 이 전략의 핵심 차별점(단일 dual 정점으로는 셀 하나를
  지나는 두 시트를 동시에 표현 못 함 — dual 큐브 케이스 조회가 이를 자연히 풀어줌).
- **언제 쓰는가:** 데이터에 **셀 하나보다 얇은 간격의 두 표면**(박막, 근접한 두 겹)이
  있고, `dc`의 특징 보존은 원하지만 `dc`의 비다양체 위험 없이 `mc`급 매니폴드 보장도
  같이 원할 때. 대가는 큐브 패스 2회(패스1: primal 셀 QEF, 패스2: dual 큐브 케이스).
- **한계:** dual 코너 스칼라를 "무엇으로 할지"가 논문에 완전히 못박혀 있지 않아 구현
  판단이 개입됨(위 참고, 문서화됨); dual 그리드 경계가 원본 narrow band 경계보다 한 셀
  더 안쪽에서 시작(패스2 후보가 패스1의 "해결된" 좌표에서 나오므로) — 희소 필드 가장자리
  에서 다른 전략보다 약간 더 넓은 밴드가 필요.
- **테스트:** `test_isosurface_dmc.cpp` — 구 edge-manifold, 박막(두께<1셀) 픽스처에서
  z>0/z<0 두 시트 모두 재현.

### 2.7 `"cms"` — Cubical Marching Squares

- **파일:** `CubicalMarchingSquaresExtractor.cpp` (신규, 이 태스크)
- **원리:** Ho, Wu, Chen, Chuang, Ouhyoung 2005. 3-D 큐브 등가면 문제를 큐브의 6개 면 각각에
  대한 **독립적인 2-D marching squares** 문제로 환원한다. 각 면은 코너 4개 → 활성 엣지
  0/2/4개; 4개(체커보드 모호)인 경우만 **2-D 점근 판정자**(mc33의 `FaceTest`와 같은
  bilinear-saddle 공식, `v0·v2 - v1·v3` 부호)로 두 대각 중 어느 쪽이 이어지는지 해소한다.
  이 결정은 **그 면 자신의 4개 코너 값에만** 의존하므로(어느 큐브가 묻는지와 무관), 표준
  `mc::CORNER` 번호 규약상 한 큐브의 "+axis" 면과 그 이웃의 대응 "-axis" 면은 **같은
  4개 격자 정점을 같은 순환 순서로** 방문한다(손으로 3축 모두 검증, 파일 헤더 기록) —
  그래서 두 이웃 큐브는 공유 면에서 **항상 같은 분할**을 계산한다(crack-free가 설계로
  보장됨, 표를 짜서 맞춘 게 아니라 증명됨). 큐브의 엣지 12개는 각각 정확히 2개 면에
  속하고, 두 면은 최대 1개 엣지만 공유하므로, 활성 엣지마다 정확히 2개의 면-분할
  연결이 생겨 — 큐브 단위 연결 그래프가 항상 **단순 폐곡선(loop)들의 분리 합집합**이
  됨을 구성적으로 보장한다(`TraceLoops`). 각 loop는: 그 큐브가 특징 셀(`emc`와 동일한
  "활성 엣지 법선 쌍 최소 내적 < 임계값" 판정, 셀 전체 1회)이면 공유 QEF 특징 정점으로,
  아니면 그 loop 자신의 중심으로 팬 삼각분할한다. 와인딩은 loop 전체에 대해 1회만
  결정(`dc`/`mtet`의 EmitOutwardTriangle과 같은 "표 대신 동적 유도" 발상을 loop
  단위로 적용) — 팬 하나가 부분적으로 뒤집히는 일이 구조적으로 없다.
- **Primal / Dual:** **혼합**으로 보는 게 정확하다 — 정점 배치는 **primal**(교점은 여전히
  큐브 **엣지** 위, `mc`/`emc`처럼; 특징 정점은 `emc`처럼 loop에 "추가"되는 것이지 `dc`
  처럼 정점을 대체하지 않음)이지만, **연결성은 dual 계열의 철학**(표를 미리 짜두는 게
  아니라 저차원 하위 문제 + 그래프 결합으로 그때그때 유도)을 따른다. 서베이 문서(§3)가
  CMS를 dual 계열에 두는 것과 일치.
- **삼각형 단위:** 가변 길이(보통 3~8) 폴리곤 loop을 팬 삼각분할 — dual 계열의 사각형처럼
  "표가 아닌 그때그때 조립된 폴리곤"이 자연 단위라는 점은 같지만, 그 폴리곤이 항상
  사각형(dc/dmc)이 아니라 큐브마다 달라지는 n각형(loop) 자체라는 점이 다르다.
- **위상/특징 보장:** **crack-free가 구성적으로 증명됨**(위 원리 항목) — 이 구현에서
  `dc`류 체커보드-공유-면 비다양체의 대응물이 나타나지 않는다: loop의 둘레(rim) 엣지는
  항상 그 면의 유일한 분할 결정에서 나오므로 두 이웃 큐브가 만드는 등장 횟수는 항상
  정확히 2(공유 면)이거나 1(필드 경계)이지 `dc`처럼 4가 될 수 없다(자세한 논증은
  `task-7-report.md` "Concerns" 참고 — **구성적 증명이지 적대적 퍼징 검증은 아님**).
  구 픽스처에서 edge-manifold 실측 확인. 특징 보존은 `emc`와 동급(같은 판정식, 같은
  QEF), 박스 크리즈 갭이 `mc`보다 작음을 실측.
- **언제 쓰는가:** 이 패밀리의 "다 되는" 선택지 — `mc33`급 위상 안정성(공유 면 합의)을
  큰 서브케이스 테이블 없이, `dc`급 특징 보존을 비다양체 위험 없이, 게다가 새들(saddle)
  큐브의 **분리된 2-loop**(mc의 고정 케이스 테이블은 미리 정해둔 하나의 삼각분할만
  가능)까지 자연스럽게 처리한다. 오프라인(비실시간) 메시 추출에서 위상 견고성과 특징
  보존을 **둘 다** 원할 때가 기본 후보.
- **한계:** 큐브당 비용이 큼(면 6개 × 2-D marching squares + 그래프 스티칭 + 조건부 QEF,
  `mc`/`mtet`의 단일 테이블 조회보다 무거움); 이 저장소에는 GPU 구현이 없음(이번 작업의
  비목표); 한 큐브가 특징 셀이면서 loop가 2개인 경우 두 loop가 **같은** 공유 특징
  정점으로 팬됨(loop별로 독립적인 QEF를 풀지 않음, `emc`의 셀 단위 판정을 그대로
  따른 결과) — 기하학적으로 타당하지만(정점 하나에 팬 2개가 만나는 것 자체는
  edge-manifold를 깨지 않음) 이번 두 픽스처(구/박스)로는 이 경로가 직접 exercise되지
  않았다.
- **테스트:** `test_isosurface_cms.cpp` — 구 정확도(RMSE<cell)+edge-manifold(100+
  삼각형), 박스 크리즈 갭이 `mc`보다 작음.

---

## 3. 선택 가이드 요약

| 전략 | Primal/Dual | 출력 단위 | 위상 보장 | 특징 보존 | 상대 비용 | 대표 사용처 |
|---|---|---|---|---|---|---|
| `mc` | Primal | 삼각형(직접) | 없음(모호성 미해결) | 없음 | 최저(표 조회 1회) | 기본값, 매끄러운 데이터 |
| `mc33` | Primal | 삼각형(직접) | **면+내부 모호성 해소** | 없음 | 낮음(+면/내부 테스트) | watertight 필수, 특징 무관 |
| `mtet` | Primal | 삼각형(직접) | **구조적으로 모호성 없음**(Euler==2 확인) | 없음 | 중간(삼각형 2~3배, 촘촘한 weld) | 표 없이 위상 보장 원할 때 |
| `emc` | Primal(+특징 정점 추가) | 삼각형(케이스+팬) | `mc`와 동급 | **있음**(셀 단위 QEF) | 낮음(+조건부 QEF) | 최소 변경으로 특징만 추가 |
| `dc` | **Dual**(셀당 1정점) | 사각형→삼각형 2개 | **약함**(체커보드 면 비다양체 가능, 문서화) | **있음**(가장 직접적) | 중간(셀마다 QEF) | 특징 최우선, 드문 비다양체 감수 |
| `dmc` | **Dual**(dc 정점 + primal 조회) | 삼각형(dual 케이스) | **crack-free**(`dc`의 약점 해소) | **있음**(dc 상속) | 높음(2-패스) | 박막/근접 이중 시트 재현 |
| `cms` | 혼합(정점=primal, 연결=dual 철학) | 가변 n-gon→팬 삼각형 | **crack-free(구성적 증명)** | **있음**(emc 동급) | 가장 높음(면 6개+스티칭) | 위상+특징 둘 다, 새들 셀 자연 처리 |

---

## 4. 테스트 프레임워크

모든 전략이 같은 해석적(analytic) 필드와 술어로 검증된다(`test/isosurface_test_util.h`,
`namespace isotest`):

- **필드 빌더:** `SphereField(radius, cellSize, halfN)`(정확한 gradient `p/|p|`),
  `BoxField(halfExtents, cellSize, halfN)`(정확한 gradient, 날카로운 모서리/코너 —
  `FromImplicit`을 통해 두 값·gradient 함수 모두 narrow band에 샘플링).
- **메시 술어:** `IsEdgeManifold`(모든 무방향 엣지의 삼각형 등장 횟수 ≤2),
  `IsWatertight`(모든 엣지가 정확히 2), `EulerCharacteristic`(V−E+F),
  `EdgeCounts`(위 세 술어의 기반).
- **정확도:** `Engine::Eval::NearestNeighbourRMSE(mesh.vertices, trueSurfacePoints)` —
  각 정점을 참 표면(예: 구는 `v.normalized()*radius`)으로 투영한 목표점과 최근접
  거리의 RMSE.
- **특징 보존 비교:** 박스의 12개 진짜 크리즈까지의 최소 거리(두 가장 작은 축 오차의
  합, `MinVertexToBoxEdgeGap`류 헬퍼 — `emc`/`dc`/`cms` 테스트 파일에 각각 로컬
  중복, 브리프의 선례를 따름)를 `mc`와 비교.
- **DISABLED 회귀 타깃 관례:** 알려져 있고 문서화된 실패(예: `dc`의 체커보드 면
  비다양체)는 테스트를 지우지 않고 `DISABLED_` 접두사로 남겨 미래 구현(Manifold DC 등)의
  회귀 타깃으로 삼는다 — `test_isosurface_dc.cpp` 참고.

---

## 5. 참고문헌

- Lorensen, Cline, *Marching Cubes*, SIGGRAPH 1987.
- Nielson, Hamann, *The Asymptotic Decider: Resolving the Ambiguity in Marching Cubes*,
  IEEE Visualization 1991.
- Chernyaev, *Marching Cubes 33*, CERN CN/95-17, 1995.
- Lewiner, Lopes, Vieira, Tavares, *Efficient Implementation of Marching Cubes' Cases with
  Topological Guarantees*, Journal of Graphics Tools 8(2), 2003.
- Doi, Koide, *An Efficient Method of Triangulating Equi-Valued Surfaces by Using
  Tetrahedral Cells*, IEICE Trans. E74-D(1), 1991.
- Kobbelt, Botsch, Schwanecke, Seidel, *Feature Sensitive Surface Extraction from Volume
  Data*, SIGGRAPH 2001.
- Ju, Losasso, Schaefer, Warren, *Dual Contouring of Hermite Data*, SIGGRAPH 2002.
- Schaefer, Warren, *Dual Marching Cubes: Primal Contouring of Dual Grids*, Pacific
  Graphics 2004 / CGF.
- Ho, Wu, Chen, Chuang, Ouhyoung, *Cubical Marching Squares: Adaptive Feature Preserving
  Surface Extraction from Volume Data*, Eurographics 2005 (CGF 24(3)).
- Schaefer, Ju, Warren, *Manifold Dual Contouring*, IEEE TVCG 13(3), 2007 (`dc`의
  비다양체 한계를 고치는 미래 작업으로 인용됨).

전체 계보(위 8개 논문이 속한 6개 시대 구분, 학습기반/성능 계열, 이 저장소의 다중해상도
적용 사례)는 [`MARCHING_CUBES_SURVEY.md`](MARCHING_CUBES_SURVEY.md)를 참고.
