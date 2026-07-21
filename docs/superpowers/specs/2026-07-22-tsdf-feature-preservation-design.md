# Sharp/Round Feature Preservation 비교 (SimpleTSDF vs DirectionalTSDF) 설계

> 배경: DirectionalTSDF 도입 목적 = 날카로운 모서리(코너/능선)에서 두 면을 별도 방향 레이어로 유지 → single TSDF가 부호거리를 평균해 뭉개는(라운딩) 코너를 보존. 이 스펙은 그 이득을 **통제된 합성 형상 + 인앱 A/B 뷰어 + 정량 지표**로 실험·시각화한다.

## 목적
동일 입력을 `SimpleTSDF`(평균 단일 필드 → 코너 라운딩)와 `DirectionalTSDF`(방향 레이어 → 코너 보존)에 통합·추출하고, 같은 카메라에서 A/B 토글로 표면을 비교한다. GT(해석적 형상) 대비 추출점 오차를 영역별(flat/curved/edge)로 산출해 "**edge에서 Directional이 크게 유리, flat/curved는 대등**"을 눈과 숫자로 확인한다.

기반은 이미 존재: 두 TSDF 클래스(`SimpleTSDF`/`DirectionalTSDF`)가 모두 위치+법선 점군을 추출(`SimpleTSDF::ExtractPointCloud`→`OrientedPointCloud{points,normals}`, `DirectionalTSDF::PointCloud`→`vector<ExtractedPoint>`), 뷰어 인프라(`PointCloudPass`/`ImGuiPass`/`Application`/트랙볼)도 있음. 신규 = 형상 fixture + 신규 앱 + ImGuiPass 소폭 일반화.

## 범위 (v1)
- **형상 2종**: `cube`(날카로운 90° 능선/코너 + 평면), `cylinder`(둥근 옆면 + 평면 캡 + 날카로운 원형 림). 각 형상은 ① 다시점 지향 샘플러 ② 해석적 최근접-표면 거리 ③ 영역 분류(flat/curved/edge)를 제공.
- **관측**: 객체 주위 다시점(구면 6~12뷰) 해석적 전면 샘플링(위치+법선) → 순차 통합. 노이즈 v1 off.
- **두 TSDF**: 같은 voxelSize/truncation. Simple=위치만 통합→MC 점군 추출; Directional=위치+법선 통합→방향 점군 추출.
- **지표**: 추출점별 `err=|해석적 최근접거리|`, 영역별 mean·max(Simple/Directional 각각). 헤드리스 `--dump`로 출력.
- **시각화**: A/B 토글(Tab+ImGui 라디오, 동일 카메라), 색 모드 error/region, GT 입력 오버레이 옵션, 영역별 오차 표.

### 범위 밖 (후속)
- 노이즈/아웃라이어 주입 슬라이더, 실 스캔 데이터, 좌우 분할 뷰포트, 오차 히스토그램, 마칭큐브 신규 구현(Simple은 기존 MC 재사용), sphere/wedge 형상, 단위 테스트(이번엔 생략 — `--dump` 수치로 대체).

## 아키텍처

### 1. 형상 fixture — `example2/shape_fixtures.h` (신규)
형상별로 아래 인터페이스를 제공(POD/함수). 좌표 단위는 mm, voxel 0.1mm 스케일에 맞춘 크기(cube 한 변 ~3mm, cylinder r~1.5mm·h~3mm).
- `SampleViews(shape, params) -> vector<View>` where `View { Eigen::Vector3f camPos; vector<Eigen::Vector3f> points; vector<Eigen::Vector3f> normals; }` — 각 가상 시점에서 **전면(normal·(cam−p)>0)** 표면만 촘촘히 샘플링.
- `float NearestDistance(shape, p)` — 점 p에서 해석적 형상 표면까지 최근접 거리(정확값; cube=면/능선/코너 케이스, cylinder=옆면/캡/림 케이스).
- `Region ClassifyRegion(shape, p)` — `enum Region { Flat, Curved, Edge }`. p의 최근접 표면 특징이 날카로운 능선/림 band(≤ 2·voxelSize) 이내면 `Edge`, 곡면이면 `Curved`, 평면이면 `Flat`.
- 재사용: 기존 `tsdf_fixtures.h`의 색/좌표 헬퍼와 겹치면 참조(중복 최소화).

### 2. 두 TSDF 실행 (앱 내부)
- `SimpleTSDF simple; simple.Build(ctx, voxelSize, truncation); for(v:views) simple.Integrate(v.points, v.camPos); auto simpleCloud = simple.ExtractPointCloud();`
- `DirectionalTSDF dir; dir.Build(ctx, voxelSize, truncation); dir.SetIntegrationQuality(<full multi-direction>); for(v:views) dir.Integrate(v.points, v.normals, v.camPos, Zero()); auto dirCloud = dir.PointCloud();` — 품질은 다방향 보존이 켜진 설정(maxDirections ≥ 2, view-angle weighting on); 정확한 필드명은 `DirectionalIntegrationQuality.h`의 `IntegrationQuality` 참조(플랜에서 확정).
- 형상/파라미터 변경 시 재실행(뷰어의 dirty→rebuild와 동일 패턴; `SimpleTSDF::Reset` 또는 재-Build로 클린 볼륨).

### 3. 지표 (앱 내 CPU) — 진짜 오라클
각 추출 점군(simple/dir)에 대해:
```
for p in cloud: e = NearestDistance(shape, p); r = ClassifyRegion(shape, p)
per-region accumulate(mean, max) of e
```
결과 표(형상별로 존재하는 영역만): `region | Simple mean/max | Dir mean/max`. `--dump` 모드에서 stdout 출력. 가설: `edge: Dir.mean ≪ Simple.mean`, `flat/curved: 대등`.

### 4. 시각화 — `example2/tsdf_feature_compare.cpp` (신규 앱)
- 인프라 재사용: `Application` + `Camera` + 트랙볼 + `RenderGraph`(수동 루프, tsdf_viewer와 동일한 between-frame rebuild 패턴) + `PointCloudPass` + `ImGuiPass`.
- 점 세트: (0) GT 입력(흰, 옵션) · (1) Simple 추출 · (2) Directional 추출. **A/B 토글 = 세트 1/2 visibility 스왑**(Tab 키 + ImGui 라디오, 카메라 불변).
- 색 모드: `error`(각 점 색 = `sdfColor`류 파랑→빨강, err를 [0, ~3·voxelSize]로 정규화) / `region`(Flat/Curved/Edge 고정색). 색은 rebuild 때 점별로 굽는다(PointVertex.rgba).
- `CompareState { int shape; int method; int colorMode; bool showInput; bool dirty; /* metrics: per-region mean/max for both */ }` 공유.
- ImGui 패널: shape 콤보[cube, cylinder], method 라디오[Simple/Directional](Tab로도 전환), colorMode 콤보[error, region], showInput 체크, 영역별 오차 표(Text), "Re-run" 버튼.

### 5. ImGuiPass 일반화 리팩터 (`example2/ImGuiPass.{h,cpp}`)
현재 `ImGuiPass::Execute`는 tsdf_viewer 전용 UI를 하드코딩. 이를 **`std::function<void()> drawUi` 콜백 주입**으로 일반화:
- 새 API: `void SetUi(std::function<void()> drawUi)` (또는 생성자 인자). `Execute`는 `NewFrame → drawUi() → Render → RenderDrawData`만 담당, 패널 내용은 각 앱이 공급.
- `tsdf_viewer.cpp`도 이 방식으로 이전(기존 `ViewerState` UI를 람다로 이동) — 두 앱이 같은 pass 공유, 중복 제거. init/teardown·load-not-clear·descriptor pool 로직은 불변.

### 6. 파일 요약
- 신규: `example2/shape_fixtures.h`, `example2/tsdf_feature_compare.cpp`.
- 수정: `example2/ImGuiPass.{h,cpp}`(콜백 일반화), `example2/tsdf_viewer.cpp`(콜백 방식으로 이전), `example2/CMakeLists.txt`(신규 타깃 + 셰이더 재사용 + `imgui`/`Engine::Render` 링크). `PointCloudPass`·셰이더는 그대로 재사용.

## 검증 (GUI라 UI 단위테스트 오라클 없음 — 단, 지표는 수치 오라클)
- **빌드**: `--target tsdf_feature_compare` 성공.
- **launch-smoke**: `--frames N` 자동종료, exit 0, 신규 validation 에러 없음(기존 `VUID-...-00067`은 범위 밖).
- **수치**: `--dump`(+ `--shape cube|cylinder`)로 영역별 오차 표 출력 → **cube/cylinder 모두 edge에서 Dir.mean < Simple.mean, flat/curved 대등** 확인. 안 맞으면 fixture/통합/영역분류 버그.
- **육안**: 사용자가 창에서 A/B 토글로 코너 라운딩(Simple) vs 보존(Directional) 확인.

## 리스크
| 리스크 | 대응 |
|---|---|
| SimpleTSDF MC 추출이 형상을 과하게 스무딩해 "불공정"하게 보임 | 그것이 곧 single-TSDF의 실제 한계 — GT 대비 오차로 정량화하면 공정. 같은 voxel/truncation 사용 |
| 다시점 샘플링이 코너 양면을 충분히 못 봐 라운딩 재현 실패 | 뷰 수/배치를 능선을 양쪽에서 보게 배치(구면 분포), 뷰 수 파라미터화 |
| 영역 band 폭(edge 판정) 임의성 | band=2·voxelSize 기본, 필요시 파라미터; flat/curved/edge를 GT 특징 기준으로 분류(추출점 위치 기준 최근접) |
| DirectionalTSDF large-N 비결정성(기존 알려진 이슈) | 형상 샘플 수를 N≲1000 근방으로 유지(voxel 0.1, 소형 형상) |
| ImGuiPass 일반화가 tsdf_viewer 회귀 유발 | 리팩터 후 tsdf_viewer도 재빌드·launch-smoke·`--dump` 수치 재확인 |
