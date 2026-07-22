# DirectionalTSDF Sub-voxel 추출 정밀화 (Gradient-refined, lite) 설계

> 배경: `DirectionalTSDF`는 방향 레이어 덕에 *field 레벨*에선 코너 양면을 분리 보존하지만, **추출단계**의 점 위치는 같은 레이어 +axis 이웃과의 선형보간 crossing을 **평균**(`posSum/crossings`)해 잡는다. 축 방향 보간+평균이라 곡면·비축정렬 표면에서 추출점이 실제 등위면에서 벗어난다. 이 스펙은 출력 형태(oriented point cloud)·통합·저장을 **모두 그대로 둔 채**, 추출 셰이더만 gradient 투영으로 바꿔 점 위치를 **sub-voxel로 정밀화**하고, feature-preservation 오라클을 확장해 그 이득을 정량 검증한다.

## 목적
재구성 "품질 유지"가 북극성이다. 등록/FPFH 파이프라인과의 호환을 위해 추출 산출물은 계속 oriented point cloud(`ExtractedPoint{position, normal, dirMask}`)로 유지하되, 각 추출점을 **실제 zero 등위면으로 1차 Newton 투영**해 위치 오차(특히 flat/curved 영역)를 낮춘다. 개선 여부는 해석적 형상 오라클로 **위치 오차 + 법선각 오차**를 영역별로 측정해 판정한다. 법선까지 더 필요하면 후속 저장형 Gradient-SDF(관측 법선 누적) 착수의 데이터 근거로 삼는다.

기반은 이미 존재: GPU 추출(`directional_tsdf_extract.comp`)이 방향별 crossing마다 위치+유한차분 gradient를 `DirectionalCandidate`로 방출하고, CPU `mergeCandidates`가 복셀당 클러스터링해 점군을 만든다. feature-preservation 실험(`example2/tsdf_feature_compare.cpp` + `example2/shape_fixtures.h`)이 다시점 합성 샘플링·해석적 최근접거리 오라클·영역 분류·`--dump`·A/B 뷰어를 이미 제공한다. 신규 = 추출 셰이더의 위치 공식 교체 + refine 토글 배선 + 오라클에 법선각 오차 추가.

## 단위 근거 (설계 전제)
통합 셰이더(`directional_tsdf_integrate.comp` L103)는 `newValue = clamp(sdf / g_truncation, -1, 1)`로 **truncation 정규화 값**을 누적한다. 따라서 추출에서 복원하는 `c = sumDW/sumW`는 metric(미터)이 아니라 정규화 값이다. 정규화 스케일은 유한차분 gradient 크기 `|grad|`(정규화 값의 voxel-index당 변화량)에 이미 반영돼 있으므로, truncation 상수를 몰라도 gradient 크기만으로 단위가 맞는 투영을 쓴다:

```
center = (v + 0.5) * voxelSize          // 복셀 중심 (world)
p      = center − c * voxelSize * grad / dot(grad, grad)
```

이는 `p = center − value/|∇value_world| · n̂` (n̂ = grad/|grad|, ∇value_world = grad/voxelSize)와 동치인 표준 1차 등위면 투영이다. 이상적 단위-SDF뿐 아니라 truncated·weighted-average 필드에도 안정적으로 동작한다.

## 범위 (v1)
- **추출 위치 정밀화**: 방출 게이트(같은 방향 레이어에서 +axis crossing 존재 = 표면이 ~1 voxel 내 통과)는 유지하고, 방출점 위치만 `posSum/crossings`(축보간 평균) → 위 gradient 투영으로 교체.
- **A/B 토글**: 셰이더에 `g_refine`(push-constant; 0 = legacy 평균, 1 = refined 투영) 추가 → 같은 볼륨·같은 통합 결과에서 두 추출 방식을 비교 가능. `DirectionalTSDF`에 `SetSubvoxelRefine(bool)` 소형 세터 + extract dispatch 배선.
- **법선**: 현재 central-difference gradient 유지(`n = grad/|grad|`). **lite는 위치가 주 개선, 법선은 소폭.** 법선 정확도의 큰 도약은 저장형(full, 범위 밖).
- **검증 오라클 확장**: `shape_fixtures.h`에 `NearestNormal(shape, p)` 추가, `tsdf_feature_compare.cpp`에 영역별 법선각 오차(°) + legacy/refined 비교 열 + `--dump` 수치.

### 범위 밖 (후속)
- 저장형 Gradient-SDF(복셀에 관측 법선 누적 → 정확 법선 + direct/photometric tracking): GPU voxel 레이아웃·통합·host-store·wire 변경 동반. **이번 오라클 수치로 착수 여부 결정.**
- 얇은 구조(thin wall/tube) 스트레스 케이스·보존 강화.
- Sharp watertight 메시(Dual Contouring/QEF).
- 추출점 법선의 trilinear 개선(측정 후 필요시).
- large-N GPU 비결정성(기존 알려진 이슈) 수정.

## 아키텍처

### 1. 추출 셰이더 — `src/shader/directional_tsdf_extract.comp` (수정)
현재 흐름: 복셀 중심 값 `c` → 3개 +axis 이웃 crossing을 선형보간(`t=c/(c-nVal)`)·평균 → `posSum/crossings` 위치, central-difference gradient 법선.

변경:
- push-constant `PC`에 `uint g_refine;` 추가(기존 필드 뒤, std140 스칼라 정렬 유지).
- 방출 게이트는 불변(`crossings == 0`이면 return, `|grad| < 1e-6`이면 return).
- 위치 계산 분기:
  - `g_refine == 0` (legacy): 기존 `pos = posSum / crossings` 유지(회귀 비교 기준).
  - `g_refine == 1` (refined): `vec3 center = (vec3(v) + 0.5) * g_voxelSize; vec3 step = c * g_voxelSize * grad / dot(grad, grad); vec3 pos = center - step;`
    - **발산 클램프**: `length(step) > g_voxelSize`이면 refined를 버리고 legacy `posSum/crossings`로 fallback(gradient가 약하거나 `c`가 큰 비선형 영역 보호).
- 법선·candidate 방출·`atomicAdd` 카운터 경로는 불변.

### 2. 토글 배선 — `DirectionalTSDF.{h,cpp}` (수정)
- 신규 API: `void SetSubvoxelRefine(bool on) { m_subvoxelRefine = on; }` + `bool m_subvoxelRefine = false;` 멤버(기본 off = 기존 동작 정확 재현).
- extract dispatch에서 push-constant에 `g_refine = m_subvoxelRefine ? 1u : 0u` 기록. 그 외 통합/추출 파이프라인 불변.

### 3. 오라클 확장 — `example2/shape_fixtures.h` (수정)
- 신규: `Eigen::Vector3f NearestNormal(int shape, const Eigen::Vector3f &p)` — 점 p의 최근접 표면 특징에서의 해석적 바깥 법선(cube = 면/능선/코너 케이스, cylinder = 옆면/캡/림 케이스). 능선·코너 등 불연속에서는 인접 면 법선의 대표값(가장 가까운 면)을 반환하며, 이 band는 `ClassifyRegion`의 Edge로 이미 분리되므로 법선각 오차 집계에서 Edge는 참고치로만 본다.
- 기존 `NearestDistance`/`ClassifyRegion`/`SampleViews`는 재사용.

### 4. 실험 앱 — `example2/tsdf_feature_compare.cpp` (수정)
- **legacy/refined 두 점군 확보**: 공개 API는 `Integrate()`(전체 프레임 파이프라인, 결과를 `m_pointCloud`에 저장)뿐이고 독립 재추출 메서드는 없다. 따라서 기본 경로는 **동일 합성 입력으로 두 번 실행** — `SetSubvoxelRefine(false)`로 Build+Integrate → 점군 A 복사, `SetSubvoxelRefine(true)`로 (재-Build 또는 두 번째 인스턴스) Build+Integrate → 점군 B. 합성 입력은 N≲1000에서 결정적이라 두 통합 볼륨이 동일해 추출 방식만 차이 남. (선택적 최적화로 extract dispatch만 재실행하는 내부 헬퍼를 열 수 있으나 v1 필수 아님.)
- 지표(영역별, flat/curved/edge):
  ```
  for p in cloud: e_pos = |NearestDistance(shape, p)|
                  e_nrm = angle(p.normal, NearestNormal(shape, p))   // 도(°)
  per-region accumulate mean/max of e_pos, e_nrm  (legacy / refined 각각)
  ```
- `--dump`: 영역별 `pos mean/max`(legacy vs refined) + `normal° mean/max`(legacy vs refined) 표를 stdout 출력.
- 뷰어: 기존 A/B 토글(Simple/Directional) 옆에 refine on/off 토글(ImGui 체크 + `--dump`와 동일 경로) 추가. 색 모드 error/region 재사용. 카메라 불변.

### 5. 파일 요약
- 수정: `src/shader/directional_tsdf_extract.comp`(refine 분기+토글), `src/Engine/Spatial/DirectionalTSDF.{h,cpp}`(세터+push-constant 배선), `example2/shape_fixtures.h`(+`NearestNormal`), `example2/tsdf_feature_compare.cpp`(법선각 오차 + refine A/B + `--dump`).
- 불변: 통합 셰이더·host-store·GPU voxel 레이아웃·`DirectionalCandidate`/`ExtractedPoint` 구조·`mergeCandidates`·등록/FPFH 경로.

## 검증
- **빌드**: `--target tsdf_feature_compare` + `--target vkspatial_tests` 성공.
- **회귀**: `g_refine == 0` 경로가 기존 추출과 **비트 동일**(같은 입력·볼륨에서 legacy 점군 불변) — 리팩터가 기존 동작을 안 깼음을 확인.
- **수치(핵심 오라클)**: `--dump --shape cube|cylinder`로 영역별 표 출력 →
  - **위치**: refined가 **curved·flat에서 mean/max 유의 감소**, **edge 무회귀**(같거나 개선).
  - **법선각**: 참고치. 개선 미미(≈legacy)하면 그게 곧 저장형(full) 착수 근거.
  - 안 맞으면(refined가 회귀) 투영 공식/단위/클램프 또는 fallback 로직 버그.
- **launch-smoke**: `--frames N` 자동종료, exit 0, 신규 validation 에러 없음.
- **육안**: 뷰어에서 refine on/off 토글로 곡면 표면점이 등위면에 더 밀착하는지 확인(카메라 불변).

## 리스크
| 리스크 | 대응 |
|---|---|
| 저장 TSDF가 truncation 정규화라 metric 가정 시 스케일 오류 | gradient-magnitude 공식(`p = center − c·voxelSize·grad/dot(grad,grad)`)이 truncation 상수 없이 단위 일치 — 설계 전제로 고정 |
| gradient 약하거나 `c` 큰 비선형 영역에서 투영 발산 | `|grad|<1e-6` skip + step 길이 > 1 voxel이면 legacy 평균 crossing으로 fallback |
| CPU `mergeCandidates`의 위치 평균이 sub-voxel 이득 희석 | 클러스터 내 점들이 거의 동일 위치라 평균도 정확; `--dump`로 실제 이득 확인, 희석 크면 후속에서 merge를 대표점 선택으로 조정(범위 밖) |
| legacy/refined 두 점군 확보(독립 재추출 API 없음) | 동일 합성 입력으로 재-Build+재통합 2회가 기본(N≲1000 결정적이라 두 볼륨 동일); extract-only 재실행 헬퍼는 선택적 최적화 |
| large-N 비결정성(기존 이슈) | feature-preservation과 동일하게 N≲1000 소형 형상 유지 |
| 법선 오라클 불연속(능선/코너) | Edge band는 `ClassifyRegion`으로 이미 분리 → 법선각 집계에서 Edge는 참고치로만 해석 |
