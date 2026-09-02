# Depth 프레임 노이즈 제거

D435 depth 프레임의 노이즈를 어디서 어떻게 거르는지, 무엇이 실제로 원인이고 무엇이 아닌지, 그리고 무엇이 아직 없는지. 측정은 전부 `capture/`(477프레임 D435 실측 녹화, 결정론적 재생) 위에서 했다.

## 1. 다섯 층

노이즈 처리가 다섯 층에 걸쳐 있고 **각 층이 볼 수 있는 정보가 다르다.** 실행 순서대로다.

| 층 | 어디서 | 볼 수 있는 것 |
|---|---|---|
| 1. 디바이스 | `RealSenseDepthProvider` | raw 스테레오 쌍 — **케이블 이쪽에서는 영원히 못 보는 정보** |
| 2. 이미지 스무딩 | `PrefilterDepth` | depth 이미지의 국소 이웃 |
| 3. 점 admission | `BackprojectDepth` | 이웃 + 복원된 3D 기하 |
| 4. 융합 가중 | TSDF integrate 커널 | 표면 대비 복셀의 위치, 시선 |
| 5. 맵 | (없음) | 여러 프레임에 걸친 관측의 일관성 |

## 2. 층 1 — 디바이스 (`RealSenseOptions`)

| 노브 | 기본 | 비고 |
|---|---|---|
| `highAccuracyPreset` | off | 매처의 거부 임계를 올린다. 점은 줄고 남은 점의 신뢰도는 오른다 |
| `minimum/maximumDepthMeters` | off | `rs2::threshold_filter` |
| `advancedModeJsonPath` | off | `rs400::advanced_mode::load_json` |

**프리셋은 `get_depth_scale()` 전에 적용한다.** 비주얼 프리셋이 `RS2_OPTION_DEPTH_UNITS`를 바꿀 수 있어, 먼저 읽으면 이전 설정의 스케일이 잡히고 재구성 전체가 **증상 없이** 오배율된다.

**이 층은 재생으로 A/B가 원리적으로 불가능하다.** 녹화는 매처가 판단을 끝낸 뒤의 raw Z16이라, 디바이스에 사는 게이트는 녹화가 실어 나를 흔적을 남기지 않는다.

## 3. 층 2 — `PrefilterDepth`

불연속 인지형 정사각 평균. 같은 표면 위 이웃만 평균하므로 스텝을 넘어 뭉개지 않는다.

- 실측: 인접 법선 불일치 **중앙값 24° → 4.7°(3×3) → 2.9°(5×5)**
- 이 저장소에서 가장 큰 단일 효과다. `realsense_scan` 기본 3, `icp_quality_diag` 기본 0

## 4. 층 3 — 점 admission (`DepthFilterOptions`)

**순서가 의미를 가진다.** 앞 게이트가 픽셀을 invalid로 만들면 뒤 게이트의 이웃 계산이 달라진다.

| 순서 | 노브 | 잡는 것 | 기본 |
|---|---|---|---|
| 1 | `minimum/maximumDepthMeters` | 센서 측정 범위 밖. **invalid로 만든다** — 단순히 안 내보내면 옆 픽셀의 법선을 여전히 결정한다 | off |
| 2 | `relative/minimumDepthJump` | 전방(`u+1`,`v+1`) 점프. **법선을 보호한다** | 항상 on |
| 3 | `symmetricDepthJumpGuard` | 8이웃 전부. **점을 보호한다** | off |
| 4 | `minimumValidNeighbours` | 이웃 지지도. "유효"가 아니라 **"같은 표면"** 이웃을 센다 | off |
| 5 | `maximumIncidenceDegrees` | 스치는 입사각 | off |

### 2번과 3번의 관계

전방 가드가 `u+1`,`v+1`만 보는 건 **버그가 아니라 범위의 문제**다. 법선이 그 두 이웃의 전방차분이므로 가드는 *법선*을 정확히 보호한다. 점의 신뢰도는 별개 질문이고, 둘은 **모든 스텝의 뒤쪽 가장자리에서 갈린다.**

### 5번은 권하지 않는다

`MapConfig::viewAngleWeight`가 기본 `true`라 융합이 이미 `cos(θ)`로 가중된다. hard 게이트는 그 위에 **이중으로** 얹히는 것이다.

## 5. 층 4 — 융합 가중 (TSDF integrate)

| 노브 | 기본 | 측정 효과 |
|---|---|---|
| `viewAngleWeight` — `dot(n, -ray)` | **on** | 층 3의 5번과 중복 |
| `confidenceWeight` (A1) — `1 − λ\|tsdf\|` | 0.5 | **대칭** — 관측한 앞쪽을 본 적 없는 뒤쪽과 똑같이 깎는다 |
| `behindSurfaceDropoff` (Voxblox eq. 5) | off | 앞쪽 온전히, 뒤쪽만 램프. 맵 복셀 −2.33% |
| `adaptiveBand` — 밴드폭 `N·σ_z(z,θ)` | off | 맵 복셀 −22.3%, ICP −21.5% |

## 6. 층 5 — 없음: 자유공간 카빙

integrate 커널은 각 점의 **±truncation 밴드만** 쓴다. 카메라와 표면 사이 자유공간에 아무것도 쓰지 않고, 감쇠·삭제 경로도 없다.

표준(Curless & Levoy 1996, Voxblox)은 센서 원점에서 광선을 따라가며 자유공간을 갱신한다. 그래서 나중 프레임이 flyer가 있던 자리를 통과하면 그 복셀을 밀어낸다. **여기서는 한 번 융합된 flyer가 영구적이다.**

층 3의 게이트들은 flyer가 **들어오지 못하게** 막는다. 들어온 뒤 지워지게 하는 것은 카빙뿐이고, 둘은 서로를 대체하지 않는다.

## 7. 시선 방향 스트리크 — 원인 규명

`capture/` raw depth 직접 분석(12프레임, prefilter 3):

| 가설 | 판정 | 근거 |
|---|---|---|
| 불연속 경계의 flying pixel | **원인** | 방출 점의 0.17%, 가려진 이웃과 **중앙값 403 mm** (p90 543, 최대 616) |
| 센서 축방향 노이즈 | **기각** | 평탄면 σ_z 0.68 mm(0.3–0.8 m) / 1.12 mm(0.8–1.5 m) — 횡방향 샘플 간격(1.8–2.4 mm)보다 **작다** |
| 소프트 보간 램프 | **기각** | 대칭 가드 적용 후 7×7 중앙값에서 50 mm 이상 벗어난 점 **0개**. 남는 종류가 없다 |
| 트런케이션 밴드 | 설계 | truncation 0.03 / voxel 0.01 → 표면마다 ±3 복셀, 약 70 mm 두께 |

즉 **국소 점프 검사가 못 보는 잔여 클래스는 없다.** 대칭 가드를 켜면 depth 이미지 안의 이상점은 사라진다.

## 8. 권장 설정

`--no-submap`, icp 트래커, prefilter 3, `capture/` 477프레임:

| 설정 | 점 유지 | 맵 복셀 | 추적 | align ms |
|---|---|---|---|---|
| 게이트 없음 | 123,471,684 | 1,000,198 | 476/476 | 70.6 |
| **moderate** `--symmetric-guard --min-neighbours 6 --behind-dropoff` | −1.19% | −4.34% | 476/476 | 69.7 |
| **aggressive** 위 + `--min-neighbours 8 --far 4.0` | **−3.58%** | **−6.73%** | 476/476 | 64.3 |

추적은 전 설정에서 476/476(프레임 0의 `NoModel` 1건 제외)이고 RMSE는 나빠지지 않는다. **신뢰 점을 조금 잃더라도 의심 점을 지우고 싶다면 aggressive가 그 트레이드다.**

`--min-neighbours 8`은 이웃 8개가 전부 유효하고 같은 표면일 것을 요구하므로, 이미지 1픽셀 테두리와 invalid 픽셀에 인접한 모든 점을 버린다. 의도된 공격성이다.

## 9. 측정 함정 — 맵 복셀 수를 품질 프록시로 쓰지 말 것

`submap = true`(기본)에서 **entry 수는 입력 점 수의 단조 함수가 아니다.**

| | 베이스라인 | 대칭 가드 | 변화 |
|---|---|---|---|
| `submap = true` | 9,124,516 | 9,878,809 | **+8.3%** |
| `--no-submap` | 2,728,887 | 2,717,576 | **−0.41%** |

원인은 [분류 커널](../src/TSDF/Memory/RegionClassifier/kernel_DenseRegionClassifier.classify.comp.glsl)의 `if (g_dense[i] != 0u) { // latched: never revert`. 분류가 latch되고 되돌아가지 않아 경로 의존적이다 — 입력이 0.27% 바뀌면 어느 블록이 언제 latch되는지가 달라지고, 한 블록이 latch될 때마다 그 부피가 base 32³에서 detail 64³로 약 8배가 된다. 실측에서 dense blocks **27→28, 단 한 개 차이**가 +8.3%를 만들었다.

기각한 대안: 해시 오버플로(insert failures 0, window refusals 0), 포즈 변화(`identity`로 고정해도 발생), 용량 클램프(버퍼는 성장).

**프런트엔드 A/B는 `--no-submap`으로 해야 한다.**

## 10. 아직 없는 것

1. **자유공간 카빙** (Curless & Levoy) — flyer를 영구적이지 않게 만드는 유일한 수단. 걸림돌 둘: point-to-plane은 **법선**을 따라 행진하는데 카빙은 **광선**을 따라야 하고(두 축을 섞으면 안 된다), directional TSDF에서 "빈 공간"을 어느 방향 레이어에 쓸지가 자명하지 않다(자유공간에는 법선이 없다)
2. **σ 가중 prefilter** (Nguyen) — 층 2를 하드 임계에서 $w=\exp(-\frac{\Delta u^2}{2\sigma_L^2}-\frac{\Delta z^2}{2\sigma_z^2})$로
3. **ICP 잔차 가중** $\sigma_z(z_{min},0)/\sigma_z$ — 현재 Huber만이라 먼 점이 가까운 점과 같은 발언권을 갖는다
4. **σ_z 계수 D435 재피팅** — 현재 계수는 Kinect v1(구조광) 피팅이다. 함수 형태는 전이되지만 상수는 아니다

## 참고

- Nguyen, Izadi & Lovell, *Modeling Kinect Sensor Noise for Improved 3D Reconstruction and Tracking*, 3DIMPVT 2012 — σ_z 모델과 KinectFusion 적용(필터·ICP 가중·트런케이션)
- Curless & Levoy, *A Volumetric Method for Building Complex Models from Range Images*, SIGGRAPH 1996 — space carving, outlier robustness
- Oleynikova et al., *Voxblox*, IROS 2017 — eq. 5의 뒤쪽 감쇠 가중치(δ = 4v, ε = v); `1/z²`는 Nguyen 모델에서
- Bylow et al. 2013 — 뒤쪽 감쇠의 원출처(Voxblox 경유)
- Weder et al., *RoutedFusion*, CVPR 2020 — 학습 기반 융합; 표면 경계·얇은 물체의 thickening artifact를 직접 겨냥
