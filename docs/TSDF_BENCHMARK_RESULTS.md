# TSDF Benchmark — 모든 구현 비교

`example2/tsdf_benchmark` 실측 (Apple M4 Max / MoltenVK). 모든 방식이 **동일한 multi-view 입력**(`fixtures::SampleViews`)을 통합하고, 정확도는 **동일 GT 메트릭**(`fixtures::NearestDistance` = 추출 표면점→해석적 표면 거리, mm)으로 측정.

**우선순위: RMSE + integrate 속도.** `integ_ms` = 순수 integration(extract 제외); Directional은 내부 `LastFrameStats.integrateMs`(추출이 Integrate 안에서 실행되므로), 나머지는 integrate 루프만 CPU 계측.

> ⚠️ **warmup 주의:** 각 GPU 파이프라인의 **첫 사용**은 셰이더/할당 warmup을 먹는다 — cube가 먼저 돌아 cube의 integ_ms가 부풀려짐(Simple 40.1 vs cylinder 4.3). **cylinder(모든 파이프라인 warm)가 공정한 정상상태 integrate 비교.**

---

## 결과

### Cylinder (정상상태 — integrate 공정 비교)
| 방식 | **integ_ms** | **acc_rmse** | mem_KB | edge | flat | curved |
|---|---|---|---|---|---|---|
| Simple(fine) | 4.3 | 0.0668 | 193.0 | 0.0614 | 0.0475 | 0.0487 |
| **Compact-Directional** | **4.5** | **0.0159** | 274.1 | 0.0191 | 0.0102 | 0.0104 |
| Compact-Dir(var-adaptive) | 17.3 | 0.0162 | 153.3 | 0.0186 | 0.0102 | 0.0109 |
| Directional | 32.1 | 0.0268 | 6784.0 | 0.0299 | 0.0118 | 0.0121 |
| AdaptiveVoxelGrid(p0.50) | 4.2 | 0.0645 | 169.6 | 0.0628 | 0.0501 | 0.0491 |
| AdaptiveVoxelGrid(p0.90) | 4.1 | 0.0731 | 66.6 | 0.0687 | 0.0608 | 0.0517 |

### Cube (integ_ms는 첫-사용 warmup 포함 — RMSE/mem은 유효)
| 방식 | integ_ms | acc_rmse | mem_KB | edge | flat |
|---|---|---|---|---|---|
| Simple(fine) | 40.1* | 0.0542 | 192.1 | 0.0496 | 0.0458 |
| Compact-Directional | 22.2* | 0.0223 | 219.4 | 0.0308 | 0.0107 |
| Compact-Dir(var-adaptive) | 12.8 | 0.0233 | 148.1 | 0.0305 | 0.0106 |
| Directional | 46.3* | 0.0222 | 6144.0 | 0.0267 | 0.0110 |
| AdaptiveVoxelGrid(p0.50) | 3.5 | 0.0572 | 174.1 | 0.0504 | 0.0480 |
| AdaptiveVoxelGrid(p0.90) | 4.4 | 0.0694 | 56.1 | 0.0497 | 0.0500 |

`*` = 해당 파이프라인 첫 사용 warmup 포함(과대). cylinder 열이 정상상태.

---

## 핵심 결론 (RMSE + integrate 우선)

### 1. Compact-Directional = 명백한 승자
**Directional과 같은(더 나은) 정확도(RMSE 0.016)를 Simple 속도(~4.5ms)로.** RMSE와 integrate 둘 다 최상위 — 정상상태 산점도에서 **좌하단(빠르고 정확)**을 독점.

### 2. Directional은 Compact에 지배(dominated)됨
같은 6-방향 표현·같은 정확도인데 integrate **~7× 느림**(32ms vs 4.5ms — 8³ block 할당 + per-point 6방향 TopK)이고 메모리 **~25× 많음**(6784 vs 274 KB). RMSE+integrate 어느 축으로도 Compact가 우월.

### 3. var-adaptive = 정확도는 같고 integrate가 느림
RMSE 0.016(Compact와 동등)이지만 fine+coarse **두 해시를 통합**해 integrate **~4× 느림**(17ms). 메모리(153KB)를 위해 속도를 내주는 트레이드오프 — RMSE+integrate가 최우선이면 Compact가 낫다.

### 4. 단일필드(Simple/AdaptiveVoxelGrid) = 빠르지만 부정확
integrate ~4ms로 빠르나 RMSE **~0.06–0.07** (projective·normal-free 편향). AdaptiveVoxelGrid는 여기에 메모리 최소화(56–67KB)를 더한 것 — 정확도가 목표가 아니라 메모리가 목표일 때.

---

## 프론티어 (RMSE vs integrate)

| 목표 | 방식 | integ_ms | acc_rmse |
|---|---|---|---|
| **정확도 + 속도 (추천)** | **Compact-Directional** | ~4.5 | 0.016–0.022 |
| 정확도 + 최소 메모리 | Compact-Dir(var-adaptive) | ~17 | 0.016–0.023 |
| 최소 메모리 (정확도 양보) | AdaptiveVoxelGrid(p0.90) | ~4 | 0.069–0.073 |
| (지배됨) | Directional | ~32 | 0.022–0.027 |

**RMSE와 integrate 속도가 최우선이면 → Compact-Directional.** 메모리까지 최소화하려면 약간의 속도를 내주고 var-adaptive.

---

## 재현
```bash
VULKAN_SDK=/usr/local cmake --build build --target tsdf_benchmark
VULKAN_SDK=/usr/local ./build/example2/tsdf_benchmark
```
`integ_ms` 열이 순수 integrate. 구현: `example2/tsdf_benchmark.cpp`. 관련: `docs/ADAPTIVE_VOXEL_GRID_VS_COMPACT_DIRECTIONAL.md`, `docs/COMPACT_VS_DIRECTIONAL_TSDF.md`, `docs/DIRECTIONAL_TSDF_INTEGRATION.md`.
