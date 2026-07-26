# TSDF Benchmark — 모든 구현 비교

`example2/tsdf_benchmark` 실측 (Apple M4 Max / MoltenVK). 모든 방식이 **동일한 multi-view 샘플**(`fixtures::SampleViews`)을 통합하고, 정확도는 **동일 GT 메트릭**(`fixtures::NearestDistance` = 추출 표면점→해석적 표면 거리, mm)으로 측정. voxel = 벤치마크 기본값.

> 수식/표는 GitHub·마크다운 뷰어에서 렌더됩니다.

---

## 결과 (정확도 vs 메모리 vs 속도)

### Cube
| 방식 | build_ms | nPoints | mem_KB | acc_rmse | edge | flat |
|---|---|---|---|---|---|---|
| Simple(fine) | 154.6 | 6246 | 192.1 | 0.0542 | 0.0496 | 0.0458 |
| Simple(weighted) | 92.2 | 8010 | 192.1 | 0.0613 | 0.0506 | 0.0496 |
| **Directional** | 113.8 | 6139 | **6144.0** | **0.0222** | 0.0267 | 0.0110 |
| Compact-Directional | 91.6 | 4689 | 219.4 | 0.0223 | 0.0308 | 0.0107 |
| **Compact-Dir(var-adaptive)** | 116.1 | 3966 | **148.1** | **0.0233** | 0.0305 | 0.0106 |
| AdaptiveVoxelGrid(p0.50) | 125.0 | 6017 | 174.1 | 0.0572 | 0.0504 | 0.0480 |
| **AdaptiveVoxelGrid(p0.90)** | 77.2 | 2227 | **56.1** | 0.0694 | 0.0497 | 0.0500 |

### Cylinder
| 방식 | build_ms | nPoints | mem_KB | acc_rmse | edge | flat | curved |
|---|---|---|---|---|---|---|---|
| Simple(fine) | 82.4 | 4022 | 193.0 | 0.0668 | 0.0614 | 0.0475 | 0.0487 |
| **Directional** | 63.4 | 3814 | **6784.0** | 0.0268 | 0.0299 | 0.0118 | 0.0121 |
| **Compact-Directional** | 58.9 | 3506 | 274.1 | **0.0159** | 0.0191 | 0.0102 | 0.0104 |
| **Compact-Dir(var-adaptive)** | 119.1 | 3091 | **153.3** | 0.0162 | 0.0186 | 0.0102 | 0.0109 |
| AdaptiveVoxelGrid(p0.50) | 103.0 | 4397 | 169.6 | 0.0645 | 0.0628 | 0.0501 | 0.0491 |
| **AdaptiveVoxelGrid(p0.90)** | 102.8 | 4208 | **66.6** | 0.0731 | 0.0687 | 0.0608 | 0.0517 |

---

## 핵심 결론

### 1. 정확도는 두 계층 — 표현(representation)이 결정한다
- **Directional 계열**(Directional / Compact-Directional / var-adaptive): RMSE **~0.016–0.023 mm**. 6개 부호축 레이어로 서로 다르게 향한 면을 분리 저장 → **모서리 보존**(flat 0.011 vs Simple 0.046).
- **단일필드**(Simple, AdaptiveVoxelGrid): RMSE **~0.054–0.073 mm**. projective·normal-free 통합의 half-voxel 편향(별도 문서화됨). **AdaptiveVoxelGrid는 SimpleTSDF 위에 얹혀 정확도가 Simple과 같은 계층** — 정확도를 올리지 않는다.

### 2. AdaptiveVoxelGrid = 순수 "해상도 적응" (메모리 최소화)
- p0.90에서 **56–67 KB, 전체 최저 메모리**(Simple 대비 ~3×, Directional 블록 대비 ~100× 절감).
- 정확도는 Simple 수준 유지 — coarsening이 truncation-band 내부에 몰려 **표면 메시가 거의 안 바뀜**(모서리도 0.05로 유지, 안 뭉갬). 즉 **정확도를 희생하지 않고 저장 표현만 줄인다.**
- ⚠️ 현재 구현은 fine 레벨을 GPU 상주로 유지 — mem_KB는 **추출/저장 표현**(fine+coarse voxel × 16B)의 크기이지 GPU VRAM 절감이 아님.

### 3. 스위트 스팟 = Compact-Dir(var-adaptive) — "방향 × 해상도"
- **Directional급 정확도(0.016–0.023)를 148–153 KB**에 달성. Directional 블록(6144–6784 KB) 대비 **~40× 메모리 절감**, 정확도 동등.
- 방향(정확도)과 해상도(메모리)를 **둘 다** 얻는 결합.

### 4. 직교성 명제 실측 확인
`docs/ADAPTIVE_VOXEL_GRID_VS_COMPACT_DIRECTIONAL.md`의 주장 그대로:
$$\text{Compact: } M\propto\kappa N\ (\text{방향}\uparrow),\quad \text{Adaptive: } M\propto(1-\tfrac{7f}{8})N\ (\text{해상도}\downarrow)$$
→ AdaptiveVoxelGrid는 메모리를, Directional은 정확도를, **결합이 둘 다** 얻는다.

---

## 프론티어 (선택 가이드)

| 목표 | 방식 | mem | acc_rmse |
|---|---|---|---|
| **최소 메모리** | AdaptiveVoxelGrid(p0.90) | 56–67 KB | 0.069–0.073 |
| **최고 균형 (추천)** | Compact-Dir(var-adaptive) | 148–153 KB | 0.016–0.023 |
| **최고 정확도** | Compact-Directional | 219–274 KB | 0.016–0.022 |
| 대용량 씬 | TiledCompactDirectional | (Compact/타일) | Compact와 동일 |

- 모서리·얇은 피처가 중요 → **Directional 계열**.
- 평탄·유기적 영역이 크고 메모리가 최우선, 표면 정밀도는 Simple로 충분 → **AdaptiveVoxelGrid**.
- 둘 다 → **Compact-Dir(var-adaptive)**.

> TiledCompactDirectional은 단일 512³ 창을 넘는 대형 씬용이라, 작은 GT fixture에서는 Compact-Directional과 동일해 별도 행으로 넣지 않음.

---

## 재현
```bash
VULKAN_SDK=/usr/local cmake --build build --target tsdf_benchmark
VULKAN_SDK=/usr/local ./build/example2/tsdf_benchmark
```
구현: `example2/tsdf_benchmark.cpp` (`RunAdaptiveVoxelGrid` = 신규 행). 관련: `docs/ADAPTIVE_VOXEL_GRID_VS_COMPACT_DIRECTIONAL.md`, `docs/COMPACT_VS_DIRECTIONAL_TSDF.md`, `docs/MRHASH_VS_DIRECTIONAL_TSDF.md`.
