# Variance-Adaptive Voxel Grids (MrHash) vs DirectionalTSDF

> 논문 [De Rebotti et al., *Resolution Where It Counts: Hash-based GPU-Accelerated 3D Reconstruction via Variance-Adaptive Voxel Grids*, ACM TOG 2025] (이하 **MrHash**) 분석과, 이 저장소의 **DirectionalTSDF**와의 상세 비교.
>
> 관련 문서: [`TSDF_IMPLEMENTATION.md`](TSDF_IMPLEMENTATION.md) (이 저장소 TSDF 해설), [`TSDF_CURRICULUM.md`](TSDF_CURRICULUM.md).

---

## 1. MrHash 분석

**한 줄 통찰**: *"해상도를 필요한 곳에만."* 복셀 크기를 **국소 TSDF 분산(variance)** 에 따라 동적으로 조절 — 기하가 복잡한 곳(에지·코너·노이즈 = 고분산)은 촘촘하게, 평평한 곳(저분산)은 성기게. **센서·시맨틱에 독립**(variance만으로 구동), **flat hash table** 하나로 혼합 해상도를 담아 옥트리 없이 O(1) 접근 + 완전 GPU 병렬.

### 방법

1. **분산 기반 적응 해상도.** 각 복셀이 TSDF 평균 `Dᵢ`, weight `Wᵢ`, **분산 `σᵢ²`** 를 유지. Welford 단일패스 온라인 업데이트(논문 식 3·5·6, `w_k = 1` 고정으로 분산 계산 단순화):

   ```
   Dᵢ ← (Wᵢ·Dᵢ + w_k·d_k) / (Wᵢ + w_k),   Wᵢ ← Wᵢ + w_k          (식 3)
   S₂,ᵢ,ₖ = S₂,ᵢ,ₖ₋₁ + (d_k − Dᵢ,ₖ₋₁)(d_k − Dᵢ,ₖ),   σᵢ² = S₂,ᵢ,ₖ / k   (식 5·6)
   ```

   처음엔 최고해상도로 시작 → 분산이 낮은 블록을 **coarse로 병합**(논문 Fig 4). 블록 단위.

2. **Flat hash table (핵심 자료구조).** 옥트리의 계층 순회(O(log N)) 대신 해상도 레벨 `n_j` 를 **해시 키에 인코딩**(레벨별 heap `h_n`)해 혼합 해상도 블록을 **단일 주소공간**에 저장(Fig 3). 상수시간 접근, 재귀·포인터 추적 없음 → GPU 친화.

3. **Multi-resolution Marching Cubes.** 해상도 경계에서 SDF 보간이 애매(Fig 5: 이웃 복셀 크기·중심 다름)한 문제를, finer 값 우선 가중 보간 + **transitional voxel**(Transvoxel[Lengyel 2010]식으로 coarse 복셀을 공유면 따라 잘라냄) + vertex collapsing으로 해결(Fig 6).

4. **통합.** LiDAR는 DDA 광선 순회(식 1), RGB-D는 투영 매핑(식 2). 표준 Curless-Levoy 융합.

5. **스트리밍.** GPU 예산 85% 도달 시 공간적 관련성으로 제거(RGB-D=프러스텀 밖, LiDAR=반경 밖).

6. **3DGS 렌더.** 복셀 그리드에서 시드한 가우시안 밀도를 **GPU quad-tree**(대비 기반 세분, Alg 1)로 적응 제어 → Novel View Synthesis.

### 결과 (vs VDBFusion · Voxblox · Supereight2 · PIN-SLAM · N³-Mapping · NKSR · GSFusion)

- **메모리**: multi-res가 평균 **1.95×–4.07× 적은** 정점/면(Table 3). RGB-D 2–7.5×↓, LiDAR 2.3–2.9×↓.
- **속도**: single-res 변형이 **최대 13× 빠름**(Table 4), 실시간. 경쟁 기법 일부는 LiDAR 시퀀스에서 실패.
- **정확도**: SOTA와 대등하거나 우수(Table 1·2).
- **트레이드오프**(Table 7, Fig 7·8): fixed-fine(single-res)이 상한. multi-res는 RGB-D(조밀)에선 거의 동등하나 **LiDAR(성김)에선 coarse 복셀이 손해**. 분산 임계 σ가 정확도↔메모리를 조절.
- **렌더**(Table 5): multi-res GS가 fixed-res보다 지각 품질↑(디테일에 splat 더, 평면에 덜).

### 한계
희소 LiDAR에서 적응 이득 감소, σ 튜닝 필요, transitional-voxel 근사(약간의 아티팩트 → vertex collapse로 완화).

---

## 2. DirectionalTSDF와 비교

### 핵심: 둘은 **직교(orthogonal)하는 축**을 적응한다

| | **DirectionalTSDF** (Splietker & Behnke '19, 이 저장소) | **MrHash** (Variance-Adaptive '25) |
|---|---|---|
| **적응하는 축** | **방향(surface orientation)** | **해상도(voxel size)** |
| **해결 문제** | 단일 필드가 마주보는 면을 합치고 코너를 뭉갬 → **날카로운 피처·얇은 구조 보존** | 균일 그리드의 메모리·연산 낭비 → **효율(복잡한 곳만 촘촘)** |
| **구동 신호** | 표면 **법선**(어느 축 레이어에 쓸지) | **TSDF 분산**(복잡/노이즈 정도) |
| **필드 구조** | 복셀당 **6 방향 레이어**(±X±Y±Z) | 복셀당 **단일 필드**, 대신 크기 가변 |
| **메모리 방향** | **더 씀** (최대 6×, residency로 완화) — 방향 충실도에 투자 | **덜 씀** (2–7×↓) — 해상도로 절약 |
| **복셀 크기** | 고정 | 가변(분산 기반) |

**요지**: DirectionalTSDF는 *"같은 위치에 서로 다른 방향의 표면을 동시에 담는"* 표현 문제를, MrHash는 *"어디에 얼마나 촘촘히 담을까"* 의 배분 문제를 푼다. **정반대 방향의 메모리 트레이드오프** — 하나는 품질 위해 더 쓰고, 하나는 효율 위해 덜 쓴다.

### 날카로운 코너에서 무엇이 다른가 (이 저장소 실험과 직접 연결)

이 저장소 `tsdf_feature_compare` 실험에서 **SimpleTSDF는 코너를 뭉갰다**(edge max 오차 0.130mm, 자세히는 [`TSDF_IMPLEMENTATION.md`](TSDF_IMPLEMENTATION.md)). 두 기법의 처리 방식:

- **MrHash**: 코너는 **고분산** → 그 근처만 **복셀을 잘게**. 코너가 더 또렷해지지만, 여전히 **단일 필드 + Marching Cubes**라서 *복셀 크기보다 날카로운 crease는 표현 못 함* — 라운딩을 **줄일 뿐 제거하진 못함**. 대신 코너에 메모리를 더 쓴다.
- **DirectionalTSDF**: 코너의 두 면을 **다른 방향 레이어**로 분리 → **평균이 일어나지 않음** → **고정(더 성긴) 해상도에서도 crease를 정확히 보존**. 세분 불필요.

> MrHash는 "코너에 해상도를 몰아줘서 완화", DirectionalTSDF는 "표현 자체로 crease 보존". 얇은 벽(0.4mm 간격 마주보는 두 면)은 MrHash가 아무리 세분해도 한 복셀에서 두 면이 만나면 합쳐질 수 있는데, DirectionalTSDF는 ±Z 레이어로 원천 분리한다 — Directional의 고유 강점.

### 나머지 축

| 축 | DirectionalTSDF (이 저장소) | MrHash |
|---|---|---|
| **자료구조** | Vulkan 컴퓨트, 방향별 그룹 + residency/streaming, 고정 해상도 | CUDA, 혼합해상도 **flat hash**(레벨=키), O(1) |
| **추출** | 방향별 지향 **점군** + 병합 (원논문은 directional MC로 coherent mesh) | **multi-res Marching Cubes** (mesh, transitional voxel) |
| **렌더** | 점군 뷰어(`tsdf_viewer`) | **3DGS NVS** + GPU quad-tree splat 제어 |
| **센서** | **법선 필요**(RGB-D+법선/지향 점군) | **센서 무관**(RGB-D·LiDAR, 법선 불요) |
| **가중** | view-angle × relWeight (이 저장소) | `w_k = 1` 고정(분산 계산 단순화) |
| **성숙도** | 표현 기법(2019) + 이 저장소의 집중 구현 | 통합·MC·스트리밍·렌더까지 **완결 시스템**, TOG'25 |

### 결합 가능성 — 사실상 시너지

둘이 직교하므로 **합치면 서로의 약점을 메운다**:

- **Directional의 약점 = 6× 메모리** → MrHash식 **variance-adaptive 해상도**로, 방향 레이어가 실제로 필요한 고분산(코너·얇은 벽) 영역에만 fine + multi-direction을 쓰고 평면은 coarse + single-direction으로 → 6× 부담 대폭 완화.
- **MrHash의 약점 = 단일 필드라 crease는 여전히 MC-라운딩** → 고분산 영역에 **방향 레이어**를 얹으면 세분 없이도 crease 보존.
- → **"방향 × 해상도" 동시 적응 TSDF** 가 자연스러운 다음 단계.

### 이 저장소 관점 시사점

1. **large-N GPU 버그를 방금 고쳤으므로**(recall 1.0 @ N≤16384, [`KNOWN_ISSUES_engine_core_large_n.md`](KNOWN_ISSUES_engine_core_large_n.md)) 이제 MrHash가 쓰는 대규모 스케일에서 GPU 경로를 신뢰할 수 있다 — 이 비교를 실제로 벤치마크할 토대가 생겼다.
2. MrHash의 **flat-hash 혼합해상도 키 인코딩**은 이 저장소의 방향별 그룹 residency 구조에 얹기 좋은 설계 참고서다(옥트리 회피 철학이 일치).
3. `SimpleTSDF` 대비 실험을 확장해 **DirectionalTSDF vs (Simple + variance-adaptive) vs 둘 결합**을 같은 GT로 3자 비교하면, "방향이 버는 것 vs 해상도가 버는 것"을 정량 분리할 수 있다.

### 첫 구현 (이 저장소)

논문의 핵심 메커니즘 = *분산이 해상도를 결정한다*. 그 토대인 **온라인 분산 추적**을 `SimpleTSDF`에 넣고, cube/cylinder fixture에서 "에지=고분산, 평면=저분산"을 검증하며, 분산 임계에 따른 메모리 절감을 정량화하는 것이 첫 단계다 (`example2/variance_adaptive_demo`, `SimpleTSDF` variance 확장). 혼합해상도 저장·multi-res MC는 후속.

---

## 벤치마크 결과 (`example2/tsdf_benchmark`, voxel 0.1mm, 해석적 GT)

구현된 TSDF들을 동일 다시점 입력·해석적 정답으로 실측. `tsdf_benchmark`는 정확도(GT 대비 RMSE·영역별)·메모리(점유 저장 바이트)·속도(chrono, 동일 방법론)를 보고한다.

**4-way 비교 (cylinder — all-coarse baseline이 유효한 깨끗한 케이스)**

| method | mem | RMSE(mm) | edge(mm) | 특징 |
|---|---|---|---|---|
| Simple (fine) | 193 KB | 0.067 | 0.061 | 기준선 |
| Simple (coarse, 2×) | 32 KB | 0.106 | 0.096 | 싸지만 에지 뭉갬 |
| **Variance-adaptive** | 106–153 KB | 0.067–0.074 | **0.061–0.063** | 에지=fine 수준 유지 + 메모리↓ |
| Directional | 6784 KB | **0.027** | **0.030** | 최고 정확도, ~35× 메모리 |

**Variance-adaptive σ sweep (실제 2-level fine+coarse, cylinder — 논문 Fig 7의 GT 재현)**

| coarsen% | mem_KB | edge(mm) | vs fine |
|---|---|---|---|
| 26% | 182 | 0.061 | 거의 동일 |
| 50% | 153 | 0.061 | 에지 그대로, −21% 메모리 |
| 75% | 106 | 0.063 | 에지 거의 그대로, −45% 메모리 |
| 90% | 65 | 0.068 | 공격적 → 에지 저하 시작 |

**결론:**
- **Directional** = 최고 품질(edge 0.030)이지만 **~35× 메모리**(블록 8³ + 방향 레이어 + 다시점 누적).
- **Variance-adaptive** = 고분산 에지는 fine 유지·저분산 영역만 coarse → **75% 셀을 coarsen해도 edge 오차가 fine과 거의 동일(0.061→0.063)**, 메모리 45%↓, naive all-coarse(edge 0.096) 대비 크게 우수. 분산이 "해상도를 어디 둘지"의 올바른 신호임을 GT로 입증.
- → **결합("방향 × 해상도": 방향 레이어를 고분산 영역에만)** 이 정량적으로 유망 — Directional 품질을 variance-adaptive 메모리로.

**Caveat**: ① cube는 all-coarse가 0점 추출(fixture 샘플 피치=2×voxel 앨리어싱, 선택 로직 버그 아님) → cube의 coarsen 셀은 hole이라 수치 의미 제한적. ② 2-level은 transitional-voxel 없는 근사(두 단일해상도 MC를 셀별 선택)라 경계 seam 가능. 속도는 `SimpleTSDF` MC가 점유와 무관하게 전체 해시용량을 스캔하는 특성에 영향받아 순수 알고리즘 비교가 아님.

---

## Directional 정확도를 유지하며 메모리 절약하기 (실측 분석)

**질문**: DirectionalTSDF 급 정확도를 메모리 절약하며 얻을 수 있나? — `tsdf_benchmark`로 여러 접근을 실측했고, 결론은 **직관과 반대**였다.

### 시도 1 (실패): 법선 view-angle 가중을 Simple에 추가
가설은 "Directional 평면 정확도는 법선 가중(`w=max(0,n·(−ray))`)에서 오니, 그것만 Simple에 넣으면 메모리 0으로 평면 정확도 회복"이었다. **틀렸다** — `Simple(weighted)`는 오히려 약간 나빠짐(cube flat 0.046→0.050). 원인: ① MC 점유 게이트(`MIN_WEIGHT`)가 w=1 가정이라 가중 시 ~16% 복셀이 탈락→+τ로 스퍼리어스 삼각형, ② 합성 8-코너 뷰가 모든 면을 거의 동일 경사로 봐서 가중이 차별화 안 됨. **Directional의 정확도 이득은 view-angle 가중이 아니라 방향별 표현(레이어+추출)에 내재**한다.

### 핵심 발견: Directional의 35× 메모리는 거의 전부 "블록 낭비"

정확도를 못 깎으니 **저장 방식**을 봐야 한다. DirectionalTSDF의 메모리는 8³=512복셀 **블록 단위**로 잡히는데, 얇은 표면 밴드 주변에선 블록 대부분이 비어 있다. GPU 활성 풀을 열어 **실제 점유(weight>0) 방향-복셀**을 센 결과:

| shape | 현재 mem (블록) | 점유 방향복셀 | **per-voxel 투영 mem** | **낭비 배수** | 평균 방향/복셀 |
|---|---|---|---|---|---|
| cube | 6144 KB | 14040 | **110 KB** | **56×** | **1.14** |
| cylinder | 6784 KB | 17542 | **137 KB** | **49×** | **1.42** |

> **결론(핵심)**: Directional의 정확도(edge 0.027, RMSE 0.022)는 **per-voxel 저장이면 ~110–137 KB — Simple(193 KB)과 대등하거나 더 적게** 달성 가능하다. **35–56×는 근본 비용이 아니라 블록 그래뉴러리티 낭비다. 정확도 손실 0.**

### 방향 레이어 프루닝: 대부분 1개 방향뿐
방향 히스토그램: 점유 복셀의 **87%(cube)/61%(cylinder)가 단일 방향**, 평균 1.14–1.42 방향/복셀. "6개 레이어"는 코너/능선 소수에서만 쓰인다. per-voxel 카운트가 이미 이 프루닝(점유 방향만)을 반영한다.

### Adaptive-Directional 하이브리드 (오늘 바로 가능한 절충점)
셀별로 **에지=Directional / 평면=Simple** 선택 + 에지는 per-voxel 계상:

| method | edge(mm) | RMSE(mm) | mem_KB |
|---|---|---|---|
| Simple(fine) | 0.050 | 0.054 | 192 |
| **Directional** | **0.027** | **0.022** | 6144 |
| **Compact-Directional (실측)** | 0.031 | **0.022** | **219** ⬅ 구현·측정됨 |
| **Compact-Dir × variance-adaptive (실측)** | 0.031 | **0.023** | **148** ⬅ 완전체, Simple 미만 |
| **Adaptive-Dir 하이브리드** | 0.031 | 0.032 | **139** |
| variance-adaptive | 0.061 | 0.067 | 106–153 |

하이브리드는 **Simple보다 적은 메모리(139–150 KB)로 Directional에 근접한 RMSE(0.032–0.043 vs Simple 0.054–0.067)** — DirectionalTSDF 저장구조를 재작성하지 않고 오늘 얻는 실효 절충점.

### Compact-Directional: 투영을 실측으로 (`CompactDirectionalTSDF`, 구현됨)

위 "per-voxel 투영"을 **실제 구현**했다: `src/Engine/Spatial/CompactDirectionalTSDF` — SimpleTSDF의 per-voxel flat-hash에 **(voxel, direction) 키**를 얹고 DirectionalTSDF의 방향별 통합(topK+view-weight)·추출(방향별 zero-crossing+gradient)을 이식. 스트리밍 residency는 안 건드림(대규모 씬용으로 직교).

| shape | method | edge | RMSE | **mem** |
|---|---|---|---|---|
| cube | Compact-Directional | 0.031 | **0.0223** | **219 KB** |
| cube | Directional | 0.027 | 0.0222 | 6144 KB |
| cylinder | Compact-Directional | **0.019** | **0.0159** | **274 KB** |
| cylinder | Directional | 0.030 | 0.0268 | 6784 KB |

> **실측 결론**: Compact-Directional은 cube에서 Directional RMSE와 동률(0.0223≈0.0222), cylinder에서는 **Directional을 능가**(0.016 vs 0.027) — 둘 다 **~25–28× 적은 메모리(219–274 KB, Simple 192–193 KB과 대등)**로. 강한 sanity check: Compact의 `FilledCount`가 DirectionalTSDF의 독립 측정 점유복셀 수와 **정확히 일치**(cube 14040, cyl 17542) — 같은 점유 (voxel,dir) 집합, 저장 그래뉴러리티만 다름. (메모리가 8B/voxel 투영의 ~2배인 건 해시 엔트리 `DirEntry`가 키 포함 16B이기 때문 — SimpleTSDF와 동일 엔트리 크기.) cylinder에서 Directional을 능가하는 건 Compact가 merge 없이 raw per-voxel 추출을 하기 때문으로 추정.

### 완전체: Compact-Directional × variance-adaptive (실측, `Compact-Dir(var-adaptive)`)

두 직교 축을 결합했다 — **per-voxel 방향 저장 × 분산 기반 해상도 적응**: 고분산 에지는 fine Compact 유지, 저분산 평면은 coarse Compact로 병합.

| shape | method | RMSE | **mem** | vs Directional |
|---|---|---|---|---|
| cube | **Compact-Dir(var-adaptive)** | **0.0233** | **148 KB** | 정확도 동률, **41× 적음** |
| cube | Directional | 0.0222 | 6144 KB | — |
| cylinder | **Compact-Dir(var-adaptive)** | **0.0162** | **153 KB** | 정확도 **능가**, **44× 적음** |
| cylinder | Directional | 0.0268 | 6784 KB | — |

> **완전체 결론**: Directional 정확도(RMSE 0.016–0.023)를 **Simple(192–193 KB)보다도 적은 148–153 KB**로 — Directional(6144–6784 KB) 대비 **~40–44× 절감**. cylinder는 공격적 σ(90% coarsen)에서 **66 KB / RMSE 0.019**(여전히 Directional 근접)까지 내려간다. 두 fixture 모두 "Directional 정확도 @ Simple 미만 메모리" **달성**.

### 종합 권고
1. **달성됨 (per-voxel 저장)**: `CompactDirectionalTSDF` → Directional 정확도 @ ~Simple 메모리(25–28×↓), **실측**.
2. **달성됨 (완전체 = per-voxel × variance-adaptive)**: `Compact-Dir(var-adaptive)` → Directional 정확도 @ **Simple 미만 메모리**(40–44×↓), **실측**.
3. **프로덕션 진행됨**: (a) origin-relative movable 512³ 창 + 유닛 테스트 ✅ (b) 실 chair 스캔 E2E 검증 ✅ (c) merge/dedup 추출(`ExtractPointCloud(...,merge=true)`, DirectionalTSDF 클러스터링 이식) ✅ — 실데이터에서 completeness를 Directional 수준(1.45)까지. **남은 것**: 대규모 씬 타일링(HW `shaderBufferInt64Atomics=false`라 단일 해시는 512³ 상한; 스트리밍 DirectionalTSDF가 상보), 2-level 대신 진짜 multi-res MC(transitional voxel).
4. **대규모 씬**: 기존 DirectionalTSDF의 스트리밍 residency는 여전히 유효(Compact은 단일 해시) — 상보적.

**Caveat**: Compact-Directional은 merge 없이 raw per-voxel 후보를 추출(그래서 nPoints가 Directional보다 적고 cylinder에선 더 정확); 키 범위(현재 512³ voxel origin-relative 창 — 이 GPU는 `shaderBufferInt64Atomics=false`라 더 넓은 64bit 키 불가, 대규모 씬은 스트리밍 DirectionalTSDF). 하이브리드/블록-낭비 수치의 일부는 여전히 투영(점유 카운트 기반)이나, **핵심 주장(Directional 정확도 @ ~Simple 메모리)은 Compact-Directional로 실측 완료**.

---

## 실 스캔 데이터 검증 (chair, `example2/tsdf_realdata_benchmark`)

합성 fixture를 넘어 **실제 chair 스캔**(scanData/frame_*.ply, ~77k pts/frame, 씬 ~827mm)으로 E2E 검증. GT가 없으므로 관측점(서브샘플 40k) 대비 최근접 RMSE로 accuracy(recon→관측)·completeness(관측→recon) 측정. 공통 voxel 2.38mm(씬이 Directional 400-voxel·Compact 512³ 창에 모두 맞음; 실측 span 348<512).

| method | mem | accuracy_rmse(mm) | completeness(mm) | nPoints | build_ms |
|---|---|---|---|---|---|
| SimpleTSDF | 3.8 MB | 2.74 | 2.13 | 73101 | 379 |
| DirectionalTSDF | **132 MB** | 2.58 | 1.47 | 59650 | 7010 |
| **CompactDirectionalTSDF** (raw) | **8.1 MB** | **2.31** | 1.67 | 94851 | 444 |
| **CompactDirectionalTSDF (merged)** | **8.1 MB** | 2.46 | **1.45** | 52755 | — |

> **실데이터 결론**: Compact이 **accuracy 최고(raw 2.31 < Directional 2.58 < Simple 2.74)**, 메모리는 Directional 대비 **16.3× 적음(8.1 vs 132 MB)**, 속도 **~16× 빠름(444 vs 7010 ms)**. `FilledCount`(520,202)가 Directional 점유복셀과 **정확히 일치**(합성과 동일한 sanity check). 827mm chair가 origin-relative 512³ 창에 들어감을 실증. **merge/dedup**(DirectionalTSDF의 클러스터링 이식, 60° hard-split로 코너 보존)을 켜면 점 수가 94851→52755(Directional 59650보다 적게)로 줄고 **completeness 1.67→1.45로 Directional(1.47)까지 따라잡음**(accuracy는 2.31→2.46로 소폭 양보하나 여전히 3자 중 최고). **핵심 주장이 실 스캔에서도 성립.**

---

## 대규모 씬 타일링 (`TiledCompactDirectionalTSDF`, `example2/tiled_compact_demo`)

단일 Compact은 512³ voxel 창 상한(HW `shaderBufferInt64Atomics=false`). **타일링**으로 이를 넘는다: 공간을 타일(core 448 voxel)로 나눠 각 타일이 Compact 512³ 창 하나, **ghost 통합**(경계 근처 점은 인접 타일에도 통합 → 밴드 경계 정확), **core-only 추출**(중복 방지). 원래 목표(100μm dental, 전악 >51mm)의 실질 해법.

**검증 (chair, 공통 voxel 2.38mm — 단일창도 됨):** Tiled(4타일) accuracy 2.48/completeness 1.45 vs 단일 Compact 2.49/1.45 → **rel-diff 0.4% / 0.3% 일치**(seam·중복 없음, ghost 통합 검증).

**대규모 (chair, fine 1.0mm — 827 voxel > 512, 단일창 불가):**

| | TileCount | nPoints | mem | accuracy | completeness |
|---|---|---|---|---|---|
| **Tiled @ 1mm** | **8** | 145,492 | **26 MB** | **1.84** | 1.48 |
| 단일 Compact @ 1mm | 1 | 75,798 (부분만) | — | — | **133** (파손) |

> **타일링 결론**: 단일 512³ 창을 넘는 1mm chair를 **타일링만 전체 복원**(단일창은 창 밖 slab을 조용히 버려 completeness 133mm로 파손). 타일 수 비례 저메모리(26 MB / 8타일 = ~3.3 MB/타일), 더 고운 voxel이라 accuracy도 향상(2.48→1.84). 단일창과 겹치는 해상도에선 정확도 동일. **임의 크기 씬을 저메모리로 — 프로덕션 large-scene 해결.**

---

## 참고문헌
- De Rebotti, Giacomini, Grisetti, Di Giammarino, *Resolution Where It Counts*, ACM TOG 2025.
- Splietker & Behnke, *Directional TSDF: Modeling Surface Orientation for Coherent Meshes*, 2019 — [arXiv:1908.05146](https://arxiv.org/pdf/1908.05146).
- Curless & Levoy, SIGGRAPH 1996 · Nießner et al. (Voxel Hashing), 2013 · Lengyel (Transvoxel), 2010 · Welford, 1962.
