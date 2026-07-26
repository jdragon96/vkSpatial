# Variance-Adaptive Voxel Grid — 개념과 코드베이스 적용 위치

> 질문: "이 개념이 어디에 적용됐나?" 에 대한 지도(map).
> 요약: **변이(variance)를 측정하는 원시값은 통합 커널에 박혀 있고, "엣지=고변이, 평면=저변이"는 검증됐으며,
> 실제 "고변이는 fine·저변이는 coarse" 선택은 벤치마크의 combine에서 측정으로 증명됐다.
> 단, 런타임 멀티해상도 저장 구조는 아직 없다(오프라인 조립 단계).**

수식은 GitHub / 마크다운 뷰어에서 렌더됩니다.

---

## 1. 개념 (원 논문: *Resolution Where It Counts*)

논문 `docs/Variance-Adaptive Voxel Grids.pdf` 의 아이디어:

- **voxel 해상도를 지역 TSDF 변이에 맞춰 적응**시킨다.
- 변이가 큰 곳(엣지·디테일) → **fine 유지**. 변이가 작은 곳(평평한 면) → **coarse로 병합**.
- 전략: **"가장 미세하게 시작해서, 저변이 voxel을 병합(start finest, merge low-variance)".**

**DirectionalTSDF와 직교(orthogonal):**

| | 무엇을 적응시키나 | 메모리 효과 |
|---|---|---|
| DirectionalTSDF | **방향(orientation)** | 날카로운 피처에 레이어 추가 → 더 씀 |
| Variance-Adaptive | **해상도(resolution)** | 평평한 곳 병합 → 덜 씀 |

두 축이 독립이라 **결합("방향 × 해상도")** 이 자연스러운 다음 수순 → 이게 아래 §4의 combine.

---

## 2. 적용 ① — 변이 측정 원시값 (foundation)

**어디에:** `src/shader/voxel_tsdf_integrate.comp` (SimpleTSDF 통합 커널)

`TSDFEntry` 의 미사용 `pad` 4바이트를 **`sumD2`(= Σ d²·w) 로 용도 변경**해서, 통합 중에 온라인으로 2차 모멘트를 쌓습니다:

```glsl
struct TSDFEntry { uint key; int sumDW; uint sumW; uint sumD2; };   // pad -> sumD2
...
atomicAdd(g_hash[slot].sumDW, int (sdf * w       * TSDF_SCALE));  // Σ d·w
atomicAdd(g_hash[slot].sumW,  uint(w             * TSDF_SCALE));  // Σ w
atomicAdd(g_hash[slot].sumD2, uint(sdf * sdf * w * TSDF_SCALE));  // Σ d²·w   ← 변이용
```

voxel별 **분산**은 병렬 아토믹 누적으로부터(온라인, Welford 아님):

$$
\sigma^2 = \mathbb{E}[d^2] - \big(\mathbb{E}[d]\big)^2
        = \frac{\text{sumD2}}{\text{sumW}} - \left(\frac{\text{sumDW}}{\text{sumW}}\right)^2
$$

**노출 API:** `SimpleTSDF::DownloadVoxels()` → `VoxelStat { center, tsdf, weight, variance }`
(`src/Engine/Spatial/SimpleTSDF.h:23,33,84`)

> 이게 개념의 **측정 프리미티브**입니다 — 여기서 나온 voxel별 $\sigma^2$ 가 이후 모든 적응의 근거.

---

## 3. 적용 ② — "변이가 엣지에 몰리는가" 검증

**어디에:**
- `test/test_simpletsdf_variance.cpp` — 단위 검증
- `example2/variance_adaptive_demo.cpp` — 시각/수치 데모

**측정 결과:**
- 실린더: 엣지/곡면 변이비 **5.25×** (엣지가 확실히 높음)
- 병합 가능(저변이) voxel 비율 **30–75%**

**⚠️ 정직한 CAVEAT (검증에서 드러난 한계):**
- 변이 신호는 **fine voxel에서만 신뢰**할 수 있음. 큐브 엣지/평면 분리비가
  fine에서 **15.7×** 지만, voxel 0.1에선 약해지고(**1.83×**), 0.2에선 **역전**됨.
- 원인: SimpleTSDF의 projective·normal-free 통합 아티팩트(이전에 짚은 그 편향).
- 다행히 이건 논문의 **"가장 미세하게 시작" 레짐과 정확히 일치** — 개념이 유효한 구간에서 쓰면 됨.

---

## 4. 적용 ③ — Compact-Directional × Variance-Adaptive 결합 (측정된 프론티어)

**어디에:** `example2/tsdf_benchmark.cpp`
- `RunVarianceAdaptiveSweep` (라인 ~339) — Simple 2-레벨 기준선 스윕
- `EvalCompactAdaptiveSigma` (라인 ~697) / `RunCompactVarianceAdaptive` (라인 ~768) — **결합 본체**
- `CompactDirectionalTSDF::DownloadEntries()` → `CompactEntry` 로 fine 후보를 내려받아 조립

**메커니즘:**
1. Compact-Directional 을 **fine 해상도**로 통합 → `DownloadEntries()` 로 fine 후보 획득
2. coarse 셀마다 **변이 백분위(percentile)** 로 판정: 고변이 → fine 유지, 저변이 → coarse로 병합
3. `CompactDirectionalTSDF::MergeCandidates` 로 fine↔coarse 경계 이음새(seam) 봉합

**측정 결과 (합성 GT):**

| | RMSE | 메모리 |
|---|---|---|
| Directional | 0.016–0.023 | 6144–6784 KB |
| **Compact × variance-adaptive** | **0.016–0.023 (동일)** | **148–153 KB** |
| (참고) Simple | — | 192–193 KB |
| 실린더 aggressive σ | 0.019 | **66 KB** |

⇒ **Directional 정확도를 그대로 유지하면서 Simple보다도 낮은 메모리, Directional 대비 ~40–44× 절감.**

---

## 5. 현재 성숙도 / 한계 (중요)

`SimpleTSDF.h:83` 주석 그대로: **"no multi-resolution storage yet."**

즉 이 개념은 현재:

| 단계 | 상태 |
|---|---|
| (a) 변이 측정 원시값 | ✅ 통합 커널에 상시 내장 |
| (b) "엣지=고변이" 검증 | ✅ 단위테스트+데모로 증명 |
| (c) fine/coarse 선택 + 프론티어 측정 | ✅ 벤치마크에서 **오프라인 조립**으로 측정 |
| (d) 런타임 멀티해상도 할당기(octree/multi-res hash) | ❌ **아직 없음** |

→ 개념은 **"적용되어 이득이 측정으로 증명"** 된 상태지만, 라이브 파이프라인에서 voxel을
동적으로 병합/분할하는 **프로덕션 자료구조는 미구현**. 실제 적응은 벤치마크가
`DownloadEntries()` 로 fine 후보를 받아 CPU에서 변이 기준으로 fine/coarse를 **선택 조립**하는
방식(오프라인 분석)입니다.

---

## 6. 적용 지점 지도 (한눈에)

| 단계 | 파일 / 심볼 | 역할 |
|---|---|---|
| 측정 | `src/shader/voxel_tsdf_integrate.comp` — `TSDFEntry.sumD2` | 온라인 Σd²·w 누적 |
| 노출 | `SimpleTSDF::DownloadVoxels()` → `VoxelStat.variance` | voxel별 σ² 회수 |
| 검증 | `test/test_simpletsdf_variance.cpp`, `example2/variance_adaptive_demo.cpp` | 엣지=고변이 증명 |
| 결합 | `example2/tsdf_benchmark.cpp` — `RunCompactVarianceAdaptive`, `EvalCompactAdaptiveSigma` | Compact×해상도 프론티어 |
| 조립 | `CompactDirectionalTSDF::DownloadEntries()` + `MergeCandidates` | fine 후보 + seam 봉합 |
| 분석 | `docs/MRHASH_VS_DIRECTIONAL_TSDF.md` | 논문 분석 + 전체 비교 |

---

## 7. 다음 단계 (원한다면)
개념을 (d) **프로덕션화**하려면: 런타임 2-레벨(또는 octree) 저장 — fine 해시 + coarse 해시를
변이 임계로 동적 배정하고, 추출 시 레벨 경계를 `MergeCandidates`로 봉합. 지금은 그 이득이
이미 측정돼 있으니, 남은 건 오프라인 선택을 **온라인 할당기**로 옮기는 엔지니어링.

---

## 참조
- 논문: `docs/Variance-Adaptive Voxel Grids.pdf`
- 분석/비교: `docs/MRHASH_VS_DIRECTIONAL_TSDF.md`
- 관련: `docs/COMPACT_VS_DIRECTIONAL_TSDF.md`, `docs/DIRECTIONAL_TSDF_INTEGRATION.md`
- 구현: `src/shader/voxel_tsdf_integrate.comp`, `src/Engine/Spatial/SimpleTSDF.{h,cpp}`, `example2/tsdf_benchmark.cpp`
