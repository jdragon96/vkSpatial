# Compact Best-TSDF v1 — 설계 (measure-first 통합)

> **한 줄 요약:** 측정으로 확정된 "최고의 TSDF" 레시피를 **CompactDirectionalTSDF(flat-hash, block 대비 ~28× 저메모리, RMSE 중립)** 위에 통합한다. v1 = **compact + stored-gradient(mode-3) + point-to-plane**. 이 셋만으로 *block DirectionalTSDF급 정확도를 ~28×↓ 메모리로* 얻는다. variance-adaptive·at-rest 양자화·full streaming은 후속.

> 상태: 설계(브레인스토밍 승인 — (C) compact-v1). 다음: writing-plans.
> 수식은 GitHub/마크다운 뷰어에서 렌더됩니다.

---

## 0. 근거 (이 세션의 측정)
tsdf_benchmark(cube/cylinder, analytic GT)로 측정된 사실:
- **저장:** block DirectionalTSDF 6144 KB(cube) vs **Compact-Directional 219 KB = ~28×↓**, RMSE 중립(0.0223 vs 0.0223). block-granularity 낭비 56×.
- **정확도(A1 point-to-plane, 이번 세션 측정):** 투영→point-to-plane 전환 시 **flat 거의 완벽**(cube 0.011→0.00001), **edge −15~32%**, **mean −45~62%**. 비용 0(integrate 시간 동일).
- **stored-gradient(mode-3):** block 경로에서 이미 머지·검증(denoised 법선 + 서브복셀 위치; cube 법선각 1.88°→0.00°).
- **결론:** compact(메모리) + stored-gradient(법선) + point-to-plane(flat/edge) = 최고 조합. 단 stored-gradient는 현재 **block 전용** → compact로 포팅이 v1의 핵심.

## 1. 목표 · 범위

### 1.1 목표
- CompactDirectionalTSDF를 **block DirectionalTSDF급 정확도**로 끌어올리되 **~28×↓ 메모리** 유지.
- extraction **RMSE 최소화**: point-to-plane(flat/edge) + stored-gradient mode-3(법선/서브복셀).

### 1.2 범위 (v1 Core)
1. **stored-gradient 포팅** (block→compact): `DirEntry` 확장 + integrate `sumN` 누적 + extract mode-3 hybrid.
2. **point-to-plane** integrate (이번 세션 검증) — 기본 on.
3. **검증**: tsdf_benchmark/tsdf_feature_compare로 v1 RMSE·메모리 A/B.

### 1.3 후속 (별도 spec, v1 아님)
- **variance-adaptive coarsening** 프로덕션화(벤치마크서 41×↓ 검증됨) — v1.5.
- **E2 at-rest 양자화**(oct16 법선 + fp16) 및 **full streaming**(host 권위 + GPU pool LRU) — **함께** 후속. 이유: 양자화는 host/at-rest 계층이 있어야 의미가 있는데, compact-v1은 GPU-resident(movable 512³ 창이 이미 bounded-VRAM). research spec `2026-07-26-highprecision-submap-tsdf-design.md`의 Phase A/B.

### 1.4 비목표
- 실시간 ICP(별도 sub-project), 메시(DC), occlusion/hit-counter(DB-TSDF).

---

## 2. 저장 구조 — `DirEntry` 확장 (16B → 24B)

현재(main): `DirEntry { uint key; int sumDW; uint sumW; uint pad; }` = 16B (`pad` 미사용).
**v1:** `pad`를 회수하고 gradient 누적 2필드 추가:
```glsl
struct DirEntry {
    uint key;    // packDirKey(voxel, dir)
    int  sumDW;  // Σ value·w·SCALE  (value = clamp(sdf/τ, -1, 1))
    uint sumW;   // Σ w·SCALE
    int  sumNx;  // Σ n·w·SCALE  (관측 법선; pad 자리 회수)
    int  sumNy;
    int  sumNz;
}; // 24B
```
- block의 `GpuTsdfVoxel`(20B, key 없음)과 동일 의미 + key 4B = 24B.
- 여전히 **block-with-gradient 대비 ~8×↓**(block은 512-voxel 그룹 낭비). C++ `struct DirEntry`/셰이더 미러/`static_assert(sizeof==24)`/`hashCapacity*24B` 할당 동기화.
- 셰이더 `DirEntry`가 쓰이는 곳: `compact_directional_{integrate,extract}.comp` 둘뿐(주소 계산 동일) → 둘 다 24B 통일.

---

## 3. Integrate — `compact_directional_integrate.comp`

기존 로직(topK 방향 선택 + 레이 마칭 + findOrInsert + atomicAdd sumDW/sumW)에 추가:

### 3.1 gradient 누적
`atomicAdd(sumDW/sumW)` 뒤에:
```glsl
atomicAdd(g_hash[slot].sumNx, int(nrm.x * w * TSDF_SCALE));
atomicAdd(g_hash[slot].sumNy, int(nrm.y * w * TSDF_SCALE));
atomicAdd(g_hash[slot].sumNz, int(nrm.z * w * TSDF_SCALE));
```
`nrm`은 이미 로컬에 있음(방향 선택·뷰가중에 사용). block 경로 `directional_tsdf_integrate.comp`와 동일.

### 3.2 point-to-plane (기본 on)
push-const `g_pointToPlane`(0/1) 추가. 현재 `sdf = depth − dot(vCenter − cam, rayDir)`(투영)를:
```glsl
float sdf = (g_pointToPlane != 0u)
                ? dot(vCenter - p, normalize(nrm))    // point-to-plane
                : depth - dot(vCenter - cam, rayDir); // projective (fallback)
```
부호는 투영과 일치(정면 = +). C++ `CompactDirectionalTSDF::SetPointToPlane(bool)`(기본 true) → integrate PC.
> block 경로에 이미 동일 토글을 넣어 A/B 측정 완료(이번 세션). 동일 변경을 compact에 이식.

---

## 4. Extract — `compact_directional_extract.comp` (mode-3 hybrid)

현재(이번 세션 리팩터): 축 zero-crossing 위치(`estimateCrossingPosition`) + **중앙차분 gradient 법선**(`estimateNormal`) → oriented point.
**v1:** 위치는 그대로(legacy zero-crossing = 최적), **법선을 저장 gradient로 교체**:
```glsl
// mode-3 hybrid: legacy zero-crossing position + stored-gradient normal
vec3 sumN = vec3(float(entry.sumNx), float(entry.sumNy), float(entry.sumNz));
float len = length(sumN);
if (len > 1e-6) surfaceNormal = sumN / len;      // denoised stored normal
else            /* fallback: central-difference gradient (estimateNormal) */;
```
- `estimateCrossingPosition`(위치)은 불변. `estimateNormal`(중앙차분)은 **fallback 전용**으로 남김(퇴화 gradient일 때).
- 이웃 프로브가 위치엔 여전히 필요(zero-crossing). 법선은 이제 voxel-local(저장 gradient).
- merge 패스 유지.

> block 경로의 검증 결과(memory): mode-2(center-projection 위치)는 위치를 ~10× 악화시켰고, **mode-3(legacy 위치 + stored 법선)이 best-of-both**. compact도 동일 채택.

---

## 5. C++ API — `CompactDirectionalTSDF`

- `struct DirEntry` 24B로 확장(+static_assert), 버퍼 할당 `hashCapacity*sizeof(DirEntry)`, `Reset()`의 empty 초기화 6필드로.
- `void SetPointToPlane(bool on)`(기본 true) + integrate PC 필드.
- 추출은 mode-3 고정(별도 모드 API 불필요 — compact는 mode-3만 지원). `DownloadEntries`가 gradient도 언팩(선택).
- `MergeCandidates` 불변.

---

## 6. 검증 (measure-first)

`tsdf_benchmark`(analytic cube/cylinder GT)의 **Compact-Directional** 행으로:
- **RMSE:** v1(compact + stored-gradient + p2p) 의 acc_rmse/edge/flat/curved 를 (a) 현재 compact(중앙차분·투영), (b) block DirectionalTSDF(mode-3+p2p) 와 비교.
- **게이트:** v1 RMSE ≤ block(mode-3+p2p) 수준, flat 거의 완벽(point-to-plane), edge 개선.
- **메모리:** v1 `N_occ·24B` vs block `groups·512·20B` → **~8~28×↓** 확인.
- **회귀:** point-to-plane off + 저장 gradient 미사용 시 기존 compact와 동일(토글로 A/B).
- **결정적:** cube 헤드라인(deterministic). cylinder는 다회 mean(base가 large-N fix 이전이면 지터).

---

## 7. 파일 구조 (writing-plans 입력)
- 수정: `src/Engine/Spatial/CompactDirectionalTSDF.{h,cpp}`(DirEntry 24B, SetPointToPlane, Reset, DownloadEntries), `src/shader/compact_directional_integrate.comp`(sumN 누적 + point-to-plane), `src/shader/compact_directional_extract.comp`(mode-3 법선), `example2/tsdf_benchmark.cpp`(compact 행에 p2p/gradient A/B 노출).
- 테스트: `test/test_compactDirectional.cpp`(24B layout, sumN 누적, mode-3 법선 정확도, point-to-plane 부호).
- 참조 재사용: block 경로의 stored-gradient(`directional_tsdf_{integrate,extract}.comp`, `2026-07-24-directional-tsdf-stored-gradient-design.md`)와 A1 토글(이번 세션 block 구현).

## 8. Open questions
- (Q1) point-to-plane 기본 on으로 두되, cylinder RMSE outlier 꼬리(이번 세션 발견)를 **robust weight(A2)**로 잡을지 v1에 포함 여부.
- (Q2) `DirEntry` 24B가 32-bit key의 movable 512³ 창 가정을 바꾸지 않음(불변) — 확인만.
- (Q3) 추출 위치도 저장 gradient로 서브복셀 정제(analytic projection) 재시도할지 — block에서 mode-2가 위치 악화였으므로 기본 제외.

## 9. 참조
- 이번 세션 측정: tsdf_benchmark A1 결과(flat/edge/mean), Compact-Directional 메모리(28×).
- 내부: `2026-07-26-highprecision-submap-tsdf-design.md`(후속 streaming/양자화), `2026-07-24-directional-tsdf-stored-gradient-design.md`(mode-3), `COMPACT_VS_DIRECTIONAL_TSDF.md`(28× 근거), 구현 `src/Engine/Spatial/CompactDirectionalTSDF.*`.

---

## 10. Compact Best-TSDF v1 — measured (2026-07-27)

Task 1-3(24B DirEntry + sumN + SetPointToPlane + mode-3 stored-gradient extract, 커밋 `1f2a336`..`1967667`) 이후, `tsdf_benchmark`의 Compact-Directional 행을 `--p2p`로 A/B 측정. `example2/tsdf_benchmark.cpp`에 `g_p2p`/`--p2p` 파싱과 `cd.SetPointToPlane(g_p2p)` 배선을 추가하고(§7 목록대로), 측정 도중 **`row.memKB`가 여전히 `FilledCount()*16.0`(구 16B pad-entry 시절 상수)로 하드코딩되어 있는 것을 발견 — Task 1이 `DirEntry`를 24B로 키웠지만 벤치마크는 갱신되지 않았음**. `sizeof(Engine::Spatial::DirEntry)`(24)로 고쳐서 측정(같은 파일, 같은 diff에 포함).

```
VULKAN_SDK=/usr/local cmake --build build --target tsdf_benchmark -j
./build/example2/tsdf_benchmark --shape both         # projective (default, g_p2p=false)
./build/example2/tsdf_benchmark --shape both --p2p    # point-to-plane
```

### 10.1 RMSE/edge/flat/curved — projective vs point-to-plane

| shape | metric | projective | p2p | Δ |
|---|---|---:|---:|---:|
| cube | acc_mean | 0.01626 | 0.00703 | −56.8% |
| cube | acc_rmse | 0.02233 | 0.01893 | −15.2% |
| cube | edge | 0.03077 | 0.02433 | −20.9% |
| cube | flat | 0.01073 | 0.00001 | −99.9% (거의 완벽) |
| cylinder | acc_mean | 0.01189 | 0.00373 | −68.6% |
| cylinder | acc_rmse | 0.01590 | 0.01141 | −28.2% |
| cylinder | edge | 0.01912 | 0.01239 | −35.2% |
| cylinder | flat | 0.01020 | 0.00001 | −99.9% (거의 완벽) |
| cylinder | curved | 0.01037 | 0.00289 | −72.1% |

패턴은 이번 세션의 block-path A1 결과(flat 거의 완벽, edge −15~32%, mean −45~62%)와 방향·규모 모두 일치 — cube edge(−20.9%)·mean(−56.8%)은 범위 내, cylinder edge(−35.2%)는 범위를 살짝 넘지만 같은 방향/성격. **compact 배선에 버그 징후 없음.**

### 10.2 메모리 — compact vs block

| shape | 모드 | compact mem_KB (24B/entry, 수정 후) | block(Directional) mem_KB | ratio |
|---|---|---:|---:|---:|
| cube | projective | 329.06 | 6144.00 | 18.67× |
| cube | p2p | 565.88 | 6144.00 | 10.86× |
| cylinder | projective | 411.14 | 6784.00 | 16.50× |
| cylinder | p2p | 621.84 | 6784.00 | 10.91× |

**"~28×" 재현 안 됨 — 두 가지 이유, 둘 다 실제 원인이 규명됨(버그 아님):**
1. **DirEntry 16B→24B (Task 1).** `~28×`는 §0/§9에 적힌 **구 16B pad-entry 기준**(219 KB, cube). 24B로 고쳐 재계산하면 `219.37 × 24/16 = 329.06` — 즉 28× → **18.67×**는 정확히 엔트리 크기 1.5배 성장의 산술적 결과(`28.01 × 16/24 = 18.67`, 실측과 소수점까지 일치). `sumN` 3필드를 되찾은 게 설계 의도(§2)였으므로 이 축소는 예상된 트레이드오프.
2. **point-to-plane이 integrate 시점 occupancy도 바꿈.** `compact_directional_integrate.comp:126-136`의 `g_pointToPlane` 토글은 extract 전용이 아니라 **트렁케이션 밴드 멤버십 테스트 자체**(`voxel2point = dot(vCenter-p, n)` vs `depth - dot(vCenter-cam, rayDir)`)를 바꾼다. Point-to-plane 밴드는 평면에 수직인 slab이라 레이-투영 밴드보다 비스듬한/edge 지오메트리 근처에서 더 많은 (voxel,dir)을 truncation 안으로 끌어들임 → `FilledCount()` 증가(cube +72.0%, cylinder +51.2%) → p2p 모드에서 ratio가 18.67×/16.50×에서 **10.86×/10.91×**로 추가 하락. 이 또한 설계된 SDF 정의 변경의 자연스러운 부작용이지 버그가 아님(§3.2에 명시된 push-constant가 정확히 이 위치에서 쓰임).

두 shape 모두, p2p에서도 **block 대비 10.9× 이상의 메모리 절감은 유지**됨.

### 10.3 stored-gradient normal — curved(cylinder) 판별

Task 3 이후 compact extract는 항상 mode-3(저장 gradient 법선)을 사용. Cylinder curved 열이 이 경로의 판별 지표:

| 경로 | curved_mm |
|---|---:|
| block Directional (projective, 이번 세션 유일하게 측정 가능한 기준) | 0.01211 |
| Compact-Directional, projective (mode-3, 이번 태스크 기준) | 0.01037 |
| Compact-Directional, p2p (mode-3) | 0.00289 |

Compact의 저장-gradient 법선은 **projective 상태에서 이미 block 품질과 동급 이상**(0.01037 < 0.01211)이고, point-to-plane을 얹으면 block보다 **4배 이상 우수**(0.00289)해진다. → Task 3의 stored-gradient 이식은 curved 지오메트리에서 실패하지 않았고, block 수준(또는 그 이상)의 법선 품질을 compact 저장 구조에서 재현했다.

### 10.4 v1 게이트 판정 — **부분 충족 (DONE_WITH_CONCERNS)**

- ✅ **flat 거의 완벽:** 두 shape 모두 −99.9%(0.01→0.00001) — 명확히 충족.
- ✅ **RMSE 개선 방향/규모:** block-path A1 패턴과 일치, 회귀 없음 — compact 배선 버그 징후 없음.
- ✅ **RMSE ≤ block(projective):** compact p2p RMSE(cube 0.01893, cyl 0.01141) < block projective RMSE(cube 0.02223, cyl 0.02677) — 직접 측정으로 확인.
- ⚠️ **RMSE ≤ block(mode-3+p2p):** **이번 세션에 직접 재측정 불가.** `DirectionalTSDF::m_pointToPlane` 기본값이 `false`이고(A1 실험, uncommitted, 커밋 위생 규칙상 손대지 않음) `tsdf_benchmark.cpp`의 `RunDirectional`이 `dir.SetPointToPlane(...)`을 호출하지 않아 "Directional" 행은 이번 실행에서도 projective로 고정. block+p2p 자체 개선폭(−45~62% mean, 이번 세션 앞서 측정)을 적용해 역산하면 block+p2p RMSE ≈ cube 0.0085~0.0122 / cyl 0.0102~0.0147 — compact p2p(cube 0.01893, cyl 0.01141)는 cylinder에서는 이 추정 구간에 걸치지만 cube에서는 벗어난다(추정치보다 높음). **하드 데이터 아님 — 후속 태스크에서 block에 p2p를 배선해 직접 비교 필요.**
- ❌ **~28× 메모리:** **미충족.** 실측 18.67×(projective)/10.86×(p2p, cube), 16.50×/10.91×(cylinder) — §10.2에서 규명한 두 가지 실제 원인(24B 엔트리 성장 + p2p의 occupancy 증가) 때문이며 compact 구현 결함은 아니다. 그럼에도 block 대비 **10.9×~18.7×** 메모리 절감은 all-mode에서 유지.

**종합:** compact 배선에 버그는 없다(RMSE 패턴이 block-path A1과 정합, flat 거의 완벽, curved가 block 동급 이상). 다만 브리프가 기대한 "~28× 불변"은 성립하지 않으며 — 원인은 Task 1의 24B 성장(예견된 트레이드오프)과 point-to-plane이 occupancy에도 영향을 주는 설계상 특성(신규 발견) 둘 다 실제(genuine)다. block(mode-3+p2p) 대비 RMSE 비교는 block 쪽 p2p가 이번 세션에 배선되어 있지 않아 완전히 검증되지 않았다. v1을 "완료"로 승격하려면 (a) block에도 `SetPointToPlane`을 CLI로 노출해 직접 비교하거나, (b) §0/§9의 "~28×" 문구를 24B 실측치(18.7×/16.5× projective, ~10.9× p2p)로 갱신하는 후속 정리가 필요.
