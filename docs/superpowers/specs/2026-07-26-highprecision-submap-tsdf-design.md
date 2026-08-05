# 고정밀(~1mm RMSE) 형상 복원 + 고정 VRAM submap 스트리밍 — 설계

> **한 줄 요약:** voxel을 줄이지 않고 **Gradient-SDF(거리 + 저장 gradient)** 로 sub-voxel 정확도를 ~1mm RMSE까지 끌어올리고, GPU 메모리는 **고정 VRAM pool에 active submap만 LRU 상주**(host RAM 권위)시켜 장면 크기와 무관하게 상한을 건다.
> **핵심 시너지:** Gradient-SDF는 추출이 **voxel-local**(이웃 불필요)이 되어 정밀도와 submap 경계(seam) 문제를 **동시에** 해결한다.

> 상태: 설계(브레인스토밍 승인 완료). 다음 단계: writing-plans로 구현 계획화.
> 수식은 GitHub/마크다운 뷰어에서 렌더됩니다.

---

## 0. 배경 · 목표 · 확정사항

### 0.1 현재 구조 (출발점)
- **CompactDirectionalTSDF** (`src/Engine/Spatial/CompactDirectionalTSDF.{h,cpp}`, `src/shader/compact_directional_{integrate,extract}.comp`): GPU(Vulkan) 희소 해시. 엔트리 `DirEntry{ key; sumDW; sumW; pad }` = 16 B. 32-bit 키 → 이동식 $512^3$ 창. 값 = 가중평균 **투영** 부호거리. 추출 = 축 zero-crossing + 중앙차분 법선 → oriented point cloud.
- **TiledCompactDirectionalTSDF** (`.../TiledCompactDirectionalTSDF.{h,cpp}`): 448³ core + ghost 타일로 공간 분할. **타일마다 영구 GPU 인스턴스** → touched 타일 전부 상주(= submap 미적용).
- **Residency 인프라** (`IResidencyBackend`, `StreamingResidencyBackend`, `UnifiedResidencyBackend`): host 권위 저장소 위에 고정 GPU pool을 두고 classify/ensure/write-back/evict를 GPU로 구동하는 **out-of-core 스트리밍이 이미 존재**. 단 **block 기반 DirectionalTSDF 경로 전용**(compact 해시 경로엔 미적용).

### 0.2 목표
- **G1 (정밀도):** voxel을 ~2.37mm로 유지하면서 재구성 **RMSE ≤ 1mm** (현재 2.31mm → sub-voxel). 방법 = Gradient-SDF 기반 sub-voxel 정제.
- **G2 (메모리):** GPU 메모리를 **고정 VRAM 예산**으로 상한. 장면 크기와 무관하게 GPU엔 active submap만 LRU 상주, host RAM이 권위 저장소.

### 0.3 비목표 (YAGNI)
- 1mm **voxel 격자**로의 축소(메모리 폭증 경로) — 채택 안 함(정밀도는 sub-voxel로 확보).
- **디스크 백업/영속성** — host RAM으로 충분(사용자 확정: "고정 VRAM 예산 경계").
- CPU 포팅 — 이 repo는 GPU 유지.

### 0.4 근거가 된 선행 결과
- **DB-TSDF 대조** ([`../../DB_TSDF_VS_COMPACT_DIRECTIONAL.md`](../../DB_TSDF_VS_COMPACT_DIRECTIONAL.md)): 투영거리의 grazing 편향, 방향 레이어링의 가치.
- **subvoxel-extraction pivot** (`2026-07-22-directional-tsdf-subvoxel-extraction-design.md`): **경량 FD-gradient 추출 정제는 flat/curved에서 회귀**(측정)하고 법선을 못 고침 → **저장형 Gradient-SDF로 pivot**, 재사용 oracle+harness 확보. 본 설계는 그 결론을 잇는다.

---

## 1. 정밀도 축 — Gradient-SDF sub-voxel

### 1.1 현재 추출의 세 한계
1. **축 방향 zero-crossing 선형보간** → 표면점이 voxel 축 엣지에 갇힘(비축 방향 표면에 부정확).
2. **중앙차분 법선** → 이웃 필요·양자화·노이즈, 경계에서 오염.
3. **투영(레이) 부호거리** → grazing 각에서 등가면이 실제 표면에서 밀림.

### 1.2 핵심: Gradient-SDF (Sommer et al., CVPR 2022)
각 (voxel, 방향) 엔트리에 스칼라 거리 $d$ **와 SDF gradient $\mathbf g$** 를 함께 저장. 그러면:

- **해석적 표면점** (축 제약·이웃 없음):
$$
\mathbf p^\* \;=\; \mathbf x_v \;-\; (d\cdot\tau)\,\hat{\mathbf g},\qquad \hat{\mathbf g}=\mathbf g/\lVert\mathbf g\rVert
$$
($d\in[-1,1]$은 저장된 정규화 TSDF값, $\tau$=truncation → $d\tau$가 미터 부호거리. voxel 중심을 gradient 반대 방향으로 표면에 투영.)
- **법선 = 저장된 $\hat{\mathbf g}$** (재추정 X) → coarse voxel·경계에서도 깨끗.

이는 한계 ①(축 제약)·②(법선)를 동시에 제거한다.

### 1.3 엔트리 레이아웃 — active(상주) vs at-rest(host) 이원화

gradient 누적엔 atomic이 필요해 상주 엔트리가 커진다. 저장 시엔 finalize해 압축한다:

| 상태 | 레이아웃 | 크기 | 용도 |
|---|---|---|---|
| **Active** (GPU pool) | `{ key(u32), sumDW(i32), sumW(u32), sumGx(i32), sumGy(i32), sumGz(i32) }` | **28 B** | `atomicAdd`로 $\sum w\,\psi$, $\sum w$, $\sum w\,\mathbf n$ 누적 |
| **At-rest** (host store, evict 시 finalize) | `{ key(u32), d(i16), n_oct(i16×2) }` | **~10 B** | oct-encoded 단위 법선 + 양자화 거리 |

**Rehydrate**(재활성): 저장된 $(d, \mathbf n)$ 와 시드 가중치 $w_0$로 accumulator 복원 — $\text{sumW}\leftarrow w_0$, $\text{sumDW}\leftarrow d\,w_0$, $\text{sumG}\leftarrow \mathbf n\,w_0$. → **상주는 정밀(28 B), 저장은 압축(~10 B)** = submap 예산과 직결.

### 1.4 통합 변경 (`compact_directional_integrate.comp`)
1. **gradient 누적:** `atomicAdd(sumG, w * n)` — $\mathbf n$은 이미 업로드되는 점 법선(추가 입력 0). 부호는 SDF 증가(외향)에 맞춤.
2. **point-to-plane 부호거리:** 기존 투영 $\psi=\text{clamp}((\mathbf p{-}\mathbf x_v)\cdot\mathbf r/\tau,\pm1)$ → **$\psi=\text{clamp}((\mathbf p{-}\mathbf x_v)\cdot\mathbf n/\tau,\pm1)$**. grazing 편향 제거(한계 ③). $\mathbf n$을 쓰므로 gradient 누적과 일관.

### 1.5 추출 변경 (`compact_directional_extract.comp`) — voxel-local
- 리팩터된 `estimateCrossingPosition` / `estimateNormal`(이웃 프로브) **제거**.
- 엔트리별: `d = sumDW/sumW`, `g = normalize(sumG)`, `p* = x_v − d·τ·g`, `normal = g` 를 바로 emit.
- **이웃 fetch 0** → §3 시너지의 근거. `MergeCandidates`는 유지(중복 dedup + 코너 보존).

### 1.6 왜 FD-lite를 다시 안 쓰는가
경량 유한차분 추출 정제는 선행 실험에서 flat/curved 회귀 + 법선 개선 불가로 이미 기각됨(0.4). **저장형** gradient만이 위치·법선을 동시에 해결한다.

### 1.7 선택적 품질 노브 (문서화만, 기본 off)
- **국소 최소자승 정제:** 후보 이웃에 평면/이차 fit → 매끄러운 면 denoise + 곡률. 이웃 접근 필요 → §2.3 브릭 모델에서 저렴(브릭 내 dense).
- **후단 표면화:** oriented point → **Screened Poisson**(Kazhdan & Hoppe 2013) 또는 **Dual Contouring**(Ju et al. 2002, Hermite=위치+법선 소비 → Gradient-SDF와 궁합). watertight mesh가 필요할 때.

---

## 2. submap 축 — 고정 VRAM 스트리밍

### 2.1 원칙
- **host RAM = 권위 저장소** (모든 submap의 finalize된 at-rest 엔트리 보관).
- **GPU = 고정 pool** (VRAM 예산). active submap만 상주, 초과 시 LRU evict(→ host write-back).
- 기존 `StreamingResidencyBackend`의 **Begin/Ensure/End** 생명주기 철학을 compact 경로로 이식.

### 2.2 Phase A — 타일 단위 LRU GPU pool (근시일, 최소 변경)
기존 타일 분할(448³ core + ghost)을 submap 단위로 재사용.

- **`TileResidencyPool`:** 고정 $K$개 GPU 해시 버퍼(= VRAM 예산). 슬롯 ↔ TileKey 매핑, LRU 큐.
- **`HostTileStore`:** `map<TileKey, vector<at-rest 엔트리>>` (host 권위).
- **생명주기 (프레임당):**
  1. `BeginFrame`: 들어온 점 → touched TileKey 집합(기존 ghost 라우팅 재사용).
  2. `EnsureResident(touched)`: 미상주 타일마다 — LRU evict(상주 타일 download→finalize→host write-back), 빈 슬롯에 대상 타일 host 엔트리 upload→rehydrate.
  3. `Integrate`: 상주 타일에 기존 커널 dispatch.
  4. `EndFrame`: dirty 타일 표시(다음 evict/추출 때 write-back).
- **Extract-on-evict:** Gradient-SDF 추출이 voxel-local이므로 **타일이 download되는 순간 바로 추출**(이웃 대기 불필요) → 전체 재상주 없이 스트리밍 추출.
- `IResidencyBackend`와 **평행한 인터페이스**로 두어 향후 통합 여지 확보.
- **경계:** VRAM $= K \times$ (per-tile 해시 바이트). $K$·per-tile 해시 크기 튜너블.

### 2.3 Phase B — 희소 브릭 voxel-hashing (타깃)
평면 해시를 **브릭 격자**로 교체(Nießner et al., SIGGRAPH Asia 2013; NanoVDB 계열).

- 표면 근처 **$B^3$ 브릭**(예 $B{=}8$)만 할당. top-level 해시: 브릭좌표 → 브릭 pool slot.
- **GPU = 고정 브릭 pool**(VRAM 예산), host = 전체 브릭. LRU 페이징 입도 = **브릭**(타일보다 세밀 → 표면 브릭만 전송).
- 브릭 내 **dense·암묵 주소**(per-voxel key 제거) → 메모리↓ + coalesced 접근 → §1.7 국소 LS가 브릭당 **1-voxel apron**으로 저렴.
- **방향 레이어 저장 = open question(§6):** 저장 gradient가 법선을 주지만 반대면 상쇄 방지(anti-aliasing)는 못 하므로 **축소된 방향 슬롯**을 브릭 voxel마다 유지할지 결정 필요.
- Phase A의 residency 생명주기·Gradient-SDF 수학을 **승계**, 저장 기질만 교체.

### 2.4 대안 노트 — 기존 backend 일반화 (2)
`IResidencyBackend`의 group payload를 compact/브릭 표현으로 일반화해 index-grid+pool+GPU 커널을 재사용하는 경로. 장점: 경로 통합·성숙 코드 재사용. 단점: backend가 512-voxel block group 모델이라 이식 수술 큼. **Phase B 진행 시 재사용 대상 후보로만** 기록(당장 채택 X).

---

## 3. 시너지 — Gradient-SDF ⇒ voxel-local ⇒ seam-free submap

정밀도 축과 submap 축이 **한 방향으로 맞물린다:**

- 현재 중앙차분 법선은 **이웃 의존** → submap/타일 경계에서 이웃이 다른 submap에 있으면 법선 오염(seam).
- Gradient-SDF 추출은 $\mathbf p^\*=\mathbf x_v-d\tau\hat{\mathbf g}$, `normal=g`로 **자기 voxel만** 참조 → **경계 seam 소멸**, extract-on-evict 가능.
- 즉 §1의 정밀도 개선이 §2의 스트리밍을 **더 쉽게** 만든다(추출 시 halo 불필요; 통합 밴드에만 ghost 필요).
- **대가:** 엔트리 +75%(16→28 B 상주) → pool당 voxel↓. **at-rest oct 양자화(§1.3)** 로 저장측 상쇄, 상주측은 pool 크기 튜닝으로 관리.

---

## 4. 메모리 · 예산 math

기호: $V$ = VRAM 예산(B), 상주 엔트리 28 B, at-rest 10 B(§1.3), $N_{occ}$ = 전체 점유 (voxel,dir) 수.

- **상주 상한:** $P_{\max} = V / 28$. 예: $V=512\,\text{MB}\Rightarrow P_{\max}\approx 19.2\text{M}$ 엔트리.
- **host 저장:** $N_{occ}\times 10\,\text{B}$. 예: $N_{occ}=100\text{M}\Rightarrow 1.0\,\text{GB}$(host RAM 여유).
- **Phase A per-tile:** per-tile 해시 = `hashCapacityPerTile`(기본 $2^{22}$ slot). 스트리밍용으론 **점유×(1/load-factor)** 로 축소 권장(기본값 그대로면 슬롯당 28 B로 타일 하나 ≈ 117 MB → $K$ 작아짐). $K\ge$ (프레임당 touched 타일 수) 여야 thrash 회피.
- **Phase B per-brick:** $B^3\times$ (voxel당 바이트) — 브릭 점유가 낮아도 낭비가 512-block보다 훨씬 작고 locality 이득.
- **프레임 전송:** (미상주 touched submap 수) × (submap 바이트). extract-on-evict로 추출 전송을 통합 전송에 흡수.

---

## 5. 검증 계획

- **oracle 재사용** (subvoxel pivot 자산): cube/plane/sphere 해석 정답. 추출점 vs 해석 표면 최근접 거리로 **RMSE / max error** 측정.
- **정밀도 게이트:** voxel ~2.37mm에서 **RMSE ≤ 1mm**, 특히 (a) 비축 방향 평면, (b) 곡면(sphere), (c) 코너에서 현재 대비 개선 확인. point-to-plane on/off, gradient 양자화(oct16) on/off ablation.
- **submap 등가성:** 동일 입력에 대해 **단일-창(all-resident) vs Phase-A(스트리밍)** 추출 결과가 (경계 포함) 수치적으로 일치 — seam 없음 증명. LRU $K$를 최소로 낮춰 thrash 상황에서도 결과 불변(메타모픽 테스트).
- **예산 준수:** 상주 엔트리 수 ≤ $P_{\max}$ 를 런타임 카운터로 assert.

---

## 6. 로드맵 · Open questions

**단계:**
1. **P0 정밀도:** 엔트리 28 B 확장 + 통합(sumG, point-to-plane) + voxel-local 추출 + oracle 게이트. (submap 없이 단일 창에서 RMSE≤1mm 먼저 확보.)
2. **P1 Phase A:** `TileResidencyPool` + `HostTileStore` + LRU + extract-on-evict. 등가성/예산 테스트.
3. **P2 Phase B:** 브릭 voxel-hashing으로 저장 기질 교체(생명주기·수학 승계). 국소 LS 정제 활성화.

**Open questions:**
- (Q1) 브릭 모델에서 방향 레이어를 몇 슬롯 유지? 저장 gradient가 6축 분리를 어디까지 대체하나(반대면 상쇄는 여전히 분리 필요).
- (Q2) gradient oct16 양자화가 RMSE에 미치는 영향(≤1mm 유지되나).
- (Q3) 타일 vs 브릭 입도의 thrash 교차점(장면·스캔 패턴별).
- (Q4) 추출 캐던스: extract-on-evict(스트리밍) vs on-demand full-sweep — 뷰어/파이프라인 요구에 따라.
- (Q5) at-rest rehydrate 시드 가중치 $w_0$ 선택이 재통합 정확도에 주는 영향.

---

## 7. 참조
- **Gradient-SDF:** C. Sommer, L. Sang, D. Schubert, D. Cremers, "Gradient-SDF: A Semi-Implicit Surface Representation for 3D Reconstruction," CVPR 2022.
- **Voxel hashing:** M. Nießner, M. Zollhöfer, S. Izadi, M. Stamminger, "Real-time 3D Reconstruction at Scale using Voxel Hashing," SIGGRAPH Asia 2013.
- **NanoVDB/OpenVDB:** K. Museth, "VDB: High-resolution sparse volumes with dynamic topology," ACM TOG 2013.
- **Dual Contouring:** T. Ju, F. Losasso, S. Schaefer, J. Warren, "Dual Contouring of Hermite Data," SIGGRAPH 2002.
- **Screened Poisson:** M. Kazhdan, H. Hoppe, "Screened Poisson Surface Reconstruction," ACM TOG 2013.
- **DirectionalTSDF:** M. Splietker, S. Behnke, "Directional TSDF," 2019.
- **내부:** [`../../DB_TSDF_VS_COMPACT_DIRECTIONAL.md`](../../DB_TSDF_VS_COMPACT_DIRECTIONAL.md), [`../../COMPACT_VS_DIRECTIONAL_TSDF.md`](../../COMPACT_VS_DIRECTIONAL_TSDF.md), `2026-07-22-directional-tsdf-subvoxel-extraction-design.md`, `2026-07-19-crossplatform-directional-tsdf-uma-design.md`.
- **구현:** `src/Engine/Spatial/{CompactDirectionalTSDF,TiledCompactDirectionalTSDF,StreamingResidencyBackend,IResidencyBackend}.*`, `src/shader/compact_directional_{integrate,extract}.comp`.
