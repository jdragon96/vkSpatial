# CompactDirectionalTSDF vs DirectionalTSDF — 무엇이 다른가

> 한 줄 요약: **둘은 완전히 똑같은 그림을 그린다. 다른 건 그 그림을 종이에 어떻게 담느냐(저장 방식)뿐이다.**
> DirectionalTSDF는 512칸짜리 상자 단위로 사고, CompactDirectionalTSDF는 필요한 칸만 낱개로 산다.
> 결과: 같은 정확도, **메모리 약 16배 절약**(실측, chair 스캔).

수식은 GitHub / 마크다운 뷰어에서 렌더됩니다. 터미널에서는 `$…$` 가 원문으로 보일 수 있습니다.

---

## 1. 초등학생을 위한 비유 — 색칠하기 🎨

그림의 **테두리(윤곽선)** 만 색칠한다고 하자. 종이는 아주 작은 네모칸(=voxel)으로 나뉘어 있다.
표면은 **얇은 껍질**이라 색칠할 칸은 전체의 아주 일부다.

두 방법 모두 **똑같은 칸에 똑같은 색을 칠한다.** 완성된 그림은 구별할 수 없다.
차이는 **네모칸을 사는 방법**이다:

### 🟦 DirectionalTSDF = "512칸 상자로만 판다"
칸을 낱개로 못 산다. **8×8×8 = 512칸이 든 상자**로만 판다.
한 칸이라도 칠하려면 상자를 통째로 뜯어야 한다.
테두리는 얇아서, 뜯은 상자 안 대부분의 칸은 **하얀 채로 버려진다.**

```
상자 하나 (8×8×8 = 512칸):
┌───────────────┐
│ ■ ■ · · · · · │   ■ = 실제 칠한 칸 (테두리)
│ · · ■ ■ · · · │   · = 상자라서 딸려온 빈 칸 (낭비)
│ · · · · ■ · · │
│ · · · · · · · │   → 512칸 중 실제로 쓰는 건 몇 칸뿐!
└───────────────┘
```

### 🟩 CompactDirectionalTSDF = "칠할 칸만 낱개로 산다"
스티커북처럼 **필요한 칸만 하나씩** 산다. 빈 칸을 살 일이 없다.
대신 낱개 스티커에는 "이건 어디 칸이야"라는 **작은 이름표(=key)** 가 붙어서, 스티커 한 장이 조금 더 크다.

```
낱개로 필요한 칸만:
■(3,7) ■(3,8) ■(4,9) ■(5,2) ...   ← 칠한 칸만, 각자 주소표 붙여서
```

**결론:** 스티커가 장당 조금 더 커도(이름표 때문), **빈 칸을 안 사니까** 전체 종이는 훨씬 적게 쓴다.

---

## 2. 똑같은 부분 (계산 = 100% 동일)

두 방식은 **적분(integration) 수학이 완전히 같다.** 자세한 식은
[`DIRECTIONAL_TSDF_INTEGRATION.md`](DIRECTIONAL_TSDF_INTEGRATION.md) 참고. 요약하면:

- 표면을 6개 부호축 레이어로 나눔 ($+X,-X,+Y,-Y,+Z,-Z$)
- 법선으로 방향 선택 (TopK): $\rho_i = (a_i/a_{\max})^{p_e}$
- 시야각 신뢰도 $\varphi = \max(0, n\cdot(-r))$
- 투영 부호거리 $\psi = \mathrm{clamp}\big((p-x_v)\cdot r / \tau,\ -1,1\big)$
- 가중 평균 $\Psi(v,d) = \dfrac{\sum \psi\,\varphi\,\rho}{\sum \varphi\,\rho}$

⇒ **각 (voxel, 방향) 칸에 들어가는 최종 숫자 $\Psi(v,d)$ 는 두 방식이 동일하다.**
그래서 그림(재구성된 표면)이 같다.

---

## 3. 다른 부분 — 저장 자료구조

핵심 정의: **$N_{occ}$ = 실제로 표면이 지나가는 (voxel, 방향) 칸의 개수.**
표면은 얇은 껍질이므로 이 값은 두 방식 모두 **똑같다** (아래 실측에서 정확히 일치).

### 🟦 DirectionalTSDF — 블록(상자) 해시

$8^3 = 512$ voxel 짜리 **블록** 단위로 메모리를 잡는다. 표면이 스치는 블록 개수를 $B$ 라 하면:

$$
M_{\text{dir}} = B \cdot 512 \cdot s
$$

- $s$ = 칸 하나당 바이트 (블록 안은 주소가 암묵적이라 **이름표 불필요**, 실측 $s = 8\,\text{B}$)
- 문제: 얇은 껍질이 뚱뚱한 상자를 지나가면 **상자 대부분이 빈다.**

**낭비 배수(waste factor):**

$$
\eta = \frac{B \cdot 512}{N_{occ}} \quad(\text{할당한 칸} \div \text{실제 쓴 칸})
$$

### 🟩 CompactDirectionalTSDF — 낱개 (voxel, 방향) 해시

칸 하나 = 해시 엔트리 하나. **실제 칸만** 저장:

$$
M_{\text{cmp}} = N_{occ} \cdot s'
$$

- $s'$ = 엔트리 하나당 바이트. 해시라서 **주소 이름표(key)를 명시 저장**해야 함
  → `DirEntry { uint key; int sumDW; uint sumW; uint pad; }` = $16\,\text{B}$
- 빈 칸 없음. 낭비 배수 $\approx 1$ (해시 load factor 여유만).

### 두 메모리의 비율 (수식으로)

$$
\frac{M_{\text{dir}}}{M_{\text{cmp}}}
= \frac{B\cdot512\cdot s}{N_{occ}\cdot s'}
= \frac{\eta \cdot s}{s'}
= \frac{\eta \cdot 8}{16}
= \frac{\eta}{2}
$$

> **핵심 통찰:** Compact은 칸당 바이트를 **2배**($8\to16$B, 이름표 값) 쓴다.
> 그런데 DirectionalTSDF의 낭비 $\eta$ 가 **수십 배**라서, 결국 $\eta/2$ 배만큼 이긴다.
> 즉 **"칸당 조금 손해, 빈 칸 안 사서 크게 이득"** 이 정확히 수식으로 드러난다.

---

## 4. 실측치 (chair 스캔, voxel 2.37mm, 12 프레임)

| 항목 | DirectionalTSDF 🟦 | CompactDirectionalTSDF 🟩 |
|---|---|---|
| 실제 표면 칸 $N_{occ}$ | 520,202 | **520,202** (정확히 같음) |
| 잡은 블록 $B$ | 33,084 | — (블록 안 씀) |
| 할당한 총 칸 $B\cdot512$ | 16,939,008 | 520,202 |
| **채움 비율** | $520202/16.9\text{M} \approx 3\%$ | $\approx 100\%$ |
| 칸당 바이트 $s$ | 8 B | 16 B |
| **메모리** | **132,336 KB** | **8,128 KB** |
| 정확도 (RMSE) | 2.58 mm | **2.31 mm** (더 좋음) |

**수치 검산:**

$$
\eta = \frac{33084 \cdot 512}{520202} = \frac{16{,}939{,}008}{520{,}202} \approx 32.6 \quad(\text{할당의 } 97\% \text{가 빈 칸})
$$

$$
M_{\text{cmp}} = 520{,}202 \times 16\,\text{B} = 8{,}323{,}232\,\text{B} \approx 8{,}128\,\text{KB} \ ✓
$$

$$
M_{\text{dir}} = 33{,}084 \times 4096\,\text{B} = 135{,}512{,}064\,\text{B} \approx 132{,}336\,\text{KB} \ ✓
$$

$$
\frac{M_{\text{dir}}}{M_{\text{cmp}}} = \frac{\eta}{2} = \frac{32.6}{2} \approx 16.3\times \ ✓
$$

> DirectionalTSDF가 **블록을 안 쓰고** 낱개로 저장했다면 이론상 최소
> $520202 \times 8\text{B} = 4{,}064\,\text{KB}$ 였을 것 — 블록 방식이 이걸 $32.6\times$ 부풀린 것.
> Compact은 이름표 때문에 그 이론 최소의 2배($8{,}128$KB)지만, 블록판보다는 $16.3\times$ 적다.

---

## 5. 대가(trade-off) — 크기 한계

공짜는 없다. Compact의 이름표(key)는 **32비트짜리**라 주소를 넣을 공간이 제한된다.

$$
\text{key} = \underbrace{(lv_x \ll 21) \mid (lv_y \ll 12) \mid (lv_z \ll 3)}_{\text{9비트/축} \Rightarrow lv \in [0,511]} \mid \underbrace{d}_{\text{3비트}}
$$

- 축당 9비트 ⇒ **한 번에 $512^3$ voxel 창(window)** 안에서만 동작 (원점 이동 가능 = *movable window*).
- 이 GPU엔 `shaderBufferInt64Atomics=false` ⇒ 64비트 이름표(더 큰 창)를 못 쓴다.
- **DirectionalTSDF의 블록 해시는 블록을 아무데나 둘 수 있어 크기 제한이 없다** (큰 장면/스트리밍에 유리).

> **큰 장면 해법:** `TiledCompactDirectionalTSDF` — 장면을 타일로 쪼개 각 타일에 $512^3$ 창을 두고
> ghost margin으로 이어붙임. chair를 1mm로 재구성(창 하나론 불가능한 827 voxel) 성공.

---

## 6. 정확도가 오히려 조금 더 좋은 이유

Compact이 2.31 vs 2.58로 약간 앞서는 건 **적분이 달라서가 아니라 추출(extraction) 때문**이다:
Compact은 낱개 voxel 후보를 그대로 뽑고, DirectionalTSDF는 블록 내부에서 한 번 병합(merge)한다.
(Compact도 `ExtractPointCloud(..., merge=true)` 로 병합 켜면 completeness가 Directional과 같아진다.)

---

## 7. 요약 표

| | DirectionalTSDF 🟦 | CompactDirectionalTSDF 🟩 |
|---|---|---|
| 적분 수학 | 동일 | 동일 |
| 저장 단위 | $8^3$ 블록(상자) | 낱개 (voxel,방향) 해시 |
| 빈 칸 낭비 | 큼 ($\eta \approx 33\times$) | 없음 |
| 칸당 바이트 | 8 B (이름표 불필요) | 16 B (이름표 포함) |
| 메모리 (chair) | 132 MB | **8 MB (16.3× 적음)** |
| 정확도 | 기준 | 같거나 약간 나음 |
| 장면 크기 | 제한 없음 (블록 자유 배치) | $512^3$ 창(이동식), 초과 시 타일링 |
| 적합 상황 | 매우 큰/스트리밍 장면 | 창에 들어오는 물체 (메모리 최소화) |

---

## 8. 한 줄 결론

> **같은 표면, 같은 정확도.** DirectionalTSDF는 "512칸 상자"로 사서 빈 칸을 잔뜩 버리고,
> CompactDirectionalTSDF는 "칠할 칸만 낱개로" 사서 이름표 값만 조금 더 내고 **메모리를 16배 아낀다.**
> 유일한 대가는 한 번에 다룰 수 있는 크기가 $512^3$ 창으로 제한된다는 것(타일링으로 해소).

---

## 참조
- 적분 수식: [`DIRECTIONAL_TSDF_INTEGRATION.md`](DIRECTIONAL_TSDF_INTEGRATION.md)
- 벤치마크/분석: [`MRHASH_VS_DIRECTIONAL_TSDF.md`](MRHASH_VS_DIRECTIONAL_TSDF.md)
- 구현: `src/TSDF/Backends/CompactDirectionalTSDF.{h,cpp}`, `src/TSDF/Backends/TiledCompactDirectionalTSDF.{h,cpp}`
- 셰이더: `src/shader/compact_directional_integrate.comp`, `src/shader/directional_tsdf_integrate.comp`
