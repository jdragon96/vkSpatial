# `TSDF` — 창 라우팅과 2레벨 구조

> `src/TSDF/TSDF.cpp`의 알고리즘, 그리고 코드를 읽고 찾은 논리적 결함 셋.
>
> 이 클래스는 **TSDF를 계산하지 않는다.** 계산은 `TSDFBackend`(현재 `AdvancedTSDF`)가 창 하나
> 안에서 한다. 이 클래스가 하는 일은 **어느 점을 어느 창에 넣을지 정하는 것**뿐이다.

---

## 1. 기호

| 기호 | 의미 | 코드 |
|---|---|---|
| $v$ | base 복셀 크기 (m) | `backendConfig.voxelSize` |
| $\tau$ | truncation 거리 (m) | `backendConfig.truncation` |
| $N$ | 창 한 변의 복셀 수 (512 고정) | `windowVoxels` |
| $W$ | 창 한 변의 월드 길이 | `m_baseWindowWorld` $= vN$ |
| $W_d$ | detail 창 한 변 | `m_detailWindowWorld` $= \tfrac{v}{2}N = \tfrac{W}{2}$ |
| $P_f$ | 프레임 $f$의 점 집합 | `points` |
| $c_f$ | 프레임 $f$의 카메라 위치 | `cameraPosition` |

detail 레벨은 복셀이 절반이므로 창이 덮는 월드 범위도 절반이다. **레벨마다 격자가 다르다.**

$$v_d = \tfrac{v}{2},\qquad \tau_d = \tfrac{\tau}{2},\qquad W_d = \tfrac{W}{2}$$

---

## 2. 한 프레임의 흐름

```
Integrate(P, N, c)
        │
        │  1. 분할            DataSplitter
        ├──────────────►  B ⊎ D = {0..|P|-1}      (base 인덱스 ⊎ detail 인덱스, 서로소)
        │
        │  2. base 레벨                        3. detail 레벨 (useSubmap 일 때)
        ▼                                     ▼
   IntegrateLevel(hashTSDF, W, B)        IntegrateLevel(hashTSDFDetail, W_d, D)
        │                                     │
        │  점 → 창 키로 버킷팅                 (동일)
        ▼
   k(p) = ⌊p / W⌋                        k_d(p) = ⌊p / W_d⌋
        │
        ▼
   버킷마다: 창을 찾거나 생성 → 그 점들만 backend->Integrate
```

### 2.1 분할

`DataSplitter::DividePoint`가 인덱스를 **서로소로** 나눈다.

$$B \cap D = \varnothing, \qquad B \cup D = \{0,\dots,|P|-1\}$$

`dense` 전략은 32³ 블록마다 네 조건을 AND해서 판정한다(자세한 것은
[`DENSE_REGION_SEPARATION.md`](DENSE_REGION_SEPARATION.md)). `none` 전략은 $D = \varnothing$.

### 2.2 창 키

$$k(p) = \left\lfloor \frac{p}{W} \right\rfloor \quad\text{(성분별)}$$

창 $k$가 덮는 월드 영역은

$$\Omega_k = [kW,\; (k+1)W)$$

창은 **첫 접촉 시 생성**되고, 원점은 $kW$로 백엔드에 전달된다(`windowMinCorner`).

### 2.3 백엔드가 실제로 쓰는 범위

백엔드는 점 $p$ 하나에 대해 반경 $\tau$의 밴드를 적분한다. 즉 점 하나가 건드리는 복셀 집합은

$$V(p) = \{\,x \in \mathbb{Z}^3 \;:\; \lVert xv - p \rVert_\infty \le \tau \,\}$$

그리고 커널은 창 로컬 좌표가 $[0, 511]$을 벗어나면 **그 복셀을 버린다**
(`kernel_AdvancedTSDF.integrate.comp.glsl`의 `packDirKey`가 `false` 반환).

---

## 3. 찾은 결함

### 결함 1 — 창 경계마다 두께 $\tau$의 이음매가 생긴다 (심각)

점은 **자기가 속한 창 하나**에만 전달된다.

$$p \;\longmapsto\; \text{창 } k(p) \text{ 하나}$$

그런데 점이 쓰는 밴드는 $\pm\tau$로 퍼진다. $p$가 경계에서 $\tau$ 이내면 밴드의 일부가 이웃 창
$\Omega_{k'}$에 속하는데, **그 창은 이 점을 받지 못한다.** 그리고 자기 창에서는 커널이
로컬 범위를 벗어난 복셀을 버린다.

```
        창 k                    │                 창 k+1
   ─────────────────────────────┼─────────────────────────────
                          p ●───┼───╳                       
                          │←τ→│←τ→│                          
                        기록됨   버려짐 (k 범위 밖)
                                 └── k+1 은 p 를 아예 못 봄
```

**결과**: 모든 창 경계에 두께 $\tau$의 껍질이 한쪽에서만 채워진다. 그 안의 복셀은 이웃 창에서
와야 할 관측을 영원히 못 받으므로 가중치가 낮고, 추출하면 이음매·구멍이 된다.

경계 면적 비율은
$$\frac{6 \cdot 2\tau \cdot W^2}{W^3} = \frac{12\tau}{W} = \frac{12\tau}{vN}$$
$\tau = 2v$, $N = 512$이면 약 **4.7%**. 창이 많아질수록 절대량이 늘어난다.

### 결함 2 — base 레벨에 dense 영역만큼 구멍이 뚫린다

분할이 **서로소**이므로 dense로 분류된 점은 base 맵에 들어가지 않는다.

$$\text{base 맵} \;\leftarrow\; \{p_i : i \in B\}, \qquad \text{detail 맵} \;\leftarrow\; \{p_i : i \in D\}$$

두 레벨은 겹치는 게 아니라 **타일처럼 나뉜다**. 파급이 둘 있다.

1. detail 창이 `maxResidentWindow`에 걸려 거부되면 그 점들은 **완전히 소실**된다 — base에도
   없기 때문이다. `WindowLimitRefusalCount`가 세기는 하지만 복구는 불가능하다.
2. `useSubmap`을 끄고 다시 빌드하면 dense 영역이 base에 없는 상태가 아니라 base에 들어가지만,
   이미 만들어진 맵을 base만으로 쓰는 경로(예: 추적기가 base만 참조)에서는 그 영역이 비어 있다.

원래 설계 의도는 `ModelSnapshot::entries`의 주석이 말하는 **"precedence-deduped base+detail"**,
즉 두 레벨이 같은 영역을 **겹쳐** 갖고 추출에서 detail을 우선하는 것이었다. 지금 구현은 타일링이다.

### 결함 3 — `Extract()`/`Download()`가 두 레벨을 중복 제거 없이 이어붙인다

```cpp
for (const WindowMap *level : {&hashTSDF, &hashTSDFDetail})
    for (...) cloud.points.insert(...);   // 그냥 concat
```

결함 2 때문에 지금은 두 레벨이 공간적으로 겹치지 않아 눈에 띄는 중복이 없다. 하지만 결함 2를
고쳐 겹치게 만드는 순간 **같은 표면이 두 번 나온다.** 두 결함은 반드시 함께 고쳐야 한다.

### 결함 4 (경미) — `maxResidentWindow`가 두 레벨을 합쳐서 센다

`WindowCount()`는 base + detail의 합이다. base 8개가 필요한 장면에서 상한을 8로 두면 detail은
하나도 못 만들고, 결함 2에 의해 그 점들은 소실된다. 레벨별 상한이 맞다.

### 결함 5 (경미) — `m_perWindowIndex`의 키가 영구히 쌓인다

매 프레임 벡터만 비우고 키는 남는다. 스캔이 넓어질수록 두 레벨 각각이 **여태 만난 모든 창**을
순회한다. 정확성 문제는 아니지만 프레임 비용이 단조 증가한다.

---

## 4. 결함 1의 수정 — apron 라우팅

점을 **밴드가 걸치는 모든 창**에 보낸다.

$$K(p) = \left\{\, k \in \mathbb{Z}^3 \;:\; \Omega_k \cap [\,p-\tau,\; p+\tau\,] \neq \varnothing \,\right\}
= \prod_{a \in \{x,y,z\}} \left[\left\lfloor \tfrac{p_a-\tau}{W} \right\rfloor,\; \left\lfloor \tfrac{p_a+\tau}{W} \right\rfloor\right]$$

$\tau \ll W$이므로 $|K(p)| = 1$이 압도적이고, 경계 근처에서만 2·4·8이 된다. 앞의 4.7% 추정이
그대로 추가 작업량의 상한이다.

**중복 적분이 아니다.** 각 창은 자기 로컬 범위 밖 복셀을 버리므로, 같은 점이 두 창에 전달돼도
각 복셀은 정확히 한 번 갱신된다. 창들의 복셀 집합이 서로소이기 때문이다.

```
        창 k                    │                 창 k+1
   ─────────────────────────────┼─────────────────────────────
                          p ●───┼───●     ← 두 창 모두 p 를 받는다
                          │←τ→│←τ→│
                        k 가 기록   k+1 이 기록
```
