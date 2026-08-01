# 11. MISO — 지도를 "조각(submap)"으로 나눠 빠르고 일관되게 (심화)
**논문:** MISO (Tao et al., 2025) · arXiv:2504.19104
**PDF:** `../11_MISO_MultiresSubmap_2025.pdf`

## 🧒 한 줄 요약
큰 씬을 **조각지도(submap)** 로 나눠 만들고, 조각들을 **특징 공간에서 맞춰** 이어붙여요. 미리 배운 지식으로 **초기값을 똑똑하게** 넣어 훨씬 빨리 수렴.

---

## 1. 조각별 다중해상도 SDF (디코더 고정)
$$h(\mathbf x;F,\theta)=D_\theta\Big(\bigoplus_{l\in[L]} f_l(\mathbf x)\Big),\qquad f_l(\mathbf x)=\sum_{i\in I_l} k_l(\mathbf x,\mathbf z_{l,i})\,\mathbf f_{l,i}.$$
- $F=\{\mathbf f_{l,i}\}$: 여러 해상도 $l$ 격자특징(**세션마다 새로 학습**), $D_\theta$: 여러 씬으로 **미리 학습해 고정**한 디코더.
- 디코더를 고정하면 온라인 문제가 **특징 $F$ 만의 최소자승**이 되어 잘 조건화·빠름.

> 🧒 "붓질 규칙(디코더)"은 미리 배워 고정, 매 조각에선 **색칠(특징)만** 새로 배워 빨라요.

## 2. 계층적 초기화 (핵심 속도 비결)

### 2.1 최적 초기화 (선형 최소자승)
레벨 $l$ 특징을, 거친 층들까지의 **잔차** $r_{1:l-1}(\mathbf x)$ 를 없애도록 초기화:
$$F_l^\star=\mathcal E\big(r_{1:l-1}(\mathbf x)\big):=-\big[J^\top J\big]^{\dagger}J^\top r_{1:l-1}(\mathbf x),$$
$J=\partial(\text{모델출력})/\partial F_l$ 는 자코비안, $[\cdot]^\dagger$ 유사역행렬. 이는 "잔차를 1스텝 Gauss–Newton으로 지우는" **정확한 초기값**.

### 2.2 학습된 근사 (비선형 일반화)
매번 $[J^\top J]^\dagger$ 를 푸는 대신, 잔차→좋은 초기특징을 뱉는 인코더 $\mathcal E_{\phi_l}$ 를 미리 학습:
$$F_l^\star\approx \mathcal E_{\phi_l}\big(r_{1:l-1}(\mathbf x)\big).$$
**효과:** 첫 시도부터 거의 정답에 가까워 Adam/GN 반복 수가 급감(scratch 초기화 대비 큰 가속).

> 🧒 큰 붓으로 대충 칠하고(거친 층), **남은 빈틈**만 작은 붓으로 채워요. 어디를 채울지 **미리 배운 감**으로 첫 붓부터 거의 맞게 시작해요.

## 3. 특징 기반 전역 정렬 (메시 없이)
조각 $u,v$ 의 겹치는 격자정점에서 **특징이 같아지도록** 조각자세 $\mathbf T_u^w$ 최적화:
$$c_l^{\text{feat}}(\mathbf T_u^w,\mathbf T_v^w)=\!\!\sum_{i\in I_l^{uv}}\!\! d\Big(f_{1:l}^u(\mathbf z_{l,i}^u),\ f_{1:l}^v\big({\mathbf T_v^w}^{-1}\mathbf T_u^w\,\mathbf z_{l,i}^u\big)\Big),$$
$$\min_{\{\mathbf T_u^w\}}\ \sum_{(u,v)\in\mathcal E} c_l^{\text{feat}}(\mathbf T_u^w,\mathbf T_v^w)+\sum_u \rho\big(\hat{\mathbf T}_u^w,\mathbf T_u^w\big),$$
거친 층 $l{=}1$ → 고운 층 $L$ 순차(coarse-to-fine)로 풀어 큰 초기오차에도 강건. 둘째 항은 오도메트리 사전(prior)에 대한 정칙화.

**왜 특징으로?** 정렬마다 메시를 뽑는 비용을 없애고(미분 가능·연속), 특징장은 부드러워 넓은 수렴 반경을 가짐.

## 4. 우리 프로젝트와의 관계
- "큰 씬을 조각으로 나눠 확장"은 우리 **`TiledAdvancedTSDF`(공간 타일링)** 과 같은 목표. MISO는 여기에 **전역 정렬(루프 보정)** + **학습된 초기화**를 더한 신경판.
- 우리가 미룬 **submap/streaming 로드맵**의 최전선 참고. §3의 "특징으로 정렬(메시 회피)"은, 우리 타일 경계 정합을 값/그래디언트로 맞추는 아이디어로 이식 가능.
