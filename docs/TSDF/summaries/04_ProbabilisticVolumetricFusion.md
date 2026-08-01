# 04. Probabilistic Volumetric Fusion — 못 믿을 측정은 적게 반영 (심화)
**논문:** Probabilistic Volumetric Fusion for Dense Monocular SLAM (Rosinol & Carlone, 2022) · arXiv:2210.01276
**PDF:** `../04_ProbabilisticVolumetricFusion_MonoSLAM_2022.pdf`

## 🧒 한 줄 요약
카메라 한 대(단안) 깊이는 부정확해요. **"이 측정 얼마나 믿을 만해?"(분산)** 를 같이 계산해서, 못 믿을 값은 지도에 조금만 반영.

---

## 1. 왜 단안 깊이가 위험한가 — 삼각측량 오차 유도
스테레오/모션 스테레오에서 깊이 $d$ 는 시차(disparity) $\delta$ 로부터:
$$d=\frac{f\,b}{\delta}\quad(f=\text{초점거리},\ b=\text{베이스라인}).$$
$\delta$ 에 대해 미분하면 깊이 민감도:
$$\Big|\frac{\partial d}{\partial\delta}\Big|=\frac{f b}{\delta^2}=\frac{d^2}{f b}\ \Rightarrow\ \sigma_d\approx\frac{d^2}{f b}\,\sigma_\delta.$$
즉 **깊이 오차는 거리 제곱 $d^2$ 에 비례해 폭증**. 멀거나 텍스처 약한 곳(큰 $\sigma_\delta$)에서 특히 위험. 그냥 평균하면 이 노이즈가 표면을 망침.

## 2. 확률적 융합 = 가우시안 곱 (완전 유도)

### 2.1 두 측정의 융합
각 측정을 독립 가우시안: $d\sim\mathcal N(\hat d_1,\sigma_1^2)$, $d\sim\mathcal N(\hat d_2,\sigma_2^2)$. 사후분포 $\propto$ 곱:
$$\mathcal N(\hat d_1,\sigma_1^2)\cdot\mathcal N(\hat d_2,\sigma_2^2)\ \propto\ \mathcal N(\mu,\sigma^2),$$
지수부 $-\tfrac12\big[\tfrac{(d-\hat d_1)^2}{\sigma_1^2}+\tfrac{(d-\hat d_2)^2}{\sigma_2^2}\big]$ 를 $d$ 에 대해 완전제곱하면:
$$\boxed{\frac1{\sigma^2}=\frac1{\sigma_1^2}+\frac1{\sigma_2^2},\qquad \mu=\sigma^2\Big(\frac{\hat d_1}{\sigma_1^2}+\frac{\hat d_2}{\sigma_2^2}\Big)}$$
정밀도(=1/분산)는 **더해지고**, 평균은 **정밀도 가중평균**.

### 2.2 TSDF 가중치와의 연결
$w_k:=1/\sigma_k^2$ 로 두면 위 융합이 정확히 우리 TSDF 갱신식:
$$D=\frac{\sum_k w_k d_k}{\sum_k w_k},\quad W=\sum_k w_k=\frac1{\sigma^2}.$$
즉 **TSDF 가중치 = 측정 정밀도**. 이게 "믿을수록 크게"의 수학적 정체(01번 §0.2 MLE와 동일 결론).

> 🧒 자신 있어 하는 친구 말은 크게, 헷갈려 하는 친구 말은 작게 반영해 정답을 정하는 것. "자신감"이 바로 $1/\sigma^2$.

## 3. 불확실성 문턱값 (이상치 제거)
$$\sigma_d^2>\tau_\sigma\ \Rightarrow\ \text{융합에서 제외}.$$
가중치가 아주 작아도 수많이 쌓이면 표면을 흐릴 수 있어, 애초에 **버리는** 하드컷이 더 안전. §1에서 본 대로 먼 픽셀($d^2$ 큰)·저텍스처가 주 타깃.

## 4. 실전 파이프라인 (단안 SLAM용)
1. 단안 SLAM/MVS 프런트엔드가 픽셀별 깊이 $\hat d$ + **불확실성 $\sigma_d$** 를 산출.
2. 각 깊이를 §1의 $\sigma_d\propto d^2$ 모델로 가중, §3로 이상치 컷.
3. 남은 깊이를 §2.2 가중치로 TSDF에 융합 → 노이즈 있는 단안 깊이로도 안정적 밀집 지도.

## 5. 우리 프로젝트와의 관계
- 우리 `AdvancedTSDF`의 **A1(confidence weight, $w\mathrel{*}=1-\lambda|\text{tsdf}|$)** 과 같은 "불확실하면 down-weight" 계열. A1은 불확실성을 **절단대 내 위치**(밴드 가장자리일수록 부정확)로 근사한 것이고, 이 논문은 **측정 통계 $\sigma_d$** 에서 직접 끌어옴 — 더 원리적. 우리 A1의 $\lambda$ 를 실제 깊이 분산으로 대체하면 이 논문에 근접.
- 05번(Indoor-Recon)의 불확실성 가중 법선손실도 같은 철학의 다른 적용.
