# 05. Indoor Recon (Hybrid + Normal Prior) — 거친+세밀, 그리고 "못 믿을 힌트는 약하게" (심화)
**논문:** Indoor Scene Reconstruction with Fine-Grained Details Using Hybrid Representation and Normal Prior Enhancement (2023) · arXiv:2309.07640
**PDF:** `../05_IndoorRecon_HybridRep_NormalPrior_2023.pdf`

## 🧒 한 줄 요약
매끈한 큰 형태(MLP)와 세밀한 무늬(평면 특징)를 **더해서** 표면을 만들고, 단안 카메라의 **법선 힌트**는 쓰되 **못 믿을 힌트는 약하게** 반영해 디테일을 지켜요.

---

## 1. 하이브리드 = 주파수 분해 (거친 + 세밀 잔차)
$$\mathbf s=\tilde{\mathbf s}+\Delta\mathbf s,\qquad \mathbf h=\tilde{\mathbf h}+\Delta\mathbf h.$$
- $\tilde{\mathbf s}$(MLP): **저주파** — 매끈한 큰 형태(벽·바닥). MLP는 스펙트럼 편향(spectral bias)으로 저주파를 잘 학습.
- $\Delta\mathbf s$(tri-plane 특징): **고주파** — 국소 디테일(장식·모서리). 격자기반 특징은 고주파를 잘 표현.

> 🧒 큰 붓으로 전체 모양을 칠하고($\tilde s$), 얇은 펜으로 무늬를 덧그려요($\Delta s$). 더하면 둘 다 얻어요. 서로 다른 **일을 나눠** 맡아 각자 잘하는 걸 해요.

## 2. SDF → 불투명도 (미분 가능 렌더, NeuS 계열)
$$\alpha_i=\max\!\left(0,\ \frac{\Phi_\tau(f_i)-\Phi_\tau(f_{i+1})}{\Phi_\tau(f_i)}\right),$$
$\Phi_\tau$=학습되는 $\tau$ 의 시그모이드. 06번 NeuS의 편향 없는 가중치를 그대로 채택 → 사진으로 표면을 정확히 학습.

## 3. 핵심: 불확실성 가중 법선 손실 (완전 유도)

### 3.1 문제
단안 추정 법선 $\hat{\mathbf n}$(사전학습 네트워크)은 **평평한 곳은 정확**, 복잡한 곳은 자주 틀림. 그냥 다 믿으면 디테일을 오히려 뭉갬.

### 3.2 손실
$$\mathcal L_{\text{prior}}=\frac1{|\mathcal R|}\sum_{\mathbf r\in\mathcal R}\big(1-\mathbf u(\mathbf r)\big)\,\big\lVert\,1-\mathbf n(\mathbf r)^\top\hat{\mathbf n}(\mathbf r)\,\big\rVert_1.$$
- $\mathbf n$: 내 SDF 기울기 법선($\nabla f/\lVert\nabla f\rVert$), $\hat{\mathbf n}$: 단안 힌트.
- $\mathbf n^\top\hat{\mathbf n}=\cos(\angle)$ → $1-\cos$ 은 두 법선 불일치(0=완전일치).
- $\mathbf u(\mathbf r)\in[0,1]$: U-Net이 픽셀별로 예측한 **힌트 불확실성**. 못 믿을수록 $(1-\mathbf u)\to0$ → 그 픽셀은 손실에 **거의 안 들어감**.

### 3.3 왜 이게 통하나
$(1-\mathbf u)$ 는 04번의 정밀도 가중 $1/\sigma^2$ 를 [0,1]로 학습판화한 것. 신뢰 영역(평면)에서는 힌트가 강하게 표면을 잡아주고, 불확실 영역(디테일)에서는 힌트를 **자동으로 놓아줘** 사진 기반 세밀 학습이 살아남음.

> 🧒 친구가 "여긴 이 방향!"이라 할 때, **자신 있어 하는 말만** 따르고 헷갈려 하는 말은 무시해요.

## 4. 전체 목표
$$\mathcal L=\lambda_c\lVert C-\hat C\rVert+\lambda_e\,\mathbb E(\lVert\nabla f\rVert-1)^2+\lambda_p\,\mathcal L_{\text{prior}}.$$
색 일치 + Eikonal(SDF 성질) + 불확실성 가중 법선.

## 5. 우리 프로젝트와의 관계
- "불확실하면 약하게"는 우리 **A1(confidence weight)** · 04번(ProbFusion)과 동일 원리를 **법선 힌트**에 적용.
- §1 "거친+세밀 분리"는 우리 directional/adaptive 저장이 **모서리 디테일 보존**을 노리는 목적과 통함(우리는 방향/해상도로, 이 논문은 주파수 분해로).
