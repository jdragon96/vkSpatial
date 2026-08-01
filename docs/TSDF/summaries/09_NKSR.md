# 09. NKSR — 점들 사이를 "학습된 커널"로 잇기 (심화)
**논문:** Neural Kernel Surface Reconstruction (Huang et al., CVPR 2023) · arXiv:2305.19590
**PDF:** `../09_NKSR_NeuralKernelSurfaceRecon_2023.pdf`

## 🧒 한 줄 요약
방향 붙은 점(oriented points)들 주위에 **"영향력 붕붕이(커널)"** 를 놓고 잘 섞어 매끈한 표면을 만들어요. 붕붕이 모양을 **데이터로 학습**하고 **좁게** 만들어 수백만 점도 몇 초에 처리.

---

## 1. RKHS와 커널 표현 (왜 가중합인가)
재생핵힐베르트공간(RKHS) $\mathcal H_K$ 에서 다음 정칙화 회귀를 풀면
$$f^\star=\arg\min_{f\in\mathcal H_K}\ \sum_i \ell\big(f;\mathbf x_i\big)+\lambda\lVert f\rVert_{\mathcal H_K}^2,$$
**표현정리(representer theorem)** 에 의해 해는 항상 데이터점에 놓인 커널의 유한합:
$$\boxed{\ f(\mathbf x)=\sum_i \alpha_i\,K_\theta(\mathbf x,\mathbf x_i)\ }$$
표면 $=\{f=0\}$. NKF/NKSR의 새 점: 커널 $K_\theta$ 를 **데이터로 학습**(고정 RBF가 아님) → 형상 사전지식 반영.

> 🧒 점마다 부드러운 언덕을 놓고 다 더해요. 언덕 높이($\alpha$)를 잘 정하면 점들을 잇는 산맥(표면)이 생겨요. 언덕 **모양**도 배워서 물체답게 만들어요.

## 2. 계수 = 선형 연립 (커널 릿지)
값 맞춤 손실 $\ell=\tfrac12(f(\mathbf x_i)-y_i)^2$ 이면 위 최소화의 정규방정식:
$$(G+\lambda I)\,\boldsymbol\alpha=\mathbf b,\qquad G_{ij}=K_\theta(\mathbf x_i,\mathbf x_j),\ \ b_i=y_i.$$
- $G$: 그램(커널) 행렬, $\lambda$: 노이즈 안정화(릿지).
- **좁은(compact support) 커널** ⇒ $|{\mathbf x_i-\mathbf x_j}|$ 크면 $K{=}0$ ⇒ $G$ 가 **희소** ⇒ conjugate-gradient 등 희소솔버로 수백만 점 몇 초.

## 3. 노이즈 강건성 = 기울기 맞추기 (핵심)
점 위치의 값 $f(\mathbf x_i)=0$ 만 맞추면 점 위치 노이즈에 취약. 대신 **법선(=기울기)** 을 맞춤:
$$\min_{\boldsymbol\alpha}\ \sum_i\big\lVert\nabla f(\mathbf x_i)-\mathbf n_i\big\rVert^2+\lambda\lVert f\rVert_{\mathcal H_K}^2.$$
$\nabla f(\mathbf x)=\sum_j\alpha_j\nabla_{\mathbf x}K_\theta(\mathbf x,\mathbf x_j)$ 도 $\boldsymbol\alpha$ 에 **선형** → 여전히 선형 연립(그램이 $\nabla K$ 로 바뀜). **왜 강건한가:** 표면의 "위치"보다 "방향(법선)"이 센서 노이즈에 덜 흔들리고, 도함수 제약이 장(場) 전체의 형태를 더 직접 규정해 국소 노이즈의 영향이 평균화됨.

> 🧒 "여기 표면 있다"보다 "여기 표면이 **이 방향을 본다**"를 맞추면, 점이 조금 흔들려도 표면 방향은 안정적이에요.

## 4. 확장성 장치
- **다중해상도 복셀 계층**: 거친→고운 커널을 계층으로 두어 전역+국소를 함께.
- **최소 학습**: dense oriented points만 있으면 어떤 데이터셋에서도 커널을 학습 가능(범용).

## 5. 우리 프로젝트와의 관계
- 우리 파이프라인의 **후처리 옵션**(oriented cloud → 매끈한 메시). 우리 추출은 표면 위 oriented points를 주므로 NKSR 입력에 바로 맞음.
- §3 "기울기(법선) 맞추기"는 우리 **stored-gradient** 철학과 동일 — 우리는 융합 단계에서 법선을 누적, NKSR은 재구성 단계에서 법선을 적합.
