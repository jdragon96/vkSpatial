# 06. NeuS — 사진만으로 SDF를 배우되 "표면이 정확히 거리=0" (심화)
**논문:** NeuS (Wang et al., 2021) · arXiv:2106.10689
**PDF:** `../06_NeuS_NeuralImplicitSurfaces_2021.pdf`

## 🧒 한 줄 요약
여러 각도 사진만으로 모양(SDF)을 배웁니다. 핵심은 "안개처럼 그리는(volume rendering)" 방식이 **진짜 표면을 정확히 거리=0**에 놓게 수식을 바로잡은 것.

---

## 1. 배경: 볼륨렌더링과 SDF
표면 $\mathcal S=\{\mathbf x:f(\mathbf x)=0\}$, $f$=신경망 SDF. 광선 $\mathbf r(t)=\mathbf o+t\mathbf v$ 를 따라 색 누적:
$$C=\int_0^\infty w(t)\,c(\mathbf r(t))\,dt,\qquad \int w(t)\,dt\approx1.$$
문제는 **가중치 $w(t)$ 를 SDF로 어떻게 만들 것인가**. 잘못 만들면 표면 위치가 편향됨.

## 2. 로지스틱(S) 밀도
$$\Phi_s(x)=\frac{1}{1+e^{-sx}},\qquad \phi_s(x)=\Phi_s'(x)=\frac{s\,e^{-sx}}{(1+e^{-sx})^2}.$$
$\phi_s$ 는 $x=0$ 에서 대칭 봉우리. $s\to\infty$ 면 디랙델타(완벽 표면). 학습 중 $s$ 증가 → 표면 선명화.

## 3. 나이브 가중치의 편향 (문제 유도)
직관적으로 $w(t)\propto\phi_s\big(f(\mathbf r(t))\big)$ 로 두자. 단일 평면을 광선이 각도 $\theta$(법선과)로 통과하면 표면 근처에서
$$f(\mathbf r(t))\approx -\cos\theta\,(t-t^\star),$$
($t^\star$=진짜 표면 교점). 그러면 $w(t)\propto\phi_s(-\cos\theta\,(t-t^\star))$ 의 **봉우리는 $t^\star$ 에 있지만**, 폭이 $1/\cos\theta$ 로 늘어나고, $\phi_s$ 를 정규화·투과율과 곱하는 순간 봉우리가 표면 **앞쪽으로 밀림**(bias). 즉 색을 맞추도록 학습하면 표면이 실제보다 카메라 쪽에 생김.

> 🧒 손전등 빛을 그냥 쓰면 "가장 밝은 곳"이 진짜 표면보다 살짝 **앞에** 생겨요. 이걸 고쳐야 해요.

## 4. 편향 없는·가림 인식 가중치 (핵심 결과)
NeuS는 불투명도(opaque density) $\rho(t)$ 를 도입해 $w(t)=T(t)\rho(t)$, $T(t)=\exp\!\big(-\int_0^t\rho\big)$ 로 두되, **두 성질**을 요구:
1. **Unbiased**: $w(t)$ 의 봉우리가 정확히 $f=0$($t=t^\star$).
2. **Occlusion-aware**: 앞 표면이 뒤 표면을 가림(투과율 단조 감소).

이를 만족하는 이산 불투명도:
$$\boxed{\ \alpha_i=\max\!\left(0,\ \frac{\Phi_s(f_i)-\Phi_s(f_{i+1})}{\Phi_s(f_i)}\right)\ }$$
색: $\ C=\sum_i T_i\alpha_i c_i,\quad T_i=\prod_{j<i}(1-\alpha_j).$

### 4.1 왜 편향이 사라지나 (스케치)
$\alpha_i$ 는 $\Phi_s$ 의 **감소분**을 정규화한 것. 단일 표면·연속 극한에서 $\alpha\,dt \to -\frac{d\Phi_s/dt}{\Phi_s}dt$, 이로부터 유도되는 $w(t)$ 의 봉우리는 $\frac{d}{dt}\big(f\circ\mathbf r\big)$ 부호가 바뀌는 지점, 즉 $f=0$ 과 **1차 근사에서 정확히 일치**(각도 $\theta$ 에 무관). $\cos\theta$ 항이 분자·분모에서 상쇄되어 편향이 제거됨.

> 🧒 "빛이 줄어드는 **속도**"로 표면을 잡으면, 각도가 비스듬해도 봉우리가 정확히 거리 0에 앉아요.

## 5. 학습 목표
$$\mathcal L=\underbrace{\sum_{\text{ray}}\lVert C-\hat C\rVert}_{\text{색 일치}}+\lambda_{\text{eik}}\underbrace{\mathbb E_{\mathbf x}\big(\lVert\nabla f(\mathbf x)\rVert-1\big)^2}_{\text{Eikonal(SDF 성질)}}.$$
마스크 불필요(mask-free) — §4의 편향 제거 덕에 자기폐색·얇은 구조도 복원.

## 6. 우리 프로젝트와의 관계
우리는 **명시적 복셀 TSDF**(값 직접 저장)라 §4의 렌더링 학습은 안 쓰지만, "표면=SDF 0-등고선", Eikonal $\lVert\nabla f\rVert=1$ 은 우리 point-to-plane·stored-gradient의 이론적 배경과 정확히 같음. NeuS의 편향 제거는 이후 08/05/10의 SDF-렌더링 SLAM 토대.
