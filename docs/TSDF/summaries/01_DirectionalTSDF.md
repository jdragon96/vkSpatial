# 01. Directional TSDF — 방향별로 거리를 따로 저장하기 (심화)
**논문:** Rendering and Tracking the Directional TSDF (Splietker & Behnke, 2021) · arXiv:2108.08115
**PDF:** `../01_DirectionalTSDF_RenderingTracking_2021.pdf`

## 🧒 한 줄 요약
얇은 벽을 앞·뒤에서 보면 보통 TSDF는 두 거리를 **평균**내 벽을 지워버려요. 방향마다 공책을 따로 쓰면(앞 공책/뒤 공책) 안 섞여요.

---

## 0. 준비: 표준 TSDF의 두 가지 거리와 "가중 평균"의 정체

### 0.1 투영식(projective) vs 점-대-평면(point-to-plane) 부호거리
카메라/센서에서 잰 표면점 $\mathbf p$, 복셀 중심 $\mathbf x$, 광선 방향 $\mathbf r=\frac{\mathbf p-\mathbf o}{\lVert\mathbf p-\mathbf o\rVert}$ ($\mathbf o$=센서).

- **투영식**(고전 Curless–Levoy): 광선을 따라 잰 깊이 차
$$d^{\text{proj}}(\mathbf x)=\operatorname{clip}\big(\,\lVert\mathbf p-\mathbf o\rVert-(\mathbf x-\mathbf o)\cdot\mathbf r,\ -\tau,\ \tau\big)$$
- **점-대-평면**: 표면 법선 $\mathbf n$ 방향의 수직거리
$$d^{\text{p2p}}(\mathbf x)=\operatorname{clip}\big((\mathbf p-\mathbf x)\cdot\mathbf n,\ -\tau,\ \tau\big)$$

두 값은 표면에 **비스듬히** 부딪힐 때 차이가 큼. $\theta$ 를 광선과 법선 사이각이라 하면 근사적으로
$$d^{\text{proj}}\approx \frac{d^{\text{p2p}}}{\cos\theta}$$
즉 투영식은 비스듬할수록 거리를 **과대평가**($1/\cos\theta$ 배). 그래서 우리 엔진은 평평면에서 거의 완벽한 point-to-plane을 기본으로 씁니다.

### 0.2 왜 "가중 평균"인가 — 최대우도(MLE) 유도
관측 $d_k$ 가 참값 $D$ 를 분산 $\sigma_k^2$ 로 재는 가우시안이라 하자: $d_k\sim\mathcal N(D,\sigma_k^2)$, 가중치 $w_k:=1/\sigma_k^2$. 로그우도
$$\log L(D)=-\tfrac12\sum_k w_k\,(d_k-D)^2+\text{const}.$$
$\frac{\partial}{\partial D}=0$ :
$$\sum_k w_k(d_k-D)=0\ \Rightarrow\ \boxed{D=\frac{\sum_k w_k d_k}{\sum_k w_k}}$$
이걸 **점화식**으로 바꾸면(온라인 갱신) 우리가 매 프레임 쓰는 식이 나옴:
$$D_{k}=\frac{W_{k-1}D_{k-1}+w_k d_k}{W_{k-1}+w_k},\qquad W_k=W_{k-1}+w_k.$$

> 🧒 가중 평균은 "**믿는 만큼 크게 반영**하는 평균". 수학적으로는 가우시안 잡음에서 **가장 그럴듯한 참값**이에요.

---

## 1. 핵심 문제: 방향을 섞으면 표면이 사라진다 (유도)

얇은 벽을 앞($+$면, 법선 $+\mathbf e$)과 뒤($-$면, 법선 $-\mathbf e$)에서 관측. 벽 두께가 복셀보다 얇아 **같은 복셀**이 앞면엔 $d_+\approx+a$, 뒷면엔 $d_-\approx-a$ 로 관측됨(부호 반대). 방향 무시하고 평균하면
$$D=\frac{w_+(+a)+w_-(-a)}{w_++w_-}\xrightarrow{w_+\approx w_-}0\ \text{(상수)}.$$
$D$ 가 위치에 상관없이 $\approx0$ → **어디서나 표면**처럼 보여 zero-crossing이 뭉개짐. 이게 모서리 라운딩·얇은 벽 소실의 수학적 원인.

## 2. 해결: 방향별 6개 공책

방향 집합 $\mathcal D=\{+X,-X,+Y,-Y,+Z,-Z\}$, 단위벡터 $\mathbf e_d$. 관측 법선 $\mathbf n$ 의 방향별 신뢰도:
$$w_d=\max(0,\ \mathbf n\cdot\mathbf e_d)^{\,p}.$$
- $\mathbf n\cdot\mathbf e_d=\cos(\angle)$ : 법선이 그 축을 향한 정도. $p$ 크면 지배축 하나만 강조.
- **Top-K 선택**: 상대세기 $r_d=(\mathbf n\cdot\mathbf e_d)^p / \max_d(\cdot)$ 가 임계(예 0.05) 이상인 방향만 유지, 최대 K개.

각 방향은 **독립 가중평균**:
$$D_d=\frac{\sum_k w_{d,k}\,d^{\text{p2p}}_k}{\sum_k w_{d,k}},\qquad W_d=\sum_k w_{d,k}.$$
앞면 관측은 $+\mathbf e$ 공책에만, 뒷면은 $-\mathbf e$ 공책에만 → **더 이상 상쇄되지 않음**. 각 공책의 zero-crossing이 살아있어 두 표면이 모두 복원됨.

## 3. 저장 그래디언트(법선)와 추출
방향 저장과 별개로, 관측 법선의 가중합을 누적:
$$\mathbf S_n=\sum_k \mathbf n_k\,w_k\ \Rightarrow\ \hat{\mathbf n}=\frac{\mathbf S_n}{\lVert\mathbf S_n\rVert}.$$
추출 시 (a) 방향 축을 따라 $D_d$ 의 **zero-crossing 위치**를 선형/에르미트 보간으로 찾고, (b) 법선은 저장된 $\hat{\mathbf n}$ 사용(중심차분보다 잡음에 강함). 우리 엔진의 "mode-3" 추출이 이것.

## 4. 이 논문의 새 점 — 렌더링 & 트래킹

### 4.1 방향 블렌딩 레이캐스팅
화면 광선 방향 $\mathbf r$ 에 대해, 눈에 보이는 방향들만 가중 블렌딩(뒤통수 방향 제외):
$$\beta_d=\max(0,\,-\mathbf r\cdot\mathbf e_d),\qquad D^{\text{render}}(\mathbf x)=\frac{\sum_d \beta_d\,W_d\,D_d}{\sum_d \beta_d\,W_d}.$$
$D^{\text{render}}=0$ 을 광선상에서 찾아 표면·법선 산출. **저장 방향과 실제 기울기가 일치**(gradient-consistent)해 법선이 정확.

### 4.2 카메라 트래킹 (point-to-plane ICP)
현재 프레임 점 $\mathbf p_k$ 를 자세 $\mathbf T=(\mathbf R,\mathbf t)$ 로 옮겨 렌더된 표면점 $\mathbf q_k$·법선 $\mathbf n_k$ 에 맞춤:
$$\min_{\boldsymbol\xi}\sum_k\Big(\mathbf n_k\cdot\big((\mathbf R(\boldsymbol\xi)\mathbf p_k+\mathbf t(\boldsymbol\xi))-\mathbf q_k\big)\Big)^2.$$
$\boldsymbol\xi\in\mathfrak{se}(3)$ (작은 자세변화). $\mathbf R\approx I+[\boldsymbol\omega]_\times$ 로 1차 선형화하면 각 항이 $\boldsymbol\xi$ 에 **선형** → 정규방정식 $\mathbf A^\top\mathbf A\,\boldsymbol\xi=\mathbf A^\top\mathbf b$ 를 Gauss–Newton으로 반복. point-to-plane은 "표면을 따라 미끄러지는" 자유도를 허용해 point-to-point보다 빠르게 수렴.

## 5. 우리 프로젝트와의 관계
- `TopKDirections`( $|n_{axis}|^p$ ) = §2의 방향 선택, `sumDW/sumW` = 방향별 가중평균, `sumN` 정규화 = §3 저장 그래디언트.
- §0.1의 point-to-plane이 우리 `SetPointToPlane(true)` 기본값(평평면 near-perfect).
- §4.2의 point-to-plane ICP는 우리가 spec만 써둔 **direct TSDF-gradient odometry**의 고전형(최신형은 10번 PIN-SLAM).
