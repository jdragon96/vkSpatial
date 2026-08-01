# 03. Variance-Adaptive Voxel Grids (MrHash) — 복잡한 곳만 촘촘하게 (심화)
**논문:** Resolution Where It Counts (De Rebotti, Giacomini, Grisetti, Di Giammarino, 2025) · arXiv:2511.21459 (ACM TOG)
**PDF:** `../03_VarianceAdaptiveVoxelGrids_2025.pdf`

## 🧒 한 줄 요약
평평한 벽은 큰 칸, 모서리는 작은 칸. 어디가 복잡한지는 **"거리 값의 들쭉날쭉함(분산)"** 으로 자동 판단해요.

---

## 1. 부호거리 두 형태 (식 1·2)
- 점군(LiDAR): $\ d_k(\mathbf x)=\operatorname{clip}\big((\mathbf p-\mathbf x)\cdot\hat{\mathbf n},-\tau,\tau\big),\ \hat{\mathbf n}=\frac{\mathbf p-\mathbf o}{\lVert\mathbf p-\mathbf o\rVert}$ (point-to-plane)
- 카메라(RGB-D): $\ d_k(\mathbf x)=\operatorname{clip}\big(d-\lVert\mathbf x\rVert,-\tau,\tau\big)$ (투영식)

가중평균(식 3, $w_k=1$ 로 고정해 분산 계산 단순화):
$$D_i\leftarrow\frac{W_iD_i+w_kd_k}{W_i+w_k},\qquad W_i\leftarrow W_i+w_k.$$

## 2. 온라인 분산 (Welford) — 완전 유도

### 2.1 목표
관측 $d_1,\dots,d_k$ 를 다 저장하지 않고, 매 스텝 **평균과 분산**을 O(1)로 갱신하고 싶다.

### 2.2 평균 점화식
$$M_k=\frac1k\sum_{i=1}^k d_i=M_{k-1}+\frac{d_k-M_{k-1}}{k}.$$

### 2.3 제곱편차합 $S_{2,k}=\sum_{i=1}^k(d_i-M_k)^2$ 의 점화식 (핵심, 식 5)
$$\boxed{S_{2,k}=S_{2,k-1}+(d_k-M_{k-1})(d_k-M_k)}$$
**증명 스케치:** $S_{2,k}-S_{2,k-1}=\sum_{i\le k}(d_i-M_k)^2-\sum_{i\le k-1}(d_i-M_{k-1})^2$. $M_k=M_{k-1}+\frac{d_k-M_{k-1}}{k}$ 를 대입해 전개하면 교차항이 소거되어 위 곱 하나만 남는다. 그러면
$$\sigma_i^2=\frac{S_{2,i,k}}{k}\quad(\text{식 6}).$$
과거 관측 저장 없이 **한 줄**로 분산 유지. 수치적으로도 안정(단일 패스).

> 🧒 시험 점수 평균만 아니라 **흔들림**도 계산기 하나로 계속 업데이트해요. 평평한 벽은 매번 비슷(작은 흔들림), 모서리는 크게 흔들려요.

### 2.4 왜 분산이 "기하 복잡도"인가
한 복셀을 여러 방향/프레임에서 보면, **평평면**에서는 모든 $d_k$ 가 같은 참 평면거리 → $\sigma^2\approx$ 센서잡음(작음). **모서리/얇은 벽**에서는 서로 다른 표면이 그 복셀을 다른 부호·크기로 관측 → $d_k$ 가 크게 흩어짐 → $\sigma^2$ 큼. 즉 $\sigma^2$ 는 센서·의미정보 없이 **표면 복잡도의 대리지표**.

## 3. 해상도 병합 규칙
$$\sigma_i^2<\theta\ \Rightarrow\ \text{coarse 블록으로 재할당(병합)},\qquad \sigma_i^2\ge\theta\ \Rightarrow\ \text{fine 유지}.$$
초기엔 모두 최고해상도(fine)로 시작 → 분산 낮은 영역만 사후 병합(보수적). 임계 $\theta$ 가 메모리/디테일 트레이드오프 노브.

## 4. 단일 flat hash로 다중 해상도

### 4.1 해시 함수 (Teschner)
$$H(x,y,z)=(x\cdot p_1\oplus y\cdot p_2\oplus z\cdot p_3)\bmod n_{\text{hash}},$$
$p_1{=}73856093,\ p_2{=}19349669,\ p_3{=}83492791$ (큰 소수), $\oplus$=XOR. 각 엔트리는 해상도레벨 $n_j$ 힙 $\mathbf h_{n_j}$ 안의 블록 포인터를 가리켜, **여러 해상도를 한 주소공간**에서 O(1) 접근(옥트리 O(log N)·포인터추적 회피 → GPU 친화).

### 4.2 전이 복셀(transitional voxel) 문제
서로 다른 해상도 블록 경계에서 (a) SDF 보간이 이웃 부재로 정의 안 됨(Fig 5), (b) Marching Cubes 정점이 겹침(Fig 6). 해법: 큰 복셀을 공유면 따라 **잘라내고**(Lengyel식), 고운쪽 값을 우선하는 가중보간 + 정점 병합으로 이음새 제거. 다중해상도 MC는 각 블록 독립·GPU 병렬.

## 5. 스트리밍 & 렌더링
GPU 메모리가 상한(예 85%)에 도달하면 관련성 낮은 블록(카메라 프러스텀 밖 / LiDAR 반경 밖)부터 축출 — **데이터 주도** 스트리밍(고전 프러스텀 컬링보다 전송 최소). 부수적으로 3D Gaussian Splatting으로 novel-view 렌더:
$$\mathbf C_p(u,v)=\sum_{m=1}^M \mathbf C_m\alpha_m\mathcal G_m(u,v)\!\!\prod_{l<m}\!(1-\alpha_l\mathcal G_l),\quad \mathcal G_m=\exp\!\big(-\tfrac12(\mathbf u-\boldsymbol\mu_m)^\top\boldsymbol\Sigma_m^{-1}(\mathbf u-\boldsymbol\mu_m)\big).$$

## 6. 우리 프로젝트와의 관계
- 우리 `AdaptiveVoxelGrid`의 **직접 근거**. §2 Welford가 우리 `SimpleTSDF::DownloadVoxels` 분산과 동일 원리.
- 우리 실측 감소가 1.02~1.07배로 작았던 건 **truncation-plateau**(절단대 내부는 어디든 분산이 낮아 과도 병합) — 이 논문의 GPU flat-hash·다중해상도 MC·스트리밍까지 통합하면 실제 대형 씬에서 이득이 커짐.
