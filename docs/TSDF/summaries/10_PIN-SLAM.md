# 10. PIN-SLAM — "움직이는 신경 점들"로 지도를 만들고 고무줄처럼 교정 (심화)
**논문:** PIN-SLAM (Pan et al., T-RO 2024) · arXiv:2401.09101
**PDF:** `../10_PIN-SLAM_2024.pdf`

## 🧒 한 줄 요약
지도를 **특징을 지닌 점(neural points)** 으로 저장해요. 나중에 "여기 왔던 곳!"(루프)을 찾으면 점들을 **고무줄처럼 당겨** 지도 전체를 한 번에 바로잡아요.

---

## 1. 신경 점 지도와 SDF 조회
지도 = 희소 최적화가능 신경점 $\{(\mathbf p_i,\mathbf e_i)\}$ (위치+특징). 임의 $\mathbf x$ 의 SDF:
$$f(\mathbf x)=\text{MLP}\!\Big(\sum_{i\in\mathcal N(\mathbf x)} w_i(\mathbf x)\,\mathbf e_i\Big),\qquad w_i(\mathbf x)=\frac{\exp(-\lVert\mathbf x-\mathbf p_i\rVert^2/\sigma^2)}{\sum_{j\in\mathcal N}\exp(\cdots)}.$$
$\mathcal N(\mathbf x)$=근접 신경점(복셀해시로 O(1) 검색). 특징을 보간 후 얕은 MLP로 SDF 디코딩.

> 🧒 어떤 지점의 거리를 알고 싶으면 **가까운 이웃 점들에게** 물어 가까운 이웃 말을 더 크게 반영해 합쳐요.

## 2. 대응점 없는 point-to-implicit 정합 (완전 유도)
새 스캔점 $\mathbf s_k$ 는 자세 $\mathbf T(\boldsymbol\xi)\in SE(3)$ 로 옮기면 **표면 위**($f=0$)여야 함. 잔차 $r_k(\boldsymbol\xi)=f\big(\mathbf T(\boldsymbol\xi)\,\mathbf s_k\big)$ 에 대해:
$$\min_{\boldsymbol\xi}\ \sum_k \rho\big(r_k(\boldsymbol\xi)\big).$$

### 2.1 가우스–뉴턴 선형화
$\mathbf q_k=\mathbf T\mathbf s_k$. 연쇄법칙으로 자코비안:
$$\mathbf J_k=\frac{\partial r_k}{\partial\boldsymbol\xi}=\underbrace{\nabla f(\mathbf q_k)^\top}_{1\times3}\ \underbrace{\frac{\partial(\mathbf T\mathbf s_k)}{\partial\boldsymbol\xi}}_{3\times6},\qquad \frac{\partial(\mathbf T\mathbf s_k)}{\partial\boldsymbol\xi}=\big[\,\mathbf I\ \ -[\mathbf q_k]_\times\,\big].$$
로버스트 가중 $w_k=\psi(r_k)$(예 Huber)까지 넣어 정규방정식:
$$\Big(\sum_k w_k\mathbf J_k^\top\mathbf J_k\Big)\,\delta\boldsymbol\xi=-\sum_k w_k\mathbf J_k^\top r_k,\qquad \mathbf T\leftarrow\exp([\delta\boldsymbol\xi]_\wedge)\,\mathbf T.$$
**핵심:** SDF의 **값 = 표면까지 거리**, **기울기 $\nabla f$ = 밀 방향**을 직접 주므로, 고전 ICP처럼 점-대-점 **대응을 찾을 필요가 없음**(correspondence-free). 한 번의 조회로 "얼마나·어느 쪽으로" 움직일지가 나옴.

> 🧒 스캔점들이 "벽 표면(거리 0)"에 딱 붙을 때까지 로봇 위치를 조금씩 밀어요. 점끼리 짝지을 필요 없이 **"거리 0에 붙어라"** 만 시키면 돼요.

## 3. 탄성 지도 + 루프 클로징
- **점진 학습**: 새 스캔으로 국소 신경점 특징 $\mathbf e_i$ 를 온라인 학습(지도 갱신).
- **탄성(elastic)**: 각 신경점을 만들 때의 국소 자세에 **붙여** 둠. 루프 발견 시 포즈그래프 최적화로 자세를 고치면 **신경점이 함께 이동** → 지도가 변형되어 이음새 없이 전역 정렬.
- **루프 탐지**: 신경점 특징으로 장소 인식.

## 4. 우리 프로젝트와의 관계
- 우리 `AdvancedTSDF`의 **point-to-plane 적분**(표면=거리0 정렬)과 §2가 같은 목표. §2의 point-to-implicit 정합은 우리가 spec만 써둔 **direct TSDF-gradient odometry**의 **최신·정석 형태** — 우리 명시적 TSDF에서도 $\nabla f$(stored-gradient)로 동일 GN을 세울 수 있음(추천 후속).
- LiDAR·RGB-D 공용 → 우리 folder-eval 파이프라인(추정 카메라)과 접목 가능.
