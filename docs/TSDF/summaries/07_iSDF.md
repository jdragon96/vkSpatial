# 07. iSDF — 하나의 신경망이 실시간으로 SDF를 배우기 (심화)
**논문:** iSDF (Ortiz et al., RSS 2022) · arXiv:2204.02296
**PDF:** `../07_iSDF_RealtimeNeuralSDF_2022.pdf`

## 🧒 한 줄 요약
복셀 격자 대신 **작은 신경망 하나** $f_\theta(\mathbf x)$ 가 "이 좌표의 거리?"를 실시간으로 배워요. 가볍고, 안 본 곳도 그럴듯하게 채워요.

---

## 1. 참 SDF의 성질 = Eikonal 방정식
부호거리 $d(\mathbf x)=\pm\min_{\mathbf y\in\mathcal S}\lVert\mathbf x-\mathbf y\rVert$ 는 다음 편미분방정식을 (거의 어디서나) 만족:
$$\lVert\nabla d(\mathbf x)\rVert=1.$$
**직관:** 표면에서 1m 멀어지면 거리도 정확히 1m 는다. 기울기 방향은 **가장 가까운 표면점 반대쪽**을 가리킴. iSDF는 이 두 성질을 손실로 직접 강제.

## 2. 세 손실 (완전판)

### 2.1 SDF 손실 — 배치 최근접거리로 "묶기(bound)"
포즈 있는 깊이영상에서 표면점들을 배치로 샘플. 임의 질의점 $\mathbf x$ 에 대해 배치 내 최근접 표면점까지 거리를 $b(\mathbf x)$ 라 하면, 삼각부등식으로
$$|d(\mathbf x)|\le b(\mathbf x)\quad(\text{상계}).$$
- **표면 근처**($|d|<\tau$, 절단대): 광선을 따른 서명거리를 근사 타깃 $\hat d$ 로 정확히 맞춤 $\ \lvert f_\theta(\mathbf x)-\hat d\rvert$.
- **자유공간**(먼 곳): 신경망이 $b(\mathbf x)$ 를 넘지 않게 상계로 유도 $\ \max\big(0,\ |f_\theta(\mathbf x)|-b(\mathbf x)\big)$ 형태.

> 🧒 방 안 여러 점에서 "제일 가까운 벽까지 몇 걸음?"을 재고, 신경망 답이 그 값을 **넘지 않게** 가르쳐요.

### 2.2 Eikonal 손실 — 축척 1 유지
$$\mathcal L_{\text{eik}}=\mathbb E_{\mathbf x}\big(\lVert\nabla f_\theta(\mathbf x)\rVert-1\big)^2.$$
$\nabla f_\theta$ 는 자동미분으로 계산. 이 항이 없으면 SDF가 찌그러져 거리·기울기가 부정확.

> 🧒 지도 축척을 "1걸음=1칸"으로 **일정하게** 지키는 규칙.

### 2.3 법선(기울기) 손실 — 표면 방향 정렬
표면 근처에서 $\nabla f_\theta$ 가 관측 법선 $\mathbf n$ 을 향하게:
$$\mathcal L_{\text{grad}}=1-\nabla f_\theta(\mathbf x)\cdot\mathbf n\quad(\text{또는 }\lVert\nabla f_\theta-\mathbf n\rVert).$$

**전체:** $\ \mathcal L=\mathcal L_{\text{sdf}}+\lambda_1\mathcal L_{\text{eik}}+\lambda_2\mathcal L_{\text{grad}}.$

## 3. 실시간 만드는 요령
- **Active sampling**: 손실(불확실성)이 큰 광선·구간을 더 자주 뽑아 학습 효율↑.
- **연속 표현**의 이점: 격자와 달리 해상도 고정 없음, 부분관측 영역을 **매끈히 보간·채움**, 노이즈 **평활화**, 압축(작은 MLP).
- 다운스트림: 로봇 경로계획은 SDF와 $\nabla f$(충돌비용의 기울기)를 바로 씀.

## 4. 우리 프로젝트와의 관계
- "거리 + 기울기(법선)"를 함께 쓰는 건 우리 **stored-gradient(∇-SDF) 추출**과 목적이 동일(정확한 법선).
- Eikonal $\lVert\nabla f\rVert=1$ 은 우리 point-to-plane 거리(법선 방향 수직거리)가 왜 "진짜 SDF"에 가까운지의 근거.
- 차이: iSDF는 **암시적(신경망)·CPU/GPU 학습**, 우리는 **명시적 복셀·GPU 융합**. 부분관측 채움은 iSDF가, 서브복셀 결정성·속도는 우리가 유리.
