# 08. ESLAM — 3D 격자 대신 "3장의 평면"으로 가볍게 (심화)
**논문:** ESLAM (Johari, Carta, Fleuret, CVPR 2023) · arXiv:2211.11704
**PDF:** `../08_ESLAM_SDFplanes_2022.pdf`

## 🧒 한 줄 요약
공간을 통째 3D 격자로 들면 무거워요. 서로 수직인 **평면 3장**(바닥·앞·옆)의 특징만 저장하고 필요할 때 합쳐 TSDF를 만들어요 → **10× 빠르고 50% 정확**.

---

## 1. 왜 tri-plane인가 — 메모리 차수
$N^3$ 3D 특징격자는 저장 $O(N^3)$. 3D를 세 좌표평면에 분해하면 $O(N^2)$ 로 급감:
$$O(N^3)\ \longrightarrow\ 3\cdot O(N^2).$$
$N{=}512$ 면 $\sim$1.3억 vs 78만 × 3 — **수백 배** 절약. 이것이 속도·메모리 이득의 근원.

## 2. tri-plane 특징 (다중 스케일)
점 $\mathbf x=(x,y,z)$ 를 세 평면에 투영, 이중선형보간 후 이어붙임(‖):
$$\mathbf f(\mathbf x)=\big[F_{xy}(x,y)\ \|\ F_{xz}(x,z)\ \|\ F_{yz}(y,z)\big].$$
기하용 coarse/fine 평면쌍 + 색용 평면을 따로 둠(저주파·고주파 분리).

> 🧒 3D 물체를 앞·옆·위 **사진 3장**으로 저장했다가, 어떤 점 정보를 물으면 세 사진의 해당 자리를 봐서 합쳐 답해요.

## 3. 특징 → TSDF + 색 (얕은 디코더)
$$(s,\ \mathbf c)=\text{MLP}_\theta\big(\mathbf f(\mathbf x)\big),$$
$s$=절단부호거리(TSDF), $\mathbf c$=색. 디코더가 **얕아** 질의당 비용이 작음(dense 격자 대비 큰 속도차).

## 4. TSDF → 렌더링 가중치 (표면에서 봉우리)
절단거리 $tr$ 로 정규화한 종모양(두 시그모이드 곱):
$$w_i=\sigma\!\Big(\frac{s_i}{tr}\Big)\,\sigma\!\Big(-\frac{s_i}{tr}\Big).$$
$s_i>0$(앞)·$s_i<0$(뒤) 둘 다 $\sigma$ 두 개 곱이 작아지고 $s_i=0$(표면)에서 최대 → 광선상 깊이·색:
$$\hat D=\frac{\sum_i w_i t_i}{\sum_i w_i},\qquad \hat{\mathbf C}=\frac{\sum_i w_i \mathbf c_i}{\sum_i w_i}.$$

> 🧒 "거리 0에서 가장 밝은 손전등"으로 표면 깊이를 찍어요.

## 5. 목표함수 (지도 + 자세 동시 최적화)
$$\mathcal L=\lambda_d\underbrace{\lVert\hat D-D^\ast\rVert}_{\text{깊이}}+\lambda_c\underbrace{\lVert\hat{\mathbf C}-\mathbf C^\ast\rVert}_{\text{색}}+\lambda_{\text{fs}}\underbrace{\sum_{\text{free}}(s-tr)^2}_{\text{자유공간=}tr}+\lambda_{\text{sdf}}\underbrace{\sum_{\text{near}}(s-\hat s)^2}_{\text{표면근처}}.$$
- **자유공간 손실**: 표면 앞은 SDF가 $tr$(절단 최대)여야 함 → 빈 공간을 확실히 비움.
- 카메라 자세 $\mathbf T$ 를 특징·디코더와 **함께** 미분·최적화(tracking+mapping). NICE-SLAM 대비 정확도 +50%, 속도 ×10.

## 6. 우리 프로젝트와의 관계
- §4의 "TSDF를 종모양 가중치로 표면 정렬"은 우리 추출의 **zero-crossing** 개념과 같은 뿌리(우리는 명시적 값에서 직접 zero-crossing을 찾음).
- tri-plane은 우리 `TiledCompactDirectionalTSDF`(공간 타일링)와는 **다른** 메모리 절약 축(2D 분해 vs 희소 타일). 두 아이디어는 결합 가능(타일 안을 tri-plane으로).
- §5 자유공간 손실 = 우리 융합의 "절단대 밖은 +$\tau$" 규칙의 학습판.
