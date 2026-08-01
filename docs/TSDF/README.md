# TSDF 논문 모음 (2020년 이후) — PDF + 수학 중심 초등학생용 정리

2020년 이후 나온 중요한 TSDF / 부호거리장(SDF) 재구성 논문 11편의 **원문 PDF**와, 논문별 **핵심 수식을 초등학생도 이해할 수 있게** 풀어쓴 정리입니다.

- 원문 PDF: 이 폴더(`docs/TSDF/*.pdf`) — 전부 arXiv 오픈액세스에서 받음, `%PDF` 헤더 검증 완료(11개, 합계 ~194MB).
- 정리 문서: [`summaries/`](summaries/) — 논문 1편당 마크다운 1개. 각 문서는 **핵심 수식(유도 포함) → 기호 뜻 → 🧒 비유 → 우리 프로젝트와의 관계** 순. (심화판: MLE·Welford·가우시안곱·NeuS 편향제거·point-to-implicit GN 등 실제 유도 수록.)
- 학습 로드맵: [`../learning/MESH_EXPERT_ROADMAP.md`](../learning/MESH_EXPERT_ROADMAP.md) · 테스트 장비: [`../learning/TEST_EQUIPMENT.md`](../learning/TEST_EQUIPMENT.md)

## 두 갈래로 보는 TSDF 발전

### A. 명시적 볼류메트릭 TSDF (값을 직접 격자에 저장 — 이 저장소의 방식)
| # | 논문 | 연도 | 핵심 개선 (한 줄) | 정리 |
|---|------|------|------------------|------|
| 01 | Directional TSDF (Rendering & Tracking) · [2108.08115](https://arxiv.org/abs/2108.08115) | 2021 | 방향(±6축)마다 거리를 **따로** 저장 → 모서리·얇은 벽 보존 | [→](summaries/01_DirectionalTSDF.md) |
| 02 | DB-TSDF (Directional Bitmask) · [2509.20081](https://arxiv.org/abs/2509.20081) | 2025 | 거리를 **비트마스크+AND**로 → GPU 없이 상수시간, 8B/복셀 | [→](summaries/02_DB-TSDF.md) |
| 03 | Variance-Adaptive Voxel Grids (MrHash) · [2511.21459](https://arxiv.org/abs/2511.21459) | 2025 | **TSDF 분산**으로 복잡한 곳만 촘촘히(다중해상도 flat-hash) | [→](summaries/03_VarianceAdaptiveVoxelGrids.md) |
| 04 | Probabilistic Volumetric Fusion · [2210.01276](https://arxiv.org/abs/2210.01276) | 2022 | 깊이 **불확실성(분산)** 으로 가중 → 못 믿을 측정은 약하게 | [→](summaries/04_ProbabilisticVolumetricFusion.md) |

### B. 신경 임플리싯 SDF (신경망이 거리를 출력 — 사진/스캔에서 학습)
| # | 논문 | 연도 | 핵심 개선 (한 줄) | 정리 |
|---|------|------|------------------|------|
| 06 | NeuS · [2106.10689](https://arxiv.org/abs/2106.10689) | 2021 | **편향 없는** 볼륨렌더링 → 표면이 정확히 SDF=0 | [→](summaries/06_NeuS.md) |
| 07 | iSDF · [2204.02296](https://arxiv.org/abs/2204.02296) | 2022 | 실시간 신경 SDF + **Eikonal**(기울기=1)·법선 손실 | [→](summaries/07_iSDF.md) |
| 08 | ESLAM · [2211.11704](https://arxiv.org/abs/2211.11704) | 2022 | 3D격자 대신 **평면 3장(tri-plane)** → 10× 빠름 | [→](summaries/08_ESLAM.md) |
| 05 | Indoor Recon (Hybrid+Normal Prior) · [2309.07640](https://arxiv.org/abs/2309.07640) | 2023 | 거친+세밀 **하이브리드** + **불확실성 가중 법선** | [→](summaries/05_IndoorRecon_HybridNormalPrior.md) |
| 09 | NKSR · [2305.19590](https://arxiv.org/abs/2305.19590) | 2023 | **학습된 커널의 가중합** + 기울기 맞추기(노이즈 강건) | [→](summaries/09_NKSR.md) |
| 10 | PIN-SLAM · [2401.09101](https://arxiv.org/abs/2401.09101) | 2024 | **신경 점** 지도 + 대응점 없는 point-to-implicit 정합 | [→](summaries/10_PIN-SLAM.md) |
| 11 | MISO · [2504.19104](https://arxiv.org/abs/2504.19104) | 2025 | **조각지도(submap)** + 계층적 초기화 + 특징기반 전역정렬 | [→](summaries/11_MISO.md) |

## 한눈에 보는 "개선 축" (무엇을 좋게 만들었나)

| 개선 축 | 문제 | 대표 논문 | 핵심 수식 아이디어 |
|---|---|---|---|
| **모서리·얇은 벽 보존** | 반대편 표면이 섞여 뭉개짐 | 01 Directional, 02 DB-TSDF | 방향별 저장 $w_d=\max(0,\mathbf n\cdot\mathbf e_d)^p$ |
| **메모리·속도** | 고해상도가 무겁고 느림 | 02 DB-TSDF, 03 MrHash, 08 ESLAM | 비트마스크 AND / 분산기반 다중해상도 / tri-plane |
| **불확실성 처리** | 노이즈·못 믿을 측정이 표면을 망침 | 04 ProbFusion, 05 IndoorRecon | 역분산 가중 $w\propto 1/\sigma^2$, 불확실성 가중 손실 |
| **정확한 표면·법선** | 표면 위치·방향이 부정확 | 06 NeuS, 07 iSDF, 09 NKSR | 편향 없는 가중치, Eikonal $\lVert\nabla f\rVert=1$, 기울기 맞추기 |
| **대규모·전역 일관성** | 오래 달리면 드리프트 | 10 PIN-SLAM, 11 MISO | point-to-implicit 정합, submap 특징정렬 |

## 이 저장소(VkLBVH)와의 연결
- **01 Directional TSDF** → 우리 `DirectionalTSDF`/`AdvancedTSDF`의 직접 기반(방향별 가중평균 `sumDW/sumW`).
- **03 MrHash** → 우리 `AdaptiveVoxelGrid`의 근거(Welford 온라인 분산).
- **04 ProbFusion / 05 IndoorRecon** → 우리 `AdvancedTSDF`의 **A1(confidence weight)** 과 같은 "불확실하면 약하게" 원리.
- **11 MISO** → 우리 `TiledAdvancedTSDF`(공간 타일링)의 신경판 + 전역정렬 로드맵 참고.
- **10 PIN-SLAM** → 우리가 spec만 써둔 **direct TSDF-gradient odometry**의 최신 정석.

## 출처 (arXiv)
[2108.08115](https://arxiv.org/abs/2108.08115) · [2509.20081](https://arxiv.org/abs/2509.20081) · [2511.21459](https://arxiv.org/abs/2511.21459) · [2210.01276](https://arxiv.org/abs/2210.01276) · [2309.07640](https://arxiv.org/abs/2309.07640) · [2106.10689](https://arxiv.org/abs/2106.10689) · [2204.02296](https://arxiv.org/abs/2204.02296) · [2211.11704](https://arxiv.org/abs/2211.11704) · [2305.19590](https://arxiv.org/abs/2305.19590) · [2401.09101](https://arxiv.org/abs/2401.09101) · [2504.19104](https://arxiv.org/abs/2504.19104)

> 참고: "2020년 이후 **중요한**" 논문을 큐레이션한 것이라 전수는 아닙니다. 볼류메트릭 TSDF(우리 엔진 계열)와 이에 직접 영향을 준 신경 SDF를 균형 있게 골랐습니다. 더 넓은 범위(예: 3DGS-SLAM, ESDF 플래닝, 동적 4D)를 원하면 추가로 받아 정리할 수 있습니다.
