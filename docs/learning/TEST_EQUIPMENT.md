# 3D Vision / Mesh 테스트 장비 가이드

당신 상황: **Apple M4 Max**(Metal/MoltenVK로 VkLBVH 엔진 구동, UMA 39GB) + TSDF·Mesh 재구성 연구 + LiDAR 스캔 관심. 아래는 "무엇을 테스트하느냐"에 따른 센서·컴퓨트 추천입니다. (가격은 대략치 — 구매 전 현시세 확인)

---

## 0. 먼저 정하기 — 3개 질문
1. **스케일?** 손바닥 물체(cm) / 방(m) / 야외(수십 m).
2. **모달리티?** RGB-D(ToF/스테레오) / LiDAR / 사진(포토그래메트리·NeuS) / 구조광(고정밀).
3. **목적?** 알고리즘 입력(노이즈 있는 실데이터) vs **정답 메시(ground truth)** 확보.

당신 파이프라인(`object_scan_viewer`/`tsdf_folder_eval`)은 **oriented-point PLY 프레임 폴더 + ground_truth.ply** 를 먹으므로, "프레임별 점군+법선을 주는 센서"면 바로 붙습니다.

---

## 1. 컴퓨트 (가장 중요) ⭐
- **M4 Max는 당신 Vulkan/Metal 엔진엔 훌륭**(UMA라 zero-copy, 큰 TSDF도 OK). 하지만 **최신 신경 3D 논문(NeuS·iSDF·ESLAM·NKSR·PIN-SLAM)은 거의 CUDA/PyTorch** → Mac에서 재현이 고통.
- **추천: NVIDIA GPU 1대 추가** (신경 3D 실험용)
  - **RTX 4090 24GB**(신품) 또는 **중고 RTX 3090 24GB**(가성비 최고, VRAM 24GB가 핵심) — NeuS/NKSR/3DGS 학습에 24GB 권장.
  - 예산 여유: **RTX 5090 32GB** / 워크스테이션 **RTX 6000 Ada 48GB**(대형 씬).
  - 로봇/임베디드 타깃이면 **Jetson Orin (AGX 64GB / NX 16GB)** — 온디바이스 SLAM 실측.
- **결론:** Mac(엔진 개발) + NVIDIA 24GB(신경 재현) 듀얼이 이상적.

---

## 2. RGB-D 카메라 (가성비·생태계 최강, 첫 센서로 추천) ⭐
| 모델 | 방식 | 강점 | 용도 |
|---|---|---|---|
| **Intel RealSense D455** | 스테레오+IR | 넓은 FoV, 실내 방 스캔, Open3D/ROS 완비 | 방 규모 TSDF 첫 실데이터 |
| **Intel RealSense D405** | 스테레오(근거리) | **근거리 고정밀**(7cm~), 물체 스캔 | 손바닥 물체 mesh |
| **Orbbec Femto Bolt/Mega** | ToF | Azure Kinect 후속(단종품 대체), 깊이 품질 좋음 | 실내 고품질 깊이 |
| **Microsoft Azure Kinect DK** | ToF | 깊이 우수(단, 단종 — 중고만) | (있으면 좋음) |
| **Luxonis OAK-D (Pro)** | 스테레오+NPU | 온디바이스 추론, 저렴 | 엣지·로봇 |
| **Stereolabs ZED 2i / X** | 스테레오 | **야외 가능**, 넓은 범위 | 야외·이동로봇 |

- **첫 구매 추천:** **RealSense D455**(방) + 필요시 **D405**(물체). 둘 다 저렴하고 Open3D `rs`·ROS 지원, PLY+법선 export 쉬움 → 당신 folder-eval에 바로.

---

## 3. LiDAR (당신 관심사 — 스캐닝 스케일) ⭐
| 모델 | 방식 | 강점 | 용도 |
|---|---|---|---|
| **Livox Mid-360** | 솔리드스테이트 | **저렴·SLAM 연구 표준**, 360° | 핸드헬드/모바일 매핑, LiDAR-TSDF |
| **Livox Avia** | 솔리드스테이트 | 전방 고밀도 | 물체·전방 스캔 |
| **Ouster OS0/OS1** | 회전식 | 고해상도·거리, ROS 완비 | 야외·자율주행급 |
| **Hesai / RoboSense** | 회전식 | 산업급 | 대형 매핑 |

- **첫 LiDAR 추천:** **Livox Mid-360** — SLAM 논문(FAST-LIO, DB-TSDF의 Newer College류)에서 사실상 표준, 가격도 착함. 당신 Directional/DB-TSDF 계열 실데이터에 최적.
- 공개 데이터로 먼저: **Newer College**(논문 02 DB-TSDF가 씀), **KITTI/MulRan** — 장비 없이도 folder-eval 검증 가능.

---

## 4. 고정밀 물체 스캐너 (정답 메시 GT 확보용) ⭐
알고리즘 정확도를 재려면 **ground truth 메시**가 필요. 여기서 정밀 스캐너가 값을 함:
| 등급 | 모델 | 정확도 | 비고 |
|---|---|---|---|
| 소비자 | **Revopoint MINI 2 / POP 3** | 0.02~0.05mm급 | 저렴, 소물체 GT |
| 소비자+ | **Shining3D EinScan SE/H** | 0.05mm급 | 중형 물체 |
| 산업 | **Zivid 2+** / **Photoneo MotionCam** | 매우 높음(수십 µm) | 비쌈, 진짜 GT |

- **추천:** 알고리즘 벤치를 진지하게 할 거면 **Revopoint**로 시작(싸고 충분한 GT). Chamfer/F-score 비교의 기준 메시 확보.

---

## 5. 포토그래메트리 / 멀티뷰 (NeuS·neural surface용)
- **미러리스 카메라**(Sony a6xxx 등) + **턴테이블** + **확산 조명** → 물체를 여러 각도 촬영 → NeuS/NKSR/COLMAP.
- 저렴 대안: **iPhone/iPad Pro** — LiDAR + 고해상 카메라 + ARKit(포즈·깊이 export). **이미 있다면 0원 첫 장비**(AR-스케일, object capture, Polycam/Record3D로 PLY+포즈 export → folder-eval 직결).

---

## 6. 예산별 추천 번들
- **💸 최소 (≈ iPhone만):** iPhone/iPad Pro LiDAR + Record3D/Polycam → PLY 프레임 export. 지금 당장 folder-eval 실데이터 확보. **0~소액.**
- **💵 입문 (~$1–1.5k):** RealSense **D455**(+D405) + 중고 **RTX 3090 24GB**. 방·물체 RGB-D + 신경 재현.
- **💰 연구 (~$3–5k):** 위 + **Livox Mid-360**(LiDAR) + **Revopoint**(GT 메시). Directional/DB-TSDF 실데이터 + 정량 벤치 풀셋.
- **🏦 본격 (~$10k+):** RTX 5090/6000 Ada + Ouster + Zivid(산업 GT).

---

## 7. 당신 파이프라인 연결 요령
- 어떤 센서든 **프레임별 oriented-point PLY**(x y z nx ny nz)로 뽑아 한 폴더에 넣으면 `tsdf_folder_eval` / `voxel_fill_debugger`가 바로 처리.
- 법선이 없으면 Open3D `estimate_normals()` 로 붙이면 됨.
- GT 메시가 있으면 표면 샘플링해 `ground_truth.ply` 로 두면 folder-eval의 accuracy/completeness/RMSE가 동작.
- **먼저 iPhone/공개데이터(Newer College)로 시작** → 알고리즘 흐름 검증 후, RealSense·Livox로 확장하는 순서를 추천.
