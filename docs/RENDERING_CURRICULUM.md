# Rendering 커리큘럼 — 래스터화부터 뉴럴 렌더링까지

**목적:** 실시간 렌더링을 **기초 → 현대 GPU 파이프라인 → 레이트레이싱/GI → 뉴럴 렌더링**의
순서로 학습하는 로드맵. 각 레벨은 이 저장소(VkLBVH)의 `Engine::Render`(Vulkan) 컴포넌트와
연결된 실습을 포함한다. 2025–2026 최신 기술(뉴럴 셰이더, DLSS 4, GPU Work Graphs,
Radiance Cascades, 3D Gaussian Splatting)까지 이어진다.

**대상:** C++/GPU 프로그래밍 기본이 있고, 실시간 렌더러를 밑바닥부터 이해하려는 사람.
**선수지식:** 선형대수(행렬·벡터), C++, 약간의 GPU/셰이더 경험.
**사용법:** 레벨은 순차적이다. 각 모듈의 **완료 기준**을 통과하면 다음으로. 실습은
`Engine::Render`와 `example2/`의 예제(`cube_render2`, `shadow_map`, `blinn_phong`)를 뼈대로 삼는다.

> 관련 문서: [ENGINE_CORE_RENDER](ENGINE_CORE_RENDER.md) · [VULKAN_PIPELINE_STAGES](VULKAN_PIPELINE_STAGES.md)

---

## 로드맵 한눈에

| 레벨 | 주제 | 핵심 산출물 | 최신도 |
|---|---|---|---|
| **L0** | 수학·GPU 실행 모델·Vulkan 기초 | 삼각형 하나 그리기 | 항상 유효 |
| **L1** | 래스터화 파이프라인 | 텍스처·깊이 있는 메시 렌더 | 항상 유효 |
| **L2** | 셰이딩 & 라이팅 (PBR·그림자) | PBR + shadow map | 표준 |
| **L3** | 현대 GPU-driven 파이프라인 | deferred/clustered, 컬링, mesh shader | 2018–현재 |
| **L4** | 레이트레이싱 & 실시간 GI | HW RT, ReSTIR, GI 기법 | 2020–현재 |
| **L5** | 뉴럴 렌더링 & 프론티어 | 업스케일링, 뉴럴 셰이더, 3DGS | 2023–2026 |

각 레벨은 **목표 / 핵심 개념 / 자료 / VkLBVH 실습 / 완료 기준**으로 구성된다.

---

## L0 — 수학·GPU 실행 모델·Vulkan 기초

**목표:** 좌표 공간과 GPU 실행 모델을 이해하고 Vulkan으로 삼각형 하나를 화면에 띄운다.

**핵심 개념**
- 좌표 공간 체인: model → world → view → clip → NDC → screen. 동차좌표, 원근 분할.
- 카메라: view/projection 행렬, FOV, near/far, 좌우수 좌표계.
- GPU 실행 모델: SIMT, warp/wavefront, 점유율(occupancy), 메모리 계층(레지스터/공유/전역).
- Vulkan 객체 모델: instance/physical/logical device, queue, swapchain, command buffer,
  render pass vs **dynamic rendering**, pipeline, descriptor set, 동기화(semaphore/fence/barrier).

**자료:** *Vulkan Tutorial* (vulkan-tutorial.com), Khronos Vulkan Spec/Samples, *Vulkan Guide* (vkguide.dev),
*Real-Time Rendering* 4th ed. Ch.2–4, *Foundations of Game Engine Development* Vol.1.

**VkLBVH 실습**
- `example2/cube_render.cpp` + `CubePass.cpp`(`cube_render2` 타깃)를 읽고 빌드해서 큐브를 띄운다.
- [ENGINE_CORE_RENDER](ENGINE_CORE_RENDER.md)로 `Engine::Core::Context`(device/queue/cmd) 생성 흐름을 추적.
- `Camera.h`의 view/projection 구성이 위 좌표 체인과 어떻게 맞는지 확인.

**완료 기준:** 회전하는 큐브를 렌더하고, 정점이 clip space까지 가는 각 변환 단계를 말로 설명할 수 있다.

---

## L1 — 래스터화 파이프라인

**목표:** 정점→프래그먼트 파이프라인 전체를 이해하고 텍스처·깊이 버퍼가 있는 메시를 렌더한다.

**핵심 개념**
- 파이프라인 단계: IA → vertex shader → (tessellation/geometry) → 래스터화 → fragment shader → ROP.
  ([VULKAN_PIPELINE_STAGES](VULKAN_PIPELINE_STAGES.md)와 대조.)
- 무게중심 좌표 보간, perspective-correct interpolation.
- Depth test / Z-buffer, depth precision(역-Z, reverse-Z), Z-fighting.
- 텍스처링: UV, 밉맵, 필터링(bilinear/trilinear/anisotropic), sRGB vs linear.
- 컬링(back-face), 클리핑, viewport/scissor, 블렌딩.

**자료:** *Real-Time Rendering* Ch.5–6, LearnOpenGL(개념 이식용), `Engine::Render::GraphicsPipeline::Build()`.

**VkLBVH 실습**
- `blinn_phong` 예제의 `Shaders/BlinnPhong.vert.glsl` / `.frag.glsl`을 분석.
- `GraphicsPipelineDescriptor` 빌더로 depth test on/off, cull mode를 바꿔 결과를 비교.
- 텍스처 샘플러를 추가해 UV 매핑된 메시를 렌더.

**완료 기준:** 텍스처·조명 없는 라이팅으로 메시를 렌더하고, reverse-Z가 왜 깊이 정밀도를 개선하는지 설명한다.

---

## L2 — 셰이딩 & 라이팅 (PBR · 그림자)

**목표:** 물리 기반 셰이딩(PBR)과 그림자 매핑을 구현한다. 현대 렌더러의 "룩"의 기반.

**핵심 개념**
- 라이팅 기본: Lambert diffuse, Blinn-Phong specular, 감쇠, 광원 종류(점/방향/스팟/면광).
- **PBR (물리 기반 렌더링):** 렌더링 방정식, microfacet BRDF(Cook-Torrance),
  정규분포함수(GGX/Trowbridge-Reitz), 기하 항(Smith), 프레넬(Schlick), 에너지 보존,
  metallic-roughness 워크플로. **IBL**(irradiance map + prefiltered env + BRDF LUT).
- 노멀 매핑, 탄젠트 공간, 감마/톤매핑, HDR, 노출.
- **그림자 매핑:** depth from light, PCF, 바이어스(peter-panning/acne), 캐스케이드(CSM),
  분산/모멘트 섀도(VSM/ESM).

**자료:** *Real-Time Rendering* Ch.9(셰이딩)·Ch.7(그림자), *Physically Based Rendering* (pbr-book.org, 오프라인이지만 이론의 정본),
Google **Filament** 문서(실시간 PBR의 실전 정본), *moving-frostbite-to-pbr* (SIGGRAPH 코스).

**VkLBVH 실습**
- `blinn_phong`을 **Cook-Torrance GGX PBR**로 확장(metallic/roughness 파라미터).
- `shadow_map` 예제(`example2/ShadowMap.cpp`)로 방향광 그림자를 켜고, PCF 커널 크기·바이어스를 튜닝.
- 여러 광원 + 노멀맵을 추가.

**완료 기준:** metallic/roughness를 바꾸며 물리적으로 그럴듯한 재질을 렌더하고, GGX·프레넬·기하 항 각각의 역할을 설명한다.

---

## L3 — 현대 GPU-driven 파이프라인

**목표:** 오늘날 게임 엔진(UE5, idTech, Frostbite)이 쓰는 대규모·GPU 주도 렌더 구조를 이해한다.

**핵심 개념**
- **셰이딩 아키텍처:** forward → deferred(G-buffer) → **Forward+ / clustered / tiled** 라이팅 →
  **visibility buffer**(deferred material). 트레이드오프(대역폭 vs 오버드로우 vs 재질 다양성).
- **GPU-driven rendering:** indirect draw(`vkCmdDrawIndirectCount`), GPU 컬링(frustum/occlusion/
  **two-pass HZB occlusion**), draw compaction, bindless 리소스(descriptor indexing).
- **Mesh shader 파이프라인:** meshlet, task/mesh shader, per-meshlet 컬링. Nanite류 가상화 지오메트리의 기반.
  최신: **GPU Work Graphs**(mesh nodes) — GPU가 스스로 작업을 생성/스케줄. NVIDIA **Mega Geometry**,
  AMD·Samsung **Dense Geometry Format(DGF)**로 RT용 지오메트리 압축.
- **시간적 기법:** TAA, 모션 벡터, 지터, 히스토리 리젝션. 업스케일링의 전제.
- 컬러 파이프라인: HDR, 톤매핑(ACES 등), 자동 노출, 블룸.

**자료:** *Real-Time Rendering* Ch.19–20, GDC/SIGGRAPH 발표(UE5 **Nanite**, *GPU-Driven Rendering Pipelines* by Haar & Aaltonen),
[GPUOpen: GPU Work Graphs mesh nodes in Vulkan](https://gpuopen.com/learn/gpu-workgraphs-mesh-nodes-vulkan/),
*A trip through the Graphics Pipeline* (Fabian Giesen).

**VkLBVH 실습**
- `Engine::Render`의 `RenderGraph`/`RenderPass` 구조를 이해하고 **deferred G-buffer 패스**를 추가.
- `Engine::Core`/`Engine::Compute`의 compute 디스패치로 **frustum culling** 컴퓨트 패스를 만들어 indirect draw로 연결.
- (심화) `VK_EXT_mesh_shader`가 지원되면 meshlet 렌더 경로를 프로토타이핑.

**완료 기준:** deferred vs Forward+의 대역폭/오버드로우 트레이드오프를 설명하고, GPU 컬링이 왜 CPU 드로우콜 병목을 없애는지 안다.

---

## L4 — 레이트레이싱 & 실시간 GI

**목표:** 하드웨어 레이트레이싱과 실시간 전역조명(GI)의 현대 기법군을 이해한다.

**핵심 개념**
- **하드웨어 RT:** `VK_KHR_ray_tracing_pipeline`/`ray_query`, **BLAS/TLAS**(가속구조 — 이 저장소의
  [BVH](BVH.md)와 개념 연결), ray-gen/closest-hit/any-hit/miss 셰이더, 인라인 RT(ray query).
- **경로추적 & 샘플링:** 몬테카를로, 중요도 샘플링, MIS, 러시안 룰렛.
- **ReSTIR:** reservoir 기반 spatiotemporal 재사용 — **ReSTIR DI**(직접광) → **GI** → **ReSTIR PT**(경로 재사용).
  적은 샘플로 수천 광원/간접광을 실시간 처리하는 현재 표준.
- **실시간 GI 기법 지형도:**
  - 프로브 기반: **DDGI**(dynamic diffuse GI), irradiance volume.
  - 복셀: voxel cone tracing(VXGI), **Lumen**(UE5, 소프트웨어+하드웨어 하이브리드).
  - 레이디언스 캐싱: **Neural Radiance Cache(NRC)**, **SHaRC**(spatial hash radiance cache).
  - **Radiance Cascades:** 레이디언스 장을 거리/각 해상도로 계단식 분해 — 2D에서 시작해 3D·게임 적용 확산(2024–2025 화제).
- **디노이징:** SVGF/A-SVGF, 그리고 뉴럴 디노이저(→ L5의 Ray Reconstruction).

**자료:** *Ray Tracing Gems* I·II(무료), *Physically Based Rendering* Ch.13–16,
Bitterli et al. **ReSTIR**(SIGGRAPH 2020) 및 후속(ReSTIR GI/PT), NVIDIA **RTXGI/DDGI**·**RTX Kit** 문서,
Alexander Sannikov **Radiance Cascades** 논문, [NVIDIA RTX Kit](https://developer.nvidia.com/rtx-kit).

**VkLBVH 실습**
- 이 저장소의 CPU/GPU **BVH**([BVH.md](BVH.md), `src/BVH/`)를 RT 가속구조의 축소판으로 보고,
  ray-triangle 교차로 **간단한 경로추적기**(오프라인, ground-truth 용)를 작성 — L5 뉴럴 기법의 레퍼런스로 활용.
- (HW RT 지원 환경) `VK_KHR_ray_query`로 인라인 RT 그림자/AO를 래스터 패스에 얹기.

**완료 기준:** ReSTIR가 "reservoir로 샘플을 재사용"하는 원리를 설명하고, DDGI/Lumen/Radiance Cascades를 각각 언제 쓰는지 구분한다.

---

## L5 — 뉴럴 렌더링 & 프론티어 (2023–2026)

**목표:** 실시간 렌더링을 재편 중인 뉴럴/방사장(radiance field) 기술을 이해하고, 이 저장소의
재구성 파이프라인(TSDF/스캔)과의 접점을 잡는다.

**핵심 개념**
- **뉴럴 업스케일링 & 프레임 생성:** DLSS / FSR / XeSS의 원리(시간적 슈퍼샘플링 + 뉴럴).
  **DLSS 4**: transformer 기반 super resolution, **Multi Frame Generation**, **Ray Reconstruction**(뉴럴 디노이징이
  수작업 디노이저를 대체). Reflex 2.
- **뉴럴 셰이더 / Cooperative Vectors:** 셰이더 안에서 텐서 연산 실행 —
  Vulkan **`VK_NV_cooperative_vector`**(및 크로스벤더 Cooperative Vectors), Slang **RTXNS**, Blackwell tensor↔shader 통합.
  응용: **뉴럴 텍스처/머티리얼 압축**, 뉴럴 BRDF, in-shader radiance cache.
- **방사장(Radiance Fields):**
  - **NeRF** 계보: 볼륨 렌더링 + MLP → **Instant-NGP**(해시 인코딩으로 실시간 학습) → Mip-NeRF 등.
  - **3D Gaussian Splatting(3DGS)** (SIGGRAPH 2023): 명시적 가우시안 프리미티브 + 타일 기반 미분가능 래스터화 →
    1080p 실시간(≥30fps) novel-view synthesis. 현재 이 분야의 주류.
  - 3DGS 변형: **2D Gaussian Splatting(2DGS)**·GOF(표면 정확도), **Mip-Splatting**(안티에일리어싱),
    **Deformable/4D GS**(동적), **FlashGS**·GEMM-GS(가속), 가지치기(Speedy-Splat). `gsplat`/`nerfstudio` 생태계.
- **하이브리드(래스터+뉴럴):** 게임 엔진이 뉴럴 셰이더·업스케일러·RT를 한 프레임에 조합하는 방향.

**자료:** Kerbl et al. **3D Gaussian Splatting**(SIGGRAPH 2023, [arXiv 2308.04079](https://arxiv.org/abs/2308.04079)),
Müller et al. **Instant-NGP**(SIGGRAPH 2022), Huang et al. **2DGS**(SIGGRAPH 2024),
[NVIDIA 뉴럴 렌더링/RTX Kit 블로그](https://developer.nvidia.com/blog/announcing-the-latest-nvidia-gaming-ai-and-neural-rendering-technologies/),
[DLSS 4 심층 해설](https://www.tomshardware.com/pc-components/gpus/nvidia-neural-rendering-deep-dive-full-details-on-dlss-4-reflex-2-mega-geometry-and-more),
`graphdeco-inria/gaussian-splatting`, `nerfstudio`/`gsplat`.

**VkLBVH 실습**
- **3DGS 뷰어**를 별도 예제로 붙이기: 학습된 `.ply` 가우시안을 로드해 타일 기반 정렬·알파 블렌딩으로 래스터화
  (이 저장소의 PLY I/O·`Engine::Render` 재사용).
- 이 저장소의 스캔 데이터셋(`scan_dataset_gen`, `ObjectScanner`)으로 만든 멀티뷰를 NeRF/3DGS 학습 입력으로 내보내,
  **TSDF 재구성 ↔ 방사장 재구성**을 같은 데이터로 비교(→ [TSDF_CURRICULUM](TSDF_CURRICULUM.md)와 교차).

**완료 기준:** 3DGS가 NeRF 대비 왜 실시간에 유리한지(명시적 프리미티브 + 미분가능 래스터화) 설명하고,
뉴럴 셰이더(cooperative vectors)가 기존 셰이더 파이프라인에 무엇을 더하는지 안다.

---

## 도구 · 프레임워크

- **API/디버깅:** Vulkan, RenderDoc, NVIDIA Nsight Graphics, PIX, Tracy(프로파일링).
- **셰이더:** GLSL/HLSL, **Slang**(뉴럴 셰이더·cooperative vectors), SPIR-V, `glslc`/`slangc`.
- **뉴럴/방사장:** PyTorch, tiny-cuda-nn, `nerfstudio`, `gsplat`, `graphdeco-inria/gaussian-splatting`.
- **벤더 SDK:** NVIDIA RTX Kit(NRC/SHaRC/RTXNS/DLSS), AMD GPUOpen(FSR, Work Graphs, DGF).

## 학습 순서 요약

L0·L1(파이프라인) → L2(PBR·그림자) → L3(GPU-driven) 까지가 **실시간 래스터 렌더러의 골격**.
L4(RT·GI)와 L5(뉴럴)는 이후 관심사에 따라 병행 가능하되, L5의 3DGS는 이 저장소의 **재구성/스캔**
파이프라인과 직접 이어지므로 TSDF·ICP 커리큘럼과 함께 보면 시너지가 크다.

## 참고문헌 (핵심)
- Akenine-Möller, Haines, Hoffman et al., *Real-Time Rendering*, 4th ed., 2018. (렌더링의 정본)
- Pharr, Jakob, Humphreys, *Physically Based Rendering*, 4th ed. (pbr-book.org)
- Karis, *Real Shading in Unreal Engine 4* / Google **Filament** 문서. (실시간 PBR)
- Bitterli et al., *Spatiotemporal Reservoir Resampling (ReSTIR)*, SIGGRAPH 2020. (+ ReSTIR GI/PT 후속)
- Müller et al., *Instant Neural Graphics Primitives (Instant-NGP)*, SIGGRAPH 2022.
- Kerbl, Kopanas, Leimkühler, Drettakis, *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023.
- Huang et al., *2D Gaussian Splatting for Geometrically Accurate Radiance Fields*, SIGGRAPH 2024.
- NVIDIA, *RTX Kit / Neural Shaders / DLSS 4* (2025), GPUOpen *GPU Work Graphs / DGF* (2025).
