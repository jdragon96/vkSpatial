# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Vulkan 컴퓨트 기반 3D 공간 알고리즘 저장소(vkSpatial). depth 센서 → 정합 → TSDF 융합 → 등가면 추출로 메시를 만드는 재구성 스택이다. 코드 주석은 영어, 문서는 한국어다.

## 빌드

```bash
cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-rel --target vkspatial_tests -j8
```

- **`src/` 밑에 파일을 추가·삭제했으면 `cmake -S . -B build-rel`을 반드시 다시 돌린다.** 모든 라이브러리 타깃이 `GLOB_RECURSE`로 소스 목록을 configure 시점에 확정하므로, 재실행하지 않으면 새 `.cpp`가 빌드에 안 들어가거나 지운 `.cpp`를 계속 컴파일하려 한다. 헤더만 고쳤으면 불필요하다.
- 요구 사항: Vulkan SDK(`$VULKAN_SDK`, shaderc/glslc 포함), Eigen3, Ceres, glfw3, GTest. vk-bootstrap은 FetchContent로 받고, SPIRV-Reflect는 서브모듈이다.
- `librealsense2`는 **선택적**이다. 없으면 `VKBVH_HAS_REALSENSE`가 정의되지 않고 `Realsense::RealSenseD435::Open`이 그 사실을 말하며 던진다 — 타입과 어댑터는 그대로 빌드되므로 카메라 없는 머신에서도 녹화 재생 경로는 전부 돈다.
- `glslc`가 없으면 창을 띄우는 example만 조용히 스킵된다(래스터 셰이더를 빌드 시점에 `.spv`로 굽기 때문). 컴퓨트 커널은 영향받지 않는다 — 아래 참조.
- 릴리스 구성만 쓴다(`-O3 -DNDEBUG`). 그래서 **`assert`는 쓰지 않는다** — 실제로 배포되는 구성에서 no-op이 된다.

## 테스트

테스트는 GTest 단일 바이너리 `vkspatial_tests` 하나다(`test/*.cpp` 전체가 GLOB됨).

```bash
cmake --build build-rel --target vkspatial_tests -j8
./build-rel/test/vkspatial_tests                              # 전체
./build-rel/test/vkspatial_tests --gtest_filter='GpuIcp*'     # 하나만
./build-rel/test/vkspatial_tests --gtest_list_tests           # 이름 확인
```

대부분의 테스트가 실제 Vulkan 디바이스를 열고 컴퓨트 커널을 돌린다. 순수 CPU인 것은 `Registration::Backend`(포즈 그래프·루프 클로저), PLY I/O, 유틸리티 정도다.

## 헤드리스 진단 도구

`example2/`에는 창 없이 도는 측정 도구가 있고, 재현·회귀 확인은 여기서 한다.

```bash
cmake --build build-rel --target tsdf_folder_eval icp_quality_diag -j8
./build-rel/example2/tsdf_folder_eval  --dir scan_out          # 폴더 → TSDF → 추출 → GT 대비 RMSE
./build-rel/example2/icp_quality_diag  --replay capture --trackers icp  # 정합 드리프트 + 잔차 RMSE
./build-rel/example2/depth_capture     --replay capture        # depth 녹화 재생 (카메라 불필요)
```

창을 띄우는 디버그 뷰어(`validation_score_lab`, `realsense_scan`, `isosurface_viewer`, `object_scan_viewer`)는 대개 `--dump`/`--no-view` 헤드리스 모드를 함께 가진다. `voxel_fill_debugger`는 헤드리스 전용이다 — 뷰어 경로가 PLY 폴더로 파이프라인을 몰던 것이라 취득 축에서 PLY가 빠지면서 같이 없어졌다.

저장소에 들어 있는 실데이터 자산(정밀도 주장은 이 위에서 측정한다):

| 경로 | 내용 |
|---|---|
| `scan_out/` | 합성 스캔 프레임 PLY. 파이프라인 소스가 아니라 오프라인 도구용이다(취득 축은 depth 전용). **이미 정합되어 있음** |
| `scanData/` | 실제 chair 스캔 프레임 PLY (~827mm 스케일). 이것도 오프라인 도구용이고 **이미 정합되어 있음**(프레임 간 NN 중앙값 0.00mm) |
| `capture/` | 실제 RealSense depth 녹화(`intrinsics.txt` + Z16 `depth_*.bin`). `--replay`로 하드웨어 없이 재생 |
| `data/`, `scans/` | ground-truth 메시(`chair.ply`, `Dragon.obj`) |

## 아키텍처

### 타깃 레이어와 의존 방향

도메인은 `src/` 바로 밑에 나란히 놓이고 **각자 자기 CMake 타깃을 가진다.** `Engine`의 하위 폴더로 넣지 않는다.

```
Common            여러 라이브러리가 함께 쓰는 자료구조 하나당 파일 하나 (PointCloud) — Eigen만 의존
Engine::Core      Vulkan 디바이스/버퍼/이미지/디스크립터/ComputePipeline — 모두의 바닥
Engine::Compute   커널 실행 배치, 스테이징
Engine::Render    GLFW 창, 렌더 그래프, 카메라  (헤드리스 경로는 이걸 안 쓴다)
BVH / TSDF / Mesh 도메인 알고리즘
Registration      정합 전부 — 아래 셋으로 나뉜다
  Frontend/         프레임 하나를 맞춘다: global(FPFH→RANSAC→Ceres) / local(point-to-plane ICP, CPU·GPU)
  Backend/          여러 프레임을 한꺼번에 푼다: 포즈 그래프·루프 클로저·Lie — 순수 Eigen
  Features/         FPFH와 매칭 — frontend의 global 쪽이 그 위에 선다
Realsense         D400 깊이 프론트엔드
Pipeline          위를 조립하는 재구성 스레드들 — 알고리즘은 두지 않는다
```

**정합은 `Registration` 타깃 하나다.** frontend·backend·features는 호출 그래프가 아니라 **어휘**를
공유해서 한 타깃에 있다 — `RegistrationConfig/Param/Result`, `RigidTransform`, `LocalGrid`가
`src/Registration/` 루트에 있는 이유가 그것이다. frontend의 두 갈래(global/local)는 **단계가 아니라
대안**이고, 어느 쪽이 도는지는 Pipeline의 `TrackerRegistry`가 고른다.

**`Common`은 `Registration` 밑이 아니라 아래다.** `PointCloud`는 Registration·Pipeline이 다 쓰므로
어느 쪽도 소유할 수 없다. Eigen 외에 의존이 없어야 모두가 링크할 수 있고, `Common`이 형제 하나라도
의존하는 순간 그 형제는 `Common`을 못 쓴다. **한 타입에 파일 하나**로 둔다 — `PointCloud.h`가 헤더
20여 개 TU에 들리므로, 거기에 `<fstream>` 같은 걸 끌어들이면 전부가 대가를 치른다(그래서
`PointCloud::Save/Load`는 선언만 헤더에 있고 본문은 `PointCloud.cpp`에 있다).

**이름이 겹치면 하나를 바꾼다.** `Registration::Backend`가 `Registration` 안에 중첩되면서
`Backend::RegistrationResult`가 `Registration::RegistrationResult`를 조용히 가렸다 — 중첩
네임스페이스에서 수식 없는 이름은 안쪽에 붙고 컴파일러는 아무 말도 안 한다. 지금은
`PairwiseRegistrationConfig/Result`로 갈라 두었다.

**`src/Pipeline`에는 알고리즘을 두지 않는다.** 스레드와, 알고리즘을 `Tracker` 같은 인터페이스로 감싸는
어댑터만 둔다. 정합 알고리즘이 거기 있다가 나온 것이 `Registration`이다.

방향이 고정된 곳이 둘 있다:

- **`TSDF`는 `Mesh`를 링크하지 않는다.** 의존은 반대로 흐른다 — `Mesh::VoxelField`가 TSDF 엔트리를 읽어 필드를 만든다. 되돌려 링크하면 순환이 된다.
- **`Pipeline`은 `Engine::Render`를 PRIVATE으로만 링크한다.** `RenderThread.h`가 렌더 타입을 전방 선언하므로, 헤드리스 테스트와 진단 도구가 GLFW를 끌고 오지 않는다. 이 경계를 깨면 테스트 바이너리가 창 스택에 묶인다.

### 알고리즘 모듈의 공통 형태

**새 모듈은 `src/Realsense` 형태로 간다.** 파이프라인이 조합을 갖고, `Algorithm/` 밑에 세부 알고리즘이
각자 파일 한 벌씩 앉는다:

```
src/<도메인>/                    예: Realsense/
  <도메인>Pipeline.h / .cpp      세부 알고리즘을 조합해 결과물을 낸다. 알고리즘은 없다
  <도메인>Pipeline.md            그 조합의 실행 흐름 (단계형)
  <도메인>Types.h                경계 전용 POD — 옵션, 카운터, 푸시상수 미러
  README.md                      모듈 설명서
  Algorithm/
    Common.glsl                  이 폴더 커널들의 공용 struct·함수 (main() 없음 → 접두사 없음)
    FPFH.h / .cpp                세부 알고리즘 하나. 자기 커널 변형만 소유한다
    FPFH.<역할>.glsl             그 알고리즘의 커널
    FPFH.md                      그 알고리즘의 흐름 + 측정 근거
```

경계는 하나다: **`Algorithm/`의 클래스는 자기 알고리즘만 알고, 파이프라인이 스테이지 사이의 버퍼와
순서를 갖는다.** `Realsense::ValidationMask`가 이걸 어겼다가 고친 실물이다 — 한 패스의 이름을 단
클래스가 읽지도 않는 버퍼 여섯 개를 들고 있었고, "무엇이 무엇 뒤에 도는가"가 60줄 메서드를 읽어야
알 수 있는 사실이 됐다.

`.md`에는 **측정된 숫자를 남긴다.** 근거 없는 기본값은 다음 사람이 되돌린다.

**기존 `src/TSDF`·`src/BVH`는 이전 형태**이고 그 안에서는 그쪽을 지킨다. 세 조각으로 나뉜다:

1. **파사드**(전역 `class TSDF`, `class BVH`) — 수명·라우팅·집계만. 알고리즘 없음. 설정의 `backend` 문자열로 구현을 고른다.
2. **전략 인터페이스**(`TSDFBackend`, `BVHBackend`, `DataSplitter`) — 순수 가상 + `Make<Domain>Backend(name)` + `<Domain>BackendNames()`.
3. **구현** — `AdvancedTSDF`, `BinaryLBVH` 같은 실제 클래스는 인터페이스를 **모른다.** `.cpp`의 익명 네임스페이스 어댑터가 감싼다. 그래서 헤더에 구현 네임스페이스가 새어나오지 않는다.

이름 → 구현 레지스트리는 이 저장소 전반의 패턴이다: 트래커(`identity`/`icp`/`icp-cpu`/`global`, `TrackerRegistry`), 등가면 추출기(`mc`/`mc33`/`mtet`/`emc`/`dc`/`dmc`/`cms`, `ExtractorRegistry`+GPU 레지스트리), TSDF 백엔드, 데이터 스플리터. **새 알고리즘은 새 클래스가 아니라 새 등록 이름으로 들어간다.**

새 모듈을 만들거나 기존 구현을 스위칭 가능하게 빼는 작업은 `.claude/skills/algorithm-module` 스킬을 먼저 읽는다 — 실제로 사람을 여러 번 막았던 함정들이 거기 있다. 그 **구조 안에서 알고리즘 자체를 어떻게 설계할지**(TSDF 자료구조에 맞추기, GPU 우선, 벤치마크 가능한 전략, 용어)는 `docs/rule/Algorithm.md`에 있다.

### 컴퓨트 셰이더는 런타임에 컴파일된다

`ComputePipeline::Build("경로")`가 shaderc로 그 자리에서 GLSL을 컴파일한다. **커널을 추가·이동해도 CMake는 건드릴 필요가 없다.**

- 커널은 그것을 로드하는 `.cpp`(또는 헤더)와 **같은 폴더**에 둔다. `main()`이 없는 공용 include는 어느 규칙에서도 접두사·접미사를 붙이지 않는다(`voxel_common.glsl`, `Realsense/Algorithm/Common.glsl`).
- 파일명 규칙이 **두 가지 공존한다**. 새 코드는 아래쪽을 쓴다.
  - `kernel_<역할>.comp.glsl` — Pipeline·TSDF·BVH·Mesh의 기존 45개가 이 규칙이다. 옮기지 않았을 뿐이므로 그 폴더 안에서는 이 규칙을 지킨다.
  - `<클래스명>.<역할>.glsl` — `src/Realsense/`가 쓴다(`ValidationMask.ScanRows.glsl`, `NormalEstimation.EstimateNormal.glsl`). 커널을 소유한 클래스가 파일명에 드러나므로, 한 폴더에 여러 클래스의 커널이 섞여도 누가 로드하는지가 이름만으로 잡힌다.
- include 해석 순서: `셰이더 자기 폴더 → VKBVH_SHADER_DIR → VKBVH_SRC_DIR`. 커널을 옮겨도 공용 include는 그대로 잡힌다.
- **셰이더 경로 오류는 컴파일이 아니라 런타임에 터진다.** 이름을 바꿀 땐 `Build("x")` 직접 호출뿐 아니라 `mk("x")` 같은 래퍼도 전부 찾을 것.
- 전략 축이 셰이더 안에 있으면(예: 해시 주소법) C++ 가상 함수가 아니라 **디스패처 `.glsl` + `-D` 매크로**로 가른다(`src/TSDF/Hash/HashStrategy.glsl`). glslang은 `#include MACRO`를 지원하지 않는다. 그리고 컴파일 캐시는 **경로 + 정렬된 매크로 정의**로 키를 잡아야 한다 — 경로만으로 키를 잡으면 두 번째 전략이 첫 번째의 SPIR-V를 조용히 받는다.

한편 래스터 셰이더(`.vert`/`.frag`)는 반대로 `add_compiled_shaders()`가 빌드 시점에 `glslc`로 굽고 경로를 `-D` 매크로로 넘긴다. 즉 **컴퓨트는 런타임, 래스터는 빌드 타임**이다.

### 재구성 파이프라인 — 3스레드

`Pipeline::Pipeline`이 세 개의 `PipelineStage` 워커를 소유하고, `CommunicationModule` 하나가 그 사이를 잇는다:

```
AcquisitionThread --Channel<Frame>-->  RegistrationThread
                  --Channel<TrackedFrame>--> IntegrationThread
                  --Mailbox<ModelSnapshot>--> 호출자/렌더 스레드
```

- 취득 전략은 `EAcquisitionSource`(Realsense / RealsenseFile)로 갈리고, 정합은 `Tracker` 인터페이스로, 융합은 TSDF 백엔드로 갈린다 — 세 축이 각각 독립적으로 교체된다.
- **취득에는 추상화가 하나뿐이다 — `IProvider::IDepthProvider`(`src/Interface/IDepthProvider.h`).** 한때 그 위에 `IFrameSource`가 한 층 더 있었고(장치를 `Frame`으로 바꾸는 전략), 축 하나에 인터페이스가 둘이라 모든 도구가 장치와 소스를 따로 골라 **잘못 짝지을 수 있었다.** 그다음엔 PLY 폴더가 같은 축에 얹혀 있었는데, 그건 애초에 depth 소스가 아니라 이미 만들어진 점군이라 `AcquisitionThread` 안에 두 번째 루프를 만들었다. 지금 그 축에 있는 것은 depth 이미지 하나뿐이다: `Realsense::RealSenseD435`(라이브)와 `Realsense::RealSenseD435Recorder`(녹화·재생) 둘 다 `IDepthProvider`이고, `AcquisitionThread`는 `DepthFrontEnd`로 프레임을 만든다. 테스트는 `makeProvider`로 **장치를** 주입한다 — 합성 depth 이미지가 카메라와 똑같은 경로를 지난다. PLY는 오프라인 도구(`FrameLoader`, `tsdf_folder_eval`, `isosurface_viewer`, `voxel_fill_debugger --dump`)의 것이다.
- **깊이는 언제나 Z16이다.** provider는 `DepthFrame::rawZ16`만 채우고(자기 버퍼를 가리킨다, 다음 `Grab`까지 유효), GPU 프론트엔드가 그대로 업로드한다. metres로 풀었다 되돌리면 드라이버가 시작한 자리로 돌아오는 데 프레임당 호스트 전체 패스를 두 번 쓴다 — `capture/`의 옛 float32 포맷이 그랬다. sigma_z가 필요로 하는 `depthScale`/`stereoBaselineMeters`는 `CameraIntrinsics`에 실려 provider가 답한다. 예전엔 소비자가 구현 타입으로 `dynamic_cast`해서 캐냈고, 캐스트가 빗나가는 재생 경로가 **조용히 기본값 sigma_z로** 점수를 매겼다.
- **`DepthFrame`은 metres와 raw Z16을 둘 다 실을 수 있고, provider는 자기가 원래 갖고 있는 쪽만 채운다.** D400은 Z16을 주고 GPU 프론트엔드는 Z16을 먹으므로, 중간에서 float으로 풀었다가 되돌리면 드라이버가 시작한 자리로 돌아오는 데 프레임당 호스트 전체 패스를 두 번 쓴다. metres가 필요한 소비자(`DepthRecorder`)가 `EnsureMetres`로 요청한다.
- `CommunicationModule(dropWhenBehind)`가 두 링크의 오버플로 정책을 함께 정한다. **라이브 센서는 `true`(오래된 프레임을 버려 지연을 묶음), 녹화 재생은 `false`(블로킹 = 무손실).** 녹화를 드롭 모드로 돌리면 느린 설정이 조용히 더 적은 프레임을 처리해서, 설정 간 비교 측정이 전부 오염된다.
- **무손실은 재현성이 아니다.** 블로킹 채널은 프레임 *개수*만 맞춘다. 맵은 latest-wins `Mailbox`로 트래커에 전달되고 정합은 `trackedFrames` 용량만큼 융합보다 앞서 달릴 수 있으므로, 프레임 N이 *어느 버전의 맵*에 정합하는지가 쓰레드 스케줄링에 달렸다. 그 맵이 정합 타깃이므로 포즈가 달라지고, 다음 맵이 달라진다 — 실행마다 발산한다. 그래서 `CommunicationModule`은 녹화 모드에서 `FrameHandshake`도 켠다(정합이 매 프레임 융합 완료를 기다림 = lock-step). **파이프라인을 통과하는 A/B 측정은 이것 없이는 무의미하다.**
- **정합에 실패한 프레임을 융합할지는 `ETrackFailure`별로 갈린다**(`ShouldFuse()`, `Pipeline/Types.h`). `TooFewInliers`/`LowOverlap`은 융합하지 않는다 — 로컬 맵이 있는데도 solve가 게이트를 못 넘긴 경우이고, 그 틀린 포즈로 오염된 맵이 다음 프레임의 정합 타깃이 된다. `NoModel`/`NoLocalTarget`은 **융합한다**: 오염시킬 맵이 애초에 없고, 거부하면 맵이 부트스트랩되지 않거나(프레임 0이 `NoModel`) 새 영역으로 자라지 못한다. 건너뛴 수는 `PipelineStats::skippedFusions`로 관측한다.
- `ModelSnapshot`은 호출자에게 넘어가는 불변 스냅샷이고, 결과 복셀뿐 아니라 `voxel`/`truncationDistance`/해시 통계/`windowLimitRefusals`/단계별 소요 시간을 함께 싣는다. 트래커는 여기 실린 `voxel`로 ICP 대응 거리를 맵 해상도에 맞춰야 한다 — 고정값을 쓰면 대응점이 너무 적게 잡히는 동시에 GPU LocalGrid 셀 수가 폭발한다.
- 워커 예외는 삼켜지지 않고 `Pipeline::CheckErrors()`로 호출자에게 다시 던져진다.
- `PipelineStage`를 상속하면 **파생 클래스 소멸자에서 직접 `Stop()`을 불러야 한다.** 베이스 소멸자의 `Stop()`은 베이스 `Interrupt()`를 부르므로 파생 상태에는 이미 늦다.

### 정합 결과 읽는 법

`TrackingResult`는 `valid` 하나로 뭉뚱그리지 않고 `ETrackFailure`로 실패 원인을 나눈다: `NoModel`(맵이 아직 없음 — 초기 프레임에서 정상) / `NoLocalTarget`(그 위치에 맵이 거의 없음) / `TooFewInliers` / `LowOverlap`(개수는 찼지만 프레임 대비 비율이 낮음) / `ImplausibleMotion`(solve가 prior에서 물리적으로 불가능한 거리를 이동 — fitness가 좋아도 오수렴; `RegistrationParam::maxStepMeters`, 트래커 기본 0.08 m). **정합 증상은 이 분류부터 확인한다** — "맵이 아직 없다"와 "엉뚱한 점에 걸렸다"는 정반대 대응을 요구하는데 `valid == false`만 보면 구분되지 않는다.

**상수속도 prior는 두 조건이 다 차야 켜진다**: 연속 2회 채택(1-프레임 간격 delta 보장) AND 직전 step > 2 cm. 실측 근거(capture/, ~5 mm/frame): 거부 직후 2-간격 delta를 재적용하는 재무장은 자기유지 주기-2 진동(477 중 229 거부)을 만들었고, 올바른 1-간격 외삽조차 저속에서는 포즈 노이즈를 재적용하는 것이라 주기-3으로 재발했다(153 거부). 직전-포즈 prior만으로 476/476 추적. 상세: `docs/ICP_REGISTRATION_QUALITY.md` §9.6.

`RegistrationParam`의 절대 하한(`minInliers`)과 비율 게이트(`minFitness`)는 역할이 다르다. 대응점 거리 어닐링(`minCorrespondenceDistance`)은 기본 OFF이며, 켜도 NN 그리드는 가장 넓은 `maxCorrDist`로 **Solve당 한 번만** 만들어진다 — 좁히는 것은 질의 반경뿐이다.

## 이 저장소의 규약

- **`namespace X`와 전역 `class X`를 동시에 만들지 않는다.** C++에서 한 TU에 공존할 수 없다. 새 도메인은 파사드 이름과 같은 네임스페이스를 열지 않는다(`class BVH`는 있고 `namespace BVH`는 없는 이유).
- **인터페이스 경계에는 네임스페이스 없는 경계 전용 타입을 둔다.** `ModelSnapshot`이 `TSDFVoxel`(전역 평범한 구조체)을 드는 이유 — 구현 타입을 들면 그걸 포함하는 모든 TU가 구현 네임스페이스를 끌고 온다.
- **설정 기본값은 감싸는 구현체의 멤버 기본값과 정확히 같아야 한다.** 노브를 노출하는 변경이 동작을 바꾸면 안 된다.
- **용량 천장은 전부 관측 가능해야 한다.** 버퍼는 성장시키고(클램프 금지), 성장 불가능한 천장(해시 슬롯, 좌표 패킹 범위, 프로빙 예산)은 실패 카운터를 노출한다(`TSDF::WindowLimitRefusalCount()`, `DenseRegionClassifier::BlockInsertFailureCount()` 등). 조용히 잘리면 증상 없이 결과만 망가진다.
- **`src/`에 로깅이 없다.** 카운터를 노출하고 판단은 호출자에게 맡긴다.
- 실패는 `throw std::runtime_error("<ClassName>::<Method>: ...")`.
- **축약어를 쓰지 않는다**: `minimumFineOccupied`, `maxPointPerFrame` — `minFineOcc` 아님.
- 주석은 **왜**를 적고 무엇을 하는지는 코드가 말하게 한다.
- 서식: `.clang-format`(4칸, 탭 없음, `ColumnLimit: 0` — 줄 길이는 사람이 정함), C++17. GLSL은 `///` 섹션 배너, Allman 중괄호(`for`는 K&R), 번호 붙인 단계 주석, 탭.
- 판정에 관여하는 테스트 단언은 뮤테이션으로 검증한다 — 단언을 통과시키는 버그를 일부러 넣고 정말 빨개지는지 본다. 실제로 "두 카운터의 합"으로 점 보존을 확인하던 테스트가 원자적 슬롯 예약 버그를 통과시킨 적이 있다.
- 새 전략을 등록하면 **레지스트리 합의 테스트**(등록된 모든 이름이 실제로 빌드되고, 모르는 이름은 던진다)와 **전략별 동작 테스트**를 함께 둔다.

## 문서

`docs/`에 알고리즘 해설과 비교 실험 결과가 있다(`TSDF_*`, `ICP_*`, `ISOSURFACE_EXTRACTION.md`, `MARCHING_CUBES_SURVEY.md`, `BVH.md`, `ENGINE_CORE_RENDER.md`, `KNOWN_ISSUES_*`). 경로·네임스페이스는 2026-08-31에 코드와 맞췄지만 **측정값과 서술은 코드보다 오래됐을 수 있다** — 문서의 시그니처를 근거로 삼기 전에 실제 파일을 연다. `docs/superpowers/plans`·`specs`는 당시 기록이라 일부러 갱신하지 않는다.
