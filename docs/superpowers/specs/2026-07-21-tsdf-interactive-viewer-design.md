# TSDF 인터랙티브 뷰어 (Vulkan + ImGui) 설계

> 배경: `example2/tsdf_slice_debug`는 입력/추출/SDF-슬라이스를 PLY로 export → CloudCompare로 봐야 했다. 이 스펙은 그 진단을 **엔진 자체 Vulkan 스택 + ImGui UI**로 인앱 인터랙티브하게 만든다.

## 배경 / 목적

DirectionalTSDF의 integrate→extract 파이프라인 검증을, 외부 뷰어 없이 **창을 띄워 실시간으로** 본다: 입력 점군·추출 점군·숨은 TSDF 볼륨의 SDF 슬라이스를 색점으로 렌더하고, ImGui 패널로 씬·품질·표시 레이어를 토글하며, 트랙볼로 회전. integrate 버그(필드 오류)와 extract 버그(0등고선 오류)를 눈으로 분리 진단한다.

실현성은 이미 확인됨: `Engine::Render`에 windowed 앱 프레임워크(`Application`/`Camera`/`RenderGraph`/트랙볼/동적렌더링 `GraphicsPipeline`)가 있고, **ImGui가 glfw+vulkan 백엔드까지 `lib/imgui/`에 벤더링**돼 있다. 새로 짤 것은 2개 pass + 뷰어 앱뿐이다.

## 범위 (v1 — 핵심 뷰어)
- **씬**: 합성 fixture 2종 — `plane`(z=0 +Z), `interproximal`(0.4mm 간격 마주보는 두 면). `tsdf_slice_debug`의 fixture/슬라이스 로직 재사용.
- **렌더**: 색점 점군(입력=흰, 추출=방향색/초록, SDF 슬라이스=diverging 색), 트랙볼 회전, 원근 카메라.
- **UI(ImGui)**: 씬 선택, 통합 품질(maxDirections 1/2 + viewAngleWeight), 레이어 표시 토글(입력/추출/슬라이스+Z/슬라이스−Z), 추출 색모드(방향/단색), 통계(점 수·추출 |z| 평균·층별 0등고선 z), 재통합 버튼.
- 씬/품질 변경 → DirectionalTSDF 재빌드·통합·추출 → 점 버퍼 재생성.

### 범위 밖 (후속)
- 실 scanData/chair 로딩, 이동식 슬라이스 평면 슬라이더, 스크린샷 저장, single-vs-multi 나란히.
- 마칭큐브/메시 렌더(우선순위 최하).

## 아키텍처

`cube_render.cpp` 패턴 재사용: `Application` + `Camera` + `RenderGraph`(pass 추가) + 트랙볼 + `app.Run()`.

### 1. `PointCloudPass` (신규 `Engine::Render::RenderPass`)
- vertex `struct PointVertex { float pos[3]; uint8_t rgba[4]; }`, `GraphicsPipelineDescriptor` — `topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST`, `.VertexBinding<PointVertex>()` + `.VertexAttribute`(pos: R32G32B32_SFLOAT, color: R8G8B8A8_UNORM), `.ColorTarget(swapFormat)`, `.DepthTarget(depthFormat)`, `.PushConstant<{Matrix4f mvp; float pointSize;}>(VERTEX)`.
- 셰이더: `pointcloud.vert`(gl_Position = mvp*pos; gl_PointSize = pointSize; out color), `pointcloud.frag`(원형 마스크로 둥근 점, out color). `example2/Shaders/` 또는 `src/shader/`.
- 여러 **점 세트**(input/extracted/sliceP/sliceN)를 각자 `Engine::Core::Buffer` + count + `bool visible`로 보유. Execute: `RenderingScope` → `Bind` → 각 visible 세트마다 vertex bind + `PushConstants(mvp)` + `vkCmdDraw(count,1,0,0)`.
- 앱이 점 세트를 교체할 수 있게 `SetPointSet(id, vertices)` / `SetVisible(id, bool)` 제공.

### 2. `ImGuiPass` (신규)
- 생성 시 벤더링된 백엔드 초기화: `ImGui::CreateContext`, `ImGui_ImplGlfw_InitForVulkan(window, true)`(GLFWwindow* 필요 — `GlfwWindow`에서 노출), `ImGui_ImplVulkan_Init(&info)` with `UseDynamicRendering=true` + `PipelineRenderingCreateInfo{colorAttachmentCount=1, pColorAttachmentFormats=&swapFormat}`, 전용 `VkDescriptorPool`, `MinImageCount/ImageCount`(SwapChain에서), Instance/PhysicalDevice/Device/Queue(Context에서).
- Execute: `ImGui_ImplVulkan_NewFrame` + `ImGui_ImplGlfw_NewFrame` + `ImGui::NewFrame` → UI 빌드(공유 상태에 기록) → `ImGui::Render` → `RenderingScope`(로드 유지) 안에서 `ImGui_ImplVulkan_RenderDrawData(draw_data, cmd)`. 렌더그래프의 **마지막 pass**로 추가(점 위에 UI).
- 소멸 시 백엔드 shutdown + 디스크립터풀 파괴.

### 3. `tsdf_viewer` 앱 (`example2/tsdf_viewer.cpp`)
- 공유 상태 `struct ViewerState { int scene; int maxDirections; bool viewAngle; bool showInput/Extracted/SliceP/SliceN; int extractColorMode; bool dirty; ... stats; };`
- `rebuild(state)`: fixture 생성(plane/interproximal) → `DirectionalTSDF.Build`+`SetIntegrationQuality`+`Integrate` → 점 세트 생성(입력, 추출[PointCloud, dirMask 색], 슬라이스[DebugDownloadGroupVoxels로 층별 SDF 색]) → `PointCloudPass.SetPointSet`. 통계 계산.
- 렌더그래프: `PointCloudPass` + `ImGuiPass`. 트랙볼로 카메라. 루프에서 `state.dirty`면 rebuild.
- `slice_debug`의 값/색/좌표 헬퍼(sdfColor/dirColor/valueAt/floorDiv8) 재사용(공유 헤더로 추출하거나 복사).

## 검증 (GUI라 단위테스트 오라클 없음)
- **빌드**: `--target tsdf_viewer` 성공.
- **실행**: 창이 뜨고 Vulkan **validation layer 에러 없이** 프레임을 렌더(콘솔에 validation 에러/경고 없음). 헤드리스 자동화는 어려우므로, 실행이 즉시 크래시/validation-error 없이 최소 N프레임 도는지로 스모크 확인 + 사용자가 창에서 육안 확인.
- **정합성**: 창에서 plane은 추출점이 흰 슬라이스 0등고선(z=0) 위, interproximal은 두 슬라이스가 z=∓0.2로 분리·추출점 두 색 분리 — `tsdf_slice_debug`의 수치(이미 검증됨)와 일치.

## 리스크
| 리스크 | 대응 |
|---|---|
| ImGui-Vulkan 초기화(디스크립터풀·image count·동적렌더링 info) | 벤더링된 백엔드가 dynamic rendering 지원; CubePass의 `RenderingScope`/swap format 재사용해 포맷 일치 |
| MoltenVK에서 dynamic rendering + ImGui 호환 | 이미 엔진이 dynamic rendering으로 cube/shadow 렌더 중 → 검증됨; ImGui도 같은 경로 |
| GLFWwindow* 접근 | `GlfwWindow`가 미노출이면 accessor 1개 추가 |
| 점 렌더 gl_PointSize (MoltenVK) | POINT_LIST + gl_PointSize 지원 확인; 미지원 시 instanced quad로 대체 |
| GUI 검증 오라클 부재 | 빌드+launch-smoke(validation 무에러) + 수치 대조(slice_debug) + 사용자 육안 |
| 세션 상태 공유(ImGuiPass↔앱) | `ViewerState`를 참조로 공유, dirty 플래그로 rebuild |
