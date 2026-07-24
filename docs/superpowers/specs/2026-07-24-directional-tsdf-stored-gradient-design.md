# 저장형 Gradient-SDF (Stored-Gradient DirectionalTSDF, measure-first GPU-only) 설계

> 배경: lite sub-voxel 추출(유한차분 gradient를 voxel-center에서 Newton 투영)은 실험으로 **기각**됐다 — flat/curved 위치를 악화시키고, 위치만 건드려 **법선은 원리적으로 개선 불가**했다(`2026-07-22-directional-tsdf-subvoxel-extraction-design.md` 실험 결과). 그런데 FPFH/등록에 실제로 먹히는 건 법선. 이 스펙은 **복셀에 관측 법선을 누적 저장**해 추출 법선을 유한차분 대신 denoised 정확 법선으로, 위치를 그 정확한 법선 방향 투영으로 개선한다. 지속성(host-store/wire/writeback)은 **이득 확인 후 후속**으로 미루는 measure-first GPU-only v1.

## 목적
DirectionalTSDF 통합 시 각 샘플의 관측 법선 `nrm`을 복셀에 가중 누적(`sumN = Σ w·n`)한다. 추출에서 `n̂ = normalize(sumN)`이 유한차분 gradient를 대체해 **정확·denoised 법선**을 주고(진짜 목표 — FPFH/등록 직결), 위치는 `p = center − c·truncation·n̂`로 그 정확한 방향에 투영해 lite가 못 넘은 legacy 위치까지 개선을 노린다. 단일 윈도우 실험에서 기존 3-way 하네스(legacy / FD / **stored**)로 정량 입증한다.

기반은 이미 존재: 통합 셰이더는 이미 `nrm`을 로컬로 들고 있고(방향 선택·뷰가중에 사용, `directional_tsdf_integrate.comp` L81), pool voxel 크기는 전부 `sizeof(GpuTsdfVoxel)` 기반이라 struct 확장 시 자동 리사이즈된다. `NearestNormal` 오라클(159c54f)과 2-테이블 `--dump` 하네스(4a5d2ea)가 법선각·위치 오차를 이미 측정한다. 신규 = voxel 3필드 확장 + 통합 누적 + 추출 mode==2 + residency upload zero-fill + 하네스 3번째 열.

## 단위 근거 (설계 전제)
- 저장 값 `c = sumDW/sumW`는 truncation 정규화(`newValue = clamp(sdf/truncation, -1, 1)`, integrate L103). 따라서 metric 부호거리 `= c·truncation`.
- integrate의 `sdf = depth − dot(vCenter−cam, rayDir)`는 **바깥(카메라쪽 자유공간)에서 +**, 안쪽에서 −. `∇sdf`는 증가 방향 = 바깥 = **바깥 표면 법선 n̂**과 정렬. 표면점 = `center − sdf·n̂ = center − c·truncation·n̂`. ✓
- `sumN` 스케일은 정규화 `normalize(sumN)`에서 상쇄되므로 방향에 무관 → 기존 `TSDF_SCALE(=10000)` 재사용(오버플로 안전: `|sumNx| ≤ sumW` 규모, int32 여유).

## 범위 (v1)
- **복셀 확장**: `GpuTsdfVoxel` 8B→20B (+`int sumNx,sumNy,sumNz`). `HostTsdfVoxel`은 **불변(8B)** — gradient 비지속(measure-first).
- **통합**: 관측 법선을 방향 레이어별로 가중 누적(atomicAdd 3필드).
- **추출**: `g_refine`(0/1) → `g_mode`(0=legacy, 1=FD-projection[기존 lite, 비교점 유지], 2=stored-gradient). mode==2 = 정확 법선 + 위치 투영.
- **residency**: upload 변환부 `sumN=0` zero-fill(Streaming), download/writeback은 sumN drop. pool 크기는 sizeof 기반 자동.
- **API+하네스**: `SetSubvoxelRefine(bool)` → `SetExtractMode(uint32_t)`; `tsdf_feature_compare --dump`를 3-way(legacy/FD/stored)로 확장, 위치+법선각 열 추가.
- **성공 기준(§검증)**: cube flat **법선각 mean이 stored ≪ legacy/FD**(결정적, 헤드라인), 그리고 stored 위치 mean ≤ legacy(flat/curved).

### 범위 밖 (후속)
- host-store/wire/writeback 지속성(eviction 넘어 gradient 유지) — HostTsdfVoxel 확장 + Streaming/Unified 직렬화 + writeback.
- 얇은 구조 스트레스, sharp 메시(DC/QEF), 20B 메모리 압축(법선 oct-encode 등).
- FD-projection(mode 1) 자체 개선(기각됐으므로 비교점으로만 유지).

## 아키텍처

### 1. 복셀 레이아웃 — `DirectionalTSDFTypes.h` (수정)
```cpp
struct GpuTsdfVoxel {
    int32_t  sumDW = 0;
    uint32_t sumW  = 0;
    int32_t  sumNx = 0, sumNy = 0, sumNz = 0; // Σ n·w·TSDF_SCALE (per direction layer)
}; // 20B
```
- `static_assert(sizeof(GpuTsdfVoxel) == 8)` → `== 20`; `offsetof(sumW)==4` 유지 + `offsetof(sumNx)==8` 등 추가.
- `HostTsdfVoxel`(8B), `DirectionalCandidate`, `ExtractedPoint`는 불변.

### 2. 통합 — `directional_tsdf_integrate.comp` (수정)
- `struct GpuVoxel { int sumDW; uint sumW; }` → `{ int sumDW; uint sumW; int sumNx; int sumNy; int sumNz; }` (20B, 셰이더-측 미러).
- 기존 `atomicAdd(sumDW…)`/`atomicAdd(sumW…)` 뒤(L113-114 이후)에:
  ```glsl
  atomicAdd(g_pool[addr].sumNx, int(nrm.x * w * TSDF_SCALE));
  atomicAdd(g_pool[addr].sumNy, int(nrm.y * w * TSDF_SCALE));
  atomicAdd(g_pool[addr].sumNz, int(nrm.z * w * TSDF_SCALE));
  ```
  `nrm`은 world-frame 관측 법선(L81), `w = viewFactor*rel[di]`(L112). 방향 레이어별 누적이라 같은 레이어의 법선은 일관(코너에서도 레이어가 분리).

### 3. 추출 — `directional_tsdf_extract.comp` (수정)
- `struct GpuVoxel`를 20B로. PC의 `uint g_refine` → **`uint g_mode`**, 그리고 **`float g_truncation` 추가**(mode==2 위치 크기용).
- 중심 voxel의 `sumNx/y/z`를 읽어:
  ```glsl
  vec3 sumN = vec3(float(center.sumNx), float(center.sumNy), float(center.sumNz));
  ...
  if (g_mode == 2u) {
      float sl = length(sumN);
      if (sl < 1e-6) { /* fallback */ pos = posSum/float(crossings); n = grad/len; }
      else {
          vec3 nhat = sumN / sl;
          vec3 voxelCenter = (vec3(v)+vec3(0.5))*g_voxelSize;
          pos = voxelCenter - c * g_truncation * nhat;   // c = sumDW/sumW
          n = nhat;                                       // denoised normal
      }
  } else if (g_mode == 1u) { /* 기존 FD-projection */ }
  else { /* legacy */ }
  ```
- mode 0/1은 법선을 기존 유한차분 유지; mode 2만 `n = normalize(sumN)`. 방출 게이트(crossing 존재)·candidate 방출·counter는 불변.

### 4. residency upload/download — `StreamingResidencyBackend.cpp` (수정), `UnifiedResidencyBackend.cpp` (감사)
- **Streaming upload**(L253-256): mapped 스테이징에 필드별 구성 → **`g.sumNx = g.sumNy = g.sumNz = 0;` 명시 추가**(mapped 메모리는 기본생성 안 됨 → 안 하면 garbage). value/weight 변환은 불변.
- **Streaming download**(L184-185, L359-361): `sumW/sumDW`만 읽어 host로 → sumN 자연 drop. 불변.
- **Unified**(L98-111): first-touch `memset(0)`가 sumN까지 0. upload 필드 구성 없음(UMA 직접). download(L150-151) sumN drop. → **코드 변경 불필요, sizeof 자동 리사이즈만 확인**.
- `kGroupBytes`/pool 크기는 전부 `sizeof(GpuTsdfVoxel)` 기반 → 자동. Streaming L13 주석 `// 4096`은 갱신(→10240).
- **layout 감사**: pool voxel을 실제 주소하는 셰이더는 **integrate + extract 둘뿐**(`register_reusable`는 `PoolIndexList`만, classify는 pool 미접근). 둘 다 20B로 통일 확인.

### 5. C++ API + dispatch — `DirectionalTSDF.{h,cpp}` (수정)
- `SetSubvoxelRefine(bool)` → **`void SetExtractMode(uint32_t mode)`**(0/1/2; 기본 0=legacy). 멤버 `uint32_t m_extractMode = 0;`.
- `ExtractPC`: `uint32_t refine` → `uint32_t mode`; **`float truncation` 추가**. dispatch(L285-288)에서 `m_extractMode`, `m_truncation` 기록.

### 6. 하네스 3-way — `example2/tsdf_feature_compare.cpp` (수정)
- `RunCompare`: DirectionalTSDF **3 인스턴스** — `SetExtractMode(0)` legacy / `(1)` FD / `(2)` stored, 동일 입력 Build+Integrate. 각 위치+법선 수집.
- `CompareResult`: 기존 refined 필드(`dirPtsRefined`/`dirStatsRefined`/`dirNormStatsRefined`)는 그대로 **mode 1(FD)** 를 담고(개명 불필요, 최소 변경), **추가로** `dirPtsStored`/`dirNormalsStored` + `dirStatsStored`(위치) + `dirNormStatsStored`(법선각)를 신설.
- `PrintReport`: 위치표·법선각표의 `DirRef.*`(=FD) 열 옆에 `Stored.*` 열 추가(legacy / FD / stored 3-way). 헤더 라벨만 `DirRef`→`FD`로 정리(수치 로직 불변).
- 뷰어는 범위 밖(refine 기본 off 원칙 유지). `--dump`만 확장.

### 7. 파일 요약
- 수정: `src/Engine/Spatial/DirectionalTSDFTypes.h`(struct+assert), `src/shader/directional_tsdf_integrate.comp`(누적), `src/shader/directional_tsdf_extract.comp`(mode 2+truncation), `src/Engine/Spatial/StreamingResidencyBackend.cpp`(upload zero-fill), `src/Engine/Spatial/DirectionalTSDF.{h,cpp}`(SetExtractMode+PC), `example2/tsdf_feature_compare.cpp`(3-way).
- 감사(무변경 확인): `UnifiedResidencyBackend.cpp`(memset 자동 0), `register_reusable`/`classify` 셰이더.
- 불변: HostTsdfVoxel, DirectionalCandidate/ExtractedPoint, mergeCandidates, 등록/FPFH 경로.

## 검증
- **빌드**: `--target vkspatial_tests tsdf_feature_compare` 성공.
- **회귀**: mode 0(legacy) 경로가 기존과 **cube --dump 비트 동일**(20B 확장·누적이 legacy 추출을 안 바꿈).
- **수치(헤드라인, 결정적 cube)**: `--dump --shape cube` →
  - **법선각 flat: `Stored.mean ≪ FD.mean(=1.8773°)/legacy`** — stored 누적법선이 유한차분을 크게 개선(이 스펙의 핵심 성공 신호).
  - **위치 flat: `Stored.mean ≤ legacy(0.0110)`** — lite(0.0115) 실패를 뒤집음. 최소 무회귀.
  - edge: 참고치(법선 불연속).
- **수치(cylinder, curved)**: `--dump --shape cylinder` → curved 위치·법선각에서 stored 개선 확인. **주의**: 브랜치 base(c4cb890)는 large-N fix(main `ddee668`) 미포함이라 cylinder는 ±1-2점 지터 → 여러 회 실행 mean으로 판정(지터는 mean을 ~0.001 이내로만 흔듦). 결정적 측정을 원하면 main으로 rebase(선택).
- **launch-smoke**: `--frames N` exit 0(뷰어 경로 회귀 없음).
- **판정**: cube flat 법선각이 유의 개선 안 되면 누적/정규화/부호 버그. 위치가 legacy보다 나쁘면 부호/truncation 크기 버그.

## 리스크
| 리스크 | 대응 |
|---|---|
| pool 바인딩 셰이더 layout 불일치 → 주소 계산 깨짐 | 감사 결과 integrate+extract 둘만 pool voxel 주소 → 둘 다 20B 통일; register_reusable/classify 무관 확인됨 |
| Streaming upload가 mapped 메모리 필드별 구성이라 sumN 미초기화 garbage | L253-256에 `g.sumNx=g.sumNy=g.sumNz=0` 명시(Unified는 memset로 자동) |
| sumN degenerate(코너/빈 관측)로 normalize 불안정 | `length(sumN)<1e-6`면 legacy(위치)+FD(법선)로 fallback |
| `c·truncation·n̂` 부호/크기 오류 | 단위 근거로 부호 고정; cube flat 위치 무회귀로 검증(틀리면 즉시 드러남) |
| 20B 메모리 2.5×(~335MB @ 기본 poolCapacity) | M4 Max UMA 여유; 부족 시 실험에서 poolCapacity 축소, 압축은 후속 |
| cylinder 비결정성(base에 large-N fix 없음) | cube를 결정적 primary 오라클로; cylinder는 다회 mean, 필요시 main rebase |
| gradient 비지속(measure-first) | 단일 윈도우 실험엔 무영향(eviction 없음); 이득 확인 후 host-store/wire 후속 |
