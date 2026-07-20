# FPFH 공용 구조 설계 스펙

**Goal:** FPFH(Fast Point Feature Histograms) 디스크립터를 `DirectionalTSDF`/`SimpleTSDF` 양쪽에서
공용으로 쓸 수 있게, 두 TSDF에서 떼어낸 **독립 CPU 모듈**로 추가한다.

**Architecture:** FPFH는 공통 `OrientedPointCloud`(points+normals)만 소비한다. 두 TSDF는 그 인터페이스를
노출하고, FPFH는 어느 TSDF에서 왔는지 모른다. 이웃탐색은 `NeighborQuery` 인터페이스 뒤로 추상화해
지금은 CPU 그리드로, 훗날 GPU BVH로 갈아끼운다.

**Tech Stack:** C++17, Eigen, header-only(순수 CPU). GPU/Vulkan 의존 없음(추출 단계 제외).

## Global Constraints
- **Additive only:** 기존 `Integrate`/`Extract` 경로·시그니처 불변. FPFH는 integrate 시점에 갱신하지 않음(post-extraction).
- **네임스페이스:** 전부 `Engine::Spatial`. 실패 시 `std::runtime_error("<Class>: ...")`.
- **CPU-first:** GPU 이웃탐색 미사용(Engine::Core large-N 비결정 버그 회피). 이웃탐색은 인터페이스로 추상화.
- **PCL 호환:** 디스크립터 33-dim(11 bins × 3 각도특징), 서브블록 합=100 정규화(상호검증 가능).
- **결정론:** 동일 입력 → 동일 출력. 회전/평행이동 불변.

---

## 구성요소

### ① `OrientedPointCloud` — `src/Engine/Spatial/OrientedPointCloud.h` (신규, header-only)
공통 통화. Eigen에만 의존(중립).
```cpp
struct OrientedPointCloud {
    std::vector<Eigen::Vector3f> points, normals;   // 평행 배열, 동일 길이
    size_t size() const; bool empty() const;
};
```

### ② `NeighborQuery` — `src/Engine/Spatial/NeighborQuery.h` (신규, header-only)
```cpp
struct NeighborQuery {                              // 이웃탐색 추상화
    virtual ~NeighborQuery() = default;
    virtual void Radius(const Eigen::Vector3f& p, float radius,
                        std::vector<uint32_t>& outIdx, std::vector<float>& outDist) const = 0;
};
class CpuGridNeighborhood : public NeighborQuery {  // 균일 해시 그리드(셀=radius), O(1) 평균
    // origin = 클라우드 bbox min → 셀 좌표 ≥ 0, 21비트/축 패킹 키.
    // 생성 시 점집합 참조(non-owning) 저장; 질의 시 3×3×3 인접 셀 스캔.
};
```

### ③ FPFH — `src/Engine/Spatial/FPFH.h` (신규, header-only)
```cpp
struct FpfhConfig    { float radius = 0.25f; int bins = 11; };   // radius는 호출자가 ~2.5×voxelSize로
struct FpfhSignature { std::array<float, 33> hist{}; };          // [0..10]=α, [11..21]=φ, [22..32]=θ
std::vector<FpfhSignature> ComputeFPFH(const OrientedPointCloud&, const FpfhConfig&, const NeighborQuery&);
std::vector<FpfhSignature> ComputeFPFH(const OrientedPointCloud&, const FpfhConfig&); // CPU 그리드 자동
```
알고리즘(Rusu 2009):
1. **pair feature**(p_s,n_s,p_t,n_t): Darboux 프레임 `u=n_s, v=(p_t−p_s)×u(정규화), w=u×v`;
   `α=v·n_t`, `φ=u·(p_t−p_s)/d`, `θ=atan2(w·n_t, u·n_t)`. 대칭성 위해 두 점 중 연결선과
   법선각이 작은 쪽을 source로 선택.
2. **SPFH_i**: 반경 내 이웃 각 pair를 11빈 히스토그램에 누적 → 서브블록 합=100 정규화.
3. **FPFH_i = SPFH_i + (1/k)·Σ_j (1/‖p_i−p_j‖)·SPFH_j** → 서브블록 합=100 재정규화.
4. 빈: α,φ∈[−1,1], θ∈[−π,π] 균등 11분할, 클램프.

### ④ 두 TSDF의 공통 추출 (수정)
- `DirectionalTSDF.h`: `OrientedPointCloud ExtractOrientedCloud() const` — 기존 `m_pointCloud`(ExtractedPoint)
  position+normal 복사(inline). 기존 API 불변.
- `SimpleTSDF.h/.cpp`: `OrientedPointCloud ExtractPointCloud() const` — MC(`voxel_tsdf_mc.comp`)를
  in-memory 실행(ExportMC의 다운로드를 private 헬퍼 `downloadMCVertices()`로 추출·공유) →
  정점 웰딩(공간 해시, 톨러런스 ~voxelSize·1e-3) → 삼각형 면적가중 노멀 → `OrientedPointCloud`.

---

## 테스트 (`test/test_fpfh.cpp`)
**CPU 전용(Context 불필요):**
- `CpuGridNeighborhood` 반경질의 = 브루트포스 일치(랜덤 점).
- **회전/평행이동 불변성**: 랜덤 오리엔티드 클라우드 → FPFH; 랜덤 (R,t)로 점·법선 변환 → FPFH 빈차 < eps.
- 동일 클라우드 → 동일 시그니처. 각 서브블록 합 ≈ 100.
- 평면/구 합성면 히스토그램 sanity.

**GPU 통합(Context 필요, 소규모 N):**
- `SimpleTSDF` 구 포인트 통합 → `ExtractPointCloud()` 비어있지 않음, 노멀 ~단위, `ComputeFPFH` size 일치.
- `DirectionalTSDF` 소규모(N<1000, large-N 버그 회피) 통합 → `ExtractOrientedCloud()` size == `PointCloud()`.

## CMake
변경 없음: 신규 헤더는 `Engine::Spatial` public include(`src/`)로 해결, 신규 테스트 `.cpp`는
`file(GLOB)`로 수집(재구성 필요), `SimpleTSDF.cpp` 수정은 기존 `GLOB_RECURSE`에 이미 포함.

## 비목표(후속)
매칭/전역정합(FPFH+RANSAC/FGR), GPU FPFH, 디스크립터 캐싱. → [ICP_CURRICULUM](../../ICP_CURRICULUM.md) L3.
