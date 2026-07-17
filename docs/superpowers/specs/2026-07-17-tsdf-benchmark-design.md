# SimpleTSDF vs DirectionalTSDF 벤치마크 설계

## 배경 / 목적

`Engine::Spatial`에는 두 개의 TSDF 재구성 시스템이 있다: `SimpleTSDF`(단일 SDF, marching-cubes 삼각형 추출, 스트리밍 캐시 없음)와 `DirectionalTSDF`(방향별 6-layer SDF, 방향별 포인트 추출, host/GPU 스트리밍 캐시). 지금까지 각각을 개별 검증했지만(eval 하니스로 DirectionalTSDF 정확도, 벤치로 submit 수), **동일 데이터에 대해 둘을 나란히 비교하는** 코드가 없다. 이 스펙은 같은 합성 스캔에 대해 두 시스템을 품질(GT 대비 RMSE)과 성능(재구성 end-to-end 시간, 출력 점 수)으로 비교하는 벤치마크를 정의한다.

핵심 관전 포인트: interproximal(마주보는 두 얇은 면) 장면에서 single-SDF는 두 표면이 상쇄되어 completeness가 나빠지고, directional은 방향 레이어 분리로 양면을 보존한다 — 이 차이를 숫자로 보인다.

## 범위

- `SimpleTSDF`에 `std::vector<Eigen::Vector3f> ExtractMCVertices() const` **additive** 추가; 기존 `ExportMC`를 이 접근자를 쓰는 얇은 래퍼로 리팩터(DRY). `test/`에 정점 반환을 확인하는 작은 GTest 하나.
- `src/Engine/Eval/SyntheticSurface.h`에 header-only `TwoPlaneSurface`(interproximal용 합성 표면) 추가.
- `example2/tsdf_benchmark.cpp` 신규: 3장면(구/평면/interproximal) × 2시스템(Simple/Directional)을 품질+성능 표로 출력.

### 범위 밖 (YAGNI)

- 정밀 메모리 측정(GPU 버퍼 크기는 파라미터에서 자명), 센서 노이즈, 벤치마크 자체의 RMSE threshold GTest 가드(DirectionalTSDF 품질 가드는 기존 `test_directionalTSDFEval.cpp`에 이미 있음).
- `SimpleTSDF`의 알고리즘/추출 방식 변경 — MC 정점을 인메모리로 노출만 한다.

## 아키텍처

### 1. `SimpleTSDF::ExtractMCVertices` (additive)

현재 `SimpleTSDF::ExportMC(path, maxTris)`는 (a) `voxel_tsdf_mc.comp` MC 커널을 돌려 삼각형 정점(`Vec4{x,y,z,w}`)을 다운로드하고 (b) PLY로 쓴다. (a)를 인메모리 반환 메서드로 분리한다:

```cpp
// SimpleTSDF.h 에 추가:
// Runs marching cubes and returns the triangle vertex positions in memory
// (3 vertices per triangle, in triangle order). Symmetric to DirectionalTSDF::PointCloud().
std::vector<Eigen::Vector3f> ExtractMCVertices(uint32_t maxTris = 500000u) const;
```

- 구현: 기존 `ExportMC` 본문에서 커널 실행 + 정점 다운로드 부분을 그대로 옮겨 `Eigen::Vector3f` 벡터(각 `Vec4`의 x,y,z)로 반환.
- `ExportMC(path, maxTris)`는 `auto verts = ExtractMCVertices(maxTris);` 후 PLY 헤더 + 정점 + face 인덱스를 쓰는 래퍼가 된다. **출력 PLY는 리팩터 전과 바이트 동일**해야 한다(정점 순서·face 인덱싱 불변).
- `Reset()`/`Integrate()`/`FilledCount()` 등 다른 API는 불변.

### 2. `TwoPlaneSurface` (Engine::Eval, header-only)

마주보는 두 유한 평면 패치. 왼쪽은 x=−gap에서 +X를 향하고, 오른쪽은 x=+gap에서 −X를 향한다(데모의 interproximal 구성과 동일).

```cpp
class TwoPlaneSurface : public Surface {
public:
    // gap: half-distance between the two planes; extent: in-plane half-size (y,z square).
    TwoPlaneSurface(float gap, float extent);
    float Distance(const Eigen::Vector3f &p) const override;      // min(|x+gap|, |x−gap|)
    Eigen::Vector3f NormalAt(const Eigen::Vector3f &p) const override; // p.x<0 → +X else −X
    std::vector<Eigen::Vector3f> SampleDense(uint32_t approxCount) const override; // union of two grids
};
```

- `Distance` = 두 무한 평면(x=±gap)까지 거리의 최소값. 재구성 점은 스캔된 패치 근처에만 생기므로 유한 패치 무시가 실용상 문제없다(`PlaneSurface`와 동일 근거).
- `SampleDense(n)`: 각 패치를 `~n/2`개씩 격자 샘플; 왼쪽 점은 x=−gap·y·z, 오른쪽은 x=+gap. 두 벡터를 합쳐 반환.
- `NormalAt`: `p.x() < 0.0f ? (+1,0,0) : (−1,0,0)`. 스캔 가시성/노멀 생성에 쓰인다.

### 3. `example2/tsdf_benchmark.cpp`

세 장면을 정의하고 각각 두 시스템으로 재구성해 한 표로 출력한다.

**장면 (`Surface` + 카메라 궤도 파라미터)**
1. **sphere**: `SphereSurface(origin, 1.0)`, orbit radius 3.0, center origin.
2. **plane**: `PlaneSurface(origin, +Z, 1.0, 1.0)`, orbit radius 3.0, center (0,0,1.5).
3. **interproximal**: `TwoPlaneSurface(gap=0.2, extent=1.0)`, orbit radius 3.0, center origin.

**시스템별 재구성 (공통 스캔 프레임)**
- 프레임 = `GenerateOrbitScan(surface, {SampleDense(4000), ...})`.
- **DirectionalTSDF**: `Build(ctx, 0.1, 0.3)`; 각 프레임 `Integrate(points, normals, cam, hint)`; 표면 점 = `PointCloud()` positions. end-to-end 시간 = 모든 Integrate 합(추출은 프레임마다 내부 포함).
- **SimpleTSDF**: `Build(ctx, 0.1, 0.3, hashCapacity=1<<20, maxPoints=1<<15)`(클리핑 방지); 각 프레임 `Integrate(points, cam)`(normal/hint 미사용); 표면 점 = `ExtractMCVertices()`. end-to-end 시간 = 모든 Integrate 합 + `ExtractMCVertices` 1회.

두 시스템 모두 "스캔 프레임 → 최종 표면 점" wall-clock을 `std::chrono::steady_clock`으로 측정(둘 다 동기 실행이라 wall-clock이 정확).

**품질**
각 시스템 표면 점에 대해 `AccuracyRMSE(pts, surface)` + `CompletenessRMSE(surface.SampleDense(8000), pts)`.

**출력 (한 표)**
```
scene         system      totalMs  surfacePts  accuracyRMSE  completenessRMSE
sphere        Simple      ...      ...         ...           ...
sphere        Directional ...      ...         ...           ...
plane         Simple      ...
plane         Directional ...
interproximal Simple      ...
interproximal Directional ...
```
CSV(`tsdf_benchmark.csv`)로도 저장. PLY export는 하지 않는다(육안 비교는 기존 eval/demo가 담당; 벤치는 수치 표에 집중).

`example2/CMakeLists.txt`에 `voxel_tsdf_mc`와 동일 패턴으로 `tsdf_benchmark` 실행 파일 등록(`Engine::Spatial` 링크).

## 데이터 흐름

```
Surface(sphere / plane / interproximal)
  → frames = GenerateOrbitScan(surface, SampleDense(4000))
  → [Directional] Build → per-frame Integrate(pts,normals,cam,hint) → PointCloud()   (timed)
  → [Simple]      Build → per-frame Integrate(pts,cam) → ExtractMCVertices()          (timed)
  → for each system: AccuracyRMSE(pts, surface), CompletenessRMSE(SampleDense(8000), pts)
  → print table + write tsdf_benchmark.csv
```

## 파일 변경 목록

| 파일 | 변경 |
|---|---|
| `src/Engine/Spatial/SimpleTSDF.h` | `ExtractMCVertices` 선언 추가 (additive) |
| `src/Engine/Spatial/SimpleTSDF.cpp` | `ExtractMCVertices` 정의; `ExportMC`를 이를 쓰는 래퍼로 리팩터 |
| `src/Engine/Eval/SyntheticSurface.h` | `TwoPlaneSurface` 추가 (header-only) |
| `example2/tsdf_benchmark.cpp` (신규) | 3장면×2시스템 품질+성능 벤치 |
| `example2/CMakeLists.txt` | `tsdf_benchmark` 실행 파일 등록 |
| `test/test_directionalTSDFEval.cpp` | `ExtractMCVertices` 정점 반환 GTest + `TwoPlaneSurface` CPU 테스트 |

`DirectionalTSDF`/`Engine::Compute`/`Engine::Core`는 변경하지 않는다.

## 검증 기준 (성공 조건)

- `ExtractMCVertices`가 비어있지 않은 정점을 반환하고, `ExportMC`의 PLY 출력이 리팩터 전과 동일(정점 순서/개수 불변) — GTest + example 실행으로 확인.
- `TwoPlaneSurface`: 손계산 Distance/NormalAt, SampleDense가 두 x=±gap 클러스터를 모두 포함 — CPU 테스트.
- 벤치 실행: 세 장면 모두 두 시스템의 totalMs·surfacePts·RMSE가 유한값으로 출력. interproximal에서 SimpleTSDF의 completenessRMSE가 DirectionalTSDF보다 크게(양면 중 한쪽/중간 상쇄) 나오는지 관찰 — 이는 관찰 결과로 보고하고 assert하지 않는다(하드 임계는 형상/파라미터 민감).
- 전체 테스트 스위트가 기존 baseline 유지(`WideBVHTest.RadiusMatchesCpuReference`만 실패).
