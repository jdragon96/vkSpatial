# GPU Dense 영역 분리 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 프레임의 점군을 GPU에서 base/detail 두 파티션으로 가르고, 어느 블록이 detail 레벨을 받을지와 그 블록이 몇 슬롯을 쓸지를 같은 패스에서 결정한다.

**Architecture:** `DenseRegionClassifier`라는 독립 컴포넌트로 만든다. dispatch 세 개 — 점당 통계 누적, 블록당 판정, 점당 atomic append. `SubmapStrategy`가 이걸 소유하고 `DividePoint`에서 호출하지만, 컴포넌트 자체는 어느 전략에도 의존하지 않으므로 진행 중인 `Memory/`→`Structure/` 이행과 무관하게 빌드·테스트된다.

**Tech Stack:** C++17, Vulkan(Engine::Core/Compute), GLSL(shaderc), GoogleTest.

**Spec:** [`docs/superpowers/specs/2026-08-15-dense-region-separation-design.md`](../specs/2026-08-15-dense-region-separation-design.md)

## Global Constraints

- 네임스페이스는 **`TSDF`**. 단일 동사 이름. 식별자에 약어 금지.
- 동결 백엔드(`SimpleTSDF` / `DirectionalTSDF` / `CompactDirectionalTSDF`)와 그 셰이더는 건드리지 않는다.
- 커널은 호출 CPP 옆에, `<CppStem>.<kernel>.comp.glsl` 규칙.
- **clamp 금지.** `maxPointPerFrame`을 넘는 프레임이 오면 버퍼를 성장시킨다. clamp는 조용히 점을 버린다.
- 세 패스 모두 호출자의 `CommandBatch`에 기록만 하고 자체 제출하지 않는다.
- 판정 임계값은 `DensityCriteria` 하나에 모아 컴포넌트 헤더에 둔다. `VolumeParams`로 올리지 않는다.
- 빌드/테스트:
  - `VULKAN_SDK=/usr/local cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release`
  - `VULKAN_SDK=/usr/local cmake --build build-rel --target vkspatial_tests -j8`
  - `./build-rel/test/vkspatial_tests --gtest_filter='DenseRegion*'`
- 파일 추가 후 cmake **configure**를 다시 돌린다 (GLOB).
- 기준선: 전체 스위트 **287 passed / 1 skipped / 0 failed**. skip은 `EngineWideBVHTest.RadiusMatchesCpu`로 이 작업 이전부터 있던 것.

## 스펙에서 정정된 사항 두 가지

**1. 기존 해시 계약을 그대로 재사용하지 않는다.** 스펙 §5는 `findOrInsert`를 재사용한다고 했으나,
그 계약은 `DirEntry` 엔트리 구조체와 `g_firstFrame` 스탬프·`g_filledCount`에 묶여 있다. 여기서 필요한
것은 **키 전용 "처음 보는 셀인가"** 하나뿐이다. `voxel_common.glsl`의 `wangHash` / `EMPTY_KEY` /
`MAX_PROBE`는 재사용하고, 10줄짜리 탐사 루프는 이 컴포넌트 전용으로 둔다. 억지로 계약을 넓히면
TSDF 쪽 커널이 쓰지 않는 분기를 지게 된다.

**2. 셀 키는 해시가 아니라 정확한 좌표로 만든다.** 32비트에 전역 셀 좌표는 안 들어가지만,
**블록 안의 지역 좌표**는 들어간다. 블록은 32 base voxel이므로 미세 셀은 축당 64개(6비트),
굵은 셀은 32개(5비트):

```
fineCellKey   = (blockIndex << 18) | (fz << 12) | (fy << 6) | fx      // 14 + 18 = 32비트
coarseCellKey = (blockIndex << 15) | (cz << 10) | (cy << 5) | cx      // 14 + 15 = 29비트
```

`blockIndex`는 프레임당 블록 레코드의 인덱스(≤16384). 이러면 충돌이 원천적으로 없어 점유 수가
정확하다. 좌표를 해싱해서 저장하면 충돌이 점유를 과소 계상하고, 그 오차가 그대로 판정에 들어간다.

## 파일 구조

| 파일 | 책임 |
|---|---|
| `src/TSDF/Structure/DenseRegionClassifier.h` (신규) | `DensityCriteria`, `BlockRecord`, 컴포넌트 인터페이스 |
| `src/TSDF/Structure/DenseRegionClassifier.cpp` (신규) | 버퍼 수명 주기, 세 dispatch 기록, 판독 |
| `src/TSDF/Structure/DenseRegionClassifier.accumulate.comp.glsl` (신규) | 패스 1 — 점당 통계 |
| `src/TSDF/Structure/DenseRegionClassifier.classify.comp.glsl` (신규) | 패스 2 — 블록당 판정 |
| `src/TSDF/Structure/DenseRegionClassifier.partition.comp.glsl` (신규) | 패스 3 — 점당 atomic append |
| `src/TSDF/Structure/DenseRegionClassifier.clear.comp.glsl` (신규) | 프레임 시작 시 셀 해시 비우기 |
| `test/test_denseRegion.cpp` (신규) | 전 태스크의 테스트 |

---

### Task 1: 통계 누적 패스

블록 레코드와 두 셀 해시를 만들고, 점당 한 번씩 통계를 모은다. 판정은 아직 하지 않는다.

**Files:**
- Create: `src/TSDF/Structure/DenseRegionClassifier.h`, `.cpp`, `.accumulate.comp.glsl`, `.clear.comp.glsl`
- Test: `test/test_denseRegion.cpp`

**Interfaces:**
- Consumes: `Engine::Core::Buffer`, `Engine::Core::ComputePipeline`, `Engine::Compute::CommandBatch`
- Produces: `TSDF::DensityCriteria`, `TSDF::BlockRecord`, `TSDF::DenseRegionClassifier::{Build, Reset, Record, BlockCount, ReadBlocks}`

- [ ] **Step 1: 실패하는 테스트 작성**

`test/test_denseRegion.cpp`:

```cpp
#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"
#include "TSDF/Structure/DenseRegionClassifier.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using Eigen::Vector3f;

namespace {

    // A planar +Z patch of `count` x `count` samples spaced `spacing` apart, centred on the origin
    // of one block. Spacing is the knob every density assertion turns.
    void MakePlane(std::vector<Vector3f> &points, std::vector<Vector3f> &normals,
                   float spacing, int count) {
        points.clear();
        normals.clear();
        const float half = 0.5f * spacing * float(count - 1);
        for (int i = 0; i < count; ++i)
            for (int j = 0; j < count; ++j) {
                points.emplace_back(float(i) * spacing - half, float(j) * spacing - half, 0.0f);
                normals.emplace_back(0.0f, 0.0f, 1.0f);
            }
    }

} // namespace

TEST(DenseRegionAccumulate, CountsPointsAndOccupancyPerBlock) {
    Engine::Core::Context context;
    TSDF::DenseRegionClassifier classifier;
    classifier.Build(context, /*baseVoxel=*/0.01f, /*blockVoxels=*/32, /*maxPointPerFrame=*/1u << 15);

    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, /*spacing=*/0.0025f, /*count=*/64); // 4096 points, one block

    Engine::Compute::CommandBatch batch(context);
    classifier.Record(points, normals, batch);
    batch.Submit();

    const std::vector<TSDF::BlockRecord> blocks = classifier.ReadBlocks();
    ASSERT_EQ(blocks.size(), 1u) << "a 0.16 m patch at 0.32 m blocks must land in one block";
    EXPECT_EQ(blocks[0].pointCount, points.size());
    EXPECT_GT(blocks[0].fineOccupied, 0u);
    EXPECT_GT(blocks[0].coarseOccupied, 0u);
    EXPECT_GE(blocks[0].fineOccupied, blocks[0].coarseOccupied)
            << "a finer grid can never have fewer occupied cells";
}

// The ratio is the whole spacing estimate: occupiedFine/occupiedCoarse = min(4, (v/s)^2).
// At s = v/4 the fine grid is fully resolved (ratio ~4); at s = 2v neither grid is (ratio ~1).
TEST(DenseRegionAccumulate, OccupancyRatioTracksSampleSpacing) {
    Engine::Core::Context context;
    const float baseVoxel = 0.01f;

    auto ratioAt = [&](float spacing, int count) {
        TSDF::DenseRegionClassifier classifier;
        classifier.Build(context, baseVoxel, 32, 1u << 15);
        std::vector<Vector3f> points, normals;
        MakePlane(points, normals, spacing, count);
        Engine::Compute::CommandBatch batch(context);
        classifier.Record(points, normals, batch);
        batch.Submit();
        const std::vector<TSDF::BlockRecord> blocks = classifier.ReadBlocks();
        EXPECT_EQ(blocks.size(), 1u) << "spacing " << spacing;
        return double(blocks[0].fineOccupied) / double(blocks[0].coarseOccupied);
    };

    EXPECT_NEAR(ratioAt(baseVoxel * 0.25f, 64), 4.0, 0.6) << "s = v/4 resolves the fine grid";
    EXPECT_NEAR(ratioAt(baseVoxel * 2.0f, 8), 1.0, 0.3) << "s = 2v resolves neither grid";
}
```

- [ ] **Step 2: 실패 확인**

```bash
VULKAN_SDK=/usr/local cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release
VULKAN_SDK=/usr/local cmake --build build-rel --target vkspatial_tests -j8
```
Expected: `fatal error: 'TSDF/Structure/DenseRegionClassifier.h' file not found`

- [ ] **Step 3: 헤더 작성**

`src/TSDF/Structure/DenseRegionClassifier.h`:

```cpp
#pragma once

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Buffer.h"
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"

#include <Eigen/Core>
#include <cstdint>
#include <memory>
#include <vector>

namespace TSDF {

    // When a block earns a detail level. All four conditions are ANDed: any one of them alone
    // admits blocks whose detail level would be waste, and the detail level is a second full tile
    // hierarchy -- the measured dominant memory cost.
    struct DensityCriteria {
        // occupiedFine / occupiedCoarse over a surface equals min(4, (v/s)^2), so 3.2 means
        // "sample spacing s <= v / sqrt(3.2) ~= 0.56 v" -- the fine grid is actually resolved.
        float occupancyRatio = 3.2f;
        // |sum(normal)| / pointCount: ~1 on a plane, lower on curvature and edges. A densely
        // scanned flat face has nothing to refine, and this is the only condition that sees that.
        float normalCoherence = 0.9f;
        // Samples per fine cell after refinement. Below this the refined voxels are noise.
        float samplesPerFineCell = 3.0f;
        // Absolute floor so a block glimpsed by a handful of points cannot latch to dense.
        uint32_t minimumFineOccupied = 64u;
    };

    // Per-block statistics, accumulated across frames. Mirrors the GPU-side layout exactly:
    // 8 tightly packed 4-byte scalars.
    struct BlockRecord {
        uint32_t blockKey;        // packed block coordinate; 0xFFFFFFFF = empty slot
        uint32_t pointCount;      // cumulative
        uint32_t coarseOccupied;  // cumulative sum of per-frame counts
        uint32_t fineOccupied;      // cumulative sum of per-frame counts
        uint32_t fineOccupiedFrame; // THIS frame's count; zeroed by the clear pass each frame
        uint32_t fineOccupiedMax;   // max over frames -- the detail table sizing input
        int32_t sumNormalX;         // fixed point, x10000
        int32_t sumNormalY;
        int32_t sumNormalZ;
    };
    static_assert(sizeof(BlockRecord) == 36, "BlockRecord must be 9 packed 4-byte scalars");

    // Decides which blocks earn a half-voxel detail level, and splits a frame's points into the
    // two levels. Owns only its own buffers; it knows nothing about any memory strategy, so it can
    // be built and tested on its own.
    class DenseRegionClassifier {
    public:
        void Build(Engine::Core::Context &context, float baseVoxel, int blockVoxels,
                   uint32_t maxPointPerFrame, const DensityCriteria &criteria = {});

        // Empties the block records. Cell hashes are per-frame and cleared by Record itself.
        void Reset();

        // Records the accumulate pass into `batch` without submitting. Grows the point buffers
        // when a frame exceeds the current capacity -- never clamps.
        void Record(const std::vector<Eigen::Vector3f> &points,
                    const std::vector<Eigen::Vector3f> &normals,
                    Engine::Compute::CommandBatch &batch);

        uint32_t BlockCount() const;

        // Occupied block records, for tests and diagnostics. Not a per-frame path.
        std::vector<BlockRecord> ReadBlocks() const;

    private:
        void growPointBuffers(uint32_t pointCount);

        Engine::Core::Context *m_context = nullptr;
        float m_baseVoxel = 0.01f;
        int m_blockVoxels = 32;
        uint32_t m_maxPointPerFrame = 0;
        DensityCriteria m_criteria;

        std::unique_ptr<Engine::Core::Buffer> m_pointBuffer;
        std::unique_ptr<Engine::Core::Buffer> m_normalBuffer;
        std::unique_ptr<Engine::Core::Buffer> m_blockRecords;
        std::unique_ptr<Engine::Core::Buffer> m_blockCount;
        std::unique_ptr<Engine::Core::Buffer> m_blockIndex;   // per point -> record index
        std::unique_ptr<Engine::Core::Buffer> m_fineCells;    // per-frame key-only hash
        std::unique_ptr<Engine::Core::Buffer> m_coarseCells;  // per-frame key-only hash

        std::unique_ptr<Engine::Core::ComputePipeline> kernel_clearCells;
        std::unique_ptr<Engine::Core::ComputePipeline> kernel_accumulate;
    };

} // namespace TSDF
```

- [ ] **Step 4: clear 커널 작성**

`src/TSDF/Structure/DenseRegionClassifier.clear.comp.glsl`:

```glsl
#version 450

/// Empties both per-frame cell hashes. They are cleared every frame because keeping them across
/// frames would mean remembering every distinct fine cell ever seen -- the same order of memory as
/// the detail TSDF this classifier is deciding about.

#include "voxel_common.glsl" // EMPTY_KEY

layout(local_size_x = 256) in;

struct BlockRecord
{
	uint blockKey;
	uint pointCount;
	uint coarseOccupied;
	uint fineOccupied;
	uint fineOccupiedFrame;
	uint fineOccupiedMax;
	int  sumNormalX;
	int  sumNormalY;
	int  sumNormalZ;
};

layout(std430, set = 0, binding = 0) buffer FineCells { uint g_fineCells[]; };
layout(std430, set = 0, binding = 1) buffer CoarseCells { uint g_coarseCells[]; };
layout(std430, set = 0, binding = 2) buffer Blocks { BlockRecord g_blocks[]; };

layout(push_constant) uniform PC { uint g_cellCapacity; uint g_blockCapacity; };

// Dispatched over max(cellCapacity, blockCapacity); each guard covers its own range. Resetting the
// per-frame occupancy counter here -- rather than in a fourth dispatch -- keeps the frame's setup
// to one kernel.
void main()
{
	uint i = gl_GlobalInvocationID.x;
	if (i < g_cellCapacity) {
		g_fineCells[i]   = EMPTY_KEY;
		g_coarseCells[i] = EMPTY_KEY;
	}
	if (i < g_blockCapacity) g_blocks[i].fineOccupiedFrame = 0u;
}
```

- [ ] **Step 5: accumulate 커널 작성**

`src/TSDF/Structure/DenseRegionClassifier.accumulate.comp.glsl`:

```glsl
#version 450

/// Pass 1 of 3. One thread per point: find (or create) the point's block record, then count the
/// point, its normal, and whether it was the first sample this frame in its coarse and fine cell.
///
/// Cell keys are exact, not hashed: a cell's coordinate WITHIN its block fits in 18 bits (fine) or
/// 15 (coarse), so (blockIndex, localCell) packs losslessly into 32 bits. Hashing the coordinate
/// instead would let collisions merge distinct cells, and that undercount would flow straight into
/// the occupancy ratio the whole decision rests on.

#include "voxel_common.glsl" // EMPTY_KEY, MAX_PROBE, wangHash

layout(local_size_x = 256) in;

struct BlockRecord
{
	uint blockKey;
	uint pointCount;
	uint coarseOccupied;
	uint fineOccupied;
	uint fineOccupiedFrame;
	uint fineOccupiedMax;
	int  sumNormalX;
	int  sumNormalY;
	int  sumNormalZ;
};

layout(std430, set = 0, binding = 0) readonly buffer Points  { float g_points[]; };
layout(std430, set = 0, binding = 1) readonly buffer Normals { float g_normals[]; };
layout(std430, set = 0, binding = 2) buffer Blocks     { BlockRecord g_blocks[]; };
layout(std430, set = 0, binding = 3) buffer BlockCount { uint g_blockCount; };
layout(std430, set = 0, binding = 4) buffer BlockIndex { uint g_blockIndex[]; };
layout(std430, set = 0, binding = 5) buffer FineCells   { uint g_fineCells[]; };
layout(std430, set = 0, binding = 6) buffer CoarseCells { uint g_coarseCells[]; };

layout(push_constant) uniform PC
{
	uint  g_numPoints;
	uint  g_blockCapacity;
	uint  g_cellCapacity;
	float g_baseVoxel;
	int   g_blockVoxels;
};

const int NORMAL_SCALE = 10000;

/// Key-only open-addressed inserts. Each returns true when THIS call claimed the slot, which is
/// what makes "distinct cells this frame" countable without a second pass. Two near-identical
/// functions rather than one: GLSL has no reference-to-array parameter to pass the table with.

/// 21-bit-per-axis block coordinate. Blocks are 32 base voxels, so this spans a 67 km cube at
/// 1 mm voxels -- far past any scan.
uint packBlockKey(ivec3 block)
{
	return ((uint(block.x + 1048576) & 0x1FFFFFu) << 11)
	     ^ ((uint(block.y + 1048576) & 0x1FFFFFu) << 5)
	     ^  (uint(block.z + 1048576) & 0x1FFFFFu);
}

uint findOrInsertBlock(uint key)
{
	uint slot = wangHash(key) % g_blockCapacity;
	for (uint probeStep = 0u; probeStep < MAX_PROBE; probeStep++) {
		uint index = (slot + probeStep) % g_blockCapacity;
		uint previousKey = atomicCompSwap(g_blocks[index].blockKey, EMPTY_KEY, key);
		if (previousKey == EMPTY_KEY) { atomicAdd(g_blockCount, 1u); return index; }
		if (previousKey == key) return index;
	}
	return EMPTY_KEY;
}

bool claimFineCell(uint key)
{
	uint slot = wangHash(key) % g_cellCapacity;
	for (uint probeStep = 0u; probeStep < MAX_PROBE; probeStep++) {
		uint index = (slot + probeStep) % g_cellCapacity;
		uint previousKey = atomicCompSwap(g_fineCells[index], EMPTY_KEY, key);
		if (previousKey == EMPTY_KEY) return true;   // this thread claimed it -> count it
		if (previousKey == key) return false;        // already counted this frame
	}
	return false;
}

bool claimCoarseCell(uint key)
{
	uint slot = wangHash(key) % g_cellCapacity;
	for (uint probeStep = 0u; probeStep < MAX_PROBE; probeStep++) {
		uint index = (slot + probeStep) % g_cellCapacity;
		uint previousKey = atomicCompSwap(g_coarseCells[index], EMPTY_KEY, key);
		if (previousKey == EMPTY_KEY) return true;
		if (previousKey == key) return false;
	}
	return false;
}

void main()
{
	uint i = gl_GlobalInvocationID.x;
	if (i >= g_numPoints) return;

	vec3 position = vec3(g_points[i * 3u], g_points[i * 3u + 1u], g_points[i * 3u + 2u]);
	vec3 normal   = vec3(g_normals[i * 3u], g_normals[i * 3u + 1u], g_normals[i * 3u + 2u]);

	// 1. Which block, and which cell within it, at both resolutions.
	float blockWorld = g_baseVoxel * float(g_blockVoxels);
	ivec3 block = ivec3(floor(position / blockWorld));
	vec3  local = position - vec3(block) * blockWorld;

	ivec3 coarseCell = clamp(ivec3(floor(local / g_baseVoxel)), ivec3(0), ivec3(g_blockVoxels - 1));
	ivec3 fineCell   = clamp(ivec3(floor(local / (g_baseVoxel * 0.5))), ivec3(0),
	                         ivec3(g_blockVoxels * 2 - 1));

	// 2. Block record.
	uint blockSlot = findOrInsertBlock(packBlockKey(block));
	if (blockSlot == EMPTY_KEY) { g_blockIndex[i] = EMPTY_KEY; return; }
	g_blockIndex[i] = blockSlot;

	// 3. Point and normal.
	atomicAdd(g_blocks[blockSlot].pointCount, 1u);
	atomicAdd(g_blocks[blockSlot].sumNormalX, int(normal.x * float(NORMAL_SCALE)));
	atomicAdd(g_blocks[blockSlot].sumNormalY, int(normal.y * float(NORMAL_SCALE)));
	atomicAdd(g_blocks[blockSlot].sumNormalZ, int(normal.z * float(NORMAL_SCALE)));

	// 4. Occupancy, counted once per cell per frame.
	uint fineKey   = (blockSlot << 18) | (uint(fineCell.z) << 12)
	               | (uint(fineCell.y) << 6) | uint(fineCell.x);
	uint coarseKey = (blockSlot << 15) | (uint(coarseCell.z) << 10)
	               | (uint(coarseCell.y) << 5) | uint(coarseCell.x);

	if (claimFineCell(fineKey)) {
		atomicAdd(g_blocks[blockSlot].fineOccupied, 1u);      // cumulative, drives the ratio
		atomicAdd(g_blocks[blockSlot].fineOccupiedFrame, 1u); // this frame only, drives sizing
	}
	if (claimCoarseCell(coarseKey)) atomicAdd(g_blocks[blockSlot].coarseOccupied, 1u);
}
```

- [ ] **Step 6: `.cpp` 작성**

`src/TSDF/Structure/DenseRegionClassifier.cpp`:

```cpp
#include "TSDF/Structure/DenseRegionClassifier.h"

#include <cstring>

namespace TSDF {

    namespace {
        // Blocks are coarse (32 base voxels a side), so a frame touches thousands, not millions.
        // Kept well below 1<<14 because the cell key packs the block's record index in 14 bits.
        constexpr uint32_t kBlockCapacity = 1u << 13;
        // One entry per distinct fine cell in a frame. Sized against maxPointPerFrame because a
        // frame can never occupy more cells than it has points.
        constexpr float kCellCapacityFactor = 2.0f;

        struct AccumulatePC {
            uint32_t numPoints;
            uint32_t blockCapacity;
            uint32_t cellCapacity;
            float voxelSize;
            int32_t blockVoxels;
        };

        struct ClearPC {
            uint32_t cellCapacity;
        };
    } // namespace

    void DenseRegionClassifier::Build(Engine::Core::Context &context, float baseVoxel,
                                      int blockVoxels, uint32_t maxPointPerFrame,
                                      const DensityCriteria &criteria) {
        m_context = &context;
        m_baseVoxel = baseVoxel;
        m_blockVoxels = blockVoxels;
        m_maxPointPerFrame = maxPointPerFrame;
        m_criteria = criteria;

        const uint32_t cellCapacity = uint32_t(float(maxPointPerFrame) * kCellCapacityFactor);

        m_blockRecords = std::make_unique<Engine::Core::Buffer>(context);
        m_blockCount = std::make_unique<Engine::Core::Buffer>(context);
        m_fineCells = std::make_unique<Engine::Core::Buffer>(context);
        m_coarseCells = std::make_unique<Engine::Core::Buffer>(context);
        m_blockIndex = std::make_unique<Engine::Core::Buffer>(context);

        m_blockRecords->AllocateHostVisibleReadback(kBlockCapacity * sizeof(BlockRecord));
        m_blockCount->AllocateHostVisibleReadback(sizeof(uint32_t));
        m_fineCells->Allocate(cellCapacity * sizeof(uint32_t));
        m_coarseCells->Allocate(cellCapacity * sizeof(uint32_t));
        m_blockIndex->Allocate(maxPointPerFrame * sizeof(uint32_t));

        growPointBuffers(maxPointPerFrame);

        kernel_clearCells = std::make_unique<Engine::Core::ComputePipeline>(context);
        kernel_clearCells->Build("TSDF/Structure/DenseRegionClassifier.clear.comp.glsl")
                .Bind(0, *m_fineCells)
                .Bind(1, *m_coarseCells);

        kernel_accumulate = std::make_unique<Engine::Core::ComputePipeline>(context);
        kernel_accumulate->Build("TSDF/Structure/DenseRegionClassifier.accumulate.comp.glsl")
                .Bind(2, *m_blockRecords)
                .Bind(3, *m_blockCount)
                .Bind(4, *m_blockIndex)
                .Bind(5, *m_fineCells)
                .Bind(6, *m_coarseCells);

        Reset();
    }

    void DenseRegionClassifier::growPointBuffers(uint32_t pointCount) {
        if (m_pointBuffer && pointCount <= m_maxPointPerFrame) return;
        // 1.5x slack so a stream of similar frames reallocates once, not every call. Growing
        // rather than clamping is deliberate: a clamp drops points with no symptom.
        const uint32_t grown = pointCount + pointCount / 2u;
        m_pointBuffer = std::make_unique<Engine::Core::Buffer>(*m_context);
        m_normalBuffer = std::make_unique<Engine::Core::Buffer>(*m_context);
        m_pointBuffer->AllocateHostVisible(grown * 3u * sizeof(float));
        m_normalBuffer->AllocateHostVisible(grown * 3u * sizeof(float));
        m_blockIndex = std::make_unique<Engine::Core::Buffer>(*m_context);
        m_blockIndex->Allocate(grown * sizeof(uint32_t));
        m_maxPointPerFrame = grown;
        if (kernel_accumulate) kernel_accumulate->Bind(4, *m_blockIndex);
    }

    void DenseRegionClassifier::Reset() {
        // Empty every block slot on the host: the record buffer is host-visible and this runs once
        // per scene, not per frame.
        auto *records = static_cast<BlockRecord *>(m_blockRecords->MappedPtr());
        std::memset(records, 0, kBlockCapacity * sizeof(BlockRecord));
        for (uint32_t i = 0; i < kBlockCapacity; ++i) records[i].blockKey = 0xFFFFFFFFu;
        m_blockRecords->MakeVisibleToGPU(kBlockCapacity * sizeof(BlockRecord));
        *static_cast<uint32_t *>(m_blockCount->MappedPtr()) = 0;
        m_blockCount->MakeVisibleToGPU(sizeof(uint32_t));
    }

    void DenseRegionClassifier::Record(const std::vector<Eigen::Vector3f> &points,
                                       const std::vector<Eigen::Vector3f> &normals,
                                       Engine::Compute::CommandBatch &batch) {
        if (!m_context || points.empty()) return;
        const uint32_t n =
                std::min(uint32_t(points.size()), uint32_t(normals.size()));
        if (n == 0) return;
        growPointBuffers(n);

        std::memcpy(m_pointBuffer->MappedPtr(), points.data(), n * 3u * sizeof(float));
        std::memcpy(m_normalBuffer->MappedPtr(), normals.data(), n * 3u * sizeof(float));
        m_pointBuffer->MakeVisibleToGPU(n * 3u * sizeof(float));
        m_normalBuffer->MakeVisibleToGPU(n * 3u * sizeof(float));

        const uint32_t cellCapacity =
                uint32_t(float(m_maxPointPerFrame) * kCellCapacityFactor);

        // The cell hashes hold one frame's occupancy, so they are emptied at the head of every
        // frame rather than at Reset.
        kernel_clearCells->Args(ClearPC{cellCapacity});
        batch.DispatchElements(*kernel_clearCells, cellCapacity);

        kernel_accumulate->Bind(0, *m_pointBuffer).Bind(1, *m_normalBuffer);
        kernel_accumulate->Args(AccumulatePC{n, kBlockCapacity, cellCapacity, m_baseVoxel,
                                             int32_t(m_blockVoxels)});
        batch.DispatchElements(*kernel_accumulate, n);
    }

    uint32_t DenseRegionClassifier::BlockCount() const {
        if (!m_blockCount) return 0;
        m_blockCount->MakeVisibleToCPU(sizeof(uint32_t));
        return *static_cast<const uint32_t *>(m_blockCount->MappedPtr());
    }

    std::vector<BlockRecord> DenseRegionClassifier::ReadBlocks() const {
        std::vector<BlockRecord> out;
        if (!m_blockRecords) return out;
        m_blockRecords->MakeVisibleToCPU(kBlockCapacity * sizeof(BlockRecord));
        const auto *records = static_cast<const BlockRecord *>(m_blockRecords->MappedPtr());
        for (uint32_t i = 0; i < kBlockCapacity; ++i)
            if (records[i].blockKey != 0xFFFFFFFFu) out.push_back(records[i]);
        return out;
    }

} // namespace TSDF
```

`<algorithm>`을 include에 추가한다 (`std::min`).

- [ ] **Step 7: 통과 확인**

```bash
VULKAN_SDK=/usr/local cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release && \
VULKAN_SDK=/usr/local cmake --build build-rel --target vkspatial_tests -j8 && \
./build-rel/test/vkspatial_tests --gtest_filter='DenseRegionAccumulate.*'
```
Expected: 2 tests PASS

- [ ] **Step 8: 회귀 확인 후 커밋**

```bash
./build-rel/test/vkspatial_tests
```
Expected: 289 passed / 1 skipped / 0 failed

```bash
git add src/TSDF/Structure/DenseRegionClassifier.h src/TSDF/Structure/DenseRegionClassifier.cpp \
        src/TSDF/Structure/DenseRegionClassifier.accumulate.comp.glsl \
        src/TSDF/Structure/DenseRegionClassifier.clear.comp.glsl test/test_denseRegion.cpp
git commit -m "feat(tsdf): accumulate per-block density statistics on the GPU"
```

---

### Task 2: 판정 패스

**Files:**
- Create: `src/TSDF/Structure/DenseRegionClassifier.classify.comp.glsl`
- Modify: `src/TSDF/Structure/DenseRegionClassifier.{h,cpp}`
- Test: `test/test_denseRegion.cpp`

**Interfaces:**
- Consumes: Task 1의 `BlockRecord`, `DensityCriteria`
- Produces: `DenseRegionClassifier::Classify(CommandBatch &)`, `DenseRegionClassifier::DenseBlockCount()`, `DenseRegionClassifier::DetailSlotEstimate()`

- [ ] **Step 1: 실패하는 테스트 작성**

`test/test_denseRegion.cpp` 끝에 추가:

```cpp
// A dense flat plane must NOT be refined: it resolves the fine grid, but there is no geometry
// detail to recover. This is the condition the old redundancy heuristic could not see.
TEST(DenseRegionClassify, DenseFlatPlaneIsNotRefined) {
    Engine::Core::Context context;
    TSDF::DenseRegionClassifier classifier;
    classifier.Build(context, 0.01f, 32, 1u << 15);

    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, 0.0025f, 64);

    Engine::Compute::CommandBatch batch(context);
    classifier.Record(points, normals, batch);
    classifier.Classify(batch);
    batch.Submit();

    EXPECT_EQ(classifier.DenseBlockCount(), 0u)
            << "normal coherence ~1 on a plane must veto refinement";
}

// A densely scanned sphere patch has both the sampling and the curvature, so it must be refined.
TEST(DenseRegionClassify, DenseCurvedSurfaceIsRefined) {
    Engine::Core::Context context;
    TSDF::DenseRegionClassifier classifier;
    classifier.Build(context, 0.01f, 32, 1u << 15);

    // A 0.05 m sphere sampled at ~0.0025 m: strong curvature across one block.
    std::vector<Vector3f> points, normals;
    const float radius = 0.05f;
    for (int a = 0; a < 180; ++a)
        for (int b = 0; b < 90; ++b) {
            const float theta = float(a) * float(M_PI) / 90.0f;
            const float phi = float(b) * float(M_PI) / 180.0f;
            const Vector3f direction(std::sin(phi) * std::cos(theta),
                                     std::sin(phi) * std::sin(theta), std::cos(phi));
            points.push_back(direction * radius);
            normals.push_back(direction);
        }

    Engine::Compute::CommandBatch batch(context);
    classifier.Record(points, normals, batch);
    classifier.Classify(batch);
    batch.Submit();

    EXPECT_GT(classifier.DenseBlockCount(), 0u);
    EXPECT_GT(classifier.DetailSlotEstimate(), 0u)
            << "a refined block must report the slots its detail table will need";
}

// Sparse sampling fails the spacing condition however curved the surface is.
TEST(DenseRegionClassify, SparseCurvedSurfaceIsNotRefined) {
    Engine::Core::Context context;
    TSDF::DenseRegionClassifier classifier;
    classifier.Build(context, 0.01f, 32, 1u << 15);

    std::vector<Vector3f> points, normals;
    const float radius = 0.05f;
    for (int a = 0; a < 24; ++a)
        for (int b = 0; b < 12; ++b) {
            const float theta = float(a) * float(M_PI) / 12.0f;
            const float phi = float(b) * float(M_PI) / 24.0f;
            const Vector3f direction(std::sin(phi) * std::cos(theta),
                                     std::sin(phi) * std::sin(theta), std::cos(phi));
            points.push_back(direction * radius);
            normals.push_back(direction);
        }

    Engine::Compute::CommandBatch batch(context);
    classifier.Record(points, normals, batch);
    classifier.Classify(batch);
    batch.Submit();

    EXPECT_EQ(classifier.DenseBlockCount(), 0u) << "spacing must veto refinement";
}
```

- [ ] **Step 2: 실패 확인**

Expected: `no member named 'Classify' in 'TSDF::DenseRegionClassifier'`

- [ ] **Step 3: classify 커널 작성**

`src/TSDF/Structure/DenseRegionClassifier.classify.comp.glsl`:

```glsl
#version 450

/// Pass 2 of 3. One thread per block record. Applies the four ANDed conditions and latches the
/// verdict: a block that has become dense never reverts, because a level that flips mid-scan
/// leaves a seam in the reconstruction.

#include "voxel_common.glsl" // EMPTY_KEY

layout(local_size_x = 256) in;

struct BlockRecord
{
	uint blockKey;
	uint pointCount;
	uint coarseOccupied;
	uint fineOccupied;
	uint fineOccupiedFrame;
	uint fineOccupiedMax;
	int  sumNormalX;
	int  sumNormalY;
	int  sumNormalZ;
};

layout(std430, set = 0, binding = 0) buffer Blocks { BlockRecord g_blocks[]; };
layout(std430, set = 0, binding = 1) buffer Dense  { uint g_dense[]; };
layout(std430, set = 0, binding = 2) buffer Totals { uint g_denseBlockCount; uint g_detailSlots; };

layout(push_constant) uniform PC
{
	uint  g_blockCapacity;
	float g_occupancyRatio;
	float g_normalCoherence;
	float g_samplesPerFineCell;
	uint  g_minimumFineOccupied;
};

const float NORMAL_SCALE = 10000.0;

void main()
{
	uint i = gl_GlobalInvocationID.x;
	if (i >= g_blockCapacity) return;
	if (g_blocks[i].blockKey == EMPTY_KEY) return;

	if (g_dense[i] != 0u) {                       // latched: never revert
		atomicAdd(g_denseBlockCount, 1u);
		atomicAdd(g_detailSlots, g_blocks[i].fineOccupiedMax);
		return;
	}

	float pointCount     = float(g_blocks[i].pointCount);
	float coarseOccupied = float(g_blocks[i].coarseOccupied);
	float fineOccupied   = float(g_blocks[i].fineOccupied);
	if (coarseOccupied < 1.0 || pointCount < 1.0) return;

	vec3 sumNormal = vec3(float(g_blocks[i].sumNormalX), float(g_blocks[i].sumNormalY),
	                      float(g_blocks[i].sumNormalZ)) / NORMAL_SCALE;

	bool resolvesFineGrid = fineOccupied >= g_occupancyRatio * coarseOccupied;
	bool hasDetail        = length(sumNormal) / pointCount < g_normalCoherence;
	bool keepsSignal      = pointCount >= g_samplesPerFineCell * fineOccupied;
	bool hasSurface       = g_blocks[i].fineOccupied >= g_minimumFineOccupied;

	if (resolvesFineGrid && hasDetail && keepsSignal && hasSurface) {
		g_dense[i] = 1u;
		atomicAdd(g_denseBlockCount, 1u);
		// fineOccupiedMax, not fineOccupied: the cumulative sum re-counts a cell once per frame,
		// so only the per-frame maximum estimates the slots the detail table actually needs.
		atomicAdd(g_detailSlots, g_blocks[i].fineOccupiedMax);
	}
}
```

- [ ] **Step 4: `fineOccupiedMax`를 classify에서 갱신**

`fineOccupied`는 프레임을 가로질러 누적되므로 그 자체로는 한 프레임의 점유를 알 수 없다.
그래서 accumulate가 `fineOccupiedFrame`(이번 프레임분)을 따로 세고, clear 패스가 프레임마다
그것을 0으로 되돌린다(Task 1). classify는 그 값으로 최댓값을 갱신한다 — 판정 직전에, 래치 분기
앞에서:

```glsl
	// Cumulative fineOccupied re-counts a cell once per frame, so only the per-frame value
	// estimates the slots the detail table actually needs.
	atomicMax(g_blocks[i].fineOccupiedMax, g_blocks[i].fineOccupiedFrame);
```

- [ ] **Step 5: C++ 배선**

`DenseRegionClassifier.h`의 public에 추가:

```cpp
        // Applies the criteria to every block record. Latched: a dense block stays dense.
        void Classify(Engine::Compute::CommandBatch &batch);

        uint32_t DenseBlockCount() const;

        // Slots the detail table needs across every dense block, from per-frame maximum occupancy
        // rather than the cumulative count, which re-counts a cell once per frame.
        uint32_t DetailSlotEstimate() const;
```

private에 `m_denseFlags`, `m_totals`, `kernel_classify`를 추가한다. `.cpp`에서:

```cpp
        struct ClassifyPC {
            uint32_t blockCapacity;
            float occupancyRatio;
            float normalCoherence;
            float samplesPerFineCell;
            uint32_t minimumFineOccupied;
        };
```

`Build`에서 할당·바인딩한다:

```cpp
        m_denseFlags = std::make_unique<Engine::Core::Buffer>(context);
        m_totals = std::make_unique<Engine::Core::Buffer>(context);
        m_denseFlags->Allocate(kBlockCapacity * sizeof(uint32_t));
        m_totals->AllocateHostVisibleReadback(2u * sizeof(uint32_t));

        kernel_classify = std::make_unique<Engine::Core::ComputePipeline>(context);
        kernel_classify->Build("TSDF/Structure/DenseRegionClassifier.classify.comp.glsl")
                .Bind(0, *m_blockRecords)
                .Bind(1, *m_denseFlags)
                .Bind(2, *m_totals);
```

`Classify`는 누계를 0으로 되돌린 뒤 dispatch를 기록한다 — 두 합계는 매 호출 전수 재계산이므로
누적하면 안 된다:

```cpp
    void DenseRegionClassifier::Classify(Engine::Compute::CommandBatch &batch) {
        if (!m_context) return;
        auto *totals = static_cast<uint32_t *>(m_totals->MappedPtr());
        totals[0] = 0;
        totals[1] = 0;
        m_totals->MakeVisibleToGPU(2u * sizeof(uint32_t));

        kernel_classify->Args(ClassifyPC{kBlockCapacity, m_criteria.occupancyRatio,
                                         m_criteria.normalCoherence,
                                         m_criteria.samplesPerFineCell,
                                         m_criteria.minimumFineOccupied});
        batch.DispatchElements(*kernel_classify, kBlockCapacity);
    }

    uint32_t DenseRegionClassifier::DenseBlockCount() const {
        if (!m_totals) return 0;
        m_totals->MakeVisibleToCPU(2u * sizeof(uint32_t));
        return static_cast<const uint32_t *>(m_totals->MappedPtr())[0];
    }

    uint32_t DenseRegionClassifier::DetailSlotEstimate() const {
        if (!m_totals) return 0;
        m_totals->MakeVisibleToCPU(2u * sizeof(uint32_t));
        return static_cast<const uint32_t *>(m_totals->MappedPtr())[1];
    }
```

`Reset`에서 `m_denseFlags`도 0으로 되돌린다 — 래치가 장면을 넘어 살아남으면 안 된다.
`Record`의 clear dispatch는 이제 `ClearPC{cellCapacity, kBlockCapacity}`를 넘기고
`max(cellCapacity, kBlockCapacity)` 개로 디스패치한다.

- [ ] **Step 6: 통과 확인 후 커밋**

```bash
VULKAN_SDK=/usr/local cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release && \
VULKAN_SDK=/usr/local cmake --build build-rel --target vkspatial_tests -j8 && \
./build-rel/test/vkspatial_tests --gtest_filter='DenseRegion*' && \
./build-rel/test/vkspatial_tests
```
Expected: 신규 3개 PASS, 전체 292 passed / 1 skipped / 0 failed

```bash
git add -A
git commit -m "feat(tsdf): classify dense blocks from spacing, curvature and sample budget"
```

---

### Task 3: 분할 패스와 `DividePoint`

**Files:**
- Create: `src/TSDF/Structure/DenseRegionClassifier.partition.comp.glsl`
- Modify: `src/TSDF/Structure/DenseRegionClassifier.{h,cpp}`
- Test: `test/test_denseRegion.cpp`

**Interfaces:**
- Consumes: Task 2의 `Classify`, `m_denseFlags`, `m_blockIndex`
- Produces: `DenseRegionClassifier::Partition(CommandBatch &)`, `DenseRegionClassifier::ReadPartition(std::vector<uint32_t> &base, std::vector<uint32_t> &detail)`

- [ ] **Step 1: 실패하는 테스트 작성**

```cpp
// The partition must be exhaustive: every input point lands in exactly one level. A point silently
// dropped here vanishes from the reconstruction with no counter to show it.
TEST(DenseRegionPartition, EveryPointLandsInExactlyOneLevel) {
    Engine::Core::Context context;
    TSDF::DenseRegionClassifier classifier;
    classifier.Build(context, 0.01f, 32, 1u << 15);

    std::vector<Vector3f> points, normals;
    const float radius = 0.05f;
    for (int a = 0; a < 180; ++a)
        for (int b = 0; b < 90; ++b) {
            const float theta = float(a) * float(M_PI) / 90.0f;
            const float phi = float(b) * float(M_PI) / 180.0f;
            const Vector3f direction(std::sin(phi) * std::cos(theta),
                                     std::sin(phi) * std::sin(theta), std::cos(phi));
            points.push_back(direction * radius);
            normals.push_back(direction);
        }

    Engine::Compute::CommandBatch batch(context);
    classifier.Record(points, normals, batch);
    classifier.Classify(batch);
    classifier.Partition(batch);
    batch.Submit();

    std::vector<uint32_t> base, detail;
    classifier.ReadPartition(base, detail);
    EXPECT_EQ(base.size() + detail.size(), points.size());
    EXPECT_GT(detail.size(), 0u) << "a curved dense surface must send points to the detail level";
}

// The verdict is a latch: a block that became dense stays dense on later frames.
TEST(DenseRegionPartition, DenseVerdictDoesNotRevert) {
    Engine::Core::Context context;
    TSDF::DenseRegionClassifier classifier;
    classifier.Build(context, 0.01f, 32, 1u << 15);

    std::vector<Vector3f> dense, denseNormals;
    const float radius = 0.05f;
    for (int a = 0; a < 180; ++a)
        for (int b = 0; b < 90; ++b) {
            const float theta = float(a) * float(M_PI) / 90.0f;
            const float phi = float(b) * float(M_PI) / 180.0f;
            const Vector3f direction(std::sin(phi) * std::cos(theta),
                                     std::sin(phi) * std::sin(theta), std::cos(phi));
            dense.push_back(direction * radius);
            denseNormals.push_back(direction);
        }

    {
        Engine::Compute::CommandBatch batch(context);
        classifier.Record(dense, denseNormals, batch);
        classifier.Classify(batch);
        batch.Submit();
    }
    const uint32_t afterDenseFrame = classifier.DenseBlockCount();
    ASSERT_GT(afterDenseFrame, 0u);

    // A sparse second frame over the same region would fail the criteria on its own.
    std::vector<Vector3f> sparse, sparseNormals;
    for (size_t i = 0; i < dense.size(); i += 40) {
        sparse.push_back(dense[i]);
        sparseNormals.push_back(denseNormals[i]);
    }
    {
        Engine::Compute::CommandBatch batch(context);
        classifier.Record(sparse, sparseNormals, batch);
        classifier.Classify(batch);
        batch.Submit();
    }
    EXPECT_GE(classifier.DenseBlockCount(), afterDenseFrame) << "the verdict must not revert";
}
```

- [ ] **Step 2: 실패 확인**

Expected: `no member named 'Partition'`

- [ ] **Step 3: partition 커널 작성**

`src/TSDF/Structure/DenseRegionClassifier.partition.comp.glsl`:

```glsl
#version 450

/// Pass 3 of 3. One thread per point: read the point's block verdict and append its index to one
/// of the two lists.
///
/// Atomic append, not a prefix sum: TSDF integration is order independent, so a stable partition
/// buys nothing and a scan pass would cost a dispatch and a dependency.

#include "voxel_common.glsl" // EMPTY_KEY

layout(local_size_x = 256) in;

layout(std430, set = 0, binding = 0) readonly buffer BlockIndex { uint g_blockIndex[]; };
layout(std430, set = 0, binding = 1) readonly buffer Dense      { uint g_dense[]; };
layout(std430, set = 0, binding = 2) buffer BaseIndex   { uint g_baseIndex[]; };
layout(std430, set = 0, binding = 3) buffer DetailIndex { uint g_detailIndex[]; };
layout(std430, set = 0, binding = 4) buffer Counts      { uint g_baseCount; uint g_detailCount; };

layout(push_constant) uniform PC { uint g_numPoints; };

void main()
{
	uint i = gl_GlobalInvocationID.x;
	if (i >= g_numPoints) return;

	uint blockSlot = g_blockIndex[i];
	// A point whose block record could not be created (block table full) still has to be
	// integrated -- send it to the base level rather than dropping it.
	bool toDetail = (blockSlot != EMPTY_KEY) && (g_dense[blockSlot] != 0u);

	if (toDetail) g_detailIndex[atomicAdd(g_detailCount, 1u)] = i;
	else          g_baseIndex[atomicAdd(g_baseCount, 1u)] = i;
}
```

- [ ] **Step 4: C++ 배선**

`m_baseIndex` / `m_detailIndex` / `m_partitionCount` 버퍼와 `kernel_partition`을 추가한다.
`Partition`은 카운터를 0으로 되돌린 뒤 dispatch를 기록한다 — 카운터는 프레임마다 리셋된다.
`ReadPartition`은 두 카운트를 읽고 그만큼만 복사한다.

인덱스 버퍼는 `growPointBuffers`에서 점 버퍼와 함께 성장시키고 재바인딩한다.

- [ ] **Step 5: 통과 확인 후 커밋**

```bash
VULKAN_SDK=/usr/local cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release && \
VULKAN_SDK=/usr/local cmake --build build-rel --target vkspatial_tests -j8 && \
./build-rel/test/vkspatial_tests --gtest_filter='DenseRegion*' && \
./build-rel/test/vkspatial_tests
```
Expected: 신규 2개 PASS, 전체 294 passed / 1 skipped / 0 failed

```bash
git add -A
git commit -m "feat(tsdf): partition a frame into base and detail levels on the GPU"
```

---

### Task 4: 큰 프레임과 성장

**Files:**
- Modify: `src/TSDF/Structure/DenseRegionClassifier.cpp`
- Test: `test/test_denseRegion.cpp`

**Interfaces:**
- Consumes: Task 3의 `Partition`, `ReadPartition`
- Produces: 없음 (기존 계약의 성장 동작을 확정)

- [ ] **Step 1: 실패하는 테스트 작성**

```cpp
// Build-time maxPointPerFrame is a hint, not a cap. A larger frame must grow the buffers, because
// clamping would drop points with no counter and no symptom -- a failure mode this repository has
// already shipped once.
TEST(DenseRegionPartition, LargeFrameGrowsInsteadOfTruncating) {
    Engine::Core::Context context;
    TSDF::DenseRegionClassifier classifier;
    classifier.Build(context, 0.01f, 32, /*maxPointPerFrame=*/1024u);

    std::vector<Vector3f> points, normals;
    MakePlane(points, normals, 0.0025f, 64); // 4096 points, four times the hint

    Engine::Compute::CommandBatch batch(context);
    classifier.Record(points, normals, batch);
    classifier.Classify(batch);
    classifier.Partition(batch);
    batch.Submit();

    std::vector<uint32_t> base, detail;
    classifier.ReadPartition(base, detail);
    EXPECT_EQ(base.size() + detail.size(), points.size())
            << "the frame must not have been truncated to the build-time hint";
}
```

- [ ] **Step 2: 실패 확인**

```bash
./build-rel/test/vkspatial_tests --gtest_filter='DenseRegionPartition.LargeFrameGrowsInsteadOfTruncating'
```
Expected: FAIL — 4096 대신 1024만 분류되거나, cell 해시 용량이 부족해 점유가 어긋난다.

- [ ] **Step 3: 성장 경로 보강**

`growPointBuffers`가 점·법선·`blockIndex`·`baseIndex`·`detailIndex`를 함께 키우고, **셀 해시도
같이 키운다** — 셀 용량은 `maxPointPerFrame`에서 파생되므로 점 버퍼만 키우면 셀 해시가 넘쳐
점유가 과소 계상된다. 모든 커널의 해당 바인딩을 재설정한다.

- [ ] **Step 4: 통과 확인 후 커밋**

```bash
VULKAN_SDK=/usr/local cmake --build build-rel --target vkspatial_tests -j8 && \
./build-rel/test/vkspatial_tests --gtest_filter='DenseRegion*' && \
./build-rel/test/vkspatial_tests
```
Expected: 신규 1개 PASS, 전체 295 passed / 1 skipped / 0 failed

```bash
git add -A
git commit -m "feat(tsdf): grow every per-frame buffer together, including the cell hashes"
```

---

## 후속으로 넘기는 것

| 항목 | 이유 |
|---|---|
| `SubmapStrategy::DividePoint` 배선 | `Memory/`→`Structure/` 이행이 진행 중이라, 이 컴포넌트는 독립적으로 완성해 두고 이행이 끝난 쪽에 붙인다 |
| detail 테이블 실제 사이징 적용 | `DetailSlotEstimate()`가 값을 내지만, 그것을 `SubmapAdvancedTSDF`의 detail 용량으로 넣는 것은 배선 단계의 일 |
| 블록 경계 apron | 스펙 §7-3 — 통계 정밀도 문제이지 정확성 문제가 아니다 |
| 블록 테이블이 가득 찰 때의 성장 | 지금은 `kBlockCapacity` 고정이고 넘치면 그 점들이 base로 간다(분할은 여전히 전수). 스캔이 8192 블록을 넘기면 필요해진다 |
