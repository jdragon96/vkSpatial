# TSDF 해시 전략 교체 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `AdvancedTSDF`의 해시 주소 지정을 이름으로 갈아끼우고, 실제 스캔으로 "버킷 해시가 타일당 메모리를 1.6~1.8배 줄이는가"에 답한다.

**Architecture:** 커널은 `#include "TSDF/Memory/Hash/HashStrategy.glsl"` 한 줄만 알고, 그 디스패처가 `#if`로 `LinearProbe.glsl` 또는 `Bucketed.glsl`을 고른다. 선택은 C++이 `ComputePipeline::Define`으로 넘기는 매크로 하나. 성장 임계값은 C++ `HashStrategy`가 들고 `maybeGrow`가 읽는다.

**Tech Stack:** C++17, Vulkan(Engine::Core/Compute), GLSL(shaderc/glslang), Eigen, GoogleTest.

**Spec:** [`docs/superpowers/specs/2026-08-14-tsdf-hash-strategy-design.md`](../specs/2026-08-14-tsdf-hash-strategy-design.md)

## Global Constraints

- 네임스페이스는 **`TSDF`** (`Engine::TSDF` 아님). `src/TSDF/`는 `src/Engine/`의 형제인 최상위 도메인.
- **함수 이름은 동사 하나** — `Build`/`Reset`/`Configure`/`Record`/`Download`/`Integrate`/`Extract`/`Stats`/`Name`/`Device`/`Define`. 복합어를 새로 만들지 않는다. 예외: `Engine::Spatial` 기존 클래스에 추가하는 접근자는 이웃 표기(`FilledCount()`)를 따른다.
- 변수 이름에 약어 금지 — `cameraPosition`, `insertFailureCount`.
- 동결 백엔드(`SimpleTSDF` / `DirectionalTSDF` / `CompactDirectionalTSDF`)와 그 셰이더는 **건드리지 않는다.**
- 커널 이동은 `AdvancedTSDF`의 5개뿐. 규칙은 `<CppStem>.<kernel>.comp.glsl`.
- **스펙 §3에서 수정된 사항**: `HASH_LOAD_FACTOR_LIMIT`을 GLSL에 두지 않는다. 성장 판정은 C++ `maybeGrow`만 하므로 GLSL에 두면 죽은 코드이자 두 번째 진실 소스가 된다. 임계값은 C++ `HashStrategy`에만 존재한다.
- 빌드/테스트:
  - `VULKAN_SDK=/usr/local cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release`
  - `VULKAN_SDK=/usr/local cmake --build build-rel --target vkspatial_tests -j8`
  - `./build-rel/test/vkspatial_tests --gtest_filter='<Suite>.*'`
- CMakeLists가 소스를 GLOB하므로 파일 추가 후 **cmake configure를 다시 돌린다.**
- 기준선: HEAD `e8a6511`, 전체 스위트 **275 passed / 1 skipped / 0 failed**. skip은 `EngineWideBVHTest.RadiusMatchesCpu`로 이 작업 이전부터 있던 것.

## 파일 구조

| 파일 | 책임 |
|---|---|
| `src/Engine/Core/ComputePipeline.{h,cpp}` (수정) | `Define()`, 탐색 경로 목록, 캐시 키에 정의 포함 |
| `src/shader/define_probe.comp.glsl` (신규) | `Define`/캐시 키 테스트 픽스처 |
| `src/TSDF/Backends/AdvancedTSDF.{integrate,extract,compact,clear,rehash}.comp.glsl` (이동) | `src/shader/`에서 옮겨온 커널 5개 |
| `src/TSDF/Memory/Hash/HashStrategy.glsl` (신규) | `#if`로 조각을 고르는 디스패처 |
| `src/TSDF/Memory/Hash/LinearProbe.glsl` (신규) | 현행 선형탐사 `findOrInsert`/`findSlot` |
| `src/TSDF/Memory/Hash/Bucketed.glsl` (신규) | 버킷 변형 (버킷 32) |
| `src/TSDF/Memory/Hash/HashStrategy.h` (신규) | C++ 기술자: 이름·매크로·임계값 + 조회 |
| `src/TSDF/Backends/AdvancedTSDF.{h,cpp}` (수정) | 전략 주입, 임계값 사용, 실패/성장 카운터 |
| `src/TSDF/Volume.h` (수정) | `VolumeParams::hashStrategy`, `Volume::Extract` |
| `src/TSDF/Memory/*Strategy.{h,cpp}` (수정) | 전략 전달, `Extract` 위임, 카운터를 `VolumeStats`에 |
| `example2/tsdf_folder_eval.cpp` (수정) | `--tsdf` / `--hash`, 비교 표 출력 |
| `test/test_tsdf_hash.cpp` (신규) | 전 태스크의 테스트 |

---

### Task 1: `ComputePipeline` — 매크로 정의와 캐시 키

캐시 함정을 **가장 먼저** 막는다. `compileFileCached`가 경로만으로 캐싱하므로, 정의를 키에 넣지 않으면 두 번째 전략이 첫 번째의 SPIR-V를 그대로 받아 A/B 전체가 조용히 무효가 된다.

**Files:**
- Modify: `src/Engine/Core/ComputePipeline.h`, `src/Engine/Core/ComputePipeline.cpp`
- Create: `src/shader/define_probe.comp.glsl`
- Test: `test/test_tsdf_hash.cpp`

**Interfaces:**
- Consumes: 없음
- Produces: `ComputePipeline &Define(const std::string &name, const std::string &value = "1")`

- [ ] **Step 1: 픽스처 셰이더 작성**

`src/shader/define_probe.comp.glsl`:

```glsl
#version 450

// Test fixture for ComputePipeline::Define and the compile cache key. local_size_x is driven
// by a preprocessor definition, so a caller can observe through GetLocalSize() which definition
// set produced the module. If the cache were keyed on the path alone, a second build with
// different definitions would hand back the first build's module and report the wrong size.
#ifndef PROBE_LOCAL_SIZE
#define PROBE_LOCAL_SIZE 1
#endif

layout(local_size_x = PROBE_LOCAL_SIZE) in;
layout(std430, set = 0, binding = 0) buffer Out { uint g_out[]; };

void main() { g_out[0] = uint(PROBE_LOCAL_SIZE); }
```

- [ ] **Step 2: 실패하는 테스트 작성**

`test/test_tsdf_hash.cpp`:

```cpp
#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"

#include <gtest/gtest.h>

TEST(ComputePipelineDefines, DefinitionsReachTheCompilerAndKeyTheCache) {
    Engine::Core::Context context;

    Engine::Core::ComputePipeline withDefinition(context);
    withDefinition.Define("PROBE_LOCAL_SIZE", "8").Build("define_probe.comp.glsl");
    EXPECT_EQ(withDefinition.GetLocalSize().width, 8u);

    // Same file, no definition. A cache keyed on the path alone would return the module built
    // above and report 8 -- that is the failure this test exists to catch.
    Engine::Core::ComputePipeline withoutDefinition(context);
    withoutDefinition.Build("define_probe.comp.glsl");
    EXPECT_EQ(withoutDefinition.GetLocalSize().width, 1u);

    // And the definition still applies after the undefined build populated the cache.
    Engine::Core::ComputePipeline again(context);
    again.Define("PROBE_LOCAL_SIZE", "4").Build("define_probe.comp.glsl");
    EXPECT_EQ(again.GetLocalSize().width, 4u);
}
```

- [ ] **Step 3: 실패 확인**

```bash
VULKAN_SDK=/usr/local cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release
VULKAN_SDK=/usr/local cmake --build build-rel --target vkspatial_tests -j8
```
Expected: 컴파일 실패 — `no member named 'Define' in 'Engine::Core::ComputePipeline'`

- [ ] **Step 4: 헤더에 `Define` 추가**

`src/Engine/Core/ComputePipeline.h`의 `AddInclude` 선언 바로 아래:

```cpp
        // Preprocessor definition applied to the next Build(path). Repeated calls accumulate;
        // re-defining a name replaces it. Definitions are part of the compile cache key, so the
        // same file built with different definitions yields different modules.
        ComputePipeline &Define(const std::string &name, const std::string &value = "1");
```

private 멤버에 (`std::unordered_map<std::string, std::string> m_includes;` 아래):

```cpp
        // std::map, not unordered: the cache key is built by walking this container, so the
        // iteration order must be deterministic across runs.
        std::map<std::string, std::string> m_defines;
```

그리고 헤더 상단 include에 `#include <map>`를 추가한다.

- [ ] **Step 5: 구현**

`src/Engine/Core/ComputePipeline.cpp`에 정의를 추가한다 (`Build` 정의 근처):

```cpp
    ComputePipeline &ComputePipeline::Define(const std::string &name, const std::string &value) {
        m_defines[name] = value;
        return *this;
    }
```

`compileFileCached`가 정의를 받도록 시그니처를 바꾼다. 익명 네임스페이스의 해당 함수에서:

```cpp
        std::vector<uint32_t> compileFileCached(const std::string &fullPath,
                                                const std::map<std::string, std::string> &defines) {
            // Cache key = path + every definition. Keyed on the path alone, a second build of the
            // same kernel with a different definition set would silently reuse the first module.
            std::string cacheKey = fullPath;
            for (const auto &entry: defines) {
                cacheKey += '|';
                cacheKey += entry.first;
                cacheKey += '=';
                cacheKey += entry.second;
            }

            {
                std::lock_guard<std::mutex> lock(cacheMutex);
                const auto it = cache.find(cacheKey);
                if (it != cache.end()) return it->second;
            }
            ...
            opts.SetIncluder(std::make_unique<FilesystemIncluder>(dir));
            for (const auto &entry: defines)
                opts.AddMacroDefinition(entry.first, entry.second);
            ...
            cache.emplace(cacheKey, spv);
            return spv;
        }
```

기존 캐시 조회/삽입이 `fullPath`를 쓰던 자리를 전부 `cacheKey`로 바꾼다. `Build(path)`의 호출은
`compileFileCached(fullPath, m_defines)`가 된다. 파일 상단에 `#include <map>`를 추가한다.

- [ ] **Step 6: 통과 확인**

```bash
VULKAN_SDK=/usr/local cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release && \
VULKAN_SDK=/usr/local cmake --build build-rel --target vkspatial_tests -j8 && \
./build-rel/test/vkspatial_tests --gtest_filter='ComputePipelineDefines.*'
```
Expected: 1 test PASS

- [ ] **Step 7: 회귀 확인 후 커밋**

```bash
./build-rel/test/vkspatial_tests
```
Expected: 276 passed / 1 skipped / 0 failed (기준선 275 + 신규 1)

```bash
git add src/Engine/Core/ComputePipeline.h src/Engine/Core/ComputePipeline.cpp \
        src/shader/define_probe.comp.glsl test/test_tsdf_hash.cpp
git commit -m "feat(core): add ComputePipeline::Define and key the compile cache on definitions"
```

---

### Task 2: `ComputePipeline` 탐색 경로 + 커널 이동

**Files:**
- Modify: `src/Engine/Core/ComputePipeline.cpp`
- Modify: `src/Engine/CMakeLists.txt` (소스 루트 정의 추가)
- Move: `src/shader/advanced_tsdf_{integrate,extract,compact,clear,rehash}.comp.glsl` → `src/TSDF/Backends/AdvancedTSDF.{integrate,extract,compact,clear,rehash}.comp.glsl`
- Modify: `src/TSDF/Backends/AdvancedTSDF.cpp` (Build 경로 5곳)

**Interfaces:**
- Consumes: Task 1의 `Define`
- Produces: `Build(path)`가 `[VKBVH_SHADER_DIR, VKBVH_SRC_DIR]` 순으로 탐색; `#include`가 `[셰이더 자기 폴더, VKBVH_SHADER_DIR, VKBVH_SRC_DIR]` 순으로 탐색

- [ ] **Step 1: 소스 루트를 컴파일 정의로 노출**

`src/Engine/CMakeLists.txt`의 `target_compile_definitions(EngineCore PRIVATE ...)` 블록에 한 줄 추가:

```cmake
        VKBVH_SRC_DIR=\"${CMAKE_SOURCE_DIR}/src\"
```

- [ ] **Step 2: 커널 이동 (순수 이동, 내용 변경 없음)**

```bash
git mv src/shader/advanced_tsdf_integrate.comp.glsl src/TSDF/Backends/AdvancedTSDF.integrate.comp.glsl
git mv src/shader/advanced_tsdf_extract.comp.glsl   src/TSDF/Backends/AdvancedTSDF.extract.comp.glsl
git mv src/shader/advanced_tsdf_compact.comp.glsl   src/TSDF/Backends/AdvancedTSDF.compact.comp.glsl
git mv src/shader/advanced_tsdf_clear.comp.glsl     src/TSDF/Backends/AdvancedTSDF.clear.comp.glsl
git mv src/shader/advanced_tsdf_rehash.comp.glsl    src/TSDF/Backends/AdvancedTSDF.rehash.comp.glsl
```

- [ ] **Step 3: `AdvancedTSDF.cpp`의 Build 경로 갱신**

다섯 곳을 새 경로로 바꾼다:

```cpp
    kernel_integratePoints->Build("TSDF/Backends/AdvancedTSDF.integrate.comp.glsl")
    kernel_compactTable->Build("TSDF/Backends/AdvancedTSDF.compact.comp.glsl")
    kernel_clearVoxel->Build("TSDF/Backends/AdvancedTSDF.clear.comp.glsl")
    kernel_rehashTable->Build("TSDF/Backends/AdvancedTSDF.rehash.comp.glsl")
```

그리고 `ExtractPointCloud` 안의 `kernel.Build("advanced_tsdf_extract.comp.glsl")`을
`kernel.Build("TSDF/Backends/AdvancedTSDF.extract.comp.glsl")`로 바꾼다.

- [ ] **Step 4: 빌드해서 실패를 확인**

```bash
VULKAN_SDK=/usr/local cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release && \
VULKAN_SDK=/usr/local cmake --build build-rel --target vkspatial_tests -j8 && \
./build-rel/test/vkspatial_tests --gtest_filter='TsdfAccessors.*'
```
Expected: 런타임 실패 — 커널 파일을 `VKBVH_SHADER_DIR` 아래에서 못 찾아 `ComputePipeline::Build` 예외.
(탐색 경로가 아직 단일 루트이기 때문이며, 이것이 다음 단계가 고치는 문제다.)

- [ ] **Step 5: 탐색 경로를 목록으로**

`src/Engine/Core/ComputePipeline.cpp`. 파일 상단의 기본값 정의 옆에 소스 루트 기본값을 추가한다:

```cpp
#ifndef VKBVH_SRC_DIR
#define VKBVH_SRC_DIR "."
#endif
```

익명 네임스페이스에 탐색 헬퍼를 추가한다:

```cpp
        // Ordered roots for resolving a kernel path and an #include. The shader directory comes
        // first so every pre-existing call site -- which passes a bare filename -- resolves exactly
        // as before; the source root lets a kernel that lives beside its calling .cpp be named by
        // its path from src/.
        const std::vector<std::string> &shaderRoots() {
            static const std::vector<std::string> roots = {VKBVH_SHADER_DIR, VKBVH_SRC_DIR};
            return roots;
        }

        // First root that actually holds the file wins. Returns the untouched relative path when
        // none do, so the caller's error message still names what was asked for.
        std::string resolveShaderPath(const std::string &relative) {
            for (const std::string &root: shaderRoots()) {
                std::string candidate = root + "/" + relative;
                std::ifstream probe(candidate);
                if (probe.good()) return candidate;
            }
            return relative;
        }
```

`FilesystemIncluder`가 디렉터리 목록을 받도록 바꾼다:

```cpp
    class FilesystemIncluder : public shaderc::CompileOptions::IncluderInterface {
    public:
        explicit FilesystemIncluder(std::vector<std::string> dirs) : m_dirs(std::move(dirs)) {}

        shaderc_include_result *GetInclude(const char *requested,
                                           shaderc_include_type,
                                           const char * /*requesting*/,
                                           size_t) override {
            for (const std::string &dir: m_dirs) {
                std::string fullPath = dir + "/" + requested;
                std::ifstream f(fullPath, std::ios::binary);
                if (!f.is_open()) continue;
                auto *r = new shaderc_include_result{};
                auto *content = new std::string(std::istreambuf_iterator<char>(f),
                                                std::istreambuf_iterator<char>());
                auto *name = new std::string(fullPath);
                r->source_name = name->c_str();
                r->source_name_length = name->size();
                r->content = content->c_str();
                r->content_length = content->size();
                r->user_data = new std::pair<std::string *, std::string *>(name, content);
                return r;
            }
            auto *r = new shaderc_include_result{};
            static const char kErr[] = "file not found";
            r->source_name = "";
            r->source_name_length = 0;
            r->content = kErr;
            r->content_length = sizeof(kErr) - 1;
            r->user_data = nullptr;
            return r;
        }

        void ReleaseInclude(shaderc_include_result *r) override { /* 기존 본문 그대로 */ }

    private:
        std::vector<std::string> m_dirs;
    };
```

`compileFileCached`에서 includer를 만드는 자리를, 셰이더 자기 폴더를 앞에 둔 목록으로 바꾼다:

```cpp
            std::vector<std::string> includeDirs = {dir};
            for (const std::string &root: shaderRoots()) includeDirs.push_back(root);
            opts.SetIncluder(std::make_unique<FilesystemIncluder>(includeDirs));
```

`Build(path)`가 이어붙이기 대신 탐색을 쓰게 바꾼다:

```cpp
    ComputePipeline &ComputePipeline::Build(const std::string &filename) {
        destroyShaderResources();
        const std::string fullPath = resolveShaderPath(filename);
        const std::vector<uint32_t> spv = compileFileCached(fullPath, m_defines);
        ...
```

파일 상단에 `#include <fstream>`이 없다면 추가한다.

- [ ] **Step 6: 통과 확인**

```bash
VULKAN_SDK=/usr/local cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release && \
VULKAN_SDK=/usr/local cmake --build build-rel --target vkspatial_tests -j8 && \
./build-rel/test/vkspatial_tests --gtest_filter='Tsdf*'
```
Expected: 전부 PASS. 이 태스크는 커널을 옮기기만 했으므로 결과가 바뀌면 안 된다.

- [ ] **Step 7: 회귀 확인 후 커밋**

```bash
./build-rel/test/vkspatial_tests
```
Expected: 276 passed / 1 skipped / 0 failed

```bash
git add -A
git commit -m "refactor(tsdf): move AdvancedTSDF kernels beside their caller, resolve shaders by search path"
```

---

### Task 3: 해시 계약 추출 (선형탐사만)

동작을 바꾸지 않고 이음매만 만든다. 이 태스크가 끝나도 결과는 지금과 **같아야** 한다.

**Files:**
- Create: `src/TSDF/Memory/Hash/LinearProbe.glsl`, `src/TSDF/Memory/Hash/HashStrategy.glsl`, `src/TSDF/Memory/Hash/HashStrategy.h`
- Modify: `src/TSDF/Backends/AdvancedTSDF.integrate.comp.glsl`, `AdvancedTSDF.extract.comp.glsl`, `AdvancedTSDF.rehash.comp.glsl`
- Modify: `src/TSDF/Backends/AdvancedTSDF.{h,cpp}`
- Test: `test/test_tsdf_hash.cpp`

**Interfaces:**
- Consumes: Task 1 `Define`, Task 2 탐색 경로
- Produces: GLSL `uint findOrInsert(uint key)` / `uint findSlot(uint key)` / `HASH_INSERT_FAILED` / `HASH_NOT_FOUND`; C++ `struct TSDF::HashStrategy { const char *name; const char *macroName; float loadFactorLimit; }` 와 `const HashStrategy &TSDF::HashStrategyByName(const std::string &)`, `TSDF::LinearProbeStrategy()`

- [ ] **Step 1: 실패하는 테스트 작성**

`test/test_tsdf_hash.cpp` 끝에 추가:

```cpp
#include "TSDF/Memory/Hash/HashStrategy.h"

TEST(TsdfHashStrategy, LinearProbeIsTheDefaultAndCarriesItsThreshold) {
    const TSDF::HashStrategy &linear = TSDF::HashStrategyByName("linear");
    EXPECT_STREQ(linear.name, "linear");
    EXPECT_FLOAT_EQ(linear.loadFactorLimit, 0.5f);
    EXPECT_STREQ(TSDF::LinearProbeStrategy().name, "linear");
}

TEST(TsdfHashStrategy, UnknownNameFallsBackToLinear) {
    // Callers report the typo; silently running a different hash than requested would corrupt
    // an A/B comparison without any visible symptom.
    EXPECT_STREQ(TSDF::HashStrategyByName("no-such-hash").name, "linear");
}
```

- [ ] **Step 2: 실패 확인**

```bash
VULKAN_SDK=/usr/local cmake --build build-rel --target vkspatial_tests -j8
```
Expected: `fatal error: 'TSDF/Memory/Hash/HashStrategy.h' file not found`

- [ ] **Step 3: `LinearProbe.glsl` 작성**

현행 커널의 탐사 루프를 그대로 옮긴다. `src/TSDF/Memory/Hash/LinearProbe.glsl`:

```glsl
/// *********************************************
/// Linear probing (current behaviour)
///
/// One slot per probe from wangHash(key), walking forward. Cheap at low load, but the probe
/// run grows quadratically as the table fills -- which is why its growth threshold is 0.5.
/// *********************************************

#define HASH_NAME "linear"

uint findOrInsert(uint key)
{
	uint slot = wangHash(key) % g_hashCapacity;
	for (uint p = 0u; p < MAX_PROBE; p++) {
		uint index = (slot + p) % g_hashCapacity;
		uint prev = atomicCompSwap(g_hash[index].key, EMPTY_KEY, key);
		if (prev == EMPTY_KEY)
		{
			atomicAdd(g_filledCount, 1u);
			g_firstFrame[index] = g_currentFrame; // slot filled for the first time -> stamp the frame
			return index;
		}
		if (prev == key) return index;
	}
	return HASH_INSERT_FAILED;
}

uint findSlot(uint key)
{
	uint slot = wangHash(key) % g_hashCapacity;
	for (uint p = 0u; p < MAX_PROBE; p++) {
		uint index = (slot + p) % g_hashCapacity;
		uint found = g_hash[index].key;
		if (found == EMPTY_KEY) return HASH_NOT_FOUND;
		if (found == key) return index;
	}
	return HASH_NOT_FOUND;
}
```

- [ ] **Step 4: 디스패처 작성**

`src/TSDF/Memory/Hash/HashStrategy.glsl`:

```glsl
/// *********************************************
/// Hash strategy dispatcher
///
/// The kernels include only this file. Which fragment it pulls in is decided by a preprocessor
/// definition the C++ side passes through ComputePipeline::Define.
///
/// Why a dispatcher instead of `#include HASH_STRATEGY_INCLUDE`: glslang does NOT macro-expand
/// #include ("must be followed by a header name"), so the include name has to be a literal.
/// This keeps each fragment a real file that editors and grep can follow.
/// *********************************************

#define HASH_INSERT_FAILED 0xFFFFFFFFu
#define HASH_NOT_FOUND     0xFFFFFFFFu

#if defined(HASH_BUCKETED)
#include "Bucketed.glsl"
#else
#include "LinearProbe.glsl"
#endif
```

- [ ] **Step 5: 커널에서 탐사 루프를 계약 호출로 교체**

⚠️ **include 위치가 중요하다.** 조각은 `g_hash` / `g_hashCapacity` / `g_filledCount` /
`g_firstFrame` / `g_currentFrame` / `DirEntry`를 참조한다. 이것들은 커널의 버퍼·push constant
선언부에서 정의되므로, 조각 include는 **`voxel_common.glsl` 아래가 아니라 그 선언들 뒤,
처음 쓰는 함수 앞**에 와야 한다. 앞에 두면 `undeclared identifier 'g_hash'`로 깨진다.

`AdvancedTSDF.integrate.comp.glsl`: `findOrInsert` 함수 정의를 통째로 지우고, 그 자리에
(= 버퍼 선언과 `packDirKey` 뒤) 다음을 넣는다:

```glsl
#include "TSDF/Memory/Hash/HashStrategy.glsl"
```

그리고 호출부의 실패 판정을 `~0u`에서 계약 상수로 바꾼다:

```glsl
	uint slot = findOrInsert(key);
	if (slot == HASH_INSERT_FAILED) return;
```

`AdvancedTSDF.extract.comp.glsl`: 같은 include를 (역시 버퍼 선언 뒤, `packDirKey` 뒤에)
추가하고, `fetchDirectionalValue`와
`fetchDirectionalValueAndNormal` 안의 탐사 루프를 `findSlot`으로 바꾼다. 앞쪽 함수는 이렇게 된다:

```glsl
bool fetchDirectionalValue(ivec3 voxel, uint direction, out float value)
{
	value = 0.0;
	uint key;
	if (!packDirKey(voxel, direction, key)) return false;
	uint slot = findSlot(key);
	if (slot == HASH_NOT_FOUND) return false;
	DirEntry entry = g_hash[slot];
	if (entry.sumW < uint(MIN_WEIGHT)) return false;
	value = float(entry.sumDW) / float(entry.sumW);
	return true;
}
```

뒤쪽 함수도 동일하게 `findSlot`으로 슬롯을 얻은 뒤, 기존의 `entry.sumW` 검사와 값·법선 복원
코드를 그대로 이어 쓴다.

`AdvancedTSDF.rehash.comp.glsl`: 이 커널은 **새 테이블**에 삽입하며 `g_new`/`g_newCapacity`라는
다른 버퍼 이름을 쓰므로 `findOrInsert` 계약과 시그니처가 맞지 않는다. **손대지 않는다.**
(리해시는 전략과 무관하게 선형 삽입으로 동작해도 정확하다 — 키가 유일하고 적재율이 낮다.)

- [ ] **Step 6: C++ 기술자 작성**

`src/TSDF/Memory/Hash/HashStrategy.h`:

```cpp
#pragma once

#include <string>

namespace TSDF {

    // One hash addressing variant: which GLSL fragment the dispatcher pulls in, and the load
    // factor past which the table must grow. The threshold lives here and ONLY here -- growth is
    // decided in C++ (AdvancedTSDF::maybeGrow), so a copy in GLSL would be dead code and a second
    // source of truth. Pairing it with the fragment is what stops linear probing from being run
    // at the bucketed threshold, which would silently drop voxels.
    struct HashStrategy {
        const char *name;           // registry name, also the label in comparison output
        const char *macroName;      // preprocessor definition; nullptr = the dispatcher's default
        float loadFactorLimit;      // grow once occupancy reaches this
    };

    inline const HashStrategy &LinearProbeStrategy() {
        static const HashStrategy strategy{"linear", nullptr, 0.5f};
        return strategy;
    }

    // Unknown names fall back to linear rather than throwing: a comparison run that silently used
    // a different hash than asked for would be worse than one that visibly used the baseline.
    // Callers that care should check the returned name.
    const HashStrategy &HashStrategyByName(const std::string &name);

} // namespace TSDF
```

`src/TSDF/Memory/Hash/HashStrategy.cpp`:

```cpp
#include "TSDF/Memory/Hash/HashStrategy.h"

namespace TSDF {

    const HashStrategy &HashStrategyByName(const std::string &name) {
        return LinearProbeStrategy(); // bucketed joins in Task 5
    }

} // namespace TSDF
```

- [ ] **Step 7: `AdvancedTSDF`가 전략을 받게 한다**

`src/TSDF/Backends/AdvancedTSDF.h`: `#include "TSDF/Memory/Hash/HashStrategy.h"`를 추가하고
`Build`의 마지막에 인자를 붙인다:

```cpp
        void Build(Engine::Core::Context &ctx,
                   float voxelSize = 0.01f,
                   float truncation = 0.03f,
                   uint32_t hashCapacity = 1u << 20,
                   uint32_t maxPoints = 1u << 15,
                   const Eigen::Vector3f &windowMinCorner =
                           Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN()),
                   const HashStrategy &hash = LinearProbeStrategy());
```

private 멤버에 추가:

```cpp
        const HashStrategy *m_hash = &LinearProbeStrategy();
```

`src/TSDF/Backends/AdvancedTSDF.cpp`의 `Build` 안, 커널을 만들기 전에 전략을 기록하고, 해시를
읽는 세 커널(integrate / extract / compact)에 정의를 건다. `Build` 시그니처에 인자를 추가한 뒤:

```cpp
        m_hash = &hash;
```

각 `ComputePipeline`을 만든 직후, `Build(...)` 호출 **앞**에 정의를 건다:

```cpp
        kernel_integratePoints = std::make_unique<Engine::Core::ComputePipeline>(ctx);
        if (m_hash->macroName) kernel_integratePoints->Define(m_hash->macroName);
        kernel_integratePoints->Build("TSDF/Backends/AdvancedTSDF.integrate.comp.glsl")
```

**정의를 거는 커널은 integrate와 extract 둘뿐이다.** compact/clear/rehash는 해시 계약을
include하지 않고 슬롯 인덱스로 훑거나 자체 삽입을 하므로, 정의를 걸면 캐시 키만 갈라져 불필요한
재컴파일이 생긴다. `ExtractPointCloud` 안에서 만드는 임시 `kernel`에 건다:

```cpp
        Engine::Core::ComputePipeline kernel(*m_ctx);
        if (m_hash->macroName) kernel.Define(m_hash->macroName);
        kernel.Build("TSDF/Backends/AdvancedTSDF.extract.comp.glsl")
```

`maybeGrow`가 전략의 임계값을 쓰게 바꾼다:

```cpp
    void AdvancedTSDF::maybeGrow() {
        if (!m_hashBuffer || m_hashCapacity == 0) return;
        const uint32_t filled = FilledCount();
        // The threshold belongs to the hash strategy: linear probing collapses well before a
        // bucketed table does, so a shared constant would either waste memory or drop voxels.
        if (double(filled) < double(m_hashCapacity) * double(m_hash->loadFactorLimit)) return;
        growHash(m_hashCapacity * 2u);
    }
```

- [ ] **Step 8: 통과 확인**

```bash
VULKAN_SDK=/usr/local cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release && \
VULKAN_SDK=/usr/local cmake --build build-rel --target vkspatial_tests -j8 && \
./build-rel/test/vkspatial_tests --gtest_filter='TsdfHashStrategy.*:Tsdf*'
```
Expected: 신규 2개 PASS + 기존 TSDF 테스트 전부 PASS (동작 무변경)

- [ ] **Step 9: 회귀 확인 후 커밋**

```bash
./build-rel/test/vkspatial_tests
```
Expected: 278 passed / 1 skipped / 0 failed

```bash
git add -A
git commit -m "refactor(tsdf): extract the hash contract into a swappable GLSL fragment"
```

---

### Task 4: 삽입 실패·성장 계측

α를 올려도 되는지 판정할 유일한 수단이다. 이게 없으면 버킷 전략을 켜도 안전한지 알 수 없다.

**Files:**
- Modify: `src/TSDF/Memory/Hash/LinearProbe.glsl`, `src/TSDF/Backends/AdvancedTSDF.integrate.comp.glsl`
- Modify: `src/TSDF/Backends/AdvancedTSDF.{h,cpp}`
- Modify: `src/TSDF/Memory/{FlatStrategy,TileStrategy,SubmapStrategy}.cpp`
- Modify: `src/Engine/Spatial/TiledDirectionalTSDF.h`, `src/TSDF/Backends/SubmapAdvancedTSDF.h` (합산 접근자)
- Test: `test/test_tsdf_hash.cpp`

**Interfaces:**
- Consumes: Task 3의 계약
- Produces: `AdvancedTSDF::InsertFailureCount() -> uint32_t`, `AdvancedTSDF::GrowCount() -> uint32_t`, `TiledDirectionalTSDF::InsertFailureCount()/GrowCount()`, `SubmapAdvancedTSDF::InsertFailureCount()/GrowCount()`

- [ ] **Step 1: 실패하는 테스트 작성**

```cpp
#include "TSDF/Volume.h"
#include "TSDF/Memory/FlatStrategy.h"

TEST(TsdfHashCounters, NormalIntegrationDropsNothing) {
    Engine::Core::Context context;
    TSDF::FlatStrategy strategy;
    TSDF::VolumeParams params;
    params.voxelSize = 0.05f;
    params.truncation = 0.15f;
    params.hashCapacity = 1u << 16;
    strategy.Build(context, params);

    std::vector<Eigen::Vector3f> points, normals;
    for (int i = -8; i <= 8; ++i)
        for (int j = -8; j <= 8; ++j) {
            points.emplace_back(float(i) * 0.0375f, float(j) * 0.0375f, 0.0f);
            normals.emplace_back(0.0f, 0.0f, 1.0f);
        }

    Engine::Compute::CommandBatch batch(context);
    strategy.Record(points, normals, Eigen::Vector3f(0.0f, 0.0f, 1.0f), batch);
    batch.Submit();

    const TSDF::VolumeStats stats = strategy.Stats();
    EXPECT_GT(stats.occupiedEntryCount, 0u);
    EXPECT_EQ(stats.insertFailureCount, 0u) << "정상 조건에서 복셀이 드롭되면 임계값이 잘못된 것";
}

TEST(TsdfHashCounters, GrowCountRisesWhenTheTableIsTooSmall) {
    Engine::Core::Context context;
    TSDF::FlatStrategy strategy;
    TSDF::VolumeParams params;
    params.voxelSize = 0.01f;
    params.truncation = 0.03f;
    params.hashCapacity = 1u << 10; // 의도적으로 작게 -> 리해시를 강제한다
    strategy.Build(context, params);

    std::vector<Eigen::Vector3f> points, normals;
    for (int i = -40; i <= 40; ++i)
        for (int j = -40; j <= 40; ++j) {
            points.emplace_back(float(i) * 0.01f, float(j) * 0.01f, 0.0f);
            normals.emplace_back(0.0f, 0.0f, 1.0f);
        }

    Engine::Compute::CommandBatch batch(context);
    strategy.Record(points, normals, Eigen::Vector3f(0.0f, 0.0f, 1.0f), batch);
    batch.Submit();

    const TSDF::VolumeStats stats = strategy.Stats();
    EXPECT_GT(stats.growCount, 0u);
    EXPECT_GT(stats.slotCapacity, 1u << 10);
    EXPECT_EQ(stats.insertFailureCount, 0u) << "성장이 제때 일어났다면 드롭은 없어야 한다";
}
```

- [ ] **Step 2: 실패 확인**

```bash
VULKAN_SDK=/usr/local cmake --build build-rel --target vkspatial_tests -j8 && \
./build-rel/test/vkspatial_tests --gtest_filter='TsdfHashCounters.*'
```
Expected: `GrowCountRisesWhenTheTableIsTooSmall`이 `growCount == 0`으로 실패 (필드가 아무도 안 채움)

- [ ] **Step 3: GLSL 카운터 추가**

`AdvancedTSDF.integrate.comp.glsl`의 버퍼 선언부에 binding 5를 추가한다:

```glsl
// Observations dropped because probing gave up. Must stay 0 in a healthy run -- a non-zero value
// means the table's load factor limit is set too high for this hash strategy.
layout(std430, set = 0, binding = 5) buffer InsertFailures
{
	uint g_insertFailureCount;
};
```

호출부에서 실패를 센다:

```glsl
	uint slot = findOrInsert(key);
	if (slot == HASH_INSERT_FAILED) { atomicAdd(g_insertFailureCount, 1u); return; }
```

- [ ] **Step 4: C++ 버퍼와 접근자 추가**

`AdvancedTSDF.h`의 private 멤버에:

```cpp
        std::unique_ptr<Engine::Core::Buffer> m_insertFailureBuffer;
        uint32_t m_growCount = 0;
```

public에 (`HashCapacity()` 아래):

```cpp
        // Observations the integrate kernel dropped because probing gave up. Zero in a healthy
        // run; non-zero means this strategy's load factor limit is too high.
        uint32_t InsertFailureCount() const;

        // Rehashes performed since Build. Each one doubled the table.
        uint32_t GrowCount() const { return m_growCount; }
```

`AdvancedTSDF.cpp`의 `Build`에서 버퍼를 만들고 바인딩한다:

```cpp
        m_insertFailureBuffer = std::make_unique<Engine::Core::Buffer>(ctx);
        m_insertFailureBuffer->AllocateHostVisibleReadback(sizeof(uint32_t));
```

`kernel_integratePoints`의 바인딩 체인에 `.Bind(5, *m_insertFailureBuffer)`를 더한다.
`Reset()`에서 0으로 지운다:

```cpp
        *static_cast<uint32_t *>(m_insertFailureBuffer->MappedPtr()) = 0;
        m_insertFailureBuffer->MakeVisibleToGPU(sizeof(uint32_t));
        m_growCount = 0;
```

접근자 정의:

```cpp
    uint32_t AdvancedTSDF::InsertFailureCount() const {
        if (!m_insertFailureBuffer) return 0;
        m_insertFailureBuffer->MakeVisibleToCPU(sizeof(uint32_t));
        return *static_cast<const uint32_t *>(m_insertFailureBuffer->MappedPtr());
    }
```

`growHash`의 끝(용량 교체 직후)에 `++m_growCount;`를 넣는다. 또한 `growHash`가 새 해시를 바인딩할
때 binding 5는 바뀌지 않으므로 재바인딩이 필요 없다.

- [ ] **Step 5: 합산 접근자 추가**

`src/Engine/Spatial/TiledDirectionalTSDF.h`의 `SlotCapacity()` 옆에:

```cpp
        // Summed across live tiles, mirroring FilledCount()/SlotCapacity().
        uint32_t InsertFailureCount() const {
            uint32_t total = 0;
            for (const auto &kv: m_tiles) total += kv.second->InsertFailureCount();
            return total;
        }

        uint32_t GrowCount() const {
            uint32_t total = 0;
            for (const auto &kv: m_tiles) total += kv.second->GrowCount();
            return total;
        }
```

`src/TSDF/Backends/SubmapAdvancedTSDF.h`의 `SlotCapacity()` 옆에:

```cpp
        uint32_t InsertFailureCount() const {
            return m_base.InsertFailureCount() + m_detail.InsertFailureCount();
        }

        uint32_t GrowCount() const { return m_base.GrowCount() + m_detail.GrowCount(); }
```

- [ ] **Step 6: 세 전략의 `Stats()`에 반영**

`FlatStrategy::Stats()`에서 주석 한 줄을 지우고 두 줄을 채운다:

```cpp
        stats.insertFailureCount = m_tsdf.InsertFailureCount();
        stats.growCount = m_tsdf.GrowCount();
```

`TileStrategy::Stats()`와 `SubmapStrategy::Stats()`에도 동일하게 추가한다.

- [ ] **Step 7: 통과 확인 후 커밋**

```bash
VULKAN_SDK=/usr/local cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release && \
VULKAN_SDK=/usr/local cmake --build build-rel --target vkspatial_tests -j8 && \
./build-rel/test/vkspatial_tests --gtest_filter='TsdfHashCounters.*' && \
./build-rel/test/vkspatial_tests
```
Expected: 신규 2개 PASS, 전체 280 passed / 1 skipped / 0 failed

```bash
git add -A
git commit -m "feat(tsdf): count dropped insertions and rehashes, surface them in VolumeStats"
```

---

### Task 5: 버킷 변형

**Files:**
- Create: `src/TSDF/Memory/Hash/Bucketed.glsl`
- Modify: `src/TSDF/Memory/Hash/HashStrategy.cpp`
- Modify: `src/TSDF/Volume.h` (`VolumeParams::hashStrategy`)
- Modify: `src/TSDF/Memory/{FlatStrategy,TileStrategy,SubmapStrategy}.cpp`, `src/Engine/Spatial/TiledDirectionalTSDF.h`, `src/TSDF/Backends/SubmapAdvancedTSDF.h` (전략 전달)
- Test: `test/test_tsdf_hash.cpp`

**Interfaces:**
- Consumes: Task 3 계약, Task 4 카운터
- Produces: `TSDF::BucketedStrategy()`; `VolumeParams::hashStrategy` (std::string, 기본 `"linear"`)

- [ ] **Step 1: 실패하는 테스트 작성**

```cpp
TEST(TsdfHashStrategy, BucketedIsRegisteredWithItsOwnThreshold) {
    const TSDF::HashStrategy &bucketed = TSDF::HashStrategyByName("bucketed");
    EXPECT_STREQ(bucketed.name, "bucketed");
    EXPECT_STREQ(bucketed.macroName, "HASH_BUCKETED");
    EXPECT_FLOAT_EQ(bucketed.loadFactorLimit, 0.8f);
}

// 같은 스캔을 두 해시로 적분하면 저장된 엔트리 집합이 같아야 한다. 주소 지정만 다를 뿐
// 무엇을 저장하는지는 동일하기 때문이다. 다르면 버킷 구현이 키를 잃고 있다는 뜻이다.
TEST(TsdfHashStrategy, BucketedStoresTheSameEntriesAsLinear) {
    std::vector<Eigen::Vector3f> points, normals;
    for (int i = -8; i <= 8; ++i)
        for (int j = -8; j <= 8; ++j) {
            points.emplace_back(float(i) * 0.0375f, float(j) * 0.0375f, 0.0f);
            normals.emplace_back(0.0f, 0.0f, 1.0f);
        }

    auto runWith = [&](const char *hashName) {
        Engine::Core::Context context;
        TSDF::FlatStrategy strategy;
        TSDF::VolumeParams params;
        params.voxelSize = 0.05f;
        params.truncation = 0.15f;
        params.hashCapacity = 1u << 16;
        params.hashStrategy = hashName;
        strategy.Build(context, params);

        Engine::Compute::CommandBatch batch(context);
        strategy.Record(points, normals, Eigen::Vector3f(0.0f, 0.0f, 1.0f), batch);
        batch.Submit();
        return strategy.Stats();
    };

    const TSDF::VolumeStats linear = runWith("linear");
    const TSDF::VolumeStats bucketed = runWith("bucketed");

    EXPECT_EQ(bucketed.occupiedEntryCount, linear.occupiedEntryCount);
    EXPECT_EQ(bucketed.insertFailureCount, 0u);
    EXPECT_GT(linear.occupiedEntryCount, 0u);
}

// 임계값이 전략을 실제로 따라가는지 -- struct 값이 아니라 관측된 성장 시점으로 확인한다.
// 같은 스캔·같은 초기 용량에서 선형탐사(0.5)는 버킷(0.8)보다 먼저 자라므로, 최종 slotCapacity가
// 더 크거나 같아야 한다. 이게 뒤집히면 임계값 배선이 끊긴 것이다.
TEST(TsdfHashStrategy, BucketedGrowsLaterThanLinear) {
    std::vector<Eigen::Vector3f> points, normals;
    for (int i = -40; i <= 40; ++i)
        for (int j = -40; j <= 40; ++j) {
            points.emplace_back(float(i) * 0.01f, float(j) * 0.01f, 0.0f);
            normals.emplace_back(0.0f, 0.0f, 1.0f);
        }

    auto capacityAfter = [&](const char *hashName) {
        Engine::Core::Context context;
        TSDF::FlatStrategy strategy;
        TSDF::VolumeParams params;
        params.voxelSize = 0.01f;
        params.truncation = 0.03f;
        params.hashCapacity = 1u << 12; // 성장이 반드시 일어나도록 작게 잡는다
        params.hashStrategy = hashName;
        strategy.Build(context, params);

        Engine::Compute::CommandBatch batch(context);
        strategy.Record(points, normals, Eigen::Vector3f(0.0f, 0.0f, 1.0f), batch);
        batch.Submit();
        return strategy.Stats();
    };

    const TSDF::VolumeStats linear = capacityAfter("linear");
    const TSDF::VolumeStats bucketed = capacityAfter("bucketed");

    EXPECT_GT(linear.growCount, 0u) << "이 픽스처는 성장을 강제해야 한다";
    EXPECT_LE(bucketed.slotCapacity, linear.slotCapacity)
            << "버킷은 임계값 0.8이라 선형탐사(0.5)보다 늦게 자라야 한다";
    EXPECT_EQ(bucketed.insertFailureCount, 0u);
}
```

- [ ] **Step 2: 실패 확인**

```bash
VULKAN_SDK=/usr/local cmake --build build-rel --target vkspatial_tests -j8
```
Expected: 컴파일 실패 — `no member named 'hashStrategy' in 'TSDF::VolumeParams'`

- [ ] **Step 3: `Bucketed.glsl` 작성**

`src/TSDF/Memory/Hash/Bucketed.glsl`:

```glsl
/// *********************************************
/// Bucketed probing
///
/// The table is read as buckets of HASH_BUCKET_SIZE contiguous slots. A probe examines a whole
/// bucket before moving to the next, so a run touches far fewer cache lines than slot-at-a-time
/// linear probing at the same load -- which is what lets the growth threshold sit at 0.8.
///
/// Single-choice (no eviction, no second hash): an entry never moves once inserted, which the
/// integrate kernel depends on -- it accumulates into the slot with atomicAdd, so a relocation
/// would land those adds on another key's entry.
/// *********************************************

#define HASH_NAME "bucketed"
#define HASH_BUCKET_SIZE 32u

uint findOrInsert(uint key)
{
	uint bucketCount = max(g_hashCapacity / HASH_BUCKET_SIZE, 1u);
	uint bucket = wangHash(key) % bucketCount;
	for (uint b = 0u; b < MAX_PROBE; b++) {
		uint base = ((bucket + b) % bucketCount) * HASH_BUCKET_SIZE;
		for (uint j = 0u; j < HASH_BUCKET_SIZE; j++) {
			uint index = base + j;
			if (index >= g_hashCapacity) break;
			uint prev = atomicCompSwap(g_hash[index].key, EMPTY_KEY, key);
			if (prev == EMPTY_KEY)
			{
				atomicAdd(g_filledCount, 1u);
				g_firstFrame[index] = g_currentFrame;
				return index;
			}
			if (prev == key) return index;
		}
	}
	return HASH_INSERT_FAILED;
}

uint findSlot(uint key)
{
	uint bucketCount = max(g_hashCapacity / HASH_BUCKET_SIZE, 1u);
	uint bucket = wangHash(key) % bucketCount;
	for (uint b = 0u; b < MAX_PROBE; b++) {
		uint base = ((bucket + b) % bucketCount) * HASH_BUCKET_SIZE;
		bool sawEmpty = false;
		for (uint j = 0u; j < HASH_BUCKET_SIZE; j++) {
			uint index = base + j;
			if (index >= g_hashCapacity) break;
			uint found = g_hash[index].key;
			if (found == key) return index;
			if (found == EMPTY_KEY) sawEmpty = true;
		}
		// An empty slot in this bucket means insertion would have stopped here, so the key
		// cannot live further along the probe run.
		if (sawEmpty) return HASH_NOT_FOUND;
	}
	return HASH_NOT_FOUND;
}
```

- [ ] **Step 4: 레지스트리에 등록**

`src/TSDF/Memory/Hash/HashStrategy.h`에 추가:

```cpp
    inline const HashStrategy &BucketedStrategy() {
        static const HashStrategy strategy{"bucketed", "HASH_BUCKETED", 0.8f};
        return strategy;
    }
```

`src/TSDF/Memory/Hash/HashStrategy.cpp`를 채운다:

```cpp
    const HashStrategy &HashStrategyByName(const std::string &name) {
        if (name == "bucketed") return BucketedStrategy();
        return LinearProbeStrategy();
    }
```

- [ ] **Step 5: `VolumeParams`에 선택 필드 추가**

`src/TSDF/Volume.h`의 `VolumeParams`에:

```cpp
        // Hash addressing variant: "linear" (default) or "bucketed". Resolved through
        // HashStrategyByName, which falls back to linear on an unknown name.
        std::string hashStrategy = "linear";
```

- [ ] **Step 6: 전략을 백엔드까지 전달**

`FlatStrategy::Build`가 전략을 넘기게 한다:

```cpp
        m_tsdf.Build(context, params.voxelSize, params.truncation, params.hashCapacity,
                     params.maxPointsPerFrame, params.windowMinCorner,
                     HashStrategyByName(params.hashStrategy));
```

타일드 경로는 타일을 만들 때 전략이 필요하다. `src/Engine/Spatial/TiledDirectionalTSDF.h`의
`Build`에 인자를 하나 더하고(`const TSDF::HashStrategy &hash = TSDF::LinearProbeStrategy()`),
멤버에 `const TSDF::HashStrategy *m_hash`로 보관한 뒤, 타일을 만드는 `GetTSDF` 안의
`tsdf->Build(...)` 호출에 `*m_hash`를 넘긴다. 헤더 상단에
`#include "TSDF/Memory/Hash/HashStrategy.h"`를 추가한다.

`TileStrategy::Build`가 그 인자를 채운다:

```cpp
        m_tsdf.Build(context, params.voxelSize, params.truncation, params.hashCapacity,
                     params.maxPointsPerFrame, HashStrategyByName(params.hashStrategy));
```

`SubmapAdvancedTSDF::Build`도 같은 방식으로 인자를 하나 더해 `m_base`/`m_detail` 양쪽에 넘기고,
`SubmapStrategy::Build`가 채운다.

- [ ] **Step 7: 통과 확인 후 커밋**

```bash
VULKAN_SDK=/usr/local cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release && \
VULKAN_SDK=/usr/local cmake --build build-rel --target vkspatial_tests -j8 && \
./build-rel/test/vkspatial_tests --gtest_filter='TsdfHash*' && \
./build-rel/test/vkspatial_tests
```
Expected: 신규 3개 PASS, 전체 283 passed / 1 skipped / 0 failed

```bash
git add -A
git commit -m "feat(tsdf): add the bucketed hash variant behind VolumeParams::hashStrategy"
```

---

### Task 6: `Volume::Extract` + 실측 하네스

이 계획의 산출물. 실제 스캔으로 비교 표를 뽑는다.

**Files:**
- Modify: `src/TSDF/Volume.h`, `src/TSDF/ComposedVolume.h`, `src/TSDF/Memory/MemoryStrategy.h`
- Modify: `src/TSDF/Memory/{FlatStrategy,TileStrategy,SubmapStrategy}.{h,cpp}`
- Modify: `example2/tsdf_folder_eval.cpp`
- Test: `test/test_tsdf_hash.cpp`

**Interfaces:**
- Consumes: Task 5의 `VolumeParams::hashStrategy`
- Produces: `Volume::Extract(bool merge) -> Engine::Core::OrientedPointCloud` (그리고 `MemoryStrategy::Extract` 동일 시그니처)

- [ ] **Step 1: 실패하는 테스트 작성**

```cpp
#include "TSDF/ComposedVolume.h"

TEST(TsdfVolumeExtract, EveryStrategyExtractsAPointCloud) {
    const TSDF::VolumeRegistry registry = TSDF::VolumeRegistry::Default();
    std::vector<Eigen::Vector3f> points, normals;
    for (int i = -8; i <= 8; ++i)
        for (int j = -8; j <= 8; ++j) {
            points.emplace_back(float(i) * 0.0375f, float(j) * 0.0375f, 0.0f);
            normals.emplace_back(0.0f, 0.0f, 1.0f);
        }

    for (const std::string &name: registry.Names()) {
        Engine::Core::Context context;
        std::unique_ptr<TSDF::Volume> volume = registry.Create(name);
        ASSERT_NE(volume, nullptr) << name;

        TSDF::VolumeParams params;
        params.voxelSize = 0.05f;
        params.truncation = 0.15f;
        params.hashCapacity = 1u << 16;
        volume->Build(context, params);
        volume->Integrate(points, normals, Eigen::Vector3f(0.0f, 0.0f, 1.0f));

        const Engine::Core::OrientedPointCloud cloud = volume->Extract(/*merge=*/true);
        EXPECT_GT(cloud.points.size(), 0u) << name;
        EXPECT_EQ(cloud.points.size(), cloud.normals.size()) << name;
    }
}
```

- [ ] **Step 2: 실패 확인**

Expected: `no member named 'Extract' in 'TSDF::Volume'`

- [ ] **Step 3: 인터페이스에 `Extract` 추가**

`src/TSDF/Volume.h`의 `Download` 선언 아래:

```cpp
        // Surface point candidates with normals, extracted on the GPU. `merge` collapses
        // near-duplicate candidates within a voxel. This is the form an evaluator compares against
        // ground truth; Download is the raw slot dump.
        virtual Engine::Core::OrientedPointCloud Extract(bool merge = true) const = 0;
```

`src/TSDF/Memory/MemoryStrategy.h`에 동일한 순수 가상 선언을 추가한다.
`src/TSDF/ComposedVolume.h`에 전달자를 추가한다:

```cpp
        Engine::Core::OrientedPointCloud Extract(bool merge = true) const override {
            return m_memory->Extract(merge);
        }
```

세 전략에 구현을 추가한다. `FlatStrategy.cpp`:

```cpp
    Engine::Core::OrientedPointCloud FlatStrategy::Extract(bool merge) const {
        if (m_context == nullptr) return {};
        return m_tsdf.ExtractPointCloud(1u << 21, merge);
    }
```

`TileStrategy.cpp` / `SubmapStrategy.cpp`:

```cpp
    Engine::Core::OrientedPointCloud TileStrategy::Extract(bool merge) const {
        if (m_context == nullptr) return {};
        return m_tsdf.ExtractPointCloud(merge);
    }
```

(타일드·서브맵의 `ExtractPointCloud`는 후보 상한 인자를 받지 않는다.)
각 헤더에 대응하는 `override` 선언을 추가한다.

- [ ] **Step 4: 하네스에 플래그 추가**

`example2/tsdf_folder_eval.cpp`의 `BuildArgParser` 체인에 두 줄을 더한다:

```cpp
                        .Option("--tsdf", "flat")
                        .Option("--hash", "linear")
```

`Engine::Core::Context ctx;` 아래의 백엔드 분기 전체를 레지스트리 경유로 바꾼다:

```cpp
        Engine::Core::Context ctx;
        const std::string tsdfName = arg.Value("--tsdf");
        const std::string hashName = arg.Value("--hash");

        const TSDF::VolumeRegistry registry = TSDF::VolumeRegistry::Default();
        std::unique_ptr<TSDF::Volume> volume = registry.Create(tsdfName);
        if (!volume) {
            std::cerr << "unknown --tsdf: " << tsdfName << "\n";
            return 2;
        }

        TSDF::VolumeParams params;
        params.voxelSize = voxel;
        params.truncation = trunc;
        params.hashCapacity = arg.Has("--tile-hash") && tsdfName != "flat" ? tileHash : hashCap;
        params.maxPointsPerFrame = maxPts;
        params.hashStrategy = hashName;
        if (tsdfName == "flat") params.windowMinCorner = windowMinCorner;
        volume->Build(ctx, params);

        TSDF::IntegrationOptions options;
        options.pointToPlane = p2p;
        options.confidenceWeight = conf;
        options.hermitePosition = hermite;
        options.quality = {3, 4, true};
        volume->Configure(options);

        for (const auto &fr: frames) volume->Integrate(fr.pts, fr.nrm, fr.cam);
        Engine::Core::OrientedPointCloud recon = volume->Extract(/*merge=*/true);
```

- [ ] **Step 5: 비교 표 출력**

추출 직후에 넣는다:

```cpp
        const TSDF::VolumeStats stats = volume->Stats();
        std::printf("\n%-10s %-8s %10s %10s %7s %7s %9s %6s %6s\n",
                    "hash", "tsdf", "occupied", "slots", "load", "tables", "tableMB",
                    "drops", "grows");
        std::printf("%-10s %-8s %10llu %10llu %7.3f %7u %9.1f %6llu %6u\n",
                    hashName.c_str(), tsdfName.c_str(),
                    (unsigned long long) stats.occupiedEntryCount,
                    (unsigned long long) stats.slotCapacity,
                    stats.LoadFactor(),
                    stats.tableCount,
                    double(stats.deviceMemoryBytes) / (1024.0 * 1024.0),
                    (unsigned long long) stats.insertFailureCount,
                    stats.growCount);
        if (stats.insertFailureCount > 0)
            std::printf("WARNING: %llu observations were dropped -- this hash's load factor limit "
                        "is too high for this scene; the numbers above understate occupancy.\n",
                        (unsigned long long) stats.insertFailureCount);
```

- [ ] **Step 6: 통과 확인**

```bash
VULKAN_SDK=/usr/local cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release && \
VULKAN_SDK=/usr/local cmake --build build-rel --target vkspatial_tests -j8 && \
./build-rel/test/vkspatial_tests --gtest_filter='TsdfVolumeExtract.*' && \
VULKAN_SDK=/usr/local cmake --build build-rel --target tsdf_folder_eval -j8
```
Expected: 테스트 PASS, 하네스 빌드 성공

- [ ] **Step 7: 실제 스캔으로 두 해시를 비교**

```bash
./build-rel/example2/tsdf_folder_eval --dir scan_out --voxel 0.01 --tsdf tile --hash linear
./build-rel/example2/tsdf_folder_eval --dir scan_out --voxel 0.01 --tsdf tile --hash bucketed
```

두 실행의 `tableMB`를 기록한다. **이 비율이 이 계획의 답이다.** `drops`가 0이 아니면 버킷의
임계값 0.8이 이 장면에 너무 높은 것이므로, 그 사실을 결과로 보고한다 (가설 기각도 결과다).
`tables`가 점유율과 무관하게 크면 메모리가 count-limited라는 뜻이고, 그 경우 결론은
"버킷 해시로는 이 문제를 못 푼다"이다.

- [ ] **Step 8: 회귀 확인 후 커밋**

```bash
./build-rel/test/vkspatial_tests
```
Expected: 284 passed / 1 skipped / 0 failed

```bash
git add -A
git commit -m "feat(tsdf): add Volume::Extract and a --tsdf/--hash comparison harness"
```

---

## 후속으로 넘기는 것

| 항목 | 이유 |
|---|---|
| SoA 레이아웃 (키 배열 분리) | 버킷 결과를 본 뒤 판단. 버킷만으로 목표에 닿으면 불필요 |
| `Integrate`/`Extract` 축 | 전략이 하나뿐이라 축이 아니다 |
| 동결 백엔드 셋 정리 | 이번 범위 밖 |
| `rehash` 커널의 전략화 | 새 테이블에 삽입하며 버퍼 이름이 달라 계약과 맞지 않는다. 키가 유일하고 적재율이 낮아 선형 삽입으로도 정확하다 |
| `Memory/`와 `Backends/`의 계층 중복 | 축이 늘어난 뒤에 판단 |
