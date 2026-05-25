# vkBVH

GPU-accelerated Linear Bounding Volume Hierarchy (LBVH) built on Vulkan compute shaders.  
Every operation — from BVH construction to spatial queries — runs entirely on the GPU.

---

### Submodule Dependencies

```
lib/SPIRV-Reflect   — SPIR-V reflection (automatic local_size detection)
lib/tinyobjloader   — OBJ mesh loading (lbvh examples)
```

---

## Build

```bash
git clone --recursive <repo-url>
cd VkLBVH

cmake -S . -B build
cmake --build build --parallel
```

If the Vulkan SDK path is not set in your environment:

```bash
cmake -S . -B build -DVULKAN_SDK=/path/to/VulkanSDK/macOS
cmake --build build --parallel
```

---

## Quick Start

```cpp
#include "vkBVH/common/vkContext.h"
#include "vkBVH/vkBVH.h"

int main() {
    // 1. Initialize Vulkan context
    VkContext ctx;
    ctx.init();

    // 2. Prepare point data
    std::vector<PointPrim> points = {
        {1.0f, 0.0f, 0.0f},
        {2.0f, 1.0f, 0.5f},
        {0.5f, 0.5f, 0.5f},
        // ...
    };

    // 3. Build BVH
    vkBVH bvh(&ctx, VKBVH_SHADER_DIR);
    bvh.Build(points);

    // 4. Radius Search: find all points within radius 2.0 of the origin
    std::vector<uint32_t> neighbors = bvh.RadiusSearch(0.f, 0.f, 0.f, 2.0f);

    // 5. KNN: find the 5 nearest points
    std::vector<uint32_t> knn = bvh.KNN(0.f, 0.f, 0.f, 5);

    ctx.shutdown();
}
```

---

## API

### `vkBVH`

```cpp
vkBVH(VkContext *ctx, const std::string &shaderDir);
```

#### BVH Construction

```cpp
// Point primitives (AABB min == max)
void Build(const std::vector<PointPrim> &points);

// Triangle primitives
void Build(const std::vector<TrianglePrim> &triangles);

// Arbitrary AABB primitives
void Build(const std::vector<Primitive> &primitives);
```

> At least 2 primitives are required.

#### Spatial Queries

```cpp
// Returns all primitive indices within a sphere
// No result count cap (internally maxResults = Length())
std::vector<uint32_t> RadiusSearch(float cx, float cy, float cz, float r);

// Returns the k nearest primitive indices (k <= 64)
// Returns fewer than k results if the total primitive count is less than k
std::vector<uint32_t> KNN(float cx, float cy, float cz, int k);
```

#### Primitive Types

```cpp
struct PointPrim    { float x, y, z; };
struct TrianglePrim { float v0[3], v1[3], v2[3]; };
struct Primitive    { uint32_t index;
                      float aabbMinX, aabbMinY, aabbMinZ;
                      float aabbMaxX, aabbMaxY, aabbMaxZ; };
```

---

## Tests

```bash
# Run all tests
./build/test/vklbvh_tests

# Filter by suite
./build/test/vklbvh_tests --gtest_filter="RadiusSearchTest*"
./build/test/vklbvh_tests --gtest_filter="KNNTest*"
./build/test/vklbvh_tests --gtest_filter="HierarchyTest*"
./build/test/vklbvh_tests --gtest_filter="MortonCodeTest*"
./build/test/vklbvh_tests --gtest_filter="RadixSortTest*"
./build/test/vklbvh_tests --gtest_filter="VkComputeTest*"
```

### Test Suites

| Suite              | File                  | What it verifies                          |
| ------------------ | --------------------- | ----------------------------------------- |
| `VkComputeTest`    | test_vkCompute.cpp    | Basic Vulkan compute functionality        |
| `MortonCodeTest`   | test_mortonCode.cpp   | GPU Morton code computation accuracy      |
| `RadixSortTest`    | test_radixSort.cpp    | 4-bit Radix Sort correctness              |
| `HierarchyTest`    | test_hierarchy.cpp    | BVH hierarchy structural validity         |
| `RadiusSearchTest` | test_radiusSearch.cpp | GPU Radius Search vs CPU brute-force      |
| `KNNTest`          | test_knn.cpp          | GPU KNN vs CPU brute-force                |

---

## Project Structure

```
VkLBVH/
├── src/
│   ├── shader/                  # GLSL shader sources
│   │   ├── bvh_mortonCode.comp
│   │   ├── bvh_radixSort_*.comp
│   │   ├── bvh_hierarchy.comp
│   │   ├── bvh_boundingBox.comp
│   │   ├── cmd_radiusSearch.comp
│   │   └── cmd_knn.comp
│   └── vkBVH/
│       ├── vkBVH.h / vkBVH.cpp  # Public API
│       ├── types.h               # Primitive, MortonCode, etc.
│       ├── command/
│       │   ├── RadiusSearch.cpp  # RadiusSearch implementation
│       │   └── KNN.cpp           # KNN implementation
│       └── common/
│           ├── vkContext.*       # Vulkan instance / device / queue
│           ├── vkGPUMemory.*     # GPU buffer alloc / upload / download
│           ├── vkComputeBase.*   # Compute kernel wrapper (Build/Bind/Dispatch)
│           └── vkScopedMemory.*  # RAII memory mapping
├── test/                         # GTest test suite
├── examples/                     # Usage examples (PCD load, PLY export)
└── lib/
    ├── SPIRV-Reflect/            # SPIR-V reflection
    └── tinyobjloader/            # OBJ loader
```

---

## License

MIT License — see [LICENSE](LICENSE)
