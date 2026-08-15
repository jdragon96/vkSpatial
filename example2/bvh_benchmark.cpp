// Benchmark: compares Engine::Spatial acceleration backends — BinaryLBVH vs WideBVH —
// by swapping the BVH facade's backend (BVHConfiguration::backend). It
// reports build time, query time, and structure size, demonstrating the interchangeable
// design and quantifying the wide BVH's compactness.
//
//   ./bvh_benchmark [--csv out.csv]
//
// Timing is wall-clock (std::chrono); ComputePipeline::Dispatch blocks on
// vkQueueWaitIdle, so measured wall time includes full GPU execution.
//
// Columns:
//   build_ms  median of 5 builds (+1 warmup)          — TRUSTWORTHY
//   knn_us    mean per-query KNN time                 — approximate
//   radius_us mean per-query RadiusSearch time        — approximate
//   nodes     structural node count                   — TRUSTWORTHY (wide << binary)
//   mem_KB    structure footprint                     — TRUSTWORTHY
//   knn_rec%  KNN recall vs CPU (|gpu ∩ true-k| / k)  — quality indicator
//
// KNOWN ISSUES (pre-existing Engine::Core, NOT the BVH algorithm — see
// docs/KNOWN_ISSUES_engine_core_large_n.md):
//   1. N is capped at 512. Engine::Core produces non-deterministic, WRONG build/query
//      results at N >= ~1000 (multi-workgroup build path). The old vkBVH is correct at
//      N=1000 with the same shaders, so the fault is in Engine::Core, not the shaders.
//   2. Under this benchmark's repeated build+dispatch load, knn_rec% dips below 100%
//      even at N <= 512 (an Engine::Core coherence issue under load). AUTHORITATIVE
//      correctness lives in test/test_spatialIndex.cpp (all pass): it builds one index
//      per fresh Context and matches a CPU brute-force reference exactly.
//   3. WideBVH::RadiusSearch returns empty on this HW (Apple/MoltenVK) — its radius_us
//      reflects an empty traversal. The old vkWideBVH fails identically.

#include "Engine/Core/Context.h"
#include "BVH/BVHTypes.h"
#include "BVH/BVH.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace Engine::Spatial;
using Clock = std::chrono::steady_clock;

namespace {

    struct BackendCase {
        const char *backend;
        uint32_t leaf;
        const char *label;
    };

    struct Query {
        float x, y, z;
    };

    std::vector<PointPrim> randomPoints(uint32_t n, uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> d(-20.0f, 20.0f);
        std::vector<PointPrim> pts(n);
        for (auto &p : pts) {
            p.x = d(rng);
            p.y = d(rng);
            p.z = d(rng);
        }
        return pts;
    }

    std::vector<Query> randomQueries(uint32_t m, uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> d(-20.0f, 20.0f);
        std::vector<Query> qs(m);
        for (auto &q : qs) {
            q.x = d(rng);
            q.y = d(rng);
            q.z = d(rng);
        }
        return qs;
    }

    std::vector<uint32_t> cpuKNN(const std::vector<PointPrim> &pts,
                                 float cx, float cy, float cz, uint32_t k) {
        std::vector<std::pair<float, uint32_t>> d;
        d.reserve(pts.size());
        for (uint32_t i = 0; i < pts.size(); ++i) {
            const float dx = pts[i].x - cx, dy = pts[i].y - cy, dz = pts[i].z - cz;
            d.emplace_back(dx * dx + dy * dy + dz * dz, i);
        }
        std::sort(d.begin(), d.end());
        const uint32_t n = std::min<uint32_t>(k, static_cast<uint32_t>(d.size()));
        std::vector<uint32_t> out(n);
        for (uint32_t i = 0; i < n; ++i) out[i] = d[i].second;
        return out;
    }

    // Recall of the GPU KNN vs the true k-nearest: |gpu ∩ true| / |true|, as a percent.
    double knnRecall(std::vector<uint32_t> gpu, std::vector<uint32_t> truth) {
        if (truth.empty()) return 100.0;
        std::sort(gpu.begin(), gpu.end());
        std::sort(truth.begin(), truth.end());
        uint32_t inter = 0;
        for (uint32_t t : truth)
            if (std::binary_search(gpu.begin(), gpu.end(), t)) ++inter;
        return 100.0 * static_cast<double>(inter) / static_cast<double>(truth.size());
    }

    struct Row {
        std::string backend;
        uint32_t n;
        double buildMs;
        double knnUs;
        double radiusUs;
        uint32_t nodeCount;
        uint32_t memBytes;
        double knnRec;
    };

    // Median build time over `repeats` fresh builds (+ 1 warmup).
    double medianBuildMs(Engine::Core::Context &ctx, const BackendCase &c,
                         const std::vector<PointPrim> &pts, int repeats) {
        {
            BVH warm;
            BVHConfiguration warmConfig;
            warmConfig.backend = c.backend;
            warmConfig.backendConfig.maxLeafPrimitives = c.leaf ? c.leaf : 4u;
            warm.Build(ctx, warmConfig);
            warm.Insert(pts);
        }
        std::vector<double> ms;
        ms.reserve(repeats);
        for (int i = 0; i < repeats; ++i) {
            BVH idx;
        BVHConfiguration idxConfig;
        idxConfig.backend = c.backend;
        idxConfig.backendConfig.maxLeafPrimitives = c.leaf ? c.leaf : 4u;
        idx.Build(ctx, idxConfig);
            const auto t0 = Clock::now();
            idx.Insert(pts);
            const auto t1 = Clock::now();
            ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(ms.begin(), ms.end());
        return ms[ms.size() / 2];
    }

    Row runBackend(Engine::Core::Context &ctx, const BackendCase &c,
                   const std::vector<PointPrim> &pts,
                   const std::vector<Query> &queries, uint32_t k, float radius) {
        Row row;
        row.backend = c.label;
        row.n = static_cast<uint32_t>(pts.size());
        row.buildMs = medianBuildMs(ctx, c, pts, 5);

        BVH idx;
        BVHConfiguration idxConfig;
        idxConfig.backend = c.backend;
        idxConfig.backendConfig.maxLeafPrimitives = c.leaf ? c.leaf : 4u;
        idx.Build(ctx, idxConfig);
        idx.Insert(pts);
        row.nodeCount = idx.Stats().nodeCount;
        row.memBytes = idx.Stats().memoryBytes;

        const auto tk0 = Clock::now();
        for (const auto &q : queries) idx.KNN(q.x, q.y, q.z, static_cast<int>(k));
        const auto tk1 = Clock::now();
        row.knnUs = std::chrono::duration<double, std::micro>(tk1 - tk0).count() /
                    static_cast<double>(queries.size());

        const auto tr0 = Clock::now();
        for (const auto &q : queries) idx.RadiusSearch(q.x, q.y, q.z, radius);
        const auto tr1 = Clock::now();
        row.radiusUs = std::chrono::duration<double, std::micro>(tr1 - tr0).count() /
                       static_cast<double>(queries.size());

        row.knnRec = knnRecall(
                idx.KNN(queries[0].x, queries[0].y, queries[0].z, static_cast<int>(k)),
                cpuKNN(pts, queries[0].x, queries[0].y, queries[0].z, k));
        return row;
    }

} // namespace

int main(int argc, char **argv) {
    std::string csvPath;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--csv" && i + 1 < argc) csvPath = argv[++i];
    }

    Engine::Core::Context ctx;

    const std::vector<BackendCase> backends = {
            {"binary", 0u, "BinaryLBVH"},
            {"wide", 4u, "Wide(leaf=4)"},
            {"wide", 8u, "Wide(leaf=8)"},
    };
    // Capped at 512 — see KNOWN ISSUE 1 (Engine::Core is non-deterministic at N >= ~1000).
    const std::vector<uint32_t> nSweep = {128u, 256u, 512u};
    const uint32_t k = 16;
    const float radius = 10.0f;
    const uint32_t queryCount = 200;

    std::vector<Row> rows;
    for (uint32_t n : nSweep) {
        const auto pts = randomPoints(n, 12345u);
        const auto queries = randomQueries(queryCount, 999u);
        for (const auto &b : backends)
            rows.push_back(runBackend(ctx, b, pts, queries, k, radius));
    }

    std::printf("%-14s %6s %9s %9s %10s %8s %9s %9s\n",
                "backend", "N", "build_ms", "knn_us", "radius_us", "nodes",
                "mem_KB", "knn_rec%");
    for (const auto &r : rows)
        std::printf("%-14s %6u %9.3f %9.3f %10.3f %8u %9.1f %8.1f\n",
                    r.backend.c_str(), r.n, r.buildMs, r.knnUs, r.radiusUs,
                    r.nodeCount, r.memBytes / 1024.0, r.knnRec);

    std::printf("\nStructure size (nodes, mem_KB) and build_ms are trustworthy and show the\n"
                "wide BVH's compactness. knn_rec%% < 100 and wide radius_us reflect pre-existing\n"
                "Engine::Core issues under load (see the header + KNOWN_ISSUES doc), not the BVH\n"
                "algorithm — correctness is validated exactly by test/test_spatialIndex.cpp.\n");

    if (!csvPath.empty()) {
        std::ofstream f(csvPath);
        f << "backend,N,build_ms,knn_us,radius_us,nodes,mem_bytes,knn_rec_pct\n";
        for (const auto &r : rows)
            f << r.backend << ',' << r.n << ',' << r.buildMs << ',' << r.knnUs << ','
              << r.radiusUs << ',' << r.nodeCount << ',' << r.memBytes << ','
              << r.knnRec << '\n';
        std::printf("\nWrote CSV to %s\n", csvPath.c_str());
    }

    return 0;
}
