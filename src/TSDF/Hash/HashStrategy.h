#pragma once

#include <string>

namespace TSDF {

    // One hash addressing variant: which GLSL fragment the dispatcher pulls in, and the load
    // factor past which the table must grow. The threshold lives here and ONLY here -- growth is
    // decided in C++ (AdvancedTSDF::maybeGrow), so a copy in GLSL would be dead code and a second
    // source of truth. Pairing it with the fragment is what stops linear probing from being run
    // at the bucketed threshold, which would silently drop voxels.
    struct HashStrategy {
        const char *name;      // registry name, also the label in comparison output
        const char *macroName; // preprocessor definition; nullptr = the dispatcher's default

        // Grow once occupancy reaches this fraction of capacity.
        //
        // READ THIS BEFORE PICKING A HIGHER LIMIT -- the limit is also the per-frame safety margin.
        // AdvancedTSDF::maybeGrow runs at the TOP of RecordIntegrateGPU and decides from
        // FilledCount(), a host readback that only reflects submissions that have already
        // COMPLETED. The frame being recorded has not run yet and its inserts are invisible to that
        // decision, so growth is always sized to STALE occupancy. Between the decision and the next
        // Record there is exactly (1 - loadFactorLimit) * capacity of free space, and one frame
        // that inserts more new entries than that fills the table mid-dispatch: the surplus
        // exhausts its probe budget, findOrInsert returns HASH_INSERT_FAILED, and those
        // observations are DROPPED (counted in VolumeStats::insertFailureCount) -- no later grow
        // can recover them.
        //
        // So the limit trades steady-state memory against burst tolerance: at 0.5 a frame may add
        // up to half the table before anything is lost, at 0.8 only a fifth. That is the real cost
        // of the bucketed strategy's 0.8, and it is why raising the limit further needs evidence
        // about the largest per-frame insert count the scene produces, not just about probe
        // behaviour at high load. Always check insertFailureCount == 0 when changing this.
        float loadFactorLimit;
    };

    inline const HashStrategy &LinearProbeStrategy() {
        static const HashStrategy strategy{"linear", nullptr, 0.5f};
        return strategy;
    }

    // Bucketed probing: a probe reads a whole HASH_BUCKET_SIZE-slot bucket per step instead of one
    // slot at a time, so it stays cheap at a higher load -- hence the 0.8 threshold vs linear's 0.5.
    // Both strategies examine the same MAX_PROBE (128) slots before giving up; see the budget
    // comment in Bucketed.glsl. The 0.8 also shrinks the per-frame burst headroom from half the
    // table to a fifth -- see loadFactorLimit above before choosing this for a live capture.
    inline const HashStrategy &BucketedStrategy() {
        static const HashStrategy strategy{"bucketed", "HASH_BUCKETED", 0.8f};
        return strategy;
    }

    // Unknown names fall back to linear rather than throwing: a comparison run that silently used
    // a different hash than asked for would be worse than one that visibly used the baseline.
    // Callers that care should check the returned name.
    const HashStrategy &HashStrategyByName(const std::string &name);

} // namespace TSDF
