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

    // Bucketed probing: a probe reads a whole HASH_BUCKET_SIZE-slot bucket per step instead of one
    // slot at a time, so it stays cheap at a higher load -- hence the 0.8 threshold vs linear's 0.5.
    inline const HashStrategy &BucketedStrategy() {
        static const HashStrategy strategy{"bucketed", "HASH_BUCKETED", 0.8f};
        return strategy;
    }

    // Unknown names fall back to linear rather than throwing: a comparison run that silently used
    // a different hash than asked for would be worse than one that visibly used the baseline.
    // Callers that care should check the returned name.
    const HashStrategy &HashStrategyByName(const std::string &name);

} // namespace TSDF
