#include "TSDF/Hash/HashStrategy.h"

namespace TSDF {

    const HashStrategy &HashStrategyByName(const std::string &name) {
        if (name == "bucketed") return BucketedStrategy();
        return LinearProbeStrategy();
    }

} // namespace TSDF
