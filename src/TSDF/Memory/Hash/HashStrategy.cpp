#include "TSDF/Memory/Hash/HashStrategy.h"

namespace TSDF {

    const HashStrategy &HashStrategyByName(const std::string &name) {
        return LinearProbeStrategy(); // bucketed joins in Task 5
    }

} // namespace TSDF
