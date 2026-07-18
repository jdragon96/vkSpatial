#include "Engine/Spatial/SpatialIndex.h"

#include "Engine/Spatial/BinaryLBVH.h"
#include "Engine/Spatial/WideBVH.h"

#include <stdexcept>

namespace Engine::Spatial {

    std::unique_ptr<SpatialIndex>
    MakeSpatialIndex(Engine::Core::Context &ctx, BVHKind kind, const BVHParams &params) {
        switch (kind) {
            case BVHKind::BinaryLBVH:
                return std::make_unique<BinaryLBVH>(ctx);
            case BVHKind::Wide:
                return std::make_unique<WideBVH>(ctx, params.maxLeafPrimitives);
        }
        throw std::runtime_error("MakeSpatialIndex: unknown BVHKind");
    }

} // namespace Engine::Spatial
