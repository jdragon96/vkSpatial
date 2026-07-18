#include "Engine/Spatial/SpatialIndex.h"

#include "Engine/Spatial/BinaryLBVH.h"

#include <stdexcept>

namespace Engine::Spatial {

    std::unique_ptr<SpatialIndex>
    MakeSpatialIndex(Engine::Core::Context &ctx, BVHKind kind, const BVHParams &params) {
        (void) params; // Wide consumes maxLeafPrimitives (Task 9); ignored by BinaryLBVH.
        switch (kind) {
            case BVHKind::BinaryLBVH:
                return std::make_unique<BinaryLBVH>(ctx);
            case BVHKind::Wide:
                throw std::runtime_error(
                        "MakeSpatialIndex: Wide backend not implemented yet");
        }
        throw std::runtime_error("MakeSpatialIndex: unknown BVHKind");
    }

} // namespace Engine::Spatial
