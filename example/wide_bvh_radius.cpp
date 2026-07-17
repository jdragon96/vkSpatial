#include "vkCommon/vkContext.h"
#include "vkSpatial/vkWideBVH.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <vector>

int main() {
    vkCommon::VkContext ctx;
    bool initialized = false;

    try {
        ctx.init();
        initialized = true;

        std::vector<vkSpatial::PointPrim> points = {
                {0.0f, 0.0f, 0.0f},
                {0.5f, 0.0f, 0.0f},
                {0.0f, 0.75f, 0.0f},
                {1.5f, 0.0f, 0.0f},
                {-0.25f, 0.25f, 0.25f},
                {0.0f, 0.0f, 2.0f},
        };

        vkSpatial::vkWideBVH bvh(&ctx);
        bvh.Build(points);

        constexpr float cx = 0.0f;
        constexpr float cy = 0.0f;
        constexpr float cz = 0.0f;
        constexpr float radius = 1.0f;

        std::vector<uint32_t> result =
                bvh.RadiusSearch(cx, cy, cz, radius);
        std::sort(result.begin(), result.end());

        std::cout << "WideBVH RadiusSearch\n";
        std::cout << "points: " << bvh.Length()
                  << "  wide nodes: " << bvh.NodeCount() << "\n";
        std::cout << "query center: (" << cx << ", " << cy << ", " << cz
                  << ")  radius: " << radius << "\n";
        std::cout << "result indices:";
        for (uint32_t index: result)
            std::cout << ' ' << index;
        std::cout << "\n";

        ctx.shutdown();
        return 0;
    } catch (const std::exception &e) {
        if (initialized)
            ctx.shutdown();
        std::cerr << "wide_bvh_radius failed: " << e.what() << "\n";
        return 1;
    }
}
