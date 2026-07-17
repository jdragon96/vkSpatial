#include <gtest/gtest.h>

#include "Engine/Core/Context.h"

using namespace Engine::Core;

TEST(ContextTest, ComputeOnlyConstructionProducesValidHandles) {
    Context ctx;  // enablePresent=false by default

    EXPECT_NE(ctx.instance, VK_NULL_HANDLE);
    EXPECT_NE(ctx.physicalDevice, VK_NULL_HANDLE);
    EXPECT_NE(ctx.device, VK_NULL_HANDLE);
    EXPECT_NE(ctx.computeQueue, VK_NULL_HANDLE);
    EXPECT_NE(ctx.cmdPool, VK_NULL_HANDLE);
    EXPECT_NE(ctx.allocator, VK_NULL_HANDLE);

    EXPECT_EQ(ctx.graphicsQueue, VK_NULL_HANDLE);
    EXPECT_EQ(ctx.graphicsCmdPool, VK_NULL_HANDLE);
    EXPECT_EQ(ctx.surface, VK_NULL_HANDLE);
}
