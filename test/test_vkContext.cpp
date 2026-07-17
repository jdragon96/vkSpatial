#include <gtest/gtest.h>

#include "vkCommon/vkContext.h"

using namespace vkCommon;

TEST(VkContextTest, ComputeOnlyInitProducesValidHandles) {
    VkContext ctx;
    ctx.init();  // enablePresent=false by default

    EXPECT_NE(ctx.instance, VK_NULL_HANDLE);
    EXPECT_NE(ctx.physDevice, VK_NULL_HANDLE);
    EXPECT_NE(ctx.device, VK_NULL_HANDLE);
    EXPECT_NE(ctx.computeQueue, VK_NULL_HANDLE);
    EXPECT_NE(ctx.cmdPool, VK_NULL_HANDLE);

    // enablePresent=false일 때는 그래픽스/프레젠트 관련 필드가 채워지지 않는다.
    EXPECT_EQ(ctx.graphicsQueue, VK_NULL_HANDLE);
    EXPECT_EQ(ctx.surface, VK_NULL_HANDLE);

    ctx.shutdown();

    // shutdown()이 실제로 해제하는 핸들만 검증한다 (computeQueue/physDevice는
    // shutdown()이 재설정하지 않는 값이라 여기서 단언하지 않는다 — vkContext.cpp:31-40 참고).
    EXPECT_EQ(ctx.instance, VK_NULL_HANDLE);
    EXPECT_EQ(ctx.device, VK_NULL_HANDLE);
    EXPECT_EQ(ctx.cmdPool, VK_NULL_HANDLE);
}
