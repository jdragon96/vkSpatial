```cpp
vkRender::SwapChainDescriptor descriptor{};
descriptor.width = static_cast<uint32_t>(framebufferWidth);
descriptor.height = static_cast<uint32_t>(framebufferHeight);
descriptor.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

// 이미지가 어떤 용도로 사용되는지 명시한다.
// 이미지 종류
// 1) VkImage
// 2) VkSwapchain
typedef enum VkImageUsageFlagBits {
    // VkImage를 CPU로 복사 가능하다록 한다.
    VK_IMAGE_USAGE_TRANSFER_SRC_BIT = 0x00000001,
    // VkImage에 직접 랜더링 하지 않고, 복사한다.
    VK_IMAGE_USAGE_TRANSFER_DST_BIT = 0x00000002,
    VK_IMAGE_USAGE_SAMPLED_BIT = 0x00000004,
    VK_IMAGE_USAGE_STORAGE_BIT = 0x00000008,
    // VkImage에 랜더링한다.
    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT = 0x00000010,
    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT = 0x00000020,
    VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT = 0x00000040,
    VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT = 0x00000080,
    VK_IMAGE_USAGE_HOST_TRANSFER_BIT = 0x00400000,
}
```

- Vulkan SwapChain 생성 과정

```
1. 해당 디바이스에서 Surface를 지원하는지 체크한다.
2.

```

## 용어 설명

- Attachment
  - Attachment는 렌더링 중 GPU가 출력 결과를 쓰거나, 특정 방식으로 읽는 이미지
  - 다시말해, 파이프라인에서 작성된 컬러 또는 뎁스값을 의미함
