#pragma once

#include <vulkan/vulkan.h>

namespace vkCommon {

    class vkScopedMemory {
    public:
        vkScopedMemory(VkDevice device, VkDeviceMemory memory,
                       VkDeviceSize offset = 0,
                       VkDeviceSize size = VK_WHOLE_SIZE);

        ~vkScopedMemory();

        // 복사 금지
        vkScopedMemory(const vkScopedMemory &) = delete;
        vkScopedMemory &operator=(const vkScopedMemory &) = delete;

        // 이동 허용
        vkScopedMemory(vkScopedMemory &&o) noexcept;
        void *data() const { return m_ptr; }

        template<typename T>
        T *as() const { return static_cast<T *>(m_ptr); }

        bool valid() const { return m_ptr != nullptr; }

    private:
        VkDevice m_device;
        VkDeviceMemory m_memory;
        void *m_ptr = nullptr;
    };

} // namespace vkCommon
