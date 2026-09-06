#pragma once

#include <memory>
#include <mutex>

namespace util {

    template<typename T>
    class Mailbox {
    public:
        void Publish(std::shared_ptr<const T> value) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_value = std::move(value);
        }

        std::shared_ptr<const T> Latest() const {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_value;
        }

    private:
        mutable std::mutex m_mutex;
        std::shared_ptr<const T> m_value;
    };

} // namespace util
