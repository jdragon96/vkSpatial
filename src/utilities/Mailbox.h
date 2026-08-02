#pragma once

#include <memory>
#include <mutex>

namespace util {

    // Single-slot "latest value" mailbox for one producer and one-or-more consumers. Publish() swaps
    // in a new immutable value (shared_ptr<const T>); Latest() returns the most recently published
    // value, or nullptr before the first Publish. Thread-safe; consumers never block the producer
    // beyond a short critical section, and a held snapshot stays valid after newer ones arrive.
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
