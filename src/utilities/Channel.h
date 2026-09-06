#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <utility>

namespace util {

    template<typename T>
    class Channel {
    public:
        explicit Channel(std::size_t capacity, bool dropOldestWhenFull = true)
            : m_capacity(capacity == 0 ? 1 : capacity), m_dropOldest(dropOldestWhenFull) {}

        Channel(const Channel &) = delete;
        Channel &operator=(const Channel &) = delete;

        // Enqueue `value`. Returns false iff the channel is closed. Honors the overflow policy.
        bool Push(T value) {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (m_closed) return false;
            if (m_queue.size() >= m_capacity) {
                if (m_dropOldest) {
                    m_queue.pop_front();
                    ++m_dropped;
                } else {
                    m_notFull.wait(lock, [&] { return m_closed || m_queue.size() < m_capacity; });
                    if (m_closed) return false;
                }
            }
            m_queue.push_back(std::move(value));
            m_notEmpty.notify_one();
            return true;
        }

        bool Pop(T &out) {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_notEmpty.wait(lock, [&] { return m_closed || !m_queue.empty(); });
            if (m_queue.empty()) return false; // closed + drained
            out = std::move(m_queue.front());
            m_queue.pop_front();
            m_notFull.notify_one();
            return true;
        }

        bool TryPop(T &out) {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_queue.empty()) return false;
            out = std::move(m_queue.front());
            m_queue.pop_front();
            m_notFull.notify_one();
            return true;
        }

        void Clear() {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_queue.clear();
            m_notFull.notify_all();
        }

        void Close() {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_closed = true;
            }
            m_notEmpty.notify_all();
            m_notFull.notify_all();
        }

        std::size_t Size() const {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_queue.size();
        }
        std::size_t Dropped() const {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_dropped;
        }
        bool Closed() const {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_closed;
        }

    private:
        mutable std::mutex m_mutex;
        std::condition_variable m_notEmpty;
        std::condition_variable m_notFull;
        std::deque<T> m_queue;
        std::size_t m_capacity;
        bool m_dropOldest;
        bool m_closed = false;
        std::size_t m_dropped = 0;
    };

} // namespace util
