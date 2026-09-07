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

        // Forget the last value. A reconfigured pipeline MUST do this: the map is the tracker's
        // alignment target, and a snapshot from the previous configuration carries the previous
        // voxel size, which is what trackers derive their correspondence distance from. Keeping it
        // makes the first frames of a new run align against the old map at the old scale, with no
        // error anywhere.
        void Clear() {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_value.reset();
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
