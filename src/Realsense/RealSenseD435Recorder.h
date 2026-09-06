#pragma once

#include "Realsense/RealSenseTypes.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>


namespace Realsense {

    class RealSenseD435Recorder : public Realsense::IDepthProvider {
    public:
        explicit RealSenseD435Recorder(std::string directory);

        RealSenseD435Recorder(std::unique_ptr<Realsense::IDepthProvider> device,
                              std::string directory);

        const Realsense::CameraIntrinsics &Intrinsics() const override {
            return m_device ? m_device->Intrinsics() : m_intrinsics;
        }

        bool Grab(Realsense::DepthFrame &out) override;

        void Close() override {
            if (m_device) m_device->Close();
        }

        int FrameCount() const { return m_device ? m_writtenFrameCount : int(m_framePaths.size()); }

    private:
        bool record(Realsense::DepthFrame &out);

        std::unique_ptr<Realsense::IDepthProvider> m_device;

        std::string m_directory;

        int m_writtenFrameCount = 0;

        Realsense::CameraIntrinsics m_intrinsics;

        std::vector<std::string> m_framePaths;

        std::vector<std::uint16_t> m_frame;

        int m_nextFrame = 0;
    };

} // namespace Realsense
