#pragma once

#include "Realsense/RealSenseTypes.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Realsense {

    struct VisualPresetEntry {
        const char *name;
        int value; // rs2_rs400_visual_preset
    };

    const std::vector<VisualPresetEntry> &VisualPresets();
    std::vector<std::string> VisualPresetNames();

    struct D435StreamOptions {
        int width = 848;
        int height = 480;
        int fps = 30;
        bool enableInfrared = true;
        std::string visualPreset = "high-accuracy";
    };

    class RealSenseD435 : public Realsense::IDepthProvider {
    public:
        RealSenseD435();
        ~RealSenseD435() override;

        RealSenseD435(const RealSenseD435 &) = delete;
        RealSenseD435 &operator=(const RealSenseD435 &) = delete;

        void Open(const D435StreamOptions &options = {});

        void Close() override;

        bool IsOpen() const;

        const CameraIntrinsics &Intrinsics() const override { return m_intrinsics; }

        bool Grab(DepthFrame &out) override;

        ValidationScoreOptions MakeScoreOptions() const;

        NormalEstimationOptions MakeNormalOptions() const;

        const std::string &VisualPresetRefusal() const { return m_visualPresetRefusal; }

        PinholeIntrinsics MakeIntrinsics() const {
            return PinholeIntrinsics{m_intrinsics.fx, m_intrinsics.fy, m_intrinsics.cx,
                                     m_intrinsics.cy};
        }

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
        CameraIntrinsics m_intrinsics;
        std::string m_visualPresetRefusal;
        bool m_infraredEnabled = false;
    };
} // namespace Realsense
