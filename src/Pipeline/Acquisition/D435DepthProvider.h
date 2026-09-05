#pragma once

#include "Pipeline/Acquisition/DepthProvider.h"

#include "Realsense/RealSenseD435.h"

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Pipeline {

    class D435DepthProvider : public IDepthProvider {
    public:
        explicit D435DepthProvider(Realsense::D435StreamOptions options = {}) {
            m_camera.Open(options);
            const Realsense::D435Calibration &calibration = m_camera.Calibration();
            m_intrinsics.fx = calibration.fx;
            m_intrinsics.fy = calibration.fy;
            m_intrinsics.cx = calibration.cx;
            m_intrinsics.cy = calibration.cy;
            m_intrinsics.width = calibration.width;
            m_intrinsics.height = calibration.height;
            m_depthScale = calibration.depthScale;
        }

        const CameraIntrinsics &Intrinsics() const override { return m_intrinsics; }

        // Handed over as the sensor produced it. The GPU front end uploads Z16 directly, so
        // unpacking here would be a full-image host pass whose result the front end would only
        // quantise back. A consumer that needs metres asks for them with EnsureMetres.
        bool Grab(DepthFrame &out) override {
            Realsense::D435Frame frame;
            if (!m_camera.Grab(frame) || !frame.depthZ16) return false;

            out.depth.clear();
            out.rawZ16 = frame.depthZ16; // the driver's buffer, valid until the next Grab
            out.depthScale = m_depthScale;
            return true;
        }

        void Close() override { m_camera.Close(); }

        // What the device answered, for a caller that has to fill sigma_z's constants.
        const Realsense::D435Calibration &Calibration() const { return m_camera.Calibration(); }

    private:
        Realsense::RealSenseD435 m_camera;
        CameraIntrinsics m_intrinsics;
        float m_depthScale = 0.0f;
    };

} // namespace Pipeline
