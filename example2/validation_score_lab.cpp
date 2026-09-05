// Validation-score lab — turn each confidence knob and watch what survives.
//
// Two modes. Without --sweep it opens a window: the frame is back-projected and every point is
// coloured by its confidence, so dragging a slider shows WHERE the score moves, not just by how
// much. Confidence is a spatial thing -- it collapses at object boundaries, at grazing incidence
// and at range -- and a single number per frame hides exactly that.
//
// The score kernel multiplies four vetoes, and every one of them has a threshold that has to be
// picked. Picking one by staring at a single frame does not work: the terms interact (a pixel the
// range fade already zeroed is invisible to the infrared gate), and the number that matters --
// how much surface you keep -- only shows up over a whole sequence.
//
// So this sweeps ONE knob at a time across a range, holding the rest at their defaults, and prints
// what fraction of the frame survives at three score thresholds plus where the losses went. One
// run produces the whole table, because two runs taken separately differ by more than the knob.
//
// With --sweep it stays headless and prints a table instead, for picking a value rather than
// looking at one: it walks ONE knob at a time across a range, holding the rest at their defaults.
//
// Usage:
//   validation_score_lab --replay <dir> [--loop]                 interactive
//   validation_score_lab --live [--preset high-accuracy]         interactive, needs a camera
//   validation_score_lab --replay <dir> --sweep k,subpixel,near,far [--frames N]   headless table
//
// --replay reads the float32 recordings in capture/ and re-quantises them to Z16, which is what
// they were before Pipeline's front end unpacked them. --live needs a camera and librealsense2.

#include "ImGuiPass.h"
#include "PointCloudPass.h"

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"
#include "Engine/Render/Application.h"
#include "Engine/Render/Camera.h"
#include "Engine/Render/GlfwWindow.h"
#include "Engine/Render/Scene.h"
#include "Pipeline/Acquisition/DepthRecording.h"
#include "Realsense/RealSenseD435.h"
#include "Realsense/Algorithm/NormalEstimation.h"
#include "Realsense/RealSensePipeline.h"
#include "utilities/ArgParser.h"

#include "imgui.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    using Realsense::DownSampleOptions;
    using Realsense::NormalEstimationOptions;
    using Realsense::ValidationScoreCounters;
    using Realsense::ValidationScoreOptions;

    // What one configuration did to the whole sequence.
    struct SweepResult {
        double meanScore = 0.0;
        double fractionAboveZero = 0.0; // any confidence at all
        double fractionAboveHalf = 0.0; // usable for weighted fusion
        double fractionAboveNine = 0.0; // trusted outright
        ValidationScoreCounters counters{};
    };

    // Z16 is what the sensor produces and what the kernel samples; the recordings in capture/ hold
    // the float metres Pipeline's front end unpacked them into. Re-quantising is lossless in
    // practice because the original quantum WAS one Z16 unit -- but only if depthScale matches the
    // scale the recording was made with, which is why it is a parameter and not a constant.
    std::vector<std::uint16_t> QuantiseToZ16(const std::vector<float> &metres, float depthScale) {
        std::vector<std::uint16_t> out(metres.size(), 0);
        const float inverseScale = 1.0f / depthScale;
        for (std::size_t i = 0; i < metres.size(); ++i) {
            if (!(metres[i] > 0.0f)) continue;
            const float units = std::round(metres[i] * inverseScale);
            // Above 65535 the sensor could not have reported it either; clamping would invent a
            // surface at 65.5 m, so the pixel becomes "no measurement" instead.
            out[i] = units <= 65535.0f ? std::uint16_t(units) : std::uint16_t(0);
        }
        return out;
    }

    SweepResult ScoreSequence(Realsense::RealSensePipeline &pipeline, Engine::Core::Context &context,
                              const std::vector<std::vector<std::uint16_t>> &frames,
                              const ValidationScoreOptions &options) {
        SweepResult result;
        double scoreSum = 0.0;
        std::uint64_t aboveZero = 0, aboveHalf = 0, aboveNine = 0, total = 0;

        for (const std::vector<std::uint16_t> &frame: frames) {
            {
                Engine::Compute::CommandBatch batch(context);
                pipeline.RecordScore(batch, frame.data(), options);
                batch.Submit();
            }
            const std::vector<float> scores = pipeline.DownloadScores();
            for (const float score: scores) {
                scoreSum += score;
                if (score > 0.0f) ++aboveZero;
                if (score >= 0.5f) ++aboveHalf;
                if (score >= 0.9f) ++aboveNine;
            }
            total += scores.size();

            const ValidationScoreCounters counters = pipeline.DownloadCounters();
            result.counters.scoredPixels += counters.scoredPixels;
            result.counters.zeroedByNoMeasurement += counters.zeroedByNoMeasurement;
            result.counters.zeroedByRange += counters.zeroedByRange;
            result.counters.zeroedByInfrared += counters.zeroedByInfrared;
            result.counters.zeroedByNeighbourSupport += counters.zeroedByNeighbourSupport;
        }

        if (total > 0) {
            result.meanScore = scoreSum / double(total);
            result.fractionAboveZero = double(aboveZero) / double(total);
            result.fractionAboveHalf = double(aboveHalf) / double(total);
            result.fractionAboveNine = double(aboveNine) / double(total);
        }
        return result;
    }

    void PrintHeader(const char *knob) {
        std::printf("\n%-14s %9s %9s %9s %9s   %9s %9s %9s\n", knob, "mean", ">0", ">=0.5", ">=0.9",
                    "noMeasure", "range", "neighbour");
        std::printf("%s\n", std::string(88, '-').c_str());
    }

    void PrintRow(const std::string &label, const SweepResult &result, std::uint64_t pixels) {
        const auto share = [pixels](std::uint32_t count) {
            return pixels > 0 ? 100.0 * double(count) / double(pixels) : 0.0;
        };
        std::printf("%-14s %9.4f %8.2f%% %8.2f%% %8.2f%%   %8.2f%% %8.2f%% %8.2f%%\n", label.c_str(),
                    result.meanScore, 100.0 * result.fractionAboveZero,
                    100.0 * result.fractionAboveHalf, 100.0 * result.fractionAboveNine,
                    share(result.counters.zeroedByNoMeasurement), share(result.counters.zeroedByRange),
                    share(result.counters.zeroedByNeighbourSupport));
    }

    std::string Format(float value) {
        char buffer[32];
        std::snprintf(buffer, sizeof buffer, "%g", double(value));
        return buffer;
    }

    std::vector<std::string> Split(const std::string &text, char separator) {
        std::vector<std::string> parts;
        std::string current;
        for (const char character: text + separator) {
            if (character == separator) {
                if (!current.empty()) parts.push_back(current);
                current.clear();
            } else {
                current += character;
            }
        }
        return parts;
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Interactive viewer
    ///////////////////////////////////////////////////////////////////////////////////////////////

    constexpr int kSetSurface = 0;
    constexpr int kSetRejected = 1;
    constexpr int kSetNormals = 2;

    // The sensor frame is +y down, +z forward; the view frame is +y up, -z forward.
    struct Vec3 {
        float x, y, z;
    };
    Vec3 SensorToView(const Vec3 &v) { return Vec3{v.x, -v.y, -v.z}; }

    struct Color {
        std::uint8_t r, g, b;
    };

    // Blue -> cyan -> green -> yellow -> red. Deliberately NOT a greyscale ramp: confidence is
    // read here as "where does this collapse", and hue separates 0.4 from 0.6 far better than
    // lightness does.
    Color ScoreRamp(float score) {
        const float t = std::clamp(score, 0.0f, 1.0f) * 4.0f;
        const int band = std::min(3, int(t));
        const float f = t - float(band);
        const float table[5][3] = {{0.15f, 0.20f, 0.80f}, {0.10f, 0.75f, 0.85f}, {0.15f, 0.80f, 0.25f}, {0.95f, 0.85f, 0.15f}, {0.90f, 0.15f, 0.10f}};
        const auto channel = [&](int c) {
            return std::uint8_t(255.0f * (table[band][c] + f * (table[band + 1][c] - table[band][c])));
        };
        return Color{channel(0), channel(1), channel(2)};
    }

    enum class EColorMode { Score,
                            Threshold,
                            Depth,
                            Normal };

    // The standard normal-map mapping, and the standard way to READ one: a flat surface must come
    // out a single flat colour. Noise shows as speckle and a smeared boundary as a gradient where
    // the geometry has a crease, both of which a score ramp hides completely.
    Color NormalRamp(const Vec3 &normal) {
        const auto channel = [](float component) {
            return std::uint8_t(255.0f * std::clamp(component * 0.5f + 0.5f, 0.0f, 1.0f));
        };
        return Color{channel(normal.x), channel(normal.y), channel(normal.z)};
    }

    struct ViewerState {
        bool streaming = true;
        bool streamEnded = false;
        int frameIndex = 0;

        ValidationScoreOptions options;
        NormalEstimationOptions normalOptions;
        DownSampleOptions downSampleOptions;
        EColorMode colorMode = EColorMode::Score;
        // 0.9, not 0.5. Measured on capture/ against pixels labelled from the depth image as
        // sitting on a cliff -- the physical definition of a flying pixel: at 0.0 they are 1.06% of
        // the delivered cloud, at 0.7 they are 0.09%, and at 0.9 none survive. Thinning is what
        // makes the bar matter this much: it removes over 99% of a surface but keeps a trail nearly
        // intact, since each trail point owns its own voxel.
        float validThreshold = 0.9f;
        // Off by default so the bar handed to the GPU IS the display bar. With it on the GPU bar is
        // 0, every measured pixel is compacted, and the threshold above stops gating anything --
        // which is how a tuned setting can look like it does nothing.
        bool showRejected = false;
        float pointSize = 2.0f;
        float rampNear = 0.3f, rampFar = 3.0f;

        // Normals drawn as segments. A stride is not decoration: at 640x480 the frame emits a few
        // hundred thousand accepted points, and one segment each is an opaque mat rather than a
        // readable field. Every 8th is dense enough to show a surface and sparse enough to see it.
        bool showNormalLines = false;
        float normalLengthMillimetres = 20.0f;
        int normalStride = 8;

        double scoreMilliseconds = 0.0;
        ValidationScoreCounters counters{};
        std::size_t validCount = 0, measuredCount = 0;
        std::size_t downloadedPoints = 0; // what the compaction actually brought back
        double compactMilliseconds = 0.0;
        Realsense::NormalEstimationCounters normalCounters{};
        Realsense::DownSampleCounters downSampleCounters{};
        std::size_t normalSegmentCount = 0;
        float medianIncidenceDegrees = 0.0f;
        float histogram[32] = {};
    };

    // Colours the compacted cloud. Back-projection already happened on the GPU, inside
    // ValidationMask.RemainValidDepth, so there is nothing to unproject here and no dead pixel to skip --
    // the array holds survivors only.
    //
    // Two sets: at or above the display threshold, and below it. Separate point sets so the
    // rejects can be hidden outright, which is the fastest way to see what a knob just cost.
    void BuildPointSets(const std::vector<Eigen::Vector3f> &points, const std::vector<float> &scores,
                        const std::vector<Eigen::Vector3f> &normals, const ViewerState &state,
                        std::vector<PointVertex> &accepted, std::vector<PointVertex> &rejected) {
        accepted.clear();
        rejected.clear();
        const float span = std::max(1e-6f, state.rampFar - state.rampNear);

        for (std::size_t i = 0; i < points.size(); ++i) {
            const Eigen::Vector3f &p = points[i];
            const Vec3 view = SensorToView(Vec3{p.x(), p.y(), p.z()});
            const float score = scores[i];

            Color color;
            switch (state.colorMode) {
                case EColorMode::Score: color = ScoreRamp(score); break;
                case EColorMode::Threshold:
                    color = score >= state.validThreshold ? Color{60, 200, 90} : Color{200, 60, 50};
                    break;
                case EColorMode::Depth: color = ScoreRamp((p.z() - state.rampNear) / span); break;
                case EColorMode::Normal:
                    // Through the same sensor-to-view flip as the position, so the colour a surface
                    // shows agrees with the way it is facing on screen.
                    color = i < normals.size()
                                    ? NormalRamp(SensorToView(Vec3{normals[i].x(), normals[i].y(),
                                                                   normals[i].z()}))
                                    : Color{40, 40, 40};
                    break;
            }

            PointVertex vertex{};
            vertex.pos[0] = view.x;
            vertex.pos[1] = view.y;
            vertex.pos[2] = view.z;
            vertex.rgba[3] = 255;
            if (score >= state.validThreshold) {
                vertex.rgba[0] = color.r;
                vertex.rgba[1] = color.g;
                vertex.rgba[2] = color.b;
                accepted.push_back(vertex);
            } else {
                // Dimmed rather than recoloured: the hue still says what its score was, so a pixel
                // sitting just under the threshold reads differently from a dead one.
                vertex.rgba[0] = std::uint8_t(color.r / 4);
                vertex.rgba[1] = std::uint8_t(color.g / 4);
                vertex.rgba[2] = std::uint8_t(color.b / 4);
                rejected.push_back(vertex);
            }
        }
    }

    // One segment per sampled accepted point: base at the point, tip one normalLength along the
    // normal. Both ends carry the normal's colour, but the base is dimmed so the segment reads as
    // growing OUTWARD -- a flat colour would leave the direction ambiguous on a silhouette.
    //
    // The same sensor-to-view flip as the positions is applied to the direction. It is a diagonal
    // +-1 map, so transforming the tip and transforming (point + normal) agree exactly.
    void BuildNormalLines(const std::vector<Eigen::Vector3f> &points,
                          const std::vector<float> &scores,
                          const std::vector<Eigen::Vector3f> &normals, const ViewerState &state,
                          std::vector<PointVertex> &segments) {
        segments.clear();
        if (!state.showNormalLines || normals.empty()) return;

        const float length = state.normalLengthMillimetres * 0.001f;
        const int stride = std::max(1, state.normalStride);

        const auto emit = [&segments](const Vec3 &position, const Color &color) {
            PointVertex vertex{};
            vertex.pos[0] = position.x;
            vertex.pos[1] = position.y;
            vertex.pos[2] = position.z;
            vertex.rgba[0] = color.r;
            vertex.rgba[1] = color.g;
            vertex.rgba[2] = color.b;
            vertex.rgba[3] = 255;
            segments.push_back(vertex);
        };

        for (std::size_t i = 0; i < points.size() && i < normals.size(); i += std::size_t(stride)) {
            // Only for points that survived the display threshold: a segment on a rejected point
            // says nothing about the surface, and doubling the clutter hides the ones that do.
            if (scores[i] < state.validThreshold) continue;

            const Eigen::Vector3f &normal = normals[i];
            // A zero normal is a pixel the estimator refused. It cannot reach here while the pass
            // is enabled -- the kernel cancels those -- but the viewer also runs with normals off.
            if (normal.squaredNorm() < 1e-8f) continue;

            const Vec3 base = SensorToView(Vec3{points[i].x(), points[i].y(), points[i].z()});
            const Vec3 direction = SensorToView(Vec3{normal.x(), normal.y(), normal.z()});
            const Color color = NormalRamp(direction);

            emit(base, Color{std::uint8_t(color.r / 3), std::uint8_t(color.g / 3),
                             std::uint8_t(color.b / 3)});
            emit(Vec3{base.x + direction.x * length, base.y + direction.y * length,
                      base.z + direction.z * length},
                 color);
        }
    }

    // Median angle between a normal and the ray back to the camera, over the compacted cloud.
    //
    // "Do these normals look right" is not answerable by eye from the sensor's own viewpoint, where
    // every one of them points at you by construction. This is the number that answers it: a real
    // scene spans a wide range and reads 30-45 degrees here, while an estimator that had collapsed
    // to the view direction would read near zero however plausible the picture looked.
    float MedianIncidenceDegrees(const std::vector<Eigen::Vector3f> &points,
                                 const std::vector<Eigen::Vector3f> &normals) {
        std::vector<float> angles;
        angles.reserve(std::min(points.size(), normals.size()));
        for (std::size_t i = 0; i < points.size() && i < normals.size(); ++i) {
            if (points[i].squaredNorm() < 1e-12f || normals[i].squaredNorm() < 1e-8f) continue;
            const float cosine = normals[i].normalized().dot(-points[i].normalized());
            angles.push_back(std::acos(std::clamp(cosine, -1.0f, 1.0f)) * 180.0f / float(M_PI));
        }
        if (angles.empty()) return 0.0f;
        std::nth_element(angles.begin(), angles.begin() + angles.size() / 2, angles.end());
        return angles[angles.size() / 2];
    }

    void SummariseScores(const std::vector<float> &scores, ViewerState &state) {
        for (float &bin: state.histogram) bin = 0.0f;
        state.validCount = 0;
        state.measuredCount = scores.size();
        for (const float score: scores) {
            if (score >= state.validThreshold) ++state.validCount;
            const int bin = std::min(31, int(std::clamp(score, 0.0f, 1.0f) * 31.999f));
            state.histogram[bin] += 1.0f;
        }
    }

    int RunViewer(const std::string &replayDirectory, bool live, const std::string &presetName,
                  bool loopReplay) {
        // The source is opened before the window: a missing camera should print one clear line
        // rather than flash an empty window first.
        Realsense::RealSenseD435 camera;
        std::unique_ptr<Pipeline::RecordedDepthProvider> recording;
        ValidationScoreOptions baseOptions;
        NormalEstimationOptions baseNormalOptions;
        int width = 0, height = 0;
        float fx = 0.0f, fy = 0.0f, cx = 0.0f, cy = 0.0f;

        if (live) {
            Realsense::D435StreamOptions streamOptions;
            streamOptions.visualPreset = presetName;
            streamOptions.enableInfrared = false; // the viewer scores depth only
            camera.Open(streamOptions);
            const Realsense::D435Calibration &calibration = camera.Calibration();
            width = calibration.width;
            height = calibration.height;
            fx = calibration.fx;
            fy = calibration.fy;
            cx = calibration.cx;
            cy = calibration.cy;
            baseOptions = camera.MakeScoreOptions();
            baseOptions.useInfrared = false;
            // The plane fit's radius is in pixels but its noise behaviour is set by the physical
            // patch it spans, so the camera picks it from its own focal length.
            baseNormalOptions = camera.MakeNormalOptions();
        } else {
            recording = std::make_unique<Pipeline::RecordedDepthProvider>(replayDirectory);
            const Pipeline::CameraIntrinsics intrinsics = recording->Intrinsics();
            width = intrinsics.width;
            height = intrinsics.height;
            fx = intrinsics.fx;
            fy = intrinsics.fy;
            cx = intrinsics.cx;
            cy = intrinsics.cy;
            baseOptions.focalLengthPixels = intrinsics.fx;
            baseOptions.baselineMeters = 0.05f; // the recording predates this module
            baseOptions.depthScale = 0.001f;
        }
        baseOptions.countRejections = true;

        std::printf("source    : %s\n", live ? "live device" : replayDirectory.c_str());
        std::printf("intrinsics: %dx%d  fx %.2f fy %.2f  cx %.2f cy %.2f\n", width, height, fx, fy,
                    cx, cy);
        std::printf("sigma_z   : subpixel %g px / f %g px / baseline %g m%s\n",
                    double(baseOptions.subpixelRms), double(baseOptions.focalLengthPixels),
                    double(baseOptions.baselineMeters), live ? "" : "  (baseline assumed)");

        Engine::Render::ApplicationDescriptor descriptor;
        descriptor.window = {1360, 860, "Validation Score Lab"};
        Engine::Render::Application app(descriptor);
        Engine::Core::Context &context = app.GetContext();

        Engine::Render::Scene scene;
        Engine::Render::Camera renderCamera;
        const VkExtent2D extent = app.GetSwapChain().Extent();
        renderCamera.SetPerspective(60.0f * 3.14159265f / 180.0f,
                                    float(extent.width) / float(extent.height), 0.02f, 60.0f);
        // The sensor sits at the origin, on the very edge of the geometry, so orbit a target
        // inside the cloud rather than the origin itself.
        //
        // The starting orientation is deliberately OFF the sensor's own axis. Every normal in a
        // single depth frame faces the camera -- that is what the orientation flip enforces, and it
        // is correct -- so viewed straight down the view axis the whole normal field points at you
        // and foreshortens into dots. From there a correct estimator is indistinguishable from a
        // broken one, which is exactly the wrong default for a tool whose job is telling them apart.
        const vkMath::Quat yaw(Eigen::AngleAxisf(0.55f, Eigen::Vector3f::UnitY()));
        const vkMath::Quat pitch(Eigen::AngleAxisf(-0.30f, Eigen::Vector3f::UnitX()));
        renderCamera.SetOrbit(vkMath::Vec3(0.0f, 0.0f, -1.5f), 2.5f, yaw * pitch);

        Engine::Render::RenderGraph graph;
        auto pointsOwned =
                std::make_unique<PointCloudPass>(context, app.GetSwapChain().Format(), VOXDBG_SHADER_DIR);
        PointCloudPass *points = pointsOwned.get();
        graph.AddPass(std::move(pointsOwned));
        auto &glfwWindow = static_cast<Engine::Render::GlfwWindow &>(app.GetWindow());
        auto imguiOwned = std::make_unique<ImGuiPass>(context, glfwWindow.Handle(),
                                                      app.GetSwapChain().Format(),
                                                      app.GetSwapChain().ImageCount());
        ImGuiPass *imgui = imguiOwned.get();
        graph.AddPass(std::move(imguiOwned));

        Realsense::RealSensePipeline pipeline(context, width, height);
        const Realsense::PinholeIntrinsics intrinsics{fx, fy, cx, cy};
        ViewerState state;
        state.options = baseOptions;
        state.normalOptions = baseNormalOptions;

        const std::size_t pixels = std::size_t(width) * height;
        std::vector<std::uint16_t> depthZ16(pixels, 0);
        std::vector<PointVertex> acceptedVertices, rejectedVertices, normalSegments;

        using Clock = std::chrono::steady_clock;

        // Separate from grabbing on purpose: while paused, moving a slider re-scores the frozen
        // frame, which is the only way to attribute a change to the knob rather than to the scene.
        std::vector<Eigen::Vector3f> compactPoints;
        std::vector<float> compactScores;
        std::vector<Eigen::Vector3f> compactNormals;

        // Scoring is the expensive half -- four terms and a 3x3 gather per pixel. Kept separate
        // from compaction on purpose: while paused, moving a score knob re-runs this on the frozen
        // frame, which is the only way to attribute a change to the knob rather than to the scene.
        auto runScore = [&] {
            ValidationScoreOptions options = state.options;
            try {
                Realsense::ValidationMask::ValidateOptions(options);
            } catch (const std::exception &) {
                return; // the panel reports it; keep the last good score on screen
            }
            const Clock::time_point started = Clock::now();
            {
                Engine::Compute::CommandBatch batch(context);
                pipeline.RecordScore(batch, depthZ16.data(), options);
                batch.Submit();
            }
            state.scoreMilliseconds =
                    std::chrono::duration<double, std::milli>(Clock::now() - started).count();
        };

        // Threshold + compaction, the cheap half. Dragging the threshold re-runs ONLY this -- the
        // whole reason the bar lives in ValidationMask.RemainValidDepth rather than in the score kernel.
        //
        // The bar handed to the GPU is not always the display bar: with "show rejected" on, the
        // viewer wants the pixels BELOW the bar too, so it compacts at 0 and splits on the CPU.
        // With it off, the GPU bar is the display bar and only survivors ever leave the device --
        // which is the production path, and the point count below makes the saving visible.
        auto runCompact = [&] {
            ValidationScoreOptions options = state.options;
            try {
                Realsense::ValidationMask::ValidateOptions(options);
            } catch (const std::exception &) {
                return;
            }
            const float gpuThreshold = state.showRejected ? 0.0f : state.validThreshold;
            const Clock::time_point started = Clock::now();
            {
                Engine::Compute::CommandBatch batch(context);
                pipeline.RecordExtract(batch, options, intrinsics, gpuThreshold,
                                       state.normalOptions, state.downSampleOptions);
                batch.Submit();
            }
            compactPoints = pipeline.DownloadValidPoints();
            compactScores = pipeline.DownloadValidScores();
            compactNormals = pipeline.DownloadValidNormals();
            state.compactMilliseconds =
                    std::chrono::duration<double, std::milli>(Clock::now() - started).count();
            state.downloadedPoints = compactPoints.size();
            state.counters = pipeline.DownloadCounters();
            state.normalCounters = pipeline.DownloadNormalCounters();
            state.downSampleCounters = pipeline.DownloadDownSampleCounters();
            SummariseScores(compactScores, state);
            state.medianIncidenceDegrees = MedianIncidenceDegrees(compactPoints, compactNormals);
        };

        auto uploadFrame = [&] {
            BuildPointSets(compactPoints, compactScores, compactNormals, state, acceptedVertices,
                           rejectedVertices);
            BuildNormalLines(compactPoints, compactScores, compactNormals, state, normalSegments);
            state.normalSegmentCount = normalSegments.size() / 2;
            // SetPointSet may reallocate a buffer the previous frame's command buffer is still
            // reading; one idle covers both uploads.
            vkDeviceWaitIdle(context.device);
            points->SetPointSet(kSetSurface, acceptedVertices);
            points->SetPointSet(kSetRejected, rejectedVertices);
            points->SetLineSet(kSetNormals, normalSegments);
            points->SetVisible(kSetRejected, state.showRejected);
            points->SetVisible(kSetNormals, state.showNormalLines);
        };

        points->SetPointSize(state.pointSize);

        imgui->SetUi([&] {
            ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(390.0f, 0.0f), ImGuiCond_Always);
            ImGui::Begin("Validation Score", nullptr,
                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoCollapse);

            ImGui::Text("%s", live ? "live device" : "replay");
            ImGui::SameLine();
            if (ImGui::Button(state.streaming ? "pause" : "resume"))
                state.streaming = !state.streaming;
            if (state.streamEnded) ImGui::TextColored(ImVec4(1, 0.7f, 0.2f, 1), "stream ended");
            if (!state.streaming)
                ImGui::TextColored(ImVec4(0.4f, 0.8f, 1, 1), "paused -- knobs re-score this frame");

            // Validation runs every frame rather than on edit: a slider dragged through an invalid
            // value must say so, not silently keep the last good picture.
            std::string invalid;
            try {
                Realsense::ValidationMask::ValidateOptions(state.options);
            } catch (const std::exception &error) {
                invalid = error.what();
            }
            if (!invalid.empty())
                ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "REFUSED: %s", invalid.c_str());

            ImGui::SeparatorText("sigma_z = s * z^2 / (f * B)");
            ImGui::SliderFloat("s subpixel px", &state.options.subpixelRms, 0.02f, 0.6f, "%.3f");
            ImGui::InputFloat("f focal px", &state.options.focalLengthPixels, 0.0f, 0.0f, "%.2f");
            ImGui::InputFloat("B baseline m", &state.options.baselineMeters, 0.0f, 0.0f, "%.4f");
            {
                // The number the two sliders above actually produce, at a representative range --
                // sigma_z is what every same-surface decision is measured in, and neither slider
                // means anything without it.
                const float z = 1.5f;
                const float denominator =
                        state.options.focalLengthPixels * state.options.baselineMeters;
                const float sigma = denominator > 0.0f
                                            ? state.options.subpixelRms * z * z / denominator
                                            : 0.0f;
                const float tau = sigma * state.options.sameSurfaceSigmaMultiplier;
                ImGui::TextDisabled("at 1.5 m: sigma_z %.2f mm, tau %.2f mm", 1000.0 * double(sigma),
                                    1000.0 * double(std::max(tau, state.options.depthScale)));
                // Below one depth quantum "same surface" degenerates into "same quantised depth",
                // which refuses every slanted surface. The kernel floors it; saying so here is what
                // stops the knob from looking broken when it has simply run out of meaning.
                if (tau < state.options.depthScale)
                    ImGui::TextColored(ImVec4(1, 0.7f, 0.2f, 1),
                                       "tau is below one depth quantum -- floored");
            }

            ImGui::SeparatorText("c_nb: tau = k * sigma_z");
            ImGui::SliderFloat("k sigma", &state.options.sameSurfaceSigmaMultiplier, 0.25f, 10.0f,
                               "%.2f");
            ImGui::TextDisabled("How far a neighbour may sit and still be");
            ImGui::TextDisabled("the same surface. 3 covers 99.7%% of the noise.");

            ImGui::SeparatorText("c_range");
            ImGui::SliderFloat("near start m", &state.options.nearFadeStart, 0.0f, 2.0f, "%.2f");
            ImGui::SliderFloat("near end m", &state.options.nearFadeEnd, 0.0f, 2.0f, "%.2f");
            ImGui::SliderFloat("far start m", &state.options.farFadeStart, 0.5f, 10.0f, "%.2f");
            ImGui::SliderFloat("far end m", &state.options.farFadeEnd, 0.5f, 12.0f, "%.2f");
            ImGui::TextDisabled("Fades, not steps -- a hard edge seams the surface.");

            ImGui::SeparatorText("depth");
            ImGui::InputFloat("scale m/unit", &state.options.depthScale, 0.0f, 0.0f, "%.5f");

            if (ImGui::Button("reset")) {
                state.options = baseOptions;
                state.normalOptions = baseNormalOptions;
            }

            ImGui::SeparatorText("Distribution");
            ImGui::PlotHistogram("##scores", state.histogram, 32, 0, "score histogram", 0.0f,
                                 FLT_MAX, ImVec2(370.0f, 70.0f));
            ImGui::SliderFloat("valid >=", &state.validThreshold, 0.0f, 1.0f, "%.2f");
            if (state.measuredCount > 0)
                ImGui::Text("valid  %zu / %zu compacted  (%.1f %%)", state.validCount,
                            state.measuredCount,
                            100.0 * double(state.validCount) / double(state.measuredCount));

            ImGui::SeparatorText("Counters");
            // Mutually exclusive and ordered by precedence, so unlike Pipeline's ValidationMask
            // these five partition the frame and DO sum to width*height.
            const std::uint32_t total = state.counters.scoredPixels +
                                        state.counters.zeroedByNoMeasurement +
                                        state.counters.zeroedByRange +
                                        state.counters.zeroedByInfrared +
                                        state.counters.zeroedByNeighbourSupport;
            ImGui::Text("scored          %u", state.counters.scoredPixels);
            ImGui::Text("no measurement  %u", state.counters.zeroedByNoMeasurement);
            ImGui::Text("zeroed by range %u", state.counters.zeroedByRange);
            ImGui::Text("zeroed by c_nb  %u", state.counters.zeroedByNeighbourSupport);
            ImGui::TextDisabled("partition, sums to %u of %zu pixels", total, pixels);

            ImGui::SeparatorText("Normals");
            // The estimator is a compile-time variant, so switching rebuilds a pipeline once and
            // then caches it -- the combo is not as expensive as it looks after the first pick.
            {
                static const std::vector<std::string> estimatorNames = Realsense::NormalEstimatorNames();
                int estimatorIndex = 0;
                for (std::size_t i = 0; i < estimatorNames.size(); ++i)
                    if (estimatorNames[i] == state.normalOptions.estimator) estimatorIndex = int(i);

                std::string items;
                for (const std::string &name: estimatorNames) items += name + '\0';
                if (ImGui::Combo("estimator", &estimatorIndex, items.c_str()))
                    state.normalOptions.estimator = estimatorNames[std::size_t(estimatorIndex)];
            }
            ImGui::Checkbox("estimate normals", &state.normalOptions.enabled);
            if (state.normalOptions.estimator == "planefit") {
                ImGui::SliderInt("fit radius px", &state.normalOptions.planeFitRadius, 1, 6);
                ImGui::SliderInt("min samples", &state.normalOptions.minimumPlaneFitSamples, 3, 40);
                const int window = 2 * state.normalOptions.planeFitRadius + 1;
                ImGui::TextDisabled("%dx%d window, %d of %d samples needed", window, window,
                                    state.normalOptions.minimumPlaneFitSamples, window * window);
            } else {
                ImGui::TextDisabled("difference stencils ignore the fit knobs");
            }
            // Two causes, never summed: out-of-domain is the estimator's window running off the
            // image and grows with the window, while no-support is the SCENE refusing the pixel.
            // Every normal faces the camera by construction, so this is never near 90 -- but it
            // being near ZERO would mean they collapsed onto the view ray, which is the one failure
            // a picture taken from the sensor's viewpoint cannot show.
            ImGui::Text("median incidence %.1f deg", double(state.medianIncidenceDegrees));
            ImGui::Text("lost, off image  %u", state.normalCounters.outOfDomain);
            ImGui::Text("lost, no support %u", state.normalCounters.noSupport);

            ImGui::Checkbox("draw normal lines", &state.showNormalLines);
            if (state.showNormalLines) {
                ImGui::SliderFloat("length mm", &state.normalLengthMillimetres, 2.0f, 100.0f, "%.0f");
                ImGui::SliderInt("draw every", &state.normalStride, 1, 64);
                ImGui::TextDisabled("%zu segments -- the base end is dimmed, so the",
                                    state.normalSegmentCount);
                ImGui::TextDisabled("bright end is the direction it points.");
            }

            ImGui::SeparatorText("DownSample");
            // What this can remove is set by the voxel against the sample spacing z/f: there is
            // nothing to thin unless z < detailVoxel * f, which is why it is off by default rather
            // than merely tunable.
            ImGui::Checkbox("thin to detail voxel", &state.downSampleOptions.enabled);
            if (state.downSampleOptions.enabled) {
                // Seeded rather than left at zero: enabling with a size of 0 is refused, and a
                // checkbox that throws the moment it is ticked is not a knob.
                float millimetres = state.downSampleOptions.detailVoxelMeters > 0.0f
                                            ? state.downSampleOptions.detailVoxelMeters * 1000.0f
                                            : 5.0f;
                ImGui::SliderFloat("detail voxel mm", &millimetres, 0.5f, 40.0f, "%.2f");
                state.downSampleOptions.detailVoxelMeters = millimetres * 0.001f;

                // The range this actually bites over, from the same z < detailVoxel * f.
                ImGui::TextDisabled("thins where z < %.2f m",
                                    double(state.downSampleOptions.detailVoxelMeters *
                                           state.options.focalLengthPixels));
                // The absolute saving is already on screen under Readback, as downloaded
                // points against the dense pixel count.
                // Both ceilings fail OPEN -- a pixel that could not be placed keeps its flag, so
                // these being non-zero means less thinning, never a hole.
                ImGui::TextDisabled("probe fail %u, out of range %u",
                                    state.downSampleCounters.insertFailures,
                                    state.downSampleCounters.outOfPackableRange);
            }

            ImGui::SeparatorText("Display");
            int mode = int(state.colorMode);
            if (ImGui::Combo("color", &mode, "score\0threshold\0depth\0normal\0"))
                state.colorMode = EColorMode(mode);
            ImGui::Checkbox("show rejected (dim)", &state.showRejected);
            if (ImGui::InputFloat("point size", &state.pointSize, 0.0f, 0.0f, "%.2f"))
                points->SetPointSize(std::clamp(state.pointSize, 1.0f, 16.0f));
            if (state.colorMode == EColorMode::Depth) {
                ImGui::InputFloat("ramp near", &state.rampNear, 0.0f, 0.0f, "%.3f");
                ImGui::InputFloat("ramp far", &state.rampFar, 0.0f, 0.0f, "%.3f");
            }

            ImGui::SeparatorText("Readback");
            // What the GPU compaction actually saved. With "show rejected" on the bar handed to
            // the GPU is 0, so everything measured comes back and the saving is only the pixels
            // the sensor never reported; turn it off and the bar is the display bar, which is the
            // production path -- only survivors ever leave the device.
            const std::size_t densePixels = pixels;
            ImGui::Text("downloaded      %zu / %zu px", state.downloadedPoints, densePixels);
            ImGui::Text("                %.2f MB of %.2f MB",
                        double(state.downloadedPoints) * 20.0 / (1024.0 * 1024.0),
                        double(densePixels) * 20.0 / (1024.0 * 1024.0));
            if (state.showRejected)
                ImGui::TextDisabled("GPU bar is 0 -- rejects are wanted for display");
            else
                ImGui::TextDisabled("GPU bar is the display bar -- production path");

            ImGui::SeparatorText("Frame");
            ImGui::Text("frame           %d", state.frameIndex);
            ImGui::Text("score (GPU)     %.2f ms", state.scoreMilliseconds);
            ImGui::Text("compact+normal  %.2f ms", state.compactMilliseconds);
            ImGui::Text("display         %.1f fps", double(ImGui::GetIO().Framerate));
            ImGui::End();
        });

        app.GetView().SetScene(&scene);
        app.GetView().SetCamera(&renderCamera);
        app.GetView().SetRenderGraph(&graph);

        // ImGui owns the cursor whenever it is over a panel, so a drag meant for a slider must not
        // also spin the camera behind it.
        Engine::Render::MouseListenerGroup mouse(app.GetWindow().Mouse());
        mouse.Add(Engine::Render::MouseEventType::Drag, [&](Engine::Render::MouseEvent &e) {
            if (ImGui::GetIO().WantCaptureMouse) return;
            if (e.button != Engine::Render::MouseButton::Left) return;
            const VkExtent2D size = app.GetWindow().FramebufferSize();
            if (size.width == 0 || size.height == 0) return;
            if (!renderCamera.IsTrackballDragging()) {
                renderCamera.BeginTrackballDrag(e.x, e.y, int(size.width), int(size.height));
                return;
            }
            renderCamera.DragTrackball(e.x, e.y, int(size.width), int(size.height));
            e.handled = true;
        });
        mouse.Add(Engine::Render::MouseEventType::ButtonUp, [&](Engine::Render::MouseEvent &e) {
            if (e.button == Engine::Render::MouseButton::Left) renderCamera.EndTrackballDrag();
        });
        mouse.Add(Engine::Render::MouseEventType::Scroll, [&](Engine::Render::MouseEvent &e) {
            if (ImGui::GetIO().WantCaptureMouse) return;
            renderCamera.SetDistance(std::clamp(
                    renderCamera.GetDistance() * std::exp(float(-e.scrollY) * 0.08f), 0.05f, 40.0f));
            e.handled = true;
        });
        Engine::Render::KeyListenerGroup keys(app.GetWindow().Keys());
        keys.Add(Engine::Render::KeyEventType::Press, [&](Engine::Render::KeyEvent &e) {
            if (e.keyCode == Engine::Render::KeyCode::Escape) app.GetWindow().RequestClose();
            if (e.keyCode == Engine::Render::KeyCode::Space) state.streaming = !state.streaming;
        });

        // Re-scoring is driven by CHANGE, not by the clock: a paused frame with untouched knobs
        // costs nothing, and a dragged slider re-scores exactly once per distinct value.
        ValidationScoreOptions appliedOptions = state.options;
        EColorMode appliedColorMode = state.colorMode;
        float appliedThreshold = state.validThreshold;
        bool appliedShowRejected = state.showRejected;
        NormalEstimationOptions appliedNormalOptions = state.normalOptions;
        DownSampleOptions appliedDownSampleOptions = state.downSampleOptions;
        bool appliedShowNormalLines = state.showNormalLines;
        float appliedNormalLength = state.normalLengthMillimetres;
        int appliedNormalStride = state.normalStride;
        bool haveFrame = false;
        Pipeline::DepthFrame depthFrame;
        Realsense::D435Frame liveFrame;

        while (!app.GetWindow().ShouldClose()) {
            app.GetWindow().PollEvents();
            const VkExtent2D size = app.GetWindow().FramebufferSize();
            if (size.width == 0 || size.height == 0) continue;

            bool needsScore = false;
            if (state.streaming && !state.streamEnded) {
                bool got = false;
                if (live) {
                    got = camera.Grab(liveFrame);
                    if (got) std::memcpy(depthZ16.data(), liveFrame.depthZ16,
                                         pixels * sizeof(std::uint16_t));
                } else {
                    got = recording->Grab(depthFrame);
                    if (got) depthZ16 = QuantiseToZ16(depthFrame.depth, state.options.depthScale);
                }
                if (!got) {
                    // A recording runs out; a device does not. Looping keeps the window useful
                    // instead of freezing on the last frame.
                    if (!live && loopReplay) {
                        recording = std::make_unique<Pipeline::RecordedDepthProvider>(replayDirectory);
                        state.frameIndex = 0;
                        continue;
                    }
                    state.streamEnded = true;
                } else {
                    ++state.frameIndex;
                    haveFrame = true;
                    needsScore = true;
                }
            }

            // Score knobs and the threshold are tracked apart because they cost different
            // things: a score knob re-runs the four-term product, the threshold re-runs one pass
            // over the mask. Conflating them would make every slider as expensive as the worst.
            const bool scoreOptionsChanged =
                    std::memcmp(&appliedOptions, &state.options, sizeof(ValidationScoreOptions)) != 0;
            if (haveFrame && scoreOptionsChanged) {
                appliedOptions = state.options;
                needsScore = true;
            }
            if (needsScore) runScore();

            // The GPU bar only moves when the threshold does, or when "show rejected" flips it
            // between the display bar and 0.
            // The normal pass lives INSIDE the compaction chain, so its knobs re-run that and not
            // the four-term score. NormalEstimationOptions holds a std::string, so this compares
            // field by field rather than with memcmp.
            const bool normalOptionsChanged =
                    appliedNormalOptions.estimator != state.normalOptions.estimator ||
                    appliedNormalOptions.planeFitRadius != state.normalOptions.planeFitRadius ||
                    appliedNormalOptions.minimumPlaneFitSamples !=
                            state.normalOptions.minimumPlaneFitSamples ||
                    appliedNormalOptions.enabled != state.normalOptions.enabled ||
                    appliedDownSampleOptions.enabled != state.downSampleOptions.enabled ||
                    appliedDownSampleOptions.detailVoxelMeters !=
                            state.downSampleOptions.detailVoxelMeters;
            const bool compactionChanged = appliedThreshold != state.validThreshold ||
                                           appliedShowRejected != state.showRejected ||
                                           normalOptionsChanged;
            const bool needsCompaction = needsScore || compactionChanged;
            if (haveFrame && needsCompaction) {
                appliedThreshold = state.validThreshold;
                appliedShowRejected = state.showRejected;
                appliedNormalOptions = state.normalOptions;
                appliedDownSampleOptions = state.downSampleOptions;
                runCompact();
            }

            // Purely CPU-side display knobs: they rebuild the vertex arrays and re-upload, but
            // touch neither the score nor the compaction. Keeping them in their own bucket is why
            // dragging the segment length does not re-run a kernel.
            const bool displayChanged = appliedColorMode != state.colorMode ||
                                        appliedShowNormalLines != state.showNormalLines ||
                                        appliedNormalLength != state.normalLengthMillimetres ||
                                        appliedNormalStride != state.normalStride;
            if (haveFrame && (needsCompaction || displayChanged)) {
                appliedColorMode = state.colorMode;
                appliedShowNormalLines = state.showNormalLines;
                appliedNormalLength = state.normalLengthMillimetres;
                appliedNormalStride = state.normalStride;
                uploadFrame();
            }

            if (!app.GetRenderer().BeginFrame(size.width, size.height)) continue;
            app.GetRenderer().Render(app.GetView());
            app.GetRenderer().EndFrame();
        }

        vkDeviceWaitIdle(context.device);
        return 0;
    }

} // namespace

int main(int argc, char **argv) {
    try {
        auto arg = util::BuildArgParser(argc, argv)
                           .Option("--replay")
                           .Option("--frames", 0)
                           .Option("--preset", "high-accuracy")
                           .Option("--sweep", "k,subpixel,near,far")
                           .Option("--loop");
        const bool live = util::HasFlag(argc, argv, "--live");
        const bool infrared = util::HasFlag(argc, argv, "--infrared");
        const std::string replayDirectory = arg.Value("--replay");
        const int frameLimit = arg.ValueInt("--frames");

        if (live == !replayDirectory.empty())
            throw std::runtime_error("pass exactly one of --replay <dir> and --live");

        // No --sweep means the window. The sweep is for PICKING a value; the viewer is for seeing
        // what the value does, and the two want opposite things from the same code.
        if (!util::HasFlag(argc, argv, "--sweep"))
            return RunViewer(replayDirectory, live, arg.Value("--preset"),
                             util::HasFlag(argc, argv, "--loop"));

        std::vector<std::vector<std::uint16_t>> frames;
        ValidationScoreOptions baseOptions;
        int width = 0, height = 0;

        if (live) {
            Realsense::RealSenseD435 camera;
            Realsense::D435StreamOptions streamOptions;
            streamOptions.visualPreset = arg.Value("--preset");
            streamOptions.enableInfrared = infrared;
            camera.Open(streamOptions);

            const Realsense::D435Calibration &calibration = camera.Calibration();
            width = calibration.width;
            height = calibration.height;
            baseOptions = camera.MakeScoreOptions();
            std::printf("live %dx%d, preset %s, fx %.2f px, baseline %.4f m, depth scale %g m/unit\n",
                        width, height, streamOptions.visualPreset.c_str(), calibration.fx,
                        calibration.baselineMeters, double(calibration.depthScale));

            // The sweep replays the SAME frames for every configuration, so they are captured up
            // front. Comparing configurations against different frames would measure the scene.
            const int wanted = frameLimit > 0 ? frameLimit : 30;
            const std::size_t pixels = std::size_t(width) * height;
            Realsense::D435Frame frame;
            while (int(frames.size()) < wanted)
                if (camera.Grab(frame))
                    frames.emplace_back(frame.depthZ16, frame.depthZ16 + pixels);
        } else {
            Pipeline::RecordedDepthProvider provider(replayDirectory);
            const Pipeline::CameraIntrinsics intrinsics = provider.Intrinsics();
            width = intrinsics.width;
            height = intrinsics.height;

            // The recording carries no baseline -- it predates this module -- so the D435 datasheet
            // value stands in, and the printout says so. Every sweep below is relative, so a
            // baseline that is off by a few percent shifts the whole table, not its shape.
            baseOptions.focalLengthPixels = intrinsics.fx;
            baseOptions.baselineMeters = 0.05f;
            baseOptions.depthScale = 0.001f;

            Pipeline::DepthFrame depthFrame;
            while (provider.Grab(depthFrame)) {
                frames.push_back(QuantiseToZ16(depthFrame.depth, baseOptions.depthScale));
                if (frameLimit > 0 && int(frames.size()) >= frameLimit) break;
            }
            std::printf("replay %s: %d frames, %dx%d, fx %.2f px, baseline %.4f m (assumed D435)\n",
                        replayDirectory.c_str(), int(frames.size()), width, height,
                        baseOptions.focalLengthPixels, baseOptions.baselineMeters);
        }

        if (frames.empty()) throw std::runtime_error("no frames captured");
        baseOptions.countRejections = true; // the whole point of this tool
        baseOptions.useInfrared = false;    // the sweep replays depth only

        Engine::Core::Context context;
        Realsense::RealSensePipeline pipeline(context, width, height);
        const std::uint64_t pixels = std::uint64_t(width) * height * frames.size();

        std::printf("\nbaseline: k=%g, subpixel=%g px, near %g..%g m, far %g..%g m\n",
                    double(baseOptions.sameSurfaceSigmaMultiplier), double(baseOptions.subpixelRms),
                    double(baseOptions.nearFadeStart), double(baseOptions.nearFadeEnd),
                    double(baseOptions.farFadeStart), double(baseOptions.farFadeEnd));

        const std::vector<std::string> requested = Split(arg.Value("--sweep"), ',');
        const auto wants = [&requested](const char *name) {
            return std::find(requested.begin(), requested.end(), name) != requested.end();
        };

        // tau = k * sigma_z. This is the knob that decides what counts as one surface, so it moves
        // the neighbour term and nothing else.
        if (wants("k")) {
            PrintHeader("k (sigma x)");
            for (const float multiplier: {0.5f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 10.0f}) {
                ValidationScoreOptions options = baseOptions;
                options.sameSurfaceSigmaMultiplier = multiplier;
                PrintRow(Format(multiplier), ScoreSequence(pipeline, context, frames, options), pixels);
            }
        }

        // sigma_z scales linearly with the subpixel error, so this moves tau the same way k does --
        // but it is a claim about the SCENE's texture, not a preference. Sweeping both shows how
        // much of the tolerance is physics and how much is choice.
        if (wants("subpixel")) {
            PrintHeader("subpixel px");
            for (const float subpixel: {0.04f, 0.08f, 0.12f, 0.20f, 0.30f, 0.50f}) {
                ValidationScoreOptions options = baseOptions;
                options.subpixelRms = subpixel;
                PrintRow(Format(subpixel), ScoreSequence(pipeline, context, frames, options), pixels);
            }
        }

        if (wants("near")) {
            PrintHeader("near fade m");
            for (const float start: {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.6f}) {
                ValidationScoreOptions options = baseOptions;
                options.nearFadeStart = start;
                options.nearFadeEnd = start + 0.2f;
                if (options.nearFadeEnd > options.farFadeStart) continue;
                PrintRow(Format(start) + ".." + Format(options.nearFadeEnd),
                         ScoreSequence(pipeline, context, frames, options), pixels);
            }
        }

        if (wants("far")) {
            PrintHeader("far fade m");
            for (const float start: {1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 4.0f, 6.0f}) {
                ValidationScoreOptions options = baseOptions;
                options.farFadeStart = start;
                options.farFadeEnd = start + 1.0f;
                if (options.farFadeStart < options.nearFadeEnd) continue;
                PrintRow(Format(start) + ".." + Format(options.farFadeEnd),
                         ScoreSequence(pipeline, context, frames, options), pixels);
            }
        }

        std::printf("\nnoMeasure/range/neighbour are shares of ALL pixels and partition the frame\n"
                    "with the scored ones, so the four sum to 100%%.\n");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "validation_score_lab: %s\n", error.what());
        return 1;
    }
}
