#pragma once

#include <string>

// Main-thread render coordinator: owns the window / swapchain / trackball camera / render loop and
// renders a swappable IRenderStrategy fed by the pipeline's latest model snapshot. macOS requires the
// window + present on the main thread, so Run() executes on the CALLING thread and blocks until the
// window closes — the pipeline's Reconstruction / ICP / Integration workers keep running on their own
// threads. Named "*Thread" for symmetry with the worker stages, but it is a main-thread loop, not a
// std::thread. Stops the pipeline on exit.
namespace Pipeline {

    class Pipeline;
    class IRenderStrategy;

    class RenderThread {
    public:
        struct Config {
            int width = 1280;
            int height = 800;
            std::string title = "Pipeline Viewer";
        };

        explicit RenderThread(Config cfg);

        // Runs the render loop on the calling (main) thread until the window closes. Blocks.
        void Run(Pipeline &pipe, IRenderStrategy &strategy);

    private:
        Config m_cfg;
    };

} // namespace Pipeline
