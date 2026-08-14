#pragma once

#include "TSDF/Volume.h"

namespace TSDF {

    // Bytes each hash slot costs on the device: the 24-byte entry plus the parallel int32
    // first-fill stamp. Memory comparisons are only meaningful if every strategy charges the
    // same rate.
    inline constexpr uint64_t kBytesPerHashSlot =
            sizeof(TSDF::AdvDirEntry) + sizeof(int32_t);

    /// *********************************************
    /// Memory axis
    /// *********************************************

    // How the volume is organised in space: one fixed window, lazily created tiles, or
    // density-adaptive submaps. This axis is pure C++ -- tiling changes which buffers get bound
    // and what goes in the origin push constant, never the kernel source. That is why it is a
    // virtual-function seam while the integrate/extract axes are shader seams.
    //
    // The operations mirror Volume one for one; ComposedVolume forwards straight through. They
    // diverge once the integrate/extract axes land and this interface keeps only storage.
    class MemoryStrategy {
    public:
        virtual ~MemoryStrategy() = default;

        MemoryStrategy(const MemoryStrategy &) = delete;
        MemoryStrategy &operator=(const MemoryStrategy &) = delete;

        virtual void Build(Engine::Core::Context &context, const VolumeParams &params) = 0;
        virtual void Reset() = 0;
        virtual void Configure(const IntegrationOptions &options) = 0;

        virtual void Record(const std::vector<Eigen::Vector3f> &points,
                            const std::vector<Eigen::Vector3f> &normals,
                            const Eigen::Vector3f &cameraPosition,
                            Engine::Compute::CommandBatch &batch) = 0;

        virtual void Download(std::vector<TSDF::AdvancedEntry> &out) const = 0;

        // Mirrors Volume::Extract; see its doc comment.
        virtual Engine::Core::OrientedPointCloud Extract(bool merge = true) const = 0;

        virtual VolumeStats Stats() const = 0;
        virtual const char *Name() const = 0;

        // Device this strategy was built on; null before Build. ComposedVolume forwards it so
        // the base Volume::Integrate can open a CommandBatch. Public here (protected on Volume)
        // because the composing class is not a subclass and still has to read it.
        virtual Engine::Core::Context *Device() const = 0;

    protected:
        MemoryStrategy() = default;
    };

} // namespace TSDF
