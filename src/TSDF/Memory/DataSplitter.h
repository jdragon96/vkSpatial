#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace Engine::Core {
    class Context;
}

struct DataSplitterConfig {
    float baseResolution = 0.01f;

    int blockVoxels = 32;

    int maxPointPerFrame = 1 << 18;
};

class DataSplitter {
public:
    struct DividePointOutput {
        std::vector<uint32_t> baseIndex;
        std::vector<uint32_t> detailIndex;

        uint32_t denseBlockCount = 0;

        uint32_t detailSlotEstimate = 0;

        uint32_t blockInsertFailureCount = 0;
        uint32_t cellInsertFailureCount = 0;

        void Clear() {
            baseIndex.clear();
            detailIndex.clear();
            denseBlockCount = 0;
            detailSlotEstimate = 0;
            blockInsertFailureCount = 0;
            cellInsertFailureCount = 0;
        }
    };

    virtual ~DataSplitter() = default;

    virtual void Build(Engine::Core::Context &context, const DataSplitterConfig &config) = 0;

    virtual void Reset() = 0;

    // One frame. Normals must be unit length and the same count as points.
    virtual void DividePoint(const std::vector<Eigen::Vector3f> &points,
                             const std::vector<Eigen::Vector3f> &normals,
                             DividePointOutput &output) = 0;

    virtual const char *Name() const = 0;
};

std::unique_ptr<DataSplitter> MakeDataSplitter(const std::string &name);

std::vector<std::string> DataSplitterNames();
