#pragma once

#include "TSDF/Memory/DataSplitter.h"

class NonFilteringStrategy final : public DataSplitter {
public:
    void Build(Engine::Core::Context &, const DataSplitterConfig &) override {}

    void Reset() override {}

    void DividePoint(const std::vector<Eigen::Vector3f> &points,
                     const std::vector<Eigen::Vector3f> &normals,
                     DividePointOutput &output) override {
        output.Clear();
        if (points.size() != normals.size())
            throw std::runtime_error("DataSplitter::DividePoint: points/normals size mismatch");

        output.baseIndex.resize(points.size());
        for (uint32_t i = 0; i < uint32_t(points.size()); ++i) output.baseIndex[i] = i;
    }

    const char *Name() const override { return "none"; }
};