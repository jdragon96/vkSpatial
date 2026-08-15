#include "TSDF/Memory/DataSplitter.h"

#include <stdexcept>

#include "Engine/Compute/CommandBatch.h"
#include "Engine/Core/Context.h"

#include "TSDF/Memory/NonFiltering/NonFiltering.h"
#include "TSDF/Memory/RegionClassifier/DenseRegionClassifier.h"

std::unique_ptr<DataSplitter> MakeDataSplitter(const std::string &name) {
    if (name == "dense") return std::make_unique<TSDF::DenseRegionStrategy>();
    if (name == "none") return std::make_unique<NonFilteringStrategy>();
    return nullptr;
}

std::vector<std::string> DataSplitterNames() {
    return {"dense", "none"};
}
