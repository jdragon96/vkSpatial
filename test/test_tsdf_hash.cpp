#include "Engine/Core/ComputePipeline.h"
#include "Engine/Core/Context.h"

#include <gtest/gtest.h>

TEST(ComputePipelineDefines, DefinitionsReachTheCompilerAndKeyTheCache) {
    Engine::Core::Context context;

    Engine::Core::ComputePipeline withDefinition(context);
    withDefinition.Define("PROBE_LOCAL_SIZE", "8").Build("define_probe.comp.glsl");
    EXPECT_EQ(withDefinition.GetLocalSize().width, 8u);

    // Same file, no definition. A cache keyed on the path alone would return the module built
    // above and report 8 -- that is the failure this test exists to catch.
    Engine::Core::ComputePipeline withoutDefinition(context);
    withoutDefinition.Build("define_probe.comp.glsl");
    EXPECT_EQ(withoutDefinition.GetLocalSize().width, 1u);

    // And the definition still applies after the undefined build populated the cache.
    Engine::Core::ComputePipeline again(context);
    again.Define("PROBE_LOCAL_SIZE", "4").Build("define_probe.comp.glsl");
    EXPECT_EQ(again.GetLocalSize().width, 4u);
}
