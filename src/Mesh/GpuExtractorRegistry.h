#pragma once

// Context-aware twin of ExtractorRegistry.h's CPU registry: every GPU extractor strategy needs an
// Engine::Core::Context& to build its compute pipeline (upload buffers, compile/dispatch a .comp
// kernel), so Create() takes one and the Factory signature threads it through. Otherwise identical
// in shape to ExtractorRegistry (Register/Create/Has/Default), so the two registries stay
// interchangeable at call sites that already know which one they're using.

#include "Engine/Core/Context.h"
#include "Mesh/IsoSurfaceExtractor.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace Mesh {

    class GpuExtractorRegistry {
    public:
        using Factory = std::function<std::unique_ptr<IsoSurfaceExtractor>(Engine::Core::Context &)>;

        void Register(const std::string &name, Factory factory) { m_factories[name] = std::move(factory); }

        std::unique_ptr<IsoSurfaceExtractor> Create(const std::string &name, Engine::Core::Context &context) const {
            const auto it = m_factories.find(name);
            return it == m_factories.end() ? nullptr : it->second(context);
        }

        bool Has(const std::string &name) const { return m_factories.count(name) != 0; }

        // Registers every GPU extractor strategy this codebase ships. "mc-gpu" is live now
        // (Task 3); "mc33-gpu" (Task 4) and "mtet-gpu" (Task 5) join this SAME function later,
        // mirroring how ExtractorRegistry::Default() accumulates its CPU strategies.
        static GpuExtractorRegistry Default();

    private:
        std::unordered_map<std::string, Factory> m_factories;
    };

} // namespace Mesh
