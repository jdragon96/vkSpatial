#pragma once

#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>

// Functional command-line reader over (argc, argv): pure lookups, no mutation, no global state,
// shared by the example tools. Convention: options are "--key value" pairs, flags are the bare
// presence of "--key". First match wins. Compose the typed getters over ArgValue.
namespace util {

    // The token that follows `key`, if the args contain "... key value ...". The returned view
    // aliases a null-terminated argv token, so numeric getters can hand .data() to atoi/atof.
    inline std::optional<std::string_view> ArgValue(int argc, char **argv, std::string_view key) {
        for (int i = 1; i + 1 < argc; ++i)
            if (key == argv[i]) return std::string_view(argv[i + 1]);
        return std::nullopt;
    }

    // Whether `key` appears anywhere as a bare flag ("--key").
    inline bool HasFlag(int argc, char **argv, std::string_view key) {
        for (int i = 1; i < argc; ++i)
            if (key == argv[i]) return true;
        return false;
    }

    // Value of "--key", or `fallback` when absent.
    inline std::string ArgString(int argc, char **argv, std::string_view key,
                                 std::string_view fallback) {
        return std::string(ArgValue(argc, argv, key).value_or(fallback));
    }
    inline int ArgInt(int argc, char **argv, std::string_view key, int fallback) {
        const std::optional<std::string_view> v = ArgValue(argc, argv, key);
        return v ? std::atoi(v->data()) : fallback;
    }
    inline float ArgFloat(int argc, char **argv, std::string_view key, float fallback) {
        const std::optional<std::string_view> v = ArgValue(argc, argv, key);
        return v ? float(std::atof(v->data())) : fallback;
    }

} // namespace util
