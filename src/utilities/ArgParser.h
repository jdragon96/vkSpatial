#pragma once

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Command-line parsing for the example tools. Two layers over the same convention (options are
// "--key value" pairs, flags are the bare presence of "--key", first match wins):
//   * stateless primitives  ArgValue / HasFlag / ArgString / ArgInt / ArgFloat  -- one-off lookups.
//   * ArgParser (fluent)     BuildArgParser(argc, argv).Must(...).Option(...) then .Value(...)  --
//                            the ergonomic front-end; declares required/optional keys (with an
//                            optional default value) and validates.
namespace util {

    // ---- stateless primitives ----

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

    // ---- ArgParser: fluent front-end ----

    // Build with BuildArgParser(argc, argv), then chain declarations:
    //   auto arg = util::BuildArgParser(argc, argv)
    //                  .Must("--dir", "usage: ... --dir <folder> ...")  // required; on miss prints
    //                  .Option("--voxel", 0.5)                          //   the message + fails
    //                  .Option("--out", "scan_out");                    // optional + default value
    //   if (!arg) return 2;                       // a Must was unsatisfied (message already printed)
    //   std::string dir = arg.Value("--dir");     // read: Value / ValueInt / ValueFloat / Has
    //   float voxel = arg.ValueFloat("--voxel");  //   no fallback -> the declared default (0.5)
    // A getter with an explicit fallback (Value(key, fb)) ignores the declared default -- use it for
    // defaults that are only known at runtime, e.g. arg.ValueFloat("--voxel", extent / 200.0f).
    // Declared defaults are stored as text and re-parsed on read, so the Option type and the read
    // type need not match (e.g. Option("--tile-hash", 1<<20) read via ValueFloat).
    class ArgParser {
    public:
        ArgParser(int argc, char **argv) : m_argc(argc), m_argv(argv) {}

        // Required "--key value": if the value is absent, print `errorMessage` to stderr and fail
        // the parse (Ok()/bool become false). Returns *this for chaining.
        ArgParser &Must(std::string_view key, std::string_view errorMessage) {
            m_declared.emplace_back(key);
            if (!ArgValue(m_argc, m_argv, key)) {
                std::fprintf(stderr, "%.*s\n", int(errorMessage.size()), errorMessage.data());
                m_ok = false;
            }
            return *this;
        }

        // Optional "--key". The 1-arg form records the key (no default); the 2-arg forms also store
        // a default value returned by the no-fallback getters when the key is absent.
        ArgParser &Option(std::string_view key) {
            m_declared.emplace_back(key);
            return *this;
        }
        ArgParser &Option(std::string_view key, std::string_view defaultValue) {
            return declare(key, std::string(defaultValue));
        }
        ArgParser &Option(std::string_view key, int defaultValue) {
            return declare(key, std::to_string(defaultValue));
        }
        ArgParser &Option(std::string_view key, double defaultValue) {
            return declare(key, std::to_string(defaultValue));
        }

        // Reads. The no-fallback form returns the declared default (or the type's zero) when the key
        // is absent; the fallback form uses the given value instead (for runtime-computed defaults).
        std::string Value(std::string_view key) const {
            if (const std::optional<std::string_view> v = ArgValue(m_argc, m_argv, key))
                return std::string(*v);
            const std::string *d = defaultFor(key);
            return d ? *d : std::string();
        }
        std::string Value(std::string_view key, std::string_view fallback) const {
            return ArgString(m_argc, m_argv, key, fallback);
        }
        int ValueInt(std::string_view key) const {
            if (const std::optional<std::string_view> v = ArgValue(m_argc, m_argv, key))
                return std::atoi(v->data());
            const std::string *d = defaultFor(key);
            return d ? std::atoi(d->c_str()) : 0;
        }
        int ValueInt(std::string_view key, int fallback) const {
            return ArgInt(m_argc, m_argv, key, fallback);
        }
        float ValueFloat(std::string_view key) const {
            if (const std::optional<std::string_view> v = ArgValue(m_argc, m_argv, key))
                return float(std::atof(v->data()));
            const std::string *d = defaultFor(key);
            return d ? float(std::atof(d->c_str())) : 0.0f;
        }
        float ValueFloat(std::string_view key, float fallback) const {
            return ArgFloat(m_argc, m_argv, key, fallback);
        }
        bool Has(std::string_view key) const { return HasFlag(m_argc, m_argv, key); }

        // True while every Must so far was satisfied.
        bool Ok() const { return m_ok; }
        explicit operator bool() const { return m_ok; }

        // Every key passed to Must/Option, in declaration order.
        const std::vector<std::string> &Declared() const { return m_declared; }

    private:
        ArgParser &declare(std::string_view key, std::string defaultValue) {
            m_declared.emplace_back(key);
            m_defaults[std::string(key)] = std::move(defaultValue);
            return *this;
        }
        const std::string *defaultFor(std::string_view key) const {
            const auto it = m_defaults.find(std::string(key));
            return it == m_defaults.end() ? nullptr : &it->second;
        }

        int m_argc;
        char **m_argv;
        bool m_ok = true;
        std::vector<std::string> m_declared;
        std::unordered_map<std::string, std::string> m_defaults;
    };

    inline ArgParser BuildArgParser(int argc, char **argv) { return ArgParser(argc, argv); }

} // namespace util
