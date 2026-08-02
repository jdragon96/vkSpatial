#pragma once

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// Command-line parsing for the example tools. Two layers over the same convention (options are
// "--key value" pairs, flags are the bare presence of "--key", first match wins):
//   * stateless primitives  ArgValue / HasFlag / ArgString / ArgInt / ArgFloat  -- one-off lookups.
//   * ArgParser (fluent)     BuildArgParser(argc, argv).Must(...).Option(...) then .Value(...)  --
//                            the ergonomic front-end; declares required/optional keys and validates.
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
    //                  .Option("--voxel");                              //   the message + fails
    //   if (!arg) return 2;                       // a Must was unsatisfied (message already printed)
    //   std::string dir = arg.Value("--dir");     // read: Value / ValueInt / ValueFloat / Has
    // Reads delegate to the stateless primitives above; the parser only tracks declared keys and
    // whether every Must was satisfied.
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

        // Optional "--key": records the key (for Declared()/self-documentation); never fails.
        ArgParser &Option(std::string_view key) {
            m_declared.emplace_back(key);
            return *this;
        }

        // Reads (independent of Must/Option declarations).
        std::string Value(std::string_view key, std::string_view fallback = "") const {
            return ArgString(m_argc, m_argv, key, fallback);
        }
        int ValueInt(std::string_view key, int fallback = 0) const {
            return ArgInt(m_argc, m_argv, key, fallback);
        }
        float ValueFloat(std::string_view key, float fallback = 0.0f) const {
            return ArgFloat(m_argc, m_argv, key, fallback);
        }
        bool Has(std::string_view key) const { return HasFlag(m_argc, m_argv, key); }

        // True while every Must so far was satisfied.
        bool Ok() const { return m_ok; }
        explicit operator bool() const { return m_ok; }

        // Every key passed to Must/Option, in declaration order.
        const std::vector<std::string> &Declared() const { return m_declared; }

    private:
        int m_argc;
        char **m_argv;
        bool m_ok = true;
        std::vector<std::string> m_declared;
    };

    inline ArgParser BuildArgParser(int argc, char **argv) { return ArgParser(argc, argv); }

} // namespace util
