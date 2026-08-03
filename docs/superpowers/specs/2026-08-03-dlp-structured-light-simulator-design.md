# DLP Structured-Light Simulator — Step 1: Interactive Projection Viewer (Design)

**Date:** 2026-08-03 · **Status:** approved (design), spec for implementation

## Goal
Port the Blender DLP addon (`samples/DLP Simulator.py`) into the current `Engine::Render`
engine. Overarching aim: **generate varied structured-light pattern data**. This spec covers
**Step 1 only**: load a mesh and project structured-light patterns onto it in a **real-time
interactive window** (orbit camera, live pattern switching), with **projector occlusion
(shadows)** and **all Blender pattern types**.

## What the Blender addon does (reference)
A virtual DLP projector + main camera at a baseline project a fringe pattern onto a mesh; the
per-fragment world position is transformed by the projector's inverse matrix into projector UV,
so surface depth deforms the fringe (the structured-light signal). Patterns are pure-NumPy
images: phase-shift, gray-code, binary, De Bruijn/Hamming/self-equalizing color stripes,
RGB-multiplexed 3-step phase, plus optional RGB crosstalk pre-compensation.

## Chosen approach — Projective texture + projector shadow-map
`example2/ShadowMap.cpp` is a near-exact template: it renders an offscreen depth map from a
light's viewpoint and, in the scene pass, transforms world positions into that light's
clip-space. A DLP projector is the same math, except the scene pass samples a **pattern
texture** (instead of only comparing depth) and modulates by a Lambert term.

Rejected alternatives:
- **Procedural-in-shader pattern** — can't express combinatorial De Bruijn/Hamming sequences.
- **Bake UVs into vertices** — parallax needs per-fragment world position; breaks dynamic
  pattern switching.

Depth-dependent fringe **parallax comes from the projective transform, not shadows**; shadows
only add the "surfaces the projector can't see stay dark" effect (needed for faithful data).

## Architecture — two isolated units

### Unit 1 — `src/Engine/StructuredLight/PatternGenerators.h` (header-only, pure CPU)
Namespace `Engine::StructuredLight`. No Vulkan dependency → unit-testable in isolation. Ports
every generator from the Python file. Output is a uniform `PatternImage` (always RGBA8:
grayscale replicated into RGB, alpha = 255) so the GPU path has a single format.

```cpp
namespace Engine::StructuredLight {

enum class PatternType { PhaseShift, GrayCode, Binary, ColorCoded, ColorPhase };
enum class ColorCodeMode { DeBruijn, Hamming, SelfEqualizing };

struct PatternParams {
    PatternType   type   = PatternType::PhaseShift;
    int           width  = 1280;
    int           height = 720;
    // phase shift
    int           steps  = 3;      // N (>=3)
    float         freq   = 16.0f;  // cycles across width
    // gray code
    int           bits   = 4;      // N_bits
    // binary
    int           binaryPatterns = 5;
    // color coded
    ColorCodeMode colorMode = ColorCodeMode::Hamming;
    int           colorK    = 5;   // colors (De Bruijn/self-eq)
    int           colorN    = 4;   // window length
    // color phase (RGB-multiplexed PSP)
    float         colorPhaseFreq = 64.0f;
    // crosstalk precompensation (color patterns only)
    bool          precompensate = false;
    float         responseMatrix[9] = {1,0,0, 0,1,0, 0,0,1}; // row-major 3x3
};

struct PatternImage {
    int width = 0, height = 0;             // channels always 4 (RGBA8)
    std::vector<uint8_t> pixels;           // size = width*height*4
};

// How many pattern steps this configuration has (phase=steps, gray=bits,
// binary=binaryPatterns, color_coded=1, color_phase=1).
int  PatternCount(const PatternParams& p);

// Generate step `step` (0-based, wraps via PatternCount). Throws std::runtime_error on an
// invalid/singular response matrix (mirrors the Python _parse_color_matrix checks).
PatternImage MakePattern(const PatternParams& p, int step);

// Solid references used by gray/color reference frames and calibration (white/black/red/...).
PatternImage MakeFlat(int width, int height, uint8_t value);
PatternImage MakeSolidColor(int width, int height, uint8_t r, uint8_t g, uint8_t b);

// UI/metadata for color-coded patterns (logical vs projected stripe counts, decode window).
struct ColorCodeInfo { int logicalStripes, projectedStripes, decodeWindow; };
ColorCodeInfo ColorCodeMetadata(const PatternParams& p);

} // namespace Engine::StructuredLight
```

Internal helpers ported 1:1 from Python (all pure functions):
`sinusoidal`, `grayCode`, `binary`, `deBruijnSequence` (recursive), `constrainedDeBruijn`
(Euler-circuit; no equal neighbors), `hammingColorSequence`, `stripeImage`,
`selfEqualizingImage`, `colorPhase` (3-step PSP into R/G/B), `precompensateColor`.

### Unit 2 — `example2/dlp_simulator` (Vulkan glue, follows `ShadowMap.cpp` conventions)
- **`example2/dlp_simulator.cpp`** — `main`: window/app, mesh load + centering, camera (trackball),
  projector pose, ImGui control panel, key handlers, `RenderGraph` wiring.
- **`example2/DlpProjectorPass.{h,cpp}`** — the `RenderPass`. Owns: mesh objects, offscreen
  projector-depth `Image` + PCF sampler, pattern `Image` + linear/clamp sampler, a descriptor
  set with two combined image samplers (`0` = projector depth, `1` = pattern), depth pipeline +
  scene pipeline. Regenerates & re-uploads the pattern texture whenever params change (dirty flag).
- **Shaders** (`example2/`, compiled by `add_compiled_shaders`):
  - `dlp_depth.vert` — projector-POV depth (position → `projVP * model * pos`).
  - `dlp_scene.vert` — outputs world position + world normal to fragment stage.
  - `dlp_scene.frag` — projective pattern sample + occlusion + Lambert (see Data flow).

### New engine helper — CPU→GPU texture upload
No image-upload helper exists in `Engine::Core` today. Add one small function (or a private
method on `DlpProjectorPass`) that: creates a host-visible staging buffer, `memcpy`s the RGBA8
pixels, then via `Engine::Core::SubmitOneShot` records: transition `UNDEFINED→TRANSFER_DST_OPTIMAL`,
`vkCmdCopyBufferToImage`, transition `TRANSFER_DST→SHADER_READ_ONLY_OPTIMAL`. The pattern `Image`
is created with `usage = SAMPLED_BIT | TRANSFER_DST_BIT`, format `VK_FORMAT_R8G8B8A8_UNORM`.
Kept minimal and local to Step 1; promotable to `Engine::Core` later if reused.

## Data flow (one frame)

1. **Param change** (pattern type / step / freq / k / n / …) sets a dirty flag →
   `PatternGenerators::MakePattern` produces RGBA8 pixels → re-upload the pattern texture.
   Unchanged frames skip regeneration.
2. **Projector matrices** (rebuilt when FOV/baseline/mesh scale change):
   `projView = LookAt(projPos, meshCenter, up)`, `projProj = Perspective(fovY, patternAspect,
   near, far)`, `projVP = projProj * projView`. `patternAspect = width/height`.
3. **Pass 1 — projector depth**: bind depth pipeline; for each object push `{model, projVP}`;
   render into the offscreen depth image (perspective). Depth-bias to avoid self-shadow acne
   (as in `ShadowMap.cpp`: `DepthBias(1.2, 1.8)`).
4. **Pass 2 — scene** (into swapchain color + main depth): bind scene pipeline + descriptor set.
   Fragment shader per pixel:
   - `clip = projVP * worldPos`; if `clip.w <= 0` → outside (black).
   - `ndc = clip.xyz / clip.w`; `uv = ndc.xy * 0.5 + 0.5`; `projDepth = ndc.z`.
   - If `uv` outside `[0,1]` → outside projector frustum → pattern = 0 (Blender `CLIP`).
   - `storedDepth = texture(projectorDepth, uv)`; if `projDepth - bias > storedDepth` →
     **shadowed** → pattern = 0.
   - Else `patternColor = texture(pattern, uv).rgb`.
   - `lambert = max(ambientFloor, dot(N, normalize(projPos - worldPos)))` (floor = 0.08,
     matching Blender). Back/oblique faces never go fully black.
   - `finalColor = patternColor * lambert + fill * whiteAmbient` (small constant fill).

## Scene, camera, projector, controls
- **Mesh**: `util::LoadPlyMesh` (PLY) / local OBJ loader (copy the small `LoadObjMesh` helper
  from `object_scan_viewer.cpp`); `util::ComputeVertexNormals`; translate bbox center to origin
  and scale longest bbox axis to ~2 (fits [-1,1]). Vertex color = white (Lambertian target).
  CLI: `dlp_simulator <mesh.ply|obj>`; with no arg, fall back to a built-in unit sphere.
- **Render camera**: trackball orbit around origin (reuse `ShadowMap.cpp` mouse handlers),
  initial distance ~3, perspective ~55° FOV.
- **Projector**: world-fixed perspective pose. Placed by orbiting the initial camera position
  around the mesh center by `baselineAngleDeg` (default 15°, ≈ Blender `proj_angle`) about world
  up; looks at the mesh center. Independent of the render camera, so orbiting the camera reveals
  the fringe parallax.
- **ImGui panel** (mirrors the Blender N-panel; `ImGuiPass` + imgui already used by sibling
  examples): pattern type; per-type params (N & freq / bits / binary count / color mode·k·n·
  precompensation·matrix / color-phase freq); projector FOV & baseline angle; light energy &
  ambient; a `◀ Step x/N ▶` navigator; live readout of logical/projected stripes and px/stripe
  for color modes.
- **Keys**: `[` / `]` prev/next step; `P` cycle pattern type; mouse drag orbit, scroll zoom;
  `Esc` quit.

## File layout & build
```
src/Engine/StructuredLight/PatternGenerators.h   (new, header-only)
example2/dlp_simulator.cpp                        (new)
example2/DlpProjectorPass.h                       (new)
example2/DlpProjectorPass.cpp                     (new)
example2/dlp_depth.vert                           (new)
example2/dlp_scene.vert                           (new)
example2/dlp_scene.frag                           (new)
test/test_patternGenerators.cpp                   (new)
```
- `example2/CMakeLists.txt`: add
  ```cmake
  add_executable(dlp_simulator dlp_simulator.cpp DlpProjectorPass.cpp ImGuiPass.cpp)
  target_link_libraries(dlp_simulator PRIVATE Engine::Render Engine::Core imgui)
  add_compiled_shaders(dlp_simulator DLP_SHADER_DIR dlp_depth.vert dlp_scene.vert dlp_scene.frag)
  ```
  (Engine::Render already globs `Math.cpp`; no extra source needed. `util::LoadPlyMesh` /
  `PlyMesh.h` are header-only under `src/utilities`, on the propagated `src` include dir.)
- `test/CMakeLists.txt`: `*.cpp` is GLOB'd — the new test is auto-collected on the next
  `cmake` configure (re-run configure so the glob re-evaluates). No target edits needed;
  the test includes `Engine/StructuredLight/PatternGenerators.h` via the `src` include dir
  propagated by `Engine::Core`.
- **No `src/Engine/CMakeLists.txt` change**: `PatternGenerators.h` is header-only, so it needs
  no new library. (Only if it later grows a `.cpp` would a new `add_library` block be required.)

## Testing
- **Unit (`test/test_patternGenerators.cpp`, GTest)** — carries the correctness weight, mirrors
  the Python invariants:
  - Dimensions & channel count (RGBA8), value range [0,255].
  - `PatternCount` per type.
  - Phase-shift: row-constant columns, per-column cosine with correct period & step offset.
  - Gray code: adjacent stripe codes differ by exactly one bit.
  - De Bruijn (constrained): every length-`n` window unique; no two equal neighbors.
  - Hamming color: adjacent codewords Hamming distance 1; windows unique.
  - Self-equalizing: each logical stripe's color/complement channel pair sums to 255.
  - Color-phase: R/G/B channels equal `sinusoidal(0/1/2, 3, freq)`.
  - Precompensation: with identity matrix, output == input; singular matrix throws.
- **Visual (acceptance for Step 1)** — run `dlp_simulator <mesh>`; confirm the fringe deforms
  with surface depth, occluded regions are dark, and every pattern type/step switches live.
  Verify via `/verify` (drive the actual window), not tests alone.

## Out of scope for Step 1 (planned next steps)
Headless per-step PNG batch export; camera/projector calibration JSON export; orbit-batch
rendering; wiring generated frames into `Engine::Pipeline::StructuredLightFrameSource` for
decoding/reconstruction. Step 1 deliberately delivers only the interactive "load mesh + project
pattern" core.
