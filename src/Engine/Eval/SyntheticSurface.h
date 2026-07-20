#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace Engine::Eval {

    // Analytic surface with a closed-form unsigned distance, a surface normal, and a dense
    // sampler for ground-truth point clouds. Header-only (all methods inline).
    class Surface {
    public:
        virtual ~Surface() = default;
        virtual float Distance(const Eigen::Vector3f &p) const = 0;
        virtual Eigen::Vector3f NormalAt(const Eigen::Vector3f &p) const = 0;
        virtual std::vector<Eigen::Vector3f> SampleDense(uint32_t approxCount) const = 0;
    };

    // Finite plane patch: point q, unit normal n, in-plane half-extents (uExtent, vExtent).
    // Distance is the infinite-plane perpendicular distance (recon points only appear near
    // the scanned patch, so the finite extent does not affect the metric in practice).
    class PlaneSurface : public Surface {
    public:
        PlaneSurface(const Eigen::Vector3f &point, const Eigen::Vector3f &normal,
                     float uExtent, float vExtent)
            : m_point(point), m_normal(normal.normalized()),
              m_uExtent(uExtent), m_vExtent(vExtent) {
            // Build an orthonormal in-plane basis (m_u, m_v) ⟂ m_normal.
            Eigen::Vector3f seed = std::abs(m_normal.x()) < 0.9f
                                           ? Eigen::Vector3f(1, 0, 0)
                                           : Eigen::Vector3f(0, 1, 0);
            m_u = (seed - m_normal * seed.dot(m_normal)).normalized();
            m_v = m_normal.cross(m_u);
        }

        float Distance(const Eigen::Vector3f &p) const override {
            return std::abs((p - m_point).dot(m_normal));
        }

        Eigen::Vector3f NormalAt(const Eigen::Vector3f &) const override { return m_normal; }

        std::vector<Eigen::Vector3f> SampleDense(uint32_t approxCount) const override {
            const int n = std::max(2, int(std::sqrt(double(approxCount))));
            std::vector<Eigen::Vector3f> out;
            out.reserve(size_t(n) * n);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) {
                    const float u = -m_uExtent + 2.0f * m_uExtent * float(i) / float(n - 1);
                    const float v = -m_vExtent + 2.0f * m_vExtent * float(j) / float(n - 1);
                    out.push_back(m_point + m_u * u + m_v * v);
                }
            return out;
        }

    private:
        Eigen::Vector3f m_point, m_normal, m_u, m_v;
        float m_uExtent, m_vExtent;
    };

    // Sphere centred at `center` with radius `radius`.
    class SphereSurface : public Surface {
    public:
        SphereSurface(const Eigen::Vector3f &center, float radius)
            : m_center(center), m_radius(radius) {}

        float Distance(const Eigen::Vector3f &p) const override {
            return std::abs((p - m_center).norm() - m_radius);
        }

        Eigen::Vector3f NormalAt(const Eigen::Vector3f &p) const override {
            const Eigen::Vector3f d = p - m_center;
            const float n = d.norm();
            return n > 1e-8f ? Eigen::Vector3f(d / n) : Eigen::Vector3f(0, 0, 1);
        }

        // Roughly uniform surface sampling via the Fibonacci sphere.
        std::vector<Eigen::Vector3f> SampleDense(uint32_t approxCount) const override {
            const uint32_t n = std::max<uint32_t>(4, approxCount);
            std::vector<Eigen::Vector3f> out;
            out.reserve(n);
            const float golden = float(M_PI) * (3.0f - std::sqrt(5.0f)); // golden angle
            for (uint32_t i = 0; i < n; ++i) {
                const float y = 1.0f - 2.0f * (float(i) + 0.5f) / float(n); // (-1, 1)
                const float r = std::sqrt(std::max(0.0f, 1.0f - y * y));
                const float theta = golden * float(i);
                out.push_back(m_center + m_radius * Eigen::Vector3f(std::cos(theta) * r, y,
                                                                    std::sin(theta) * r));
            }
            return out;
        }

    private:
        Eigen::Vector3f m_center;
        float m_radius;
    };

    // Axis-aligned box centred at `center` with half-extents `half`. Edges/corners exercise
    // surfaces the sphere doesn't.
    class BoxSurface : public Surface {
    public:
        BoxSurface(const Eigen::Vector3f &center, const Eigen::Vector3f &half)
            : m_center(center), m_half(half.cwiseAbs()) {}

        float Distance(const Eigen::Vector3f &p) const override {
            const Eigen::Vector3f q = (p - m_center).cwiseAbs() - m_half;
            const float outside = q.cwiseMax(0.0f).norm();
            const float inside = std::min(q.maxCoeff(), 0.0f);
            return std::abs(outside + inside);
        }

        Eigen::Vector3f NormalAt(const Eigen::Vector3f &p) const override {
            const Eigen::Vector3f d = p - m_center;
            const Eigen::Vector3f a = d.cwiseAbs() - m_half;
            int axis = 0;
            a.maxCoeff(&axis); // dominant (nearest) face axis
            Eigen::Vector3f n = Eigen::Vector3f::Zero();
            n[axis] = d[axis] >= 0.0f ? 1.0f : -1.0f;
            return n;
        }

        std::vector<Eigen::Vector3f> SampleDense(uint32_t approxCount) const override {
            const int perFace = std::max(2, int(std::sqrt(double(approxCount) / 6.0)));
            std::vector<Eigen::Vector3f> out;
            out.reserve(size_t(perFace) * perFace * 6);
            const int inFace[3][2] = {{1, 2}, {0, 2}, {0, 1}};
            for (int axis = 0; axis < 3; ++axis)
                for (int s = 0; s < 2; ++s) {
                    const float sign = s == 0 ? -1.0f : 1.0f;
                    const int u = inFace[axis][0], v = inFace[axis][1];
                    for (int i = 0; i < perFace; ++i)
                        for (int j = 0; j < perFace; ++j) {
                            Eigen::Vector3f p = m_center;
                            p[axis] += sign * m_half[axis];
                            p[u] += -m_half[u] + 2.0f * m_half[u] * float(i) / float(perFace - 1);
                            p[v] += -m_half[v] + 2.0f * m_half[v] * float(j) / float(perFace - 1);
                            out.push_back(p);
                        }
                }
            return out;
        }

    private:
        Eigen::Vector3f m_center, m_half;
    };

    // Torus centred at `center`, symmetry axis +Y, major radius R (centre→tube centre), minor
    // radius r (tube). Non-convex — exercises self-occlusion in the visibility/scan step.
    class TorusSurface : public Surface {
    public:
        TorusSurface(const Eigen::Vector3f &center, float majorRadius, float minorRadius)
            : m_center(center), m_R(majorRadius), m_r(minorRadius) {}

        float Distance(const Eigen::Vector3f &p) const override {
            const Eigen::Vector3f d = p - m_center;
            const float xz = std::sqrt(d.x() * d.x() + d.z() * d.z());
            const float qx = xz - m_R;
            return std::abs(std::sqrt(qx * qx + d.y() * d.y()) - m_r);
        }

        Eigen::Vector3f NormalAt(const Eigen::Vector3f &p) const override {
            const Eigen::Vector3f d = p - m_center;
            const float xz = std::sqrt(d.x() * d.x() + d.z() * d.z());
            const Eigen::Vector3f ring = xz > 1e-8f
                    ? Eigen::Vector3f(m_R * d.x() / xz, 0.0f, m_R * d.z() / xz)
                    : Eigen::Vector3f(m_R, 0.0f, 0.0f);
            const Eigen::Vector3f n = d - ring;
            const float len = n.norm();
            return len > 1e-8f ? Eigen::Vector3f(n / len) : Eigen::Vector3f(0, 1, 0);
        }

        std::vector<Eigen::Vector3f> SampleDense(uint32_t approxCount) const override {
            const int n = std::max(4, int(std::sqrt(double(approxCount))));
            std::vector<Eigen::Vector3f> out;
            out.reserve(size_t(n) * n);
            for (int i = 0; i < n; ++i) {
                const float u = 2.0f * float(M_PI) * float(i) / float(n);
                for (int j = 0; j < n; ++j) {
                    const float v = 2.0f * float(M_PI) * float(j) / float(n);
                    const float cx = m_R + m_r * std::cos(v);
                    out.push_back(m_center + Eigen::Vector3f(cx * std::cos(u), m_r * std::sin(v),
                                                             cx * std::sin(u)));
                }
            }
            return out;
        }

    private:
        Eigen::Vector3f m_center;
        float m_R, m_r;
    };

} // namespace Engine::Eval
