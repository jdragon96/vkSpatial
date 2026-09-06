#pragma once

// Minimal SE(3)/SO(3) Lie-group helpers for the SLAM backend.
//
// These implement the exponential/logarithm maps that let us optimise camera
// poses on the manifold (design note 00 §4.2): a pose update is applied as
//   T <- T * Exp(delta),  delta in R^6,
// which keeps T a valid rigid transform no matter how the optimiser nudges it.
//
// Twist convention: xi = [rho; phi], rho = translation part (first 3),
// phi = rotation part (last 3).  Eigen-only, double precision, header-only.

#include <Eigen/Dense>

#include <cmath>

namespace Registration::Backend {

    using Vec6 = Eigen::Matrix<double, 6, 1>;
    using Mat6 = Eigen::Matrix<double, 6, 6>;

    // Skew-symmetric matrix [w]_x such that [w]_x v == w.cross(v).
    inline Eigen::Matrix3d Skew(const Eigen::Vector3d &w) {
        Eigen::Matrix3d s;
        s << 0.0, -w.z(), w.y(),
                w.z(), 0.0, -w.x(),
                -w.y(), w.x(), 0.0;
        return s;
    }

    // SO(3) exponential (Rodrigues' formula) with a small-angle Taylor fallback.
    inline Eigen::Matrix3d SO3Exp(const Eigen::Vector3d &phi) {
        const double th2 = phi.squaredNorm();
        const double th = std::sqrt(th2);
        const Eigen::Matrix3d W = Skew(phi);
        if (th < 1e-8) {
            return Eigen::Matrix3d::Identity() + W + 0.5 * W * W;
        }
        const double a = std::sin(th) / th;
        const double b = (1.0 - std::cos(th)) / th2;
        return Eigen::Matrix3d::Identity() + a * W + b * W * W;
    }

    // SO(3) logarithm: rotation matrix -> rotation vector.
    inline Eigen::Vector3d SO3Log(const Eigen::Matrix3d &R) {
        double cosTh = 0.5 * (R.trace() - 1.0);
        cosTh = std::max(-1.0, std::min(1.0, cosTh));
        const double th = std::acos(cosTh);
        Eigen::Vector3d w(R(2, 1) - R(1, 2),
                          R(0, 2) - R(2, 0),
                          R(1, 0) - R(0, 1));
        if (th < 1e-8) {
            return 0.5 * w; // small angle: (R - R^T)^v / 2
        }
        return (th / (2.0 * std::sin(th))) * w;
    }

    // Left Jacobian of SO(3) (the "V" matrix used by the SE(3) exp/log).
    inline Eigen::Matrix3d LeftJacobianSO3(const Eigen::Vector3d &phi) {
        const double th2 = phi.squaredNorm();
        const double th = std::sqrt(th2);
        const Eigen::Matrix3d W = Skew(phi);
        if (th < 1e-8) {
            return Eigen::Matrix3d::Identity() + 0.5 * W + (1.0 / 6.0) * W * W;
        }
        const double b = (1.0 - std::cos(th)) / th2;
        const double c = (th - std::sin(th)) / (th2 * th);
        return Eigen::Matrix3d::Identity() + b * W + c * W * W;
    }

    // Rigid transform T = [R t; 0 1] with the operations pose-graph code needs.
    struct SE3 {
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        Eigen::Vector3d t = Eigen::Vector3d::Zero();

        static SE3 Identity() { return SE3{}; }

        SE3 inverse() const {
            SE3 o;
            o.R = R.transpose();
            o.t = -(o.R * t);
            return o;
        }

        // Composition: (this * b) applied to a point = this applied to (b applied to point).
        SE3 operator*(const SE3 &b) const {
            SE3 o;
            o.R = R * b.R;
            o.t = R * b.t + t;
            return o;
        }

        Eigen::Vector3d operator*(const Eigen::Vector3d &p) const { return R * p + t; }

        Eigen::Vector3d translation() const { return t; }
    };

    // SE(3) exponential: twist xi = [rho; phi] -> rigid transform.
    inline SE3 SE3Exp(const Vec6 &xi) {
        const Eigen::Vector3d rho = xi.head<3>();
        const Eigen::Vector3d phi = xi.tail<3>();
        SE3 T;
        T.R = SO3Exp(phi);
        T.t = LeftJacobianSO3(phi) * rho;
        return T;
    }

    // SE(3) logarithm: rigid transform -> twist xi = [rho; phi].
    inline Vec6 SE3Log(const SE3 &T) {
        const Eigen::Vector3d phi = SO3Log(T.R);
        const Eigen::Matrix3d Vinv = LeftJacobianSO3(phi).inverse();
        Vec6 xi;
        xi.head<3>() = Vinv * T.t;
        xi.tail<3>() = phi;
        return xi;
    }

} // namespace Registration::Backend
