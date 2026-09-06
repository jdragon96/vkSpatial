#pragma once

// SE(3) pose-graph optimiser — the core of the SLAM backend.
//
// Vertices are keyframe world poses T_world_cam. Edges are relative-pose
// measurements Z_ij (from frontend odometry or from loop closure) with a 6x6
// information matrix Omega (inverse covariance; how much we trust the edge).
//
// It minimises the standard pose-graph objective (design note 00 §4.4):
//     min_T  sum_(i,j)  e_ij^T Omega_ij e_ij ,
//     e_ij = Log( Z_ij^{-1} * T_i^{-1} * T_j )  in R^6,
// via Gauss-Newton with Levenberg-Marquardt damping. Poses are updated on the
// manifold as  T_i <- T_i * Exp(delta_i)  (right perturbation). At least one
// vertex should be fixed to anchor the global gauge (e.g. vertex 0).
//
// This first version uses numerical Jacobians (finite differences of the
// residual against a right perturbation). They are exact-to-eps and match the
// update rule, so the linearisation is consistent; analytic Jacobians can
// replace them later for speed without changing behaviour.

#include "Registration/Backend/Lie.h"

#include <vector>

namespace Registration::Backend {

    class PoseGraph {
    public:
        struct Options {
            int maxIterations = 100;
            double convergenceDeltaNorm = 1e-9; // stop when ||delta|| below this
            double convergenceChi2Delta = 1e-12; // stop when chi2 improvement below this
            double initialLambda = 1e-4;          // LM damping
            bool verbose = false;
        };

        struct Report {
            int iterations = 0;
            double initialChi2 = 0.0;
            double finalChi2 = 0.0;
            bool converged = false;
        };

        // Adds a vertex with an initial pose estimate; returns its index.
        // Fixed vertices are held constant (used to anchor the gauge).
        int addVertex(const SE3 &pose, bool fixed = false);

        // Adds a relative-pose constraint i -> j: measuredIJ ~ T_i^{-1} T_j.
        void addEdge(int i, int j, const SE3 &measuredIJ,
                     const Mat6 &information = Mat6::Identity());

        Report optimize(const Options &opt);
        Report optimize() { return optimize(Options{}); }

        const SE3 &pose(int i) const { return m_poses[i]; }
        const std::vector<SE3> &poses() const { return m_poses; }
        int numVertices() const { return static_cast<int>(m_poses.size()); }
        int numEdges() const { return static_cast<int>(m_edges.size()); }

        // Total weighted squared error of the current estimate.
        double chi2() const { return chi2Of(m_poses); }

    private:
        struct Edge {
            int i = 0;
            int j = 0;
            SE3 z;                        // measured relative transform i->j
            Mat6 info = Mat6::Identity(); // information matrix Omega
        };

        std::vector<SE3> m_poses;
        std::vector<char> m_fixed; // vector<bool> avoided for stable refs
        std::vector<Edge> m_edges;

        static Vec6 relResidual(const SE3 &Ti, const SE3 &Tj, const SE3 &Z) {
            return SE3Log(Z.inverse() * Ti.inverse() * Tj);
        }

        void jacobians(const Edge &e, const std::vector<SE3> &poses,
                       Mat6 &Ji, Mat6 &Jj) const;

        double chi2Of(const std::vector<SE3> &poses) const;
    };

} // namespace Registration::Backend
