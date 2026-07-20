#include "Engine/Backend/PoseGraph.h"

#include <cmath>

namespace Engine::Backend {

    int PoseGraph::addVertex(const SE3 &pose, bool fixed) {
        m_poses.push_back(pose);
        m_fixed.push_back(fixed ? 1 : 0);
        return static_cast<int>(m_poses.size()) - 1;
    }

    void PoseGraph::addEdge(int i, int j, const SE3 &measuredIJ, const Mat6 &information) {
        Edge e;
        e.i = i;
        e.j = j;
        e.z = measuredIJ;
        e.info = information;
        m_edges.push_back(e);
    }

    double PoseGraph::chi2Of(const std::vector<SE3> &poses) const {
        double sum = 0.0;
        for (const Edge &e : m_edges) {
            const Vec6 r = relResidual(poses[e.i], poses[e.j], e.z);
            sum += r.transpose() * e.info * r;
        }
        return sum;
    }

    // Finite-difference Jacobians of the residual against a right perturbation of
    // T_i and T_j: J.col(k) = ( e(T * Exp(eps*e_k)) - e(T) ) / eps.
    void PoseGraph::jacobians(const Edge &e, const std::vector<SE3> &poses,
                              Mat6 &Ji, Mat6 &Jj) const {
        const double eps = 1e-6;
        const SE3 &Ti = poses[e.i];
        const SE3 &Tj = poses[e.j];
        const Vec6 r0 = relResidual(Ti, Tj, e.z);
        for (int k = 0; k < 6; ++k) {
            Vec6 dk = Vec6::Zero();
            dk(k) = eps;
            const SE3 step = SE3Exp(dk);
            Ji.col(k) = (relResidual(Ti * step, Tj, e.z) - r0) / eps;
            Jj.col(k) = (relResidual(Ti, Tj * step, e.z) - r0) / eps;
        }
    }

    PoseGraph::Report PoseGraph::optimize(const Options &opt) {
        Report rep;
        rep.initialChi2 = chi2();
        rep.finalChi2 = rep.initialChi2;

        // Map each free vertex to a 6-DoF block in the state vector.
        std::vector<int> stateIndex(m_poses.size(), -1);
        int nFree = 0;
        for (size_t v = 0; v < m_poses.size(); ++v) {
            if (!m_fixed[v]) stateIndex[v] = nFree++;
        }
        if (nFree == 0 || m_edges.empty()) {
            rep.converged = true;
            return rep;
        }

        const int N = 6 * nFree;
        double lambda = opt.initialLambda;
        double curChi2 = rep.initialChi2;

        for (int iter = 0; iter < opt.maxIterations; ++iter) {
            Eigen::MatrixXd H = Eigen::MatrixXd::Zero(N, N);
            Eigen::VectorXd b = Eigen::VectorXd::Zero(N);

            for (const Edge &e : m_edges) {
                const Vec6 r = relResidual(m_poses[e.i], m_poses[e.j], e.z);
                Mat6 Ji, Jj;
                jacobians(e, m_poses, Ji, Jj);
                const Mat6 &O = e.info;
                const int si = stateIndex[e.i];
                const int sj = stateIndex[e.j];
                if (si >= 0) {
                    H.block<6, 6>(6 * si, 6 * si) += Ji.transpose() * O * Ji;
                    b.segment<6>(6 * si) += Ji.transpose() * O * r;
                }
                if (sj >= 0) {
                    H.block<6, 6>(6 * sj, 6 * sj) += Jj.transpose() * O * Jj;
                    b.segment<6>(6 * sj) += Jj.transpose() * O * r;
                }
                if (si >= 0 && sj >= 0) {
                    H.block<6, 6>(6 * si, 6 * sj) += Ji.transpose() * O * Jj;
                    H.block<6, 6>(6 * sj, 6 * si) += Jj.transpose() * O * Ji;
                }
            }

            // Levenberg-Marquardt: scale the diagonal by (1 + lambda).
            Eigen::MatrixXd Hd = H;
            for (int k = 0; k < N; ++k) {
                Hd(k, k) += lambda * Hd(k, k) + 1e-12;
            }
            const Eigen::VectorXd dx = Hd.ldlt().solve(-b);

            // Tentatively apply the step (right perturbation on the manifold).
            std::vector<SE3> trial = m_poses;
            for (size_t v = 0; v < m_poses.size(); ++v) {
                const int s = stateIndex[v];
                if (s < 0) continue;
                trial[v] = m_poses[v] * SE3Exp(dx.segment<6>(6 * s));
            }
            const double newChi2 = chi2Of(trial);

            rep.iterations = iter + 1;
            if (newChi2 < curChi2) {
                const double improvement = curChi2 - newChi2;
                m_poses.swap(trial);
                curChi2 = newChi2;
                lambda = std::max(lambda * 0.5, 1e-9);
                if (dx.norm() < opt.convergenceDeltaNorm ||
                    improvement < opt.convergenceChi2Delta) {
                    rep.converged = true;
                    break;
                }
            } else {
                lambda *= 4.0;
                if (lambda > 1e12) break; // stuck
            }
        }

        rep.finalChi2 = curChi2;
        return rep;
    }

} // namespace Engine::Backend
