#pragma once

#include "Engine/Eval/SyntheticSurface.h"

#include <Eigen/Core>
#include <cmath>
#include <limits>
#include <vector>

namespace Engine::Eval {

    // Accuracy: how close reconstructed points sit to the true surface (exact, closed-form).
    // Empty reconstruction → +inf so a regression guard necessarily fails.
    inline float AccuracyRMSE(const std::vector<Eigen::Vector3f> &reconPoints,
                              const Surface &surface) {
        if (reconPoints.empty()) return std::numeric_limits<float>::infinity();
        double sumSq = 0.0;
        for (const auto &p : reconPoints) {
            const float d = surface.Distance(p);
            sumSq += double(d) * double(d);
        }
        return float(std::sqrt(sumSq / double(reconPoints.size())));
    }

    // Nearest-neighbour distance RMSE from each point in `from` to the closest in `to`
    // (CPU brute-force). Empty `from` or `to` → +inf.
    inline float NearestNeighbourRMSE(const std::vector<Eigen::Vector3f> &from,
                                      const std::vector<Eigen::Vector3f> &to) {
        if (from.empty() || to.empty()) return std::numeric_limits<float>::infinity();
        double sumSq = 0.0;
        for (const auto &a : from) {
            float best = std::numeric_limits<float>::max();
            for (const auto &b : to) {
                const float d2 = (a - b).squaredNorm();
                if (d2 < best) best = d2;
            }
            sumSq += double(best);
        }
        return float(std::sqrt(sumSq / double(from.size())));
    }

    // Completeness: how well the true surface is covered by reconstruction —
    // for each GT point, nearest reconstructed point distance.
    inline float CompletenessRMSE(const std::vector<Eigen::Vector3f> &gtDense,
                                  const std::vector<Eigen::Vector3f> &reconPoints) {
        return NearestNeighbourRMSE(gtDense, reconPoints);
    }

    // Cross-check accuracy via nearest GT (bounded by GT sampling density).
    inline float ReconToGtNnRMSE(const std::vector<Eigen::Vector3f> &reconPoints,
                                 const std::vector<Eigen::Vector3f> &gtDense) {
        return NearestNeighbourRMSE(reconPoints, gtDense);
    }

} // namespace Engine::Eval
