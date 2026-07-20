#include "Engine/Registration/FeatureMatching.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Engine::Registration {

    namespace {

        struct RatedCorrespondence {
            Correspondence corr;
            float ratio; // 1st-NN dist / 2nd-NN dist, lower = more distinctive match
        };

    } // namespace

    std::vector<Correspondence> MatchFeatures(const std::vector<Fpfh33> &srcF, const std::vector<Fpfh33> &tgtF,
                                               float ratioThr, int numMaxCorr) {
        std::vector<Correspondence> result;

        // Can't form a 1st/2nd-NN ratio with fewer than 2 targets; nothing distinctive can
        // be asserted, so skip the ratio test entirely and return no correspondences.
        if (srcF.empty() || tgtF.size() < 2) return result;

        std::vector<RatedCorrespondence> rated;
        rated.reserve(srcF.size());

        // Brute-force O(|srcF| * |tgtF|) linear scan over 33-D descriptors, fine at M1
        // scale (srcF/tgtF are downsampled/keypoint-scale clouds). M2 speedup: build a
        // kd-tree (e.g. nanoflann) once over tgtF and reuse it across all src queries
        // instead of rescanning tgtF per src point.
        for (size_t i = 0; i < srcF.size(); ++i) {
            float bestD2 = std::numeric_limits<float>::max();
            float secondD2 = std::numeric_limits<float>::max();
            int bestJ = -1;

            for (size_t j = 0; j < tgtF.size(); ++j) {
                const float d2 = (srcF[i] - tgtF[j]).squaredNorm();
                if (d2 < bestD2) {
                    secondD2 = bestD2;
                    bestD2 = d2;
                    bestJ = int(j);
                } else if (d2 < secondD2) {
                    secondD2 = d2;
                }
            }

            if (bestJ < 0 || secondD2 <= 0.0f) continue; // guard div-by-zero (degenerate duplicate descriptors)

            const float ratio = std::sqrt(bestD2) / std::sqrt(secondD2);
            if (ratio < ratioThr) {
                rated.push_back(RatedCorrespondence{Correspondence{int(i), bestJ}, ratio});
            }
        }

        // Best (most distinctive, lowest ratio) matches first, so the cap below keeps the
        // strongest correspondences.
        std::sort(rated.begin(), rated.end(),
                  [](const RatedCorrespondence &a, const RatedCorrespondence &b) { return a.ratio < b.ratio; });

        if (numMaxCorr >= 0 && int(rated.size()) > numMaxCorr) rated.resize(size_t(numMaxCorr));

        result.reserve(rated.size());
        for (const auto &r: rated) result.push_back(r.corr);
        return result;
    }

} // namespace Engine::Registration
