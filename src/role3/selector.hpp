#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace jacobi_selection {

inline constexpr std::array<int, 8> kBatchSizes{2, 4, 8, 16, 32, 64, 100, 128};

struct Probe {
    int batch;
    double setup_ms;
    double replay_ms;
    bool reusable;
};

struct Decision {
    bool use_graph = false;
    int batch = 0;
    double predicted_remaining_ms = 0.0;
    bool valid_model = false;
    std::string reason;
};

inline std::vector<int> valid_batches(int iterations) {
    std::vector<int> result;
    for (int b : kBatchSizes) {
        if (b <= iterations) result.push_back(b);
    }
    return result;
}

inline std::vector<int> expected_probe_batches(int iterations) {
    const auto valid = valid_batches(iterations);
    if (valid.size() < 3) return {};
    return {valid.front(), valid[(valid.size() - 1) / 2], valid.back()};
}

inline Decision select_policy(int iterations, double ordinary_ms_per_iteration,
                              const std::vector<Probe>& probes) {
    Decision fallback;
    fallback.reason = "ordinary fallback";

    if (iterations <= 0 || !std::isfinite(ordinary_ms_per_iteration) ||
        ordinary_ms_per_iteration <= 0.0) {
        fallback.reason = "invalid iteration count or ordinary timing";
        return fallback;
    }

    fallback.predicted_remaining_ms =
        iterations * ordinary_ms_per_iteration;

    const auto expected = expected_probe_batches(iterations);
    if (expected.empty()) {
        fallback.reason = "fewer than three valid batches";
        return fallback;
    }
    if (probes.size() != expected.size()) {
        fallback.reason = "expected exactly three probes";
        return fallback;
    }

    Decision best = fallback;
    best.valid_model = true;
    best.reason = "ordinary predicted cheapest";

    // Development measurements showed that extrapolating an affine fit toward
    // unmeasured batches was unstable. Compare the three measured policies
    // directly, while still charging setup again when a probe was destroyed.
    for (int batch : expected) {
        const auto it = std::find_if(
            probes.begin(), probes.end(),
            [batch](const Probe& p) { return p.batch == batch; });

        if (it == probes.end() || !std::isfinite(it->setup_ms) ||
            !std::isfinite(it->replay_ms) || it->setup_ms < 0.0 ||
            it->replay_ms <= 0.0) {
            fallback.reason = "missing or invalid probe";
            return fallback;
        }

        const int full_replays = iterations / batch;
        const int remainder = iterations % batch;
        const double predicted =
            (it->reusable ? 0.0 : it->setup_ms) +
            full_replays * it->replay_ms +
            remainder * ordinary_ms_per_iteration;

        if (predicted < best.predicted_remaining_ms) {
            best.use_graph = true;
            best.batch = batch;
            best.predicted_remaining_ms = predicted;
            best.reason = "measured graph probe predicted cheapest";
        }
    }

    return best;
}

} // namespace jacobi_selection
