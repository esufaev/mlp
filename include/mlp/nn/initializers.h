#pragma once

#include "mlp/core/linalg.h"
#include "mlp/core/random.h"

#include <fmt/format.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mlp
{
    enum class initializer
    {
        he_uniform,
        he_normal,
        glorot_uniform,
        zeros
    };

    [[nodiscard]] inline initializer initializer_from_string(std::string_view name)
    {
        if (name == "heUniform" || name == "he_uniform")
            return initializer::he_uniform;
        if (name == "heNormal" || name == "he_normal")
            return initializer::he_normal;
        if (name == "glorotUniform" || name == "glorot_uniform" || name == "xavier")
            return initializer::glorot_uniform;
        if (name == "zeros")
            return initializer::zeros;

        throw std::invalid_argument(fmt::format(
            "unknown initializer '{}' (expected heUniform, heNormal, glorotUniform or zeros)", name));
    }

    [[nodiscard]] inline std::string to_string(initializer initializer)
    {
        switch (initializer)
        {
        case initializer::he_uniform:
            return "heUniform";
        case initializer::he_normal:
            return "heNormal";
        case initializer::glorot_uniform:
            return "glorotUniform";
        case initializer::zeros:
            return "zeros";
        }
        return "heUniform";
    }

    inline void initialize(matrix& weights, initializer initializer, rng& rng)
    {
        const auto fan_in = static_cast<double>(weights.rows());
        const auto fan_out = static_cast<double>(weights.cols());

        switch (initializer)
        {
        case initializer::he_uniform:
        {
            const double limit = std::sqrt(6.0 / fan_in);
            for (Eigen::Index k = 0; k < weights.size(); ++k)
                weights(k) = rng.uniform(-limit, limit);
            break;
        }
        case initializer::he_normal:
        {
            const double stddev = std::sqrt(2.0 / fan_in);
            for (Eigen::Index k = 0; k < weights.size(); ++k)
                weights(k) = rng.normal(0.0, stddev);
            break;
        }
        case initializer::glorot_uniform:
        {
            const double limit = std::sqrt(6.0 / (fan_in + fan_out));
            for (Eigen::Index k = 0; k < weights.size(); ++k)
                weights(k) = rng.uniform(-limit, limit);
            break;
        }
        case initializer::zeros:
            weights.fill(0.0);
            break;
        }
    }
}
