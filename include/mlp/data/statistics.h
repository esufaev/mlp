#pragma once

#include "mlp/core/linalg.h"
#include "mlp/data/dataset.h"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlp
{
struct feature_summary
{
    std::size_t index{0};
    std::size_t count{0};
    double mean{0.0};
    double stddev{0.0};
    double minimum{0.0};
    double first_quartile{0.0};
    double median{0.0};
    double third_quartile{0.0};
    double maximum{0.0};
};

namespace detail
{

[[nodiscard]] inline double quantile(const std::vector<double> &sorted, double fraction)
{
    if (sorted.empty())
        throw std::invalid_argument("quantile: empty sample");
    if (sorted.size() == 1)
        return sorted.front();

    const double position = fraction * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

[[nodiscard]] inline std::vector<double> column(const matrix &inputs, std::size_t index)
{
    std::vector<double> values(static_cast<std::size_t>(inputs.rows()));
    for (Eigen::Index i = 0; i < inputs.rows(); ++i)
        values[static_cast<std::size_t>(i)] = inputs(i, static_cast<Eigen::Index>(index));
    return values;
}

[[nodiscard]] inline double mean_of(const std::vector<double> &values)
{
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

[[nodiscard]] inline double stddev_of(const std::vector<double> &values, double mean)
{
    double total = 0.0;
    for (const double value : values)
        total += (value - mean) * (value - mean);
    return std::sqrt(total / static_cast<double>(values.size()));
}

[[nodiscard]] inline double pearson(const std::vector<double> &left,
                                    const std::vector<double> &right)
{
    const double left_mean = mean_of(left);
    const double right_mean = mean_of(right);

    double covariance = 0.0;
    double left_variance = 0.0;
    double right_variance = 0.0;
    for (std::size_t i = 0; i < left.size(); ++i)
    {
        const double left_deviation = left[i] - left_mean;
        const double right_deviation = right[i] - right_mean;
        covariance += left_deviation * right_deviation;
        left_variance += left_deviation * left_deviation;
        right_variance += right_deviation * right_deviation;
    }

    const double denominator = std::sqrt(left_variance * right_variance);
    return denominator < 1e-12 ? 0.0 : covariance / denominator;
}
} // namespace detail

[[nodiscard]] inline std::vector<feature_summary> describe(const dataset &dataset)
{
    std::vector<feature_summary> summaries;
    summaries.reserve(dataset.feature_count());

    for (std::size_t j = 0; j < dataset.feature_count(); ++j)
    {
        std::vector<double> values = detail::column(dataset.inputs, j);
        std::sort(values.begin(), values.end());

        feature_summary summary;
        summary.index = j;
        summary.count = values.size();
        summary.mean = detail::mean_of(values);
        summary.stddev = detail::stddev_of(values, summary.mean);
        summary.minimum = values.front();
        summary.first_quartile = detail::quantile(values, 0.25);
        summary.median = detail::quantile(values, 0.50);
        summary.third_quartile = detail::quantile(values, 0.75);
        summary.maximum = values.back();
        summaries.push_back(std::move(summary));
    }

    return summaries;
}

[[nodiscard]] inline matrix correlation_matrix(const matrix &inputs)
{
    const auto features = static_cast<Eigen::Index>(inputs.cols());
    std::vector<std::vector<double>> columns(static_cast<std::size_t>(features));
    for (Eigen::Index j = 0; j < features; ++j)
        columns[static_cast<std::size_t>(j)] =
            detail::column(inputs, static_cast<std::size_t>(j));

    matrix result = matrix::Zero(features, features);
    for (Eigen::Index i = 0; i < features; ++i)
    {
        result(i, i) = 1.0;
        for (Eigen::Index j = i + 1; j < features; ++j)
        {
            const double correlation = detail::pearson(columns[static_cast<std::size_t>(i)],
                                                       columns[static_cast<std::size_t>(j)]);
            result(i, j) = correlation;
            result(j, i) = correlation;
        }
    }
    return result;
}

[[nodiscard]] inline std::vector<double> correlation_with_labels(const dataset &dataset)
{
    std::vector<double> targets(dataset.size());
    for (std::size_t i = 0; i < dataset.size(); ++i)
        targets[i] = static_cast<double>(dataset.labels[i]);

    std::vector<double> correlations(dataset.feature_count());
    for (std::size_t j = 0; j < dataset.feature_count(); ++j)
        correlations[j] = detail::pearson(detail::column(dataset.inputs, j), targets);
    return correlations;
}

[[nodiscard]] inline std::string format_summary_table(const std::vector<feature_summary> &summaries)
{
    std::string output =
        fmt::format("{:>8}{:>7}{:>12}{:>12}{:>12}{:>12}{:>12}{:>12}{:>12}\n", "column", "count",
                    "mean", "std", "min", "25%", "50%", "75%", "max");
    output += std::string(99, '-') + '\n';

    for (const feature_summary &summary : summaries)
        output += fmt::format(
            "{:>8}{:>7}{:>12.4g}{:>12.4g}{:>12.4g}{:>12.4g}{:>12.4g}{:>12.4g}{:>12.4g}\n",
            summary.index, summary.count, summary.mean, summary.stddev, summary.minimum,
            summary.first_quartile, summary.median, summary.third_quartile, summary.maximum);
    return output;
}

[[nodiscard]] inline std::string format_class_balance(const dataset &dataset)
{
    const std::vector<std::size_t> counts = dataset.class_counts();
    const auto total = static_cast<double>(dataset.size());

    std::string output =
        fmt::format("samples: {}, features: {}\n", dataset.size(), dataset.feature_count());
    for (std::size_t c = 0; c < counts.size(); ++c)
        output += fmt::format("  class {} ({}): {:>4}  {:>6.2f}%\n", c, dataset.class_names[c],
                              counts[c], static_cast<double>(counts[c]) / total * 100.0);
    return output;
}

[[nodiscard]] inline std::string format_label_correlations(const dataset &dataset,
                                                           std::size_t top_count = 10)
{
    const std::vector<double> correlations = correlation_with_labels(dataset);

    std::vector<std::size_t> order(correlations.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right)
              { return std::abs(correlations[left]) > std::abs(correlations[right]); });

    const std::size_t shown = std::min(top_count, order.size());
    std::string output =
        fmt::format("top {} columns by |correlation| with the diagnosis:\n", shown);
    for (std::size_t i = 0; i < shown; ++i)
        output += fmt::format("  column {:>2}{:>+12.3f}\n", order[i], correlations[order[i]]);
    return output;
}

[[nodiscard]] inline std::string format_correlation_highlights(const dataset &dataset,
                                                               double threshold = 0.9)
{
    const matrix correlations = correlation_matrix(dataset.inputs);

    std::string output =
        fmt::format("column pairs with |correlation| >= {:.2f} (redundant inputs):\n", threshold);
    std::size_t found = 0;
    for (Eigen::Index i = 0; i < correlations.rows(); ++i)
        for (Eigen::Index j = i + 1; j < correlations.cols(); ++j)
            if (std::abs(correlations(i, j)) >= threshold)
            {
                output += fmt::format("  column {:>2} and column {:>2}{:>+12.3f}\n", i, j,
                                      correlations(i, j));
                ++found;
            }

    if (found == 0)
        output += "  none\n";
    return output;
}
} // namespace mlp
