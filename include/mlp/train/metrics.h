#pragma once

#include "mlp/core/linalg.h"

#include <fmt/format.h>

#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlp
{
struct confusion_matrix
{
	std::vector<std::vector<std::size_t>> counts;

	[[nodiscard]] std::size_t class_count() const noexcept { return counts.size(); }

	[[nodiscard]] std::size_t total() const noexcept
	{
		std::size_t sum = 0;
		for (const std::vector<std::size_t>& row : counts)
			sum += std::accumulate(row.begin(), row.end(), std::size_t{0});
		return sum;
	}

	[[nodiscard]] std::size_t correct() const noexcept
	{
		std::size_t sum = 0;
		for (std::size_t i = 0; i < counts.size(); ++i)
			sum += counts[i][i];
		return sum;
	}

	[[nodiscard]] std::size_t predicted_as(std::size_t klass) const
	{
		std::size_t sum = 0;
		for (const std::vector<std::size_t>& row : counts)
			sum += row[klass];
		return sum;
	}

	[[nodiscard]] std::size_t support(std::size_t klass) const
	{ return std::accumulate(counts[klass].begin(), counts[klass].end(), std::size_t{0}); }
};

[[nodiscard]] inline double accuracy(const std::vector<int>& predicted,
									 const std::vector<int>& actual)
{
	if (predicted.size() != actual.size())
		throw std::invalid_argument(
			fmt::format("accuracy: {} predictions but {} labels", predicted.size(), actual.size()));
	if (predicted.empty())
		throw std::invalid_argument("accuracy: empty input");

	std::size_t hits = 0;
	for (std::size_t i = 0; i < predicted.size(); ++i)
		if (predicted[i] == actual[i])
			++hits;
	return static_cast<double>(hits) / static_cast<double>(predicted.size());
}

[[nodiscard]] inline confusion_matrix make_confusion_matrix(const std::vector<int>& predicted,
															const std::vector<int>& actual,
															std::size_t class_count)
{
	if (predicted.size() != actual.size())
		throw std::invalid_argument(
			fmt::format("make_confusion_matrix: {} predictions but {} labels", predicted.size(),
						actual.size()));

	confusion_matrix matrix;
	matrix.counts.assign(class_count, std::vector<std::size_t>(class_count, 0));

	for (std::size_t i = 0; i < predicted.size(); ++i)
	{
		const auto actual_class = static_cast<std::size_t>(actual[i]);
		const auto predicted_class = static_cast<std::size_t>(predicted[i]);
		if (actual[i] < 0 || predicted[i] < 0 || actual_class >= class_count ||
			predicted_class >= class_count)
			throw std::out_of_range(fmt::format(
				"make_confusion_matrix: sample {} has labels ({}, {}) outside of [0, {})", i,
				actual[i], predicted[i], class_count));
		++matrix.counts[actual_class][predicted_class];
	}
	return matrix;
}

namespace detail
{
inline void require_valid_class(const confusion_matrix& matrix, std::size_t klass, const char* who)
{
	if (klass >= matrix.class_count())
		throw std::out_of_range(
			fmt::format("{}: class {} is outside of {} classes", who, klass, matrix.class_count()));
}
} // namespace detail

[[nodiscard]] inline double precision(const confusion_matrix& matrix, std::size_t positive_class)
{
	detail::require_valid_class(matrix, positive_class, "precision");
	const std::size_t flagged = matrix.predicted_as(positive_class);
	return flagged == 0 ? 0.0
						: static_cast<double>(matrix.counts[positive_class][positive_class]) /
							  static_cast<double>(flagged);
}

[[nodiscard]] inline double recall(const confusion_matrix& matrix, std::size_t positive_class)
{
	detail::require_valid_class(matrix, positive_class, "recall");
	const std::size_t present = matrix.support(positive_class);
	return present == 0 ? 0.0
						: static_cast<double>(matrix.counts[positive_class][positive_class]) /
							  static_cast<double>(present);
}

[[nodiscard]] inline double f1_score(const confusion_matrix& matrix, std::size_t positive_class)
{
	const double p = precision(matrix, positive_class);
	const double r = recall(matrix, positive_class);
	return (p + r) < 1e-12 ? 0.0 : 2.0 * p * r / (p + r);
}

[[nodiscard]] inline std::vector<int> argmax_rows(const matrix& probabilities)
{
	std::vector<int> classes(static_cast<std::size_t>(probabilities.rows()));
	for (Eigen::Index i = 0; i < probabilities.rows(); ++i)
	{
		Eigen::Index best = 0;
		probabilities.row(i).maxCoeff(&best);
		classes[static_cast<std::size_t>(i)] = static_cast<int>(best);
	}
	return classes;
}

[[nodiscard]] inline std::string format_confusion_matrix(const confusion_matrix& matrix,
														 const std::vector<std::string>& names)
{
	std::string output = fmt::format("{:<14}", "actual \\ pred");
	for (std::size_t j = 0; j < matrix.class_count(); ++j)
		output += fmt::format("{:>10}", j < names.size() ? names[j] : std::to_string(j));
	output += '\n';

	for (std::size_t i = 0; i < matrix.class_count(); ++i)
	{
		output += fmt::format("{:<14}", i < names.size() ? names[i] : std::to_string(i));
		for (std::size_t j = 0; j < matrix.class_count(); ++j)
			output += fmt::format("{:>10}", matrix.counts[i][j]);
		output += '\n';
	}
	return output;
}

[[nodiscard]] inline std::string format_classification_report(const confusion_matrix& matrix,
															  const std::vector<std::string>& names)
{
	std::string output = fmt::format("{:<14}{:>12}{:>12}{:>12}{:>10}\n", "class", "precision",
									 "recall", "f1-score", "support");
	for (std::size_t c = 0; c < matrix.class_count(); ++c)
		output += fmt::format("{:<14}{:>12.4f}{:>12.4f}{:>12.4f}{:>10}\n",
							  c < names.size() ? names[c] : std::to_string(c), precision(matrix, c),
							  recall(matrix, c), f1_score(matrix, c), matrix.support(c));
	return output;
}
} // namespace mlp
