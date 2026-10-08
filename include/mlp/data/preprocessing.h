#pragma once

#include "mlp/core/linalg.h"
#include "mlp/core/random.h"
#include "mlp/data/dataset.h"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mlp
{
class standard_scaler
{
  public:
	standard_scaler() = default;

	standard_scaler(std::vector<double> means, std::vector<double> stddevs)
		: m_means(std::move(means)), m_stddevs(std::move(stddevs))
	{
		if (m_means.size() != m_stddevs.size())
			throw std::invalid_argument(
				"standard_scaler: means and stddevs have different lengths");
	}

	void fit(const matrix& inputs)
	{
		if (inputs.rows() == 0)
			throw std::invalid_argument("standard_scaler::fit: no samples");

		const auto features = static_cast<std::size_t>(inputs.cols());
		const auto sample_count = static_cast<double>(inputs.rows());

		m_means.assign(features, 0.0);
		m_stddevs.assign(features, 0.0);

		Eigen::Map<Eigen::RowVectorXd> means(m_means.data(), inputs.cols());
		Eigen::Map<Eigen::RowVectorXd> stddevs(m_stddevs.data(), inputs.cols());

		means = inputs.colwise().sum() / sample_count;
		stddevs = ((inputs.rowwise() - means).array().square().colwise().sum() / sample_count)
					  .sqrt()
					  .max(1e-12);
	}

	[[nodiscard]] matrix transform(const matrix& inputs) const
	{
		if (!fitted())
			throw std::logic_error("standard_scaler::transform: the scaler has not been fitted");
		if (static_cast<std::size_t>(inputs.cols()) != m_means.size())
			throw std::invalid_argument(
				fmt::format("standard_scaler::transform: {} features, the scaler was fitted on {}",
							inputs.cols(), m_means.size()));

		return (inputs.rowwise() - means_row()).array().rowwise() / stddevs_row().array();
	}

	[[nodiscard]] matrix inverse_transform(const matrix& inputs) const
	{
		if (!fitted())
			throw std::logic_error(
				"standard_scaler::inverse_transform: the scaler has not been fitted");
		if (static_cast<std::size_t>(inputs.cols()) != m_means.size())
			throw std::invalid_argument(
				"standard_scaler::inverse_transform: feature count mismatch");

		return matrix(inputs.array().rowwise() * stddevs_row().array()).rowwise() + means_row();
	}

	[[nodiscard]] matrix fit_transform(const matrix& inputs)
	{
		fit(inputs);
		return transform(inputs);
	}

	[[nodiscard]] const std::vector<double>& means() const noexcept { return m_means; }
	[[nodiscard]] const std::vector<double>& stddevs() const noexcept { return m_stddevs; }
	[[nodiscard]] bool fitted() const noexcept { return !m_means.empty(); }

	void save(std::ostream& stream) const
	{
		stream << "scaler " << m_means.size() << '\n';
		for (std::size_t j = 0; j < m_means.size(); ++j)
			stream << fmt::format("{:.17g} {:.17g}\n", m_means[j], m_stddevs[j]);
	}

	void load(std::istream& stream)
	{
		std::string keyword;
		std::size_t features = 0;
		if (!(stream >> keyword >> features) || keyword != "scaler")
			throw std::runtime_error("standard_scaler::load: expected a 'scaler' section");

		m_means.assign(features, 0.0);
		m_stddevs.assign(features, 0.0);
		for (std::size_t j = 0; j < features; ++j)
			if (!(stream >> m_means[j] >> m_stddevs[j]))
				throw std::runtime_error(
					fmt::format("standard_scaler::load: truncated at feature {}", j));
	}

  private:
	[[nodiscard]] Eigen::Map<const Eigen::RowVectorXd> means_row() const
	{
		return Eigen::Map<const Eigen::RowVectorXd>(
			m_means.data(), static_cast<Eigen::Index>(m_means.size()));
	}

	[[nodiscard]] Eigen::Map<const Eigen::RowVectorXd> stddevs_row() const
	{
		return Eigen::Map<const Eigen::RowVectorXd>(
			m_stddevs.data(), static_cast<Eigen::Index>(m_stddevs.size()));
	}

	std::vector<double> m_means;
	std::vector<double> m_stddevs;
};

struct split_result
{
	dataset train;
	dataset validation;
};

[[nodiscard]] inline split_result train_test_split(const dataset& dataset, double train_ratio,
												   rng& rng, bool stratify = true)
{
	if (train_ratio <= 0.0 || train_ratio >= 1.0)
		throw std::invalid_argument(
			fmt::format("train_test_split: ratio {} is outside of (0, 1)", train_ratio));
	if (dataset.size() < 2)
		throw std::invalid_argument("train_test_split: need at least two samples");

	std::vector<std::size_t> train_indices;
	std::vector<std::size_t> validation_indices;

	if (stratify)
	{

		std::vector<std::vector<std::size_t>> per_class(dataset.class_count());
		for (std::size_t i = 0; i < dataset.size(); ++i)
			per_class[static_cast<std::size_t>(dataset.labels[i])].push_back(i);

		for (std::vector<std::size_t>& indices : per_class)
		{
			rng.shuffle(indices);
			const auto train_size = static_cast<std::size_t>(
				std::llround(static_cast<double>(indices.size()) * train_ratio));
			for (std::size_t i = 0; i < indices.size(); ++i)
				(i < train_size ? train_indices : validation_indices).push_back(indices[i]);
		}

		rng.shuffle(train_indices);
		rng.shuffle(validation_indices);
	}
	else
	{
		std::vector<std::size_t> indices = rng.permutation(dataset.size());
		const auto train_size = static_cast<std::size_t>(
			std::llround(static_cast<double>(indices.size()) * train_ratio));
		train_indices.assign(indices.begin(),
							 indices.begin() + static_cast<std::ptrdiff_t>(train_size));
		validation_indices.assign(indices.begin() + static_cast<std::ptrdiff_t>(train_size),
								  indices.end());
	}

	if (train_indices.empty() || validation_indices.empty())
		throw std::invalid_argument(
			fmt::format("train_test_split: ratio {} leaves one of the splits empty", train_ratio));

	return split_result{dataset.select(train_indices), dataset.select(validation_indices)};
}

[[nodiscard]] inline matrix one_hot(const std::vector<int>& labels, std::size_t class_count)
{
	matrix result = matrix::Zero(static_cast<Eigen::Index>(labels.size()),
								 static_cast<Eigen::Index>(class_count));
	for (std::size_t i = 0; i < labels.size(); ++i)
	{
		const auto index = static_cast<std::size_t>(labels[i]);
		if (labels[i] < 0 || index >= class_count)
			throw std::out_of_range(
				fmt::format("one_hot: label {} is outside of [0, {})", labels[i], class_count));
		result(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(index)) = 1.0;
	}
	return result;
}
} // namespace mlp
