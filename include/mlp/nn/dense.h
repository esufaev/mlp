#pragma once

#include "mlp/nn/initializers.h"
#include "mlp/nn/module.h"

#include <fmt/format.h>

#include <cstddef>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlp
{
class dense final : public module
{
  public:
	dense(std::size_t input_size, std::size_t output_size)
		: m_weights(tensor::parameter(matrix::Zero(static_cast<Eigen::Index>(input_size),
												   static_cast<Eigen::Index>(output_size)))),
		  m_bias(tensor::parameter(matrix::Zero(1, static_cast<Eigen::Index>(output_size))))
	{
		if (input_size == 0 || output_size == 0)
			throw std::invalid_argument(
				fmt::format("dense: [{} x {}] is not a usable shape", input_size, output_size));
	}

	dense(std::size_t input_size, std::size_t output_size, initializer initializer, rng& rng)
		: dense(input_size, output_size)
	{ initialize(m_weights.mutable_value(), initializer, rng); }

	tensor forward(const tensor& input) override
	{
		if (input.cols() != input_size())
			throw std::invalid_argument(
				fmt::format("dense::forward: input {} does not match weights {}",
							shape_string(input.value()), shape_string(m_weights.value())));

		return matmul(input, m_weights) + m_bias;
	}

	std::vector<tensor> parameters() override { return {m_weights, m_bias}; }

	[[nodiscard]] std::string type() const override { return "dense"; }

	[[nodiscard]] std::string config() const override
	{ return fmt::format("{} {}", input_size(), output_size()); }

	void write_parameters(std::ostream& stream) const override
	{
		const matrix& weights = m_weights.value();
		const matrix& bias = m_bias.value();

		for (Eigen::Index i = 0; i < weights.rows(); ++i)
		{
			for (Eigen::Index j = 0; j < weights.cols(); ++j)
				stream << fmt::format("{}{:.17g}", j == 0 ? "" : " ", weights(i, j));
			stream << '\n';
		}
		for (Eigen::Index j = 0; j < bias.cols(); ++j)
			stream << fmt::format("{}{:.17g}", j == 0 ? "" : " ", bias(0, j));
		stream << '\n';
	}

	void read_parameters(std::istream& stream) override
	{
		matrix& weights = m_weights.mutable_value();
		matrix& bias = m_bias.mutable_value();

		for (Eigen::Index i = 0; i < weights.rows(); ++i)
			for (Eigen::Index j = 0; j < weights.cols(); ++j)
				if (!(stream >> weights(i, j)))
					throw std::runtime_error(
						fmt::format("dense::read_parameters: truncated at weight ({}, {})", i, j));

		for (Eigen::Index j = 0; j < bias.cols(); ++j)
			if (!(stream >> bias(0, j)))
				throw std::runtime_error(
					fmt::format("dense::read_parameters: truncated at bias {}", j));
	}

	[[nodiscard]] std::size_t input_size() const noexcept
	{ return static_cast<std::size_t>(m_weights.value().rows()); }

	[[nodiscard]] std::size_t output_size() const noexcept
	{ return static_cast<std::size_t>(m_weights.value().cols()); }

	[[nodiscard]] tensor& weights() noexcept { return m_weights; }
	[[nodiscard]] const tensor& weights() const noexcept { return m_weights; }
	[[nodiscard]] tensor& bias() noexcept { return m_bias; }
	[[nodiscard]] const tensor& bias() const noexcept { return m_bias; }

  private:
	tensor m_weights;
	tensor m_bias;
};
} // namespace mlp
