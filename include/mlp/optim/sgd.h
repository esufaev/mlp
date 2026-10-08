#pragma once

#include "mlp/optim/optimizer.h"

#include <fmt/format.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlp
{
class sgd final : public optimizer
{
  public:
	explicit sgd(double learning_rate, double momentum = 0.0)
		: m_learning_rate(learning_rate), m_momentum(momentum)
	{
		if (learning_rate <= 0.0)
			throw std::invalid_argument(
				fmt::format("sgd: learning rate {} must be positive", learning_rate));
		if (momentum < 0.0 || momentum >= 1.0)
			throw std::invalid_argument(
				fmt::format("sgd: momentum {} is outside of [0, 1)", momentum));
	}

	void step(const std::vector<tensor>& parameters) override
	{
		const bool use_momentum = m_momentum > 0.0;
		if (use_momentum && m_velocities.size() != parameters.size())
		{
			m_velocities.clear();
			m_velocities.reserve(parameters.size());
			for (const tensor& parameter : parameters)
				m_velocities.push_back(
					matrix::Zero(parameter.value().rows(), parameter.value().cols()));
		}

		for (std::size_t i = 0; i < parameters.size(); ++i)
		{
			matrix& value = parameters[i].mutable_value();
			const matrix& gradient = parameters[i].grad();

			if (value.rows() != gradient.rows() || value.cols() != gradient.cols())
				throw std::invalid_argument(
					fmt::format("sgd::step: parameter {} is {} but its gradient is {}", i,
								shape_string(value), shape_string(gradient)));

			if (use_momentum)
			{
				matrix& velocity = m_velocities[i];
				velocity = m_momentum * velocity - m_learning_rate * gradient;
				value += velocity;
			}
			else
			{
				value -= m_learning_rate * gradient;
			}
		}
	}

	[[nodiscard]] std::string name() const override { return "sgd"; }
	[[nodiscard]] double learning_rate() const override { return m_learning_rate; }
	[[nodiscard]] double momentum() const noexcept { return m_momentum; }

	void set_learning_rate(double learning_rate) noexcept { m_learning_rate = learning_rate; }

  private:
	double m_learning_rate;
	double m_momentum;
	std::vector<matrix> m_velocities;
};
} // namespace mlp
