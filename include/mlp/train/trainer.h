#pragma once

#include "mlp/autograd/tensor.h"
#include "mlp/core/linalg.h"
#include "mlp/core/random.h"
#include "mlp/data/preprocessing.h"
#include "mlp/nn/loss.h"
#include "mlp/nn/network.h"
#include "mlp/optim/sgd.h"
#include "mlp/train/history.h"
#include "mlp/train/metrics.h"

#include <fmt/format.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mlp
{
struct train_config
{
	std::size_t epochs{70};
	std::size_t batch_size{8};
	double learning_rate{0.0314};
	double momentum{0.0};
	std::string loss{"binaryCrossentropy"};
	std::uint64_t seed{42};
	bool verbose{true};
	std::size_t early_stopping_patience{0};
};

struct evaluation
{
	double loss{0.0};
	double accuracy{0.0};
};

[[nodiscard]] inline evaluation evaluate(network& network, const matrix& inputs,
										 const std::vector<int>& labels,
										 const std::string& loss_name, std::size_t class_count = 2)
{
	const no_grad guard;
	const std::unique_ptr<loss> loss = make_loss(loss_name);
	const matrix probabilities = network.forward(inputs);
	const matrix targets = one_hot(labels, class_count);

	return evaluation{loss->value(probabilities, targets),
					  mlp::accuracy(argmax_rows(probabilities), labels)};
}

namespace detail
{
[[nodiscard]] inline std::vector<matrix> snapshot(const std::vector<tensor>& parameters)
{
	std::vector<matrix> copy;
	copy.reserve(parameters.size());
	for (const tensor& parameter : parameters)
		copy.push_back(parameter.value());
	return copy;
}

inline void restore(const std::vector<tensor>& parameters, const std::vector<matrix>& values)
{
	for (std::size_t i = 0; i < parameters.size(); ++i)
		parameters[i].mutable_value() = values[i];
}

[[nodiscard]] inline std::size_t digits(std::size_t value)
{
	std::size_t count = 1;
	while (value >= 10)
	{
		value /= 10;
		++count;
	}
	return count;
}
} // namespace detail

class trainer
{
  public:
	explicit trainer(train_config config) : m_config(std::move(config))
	{
		if (m_config.epochs == 0)
			throw std::invalid_argument("trainer: epochs must be positive");
		if (m_config.batch_size == 0)
			throw std::invalid_argument("trainer: batch size must be positive");
	}

	history fit(network& network, const matrix& train_inputs, const std::vector<int>& train_labels,
				const matrix& validation_inputs, const std::vector<int>& validation_labels,
				std::size_t class_count = 2)
	{
		if (static_cast<std::size_t>(train_inputs.rows()) != train_labels.size())
			throw std::invalid_argument(fmt::format("trainer::fit: {} training rows but {} labels",
													train_inputs.rows(), train_labels.size()));
		if (train_inputs.rows() == 0)
			throw std::invalid_argument("trainer::fit: the training set is empty");
		if (static_cast<std::size_t>(validation_inputs.rows()) != validation_labels.size())
			throw std::invalid_argument(
				fmt::format("trainer::fit: {} validation rows but {} labels",
							validation_inputs.rows(), validation_labels.size()));

		const std::unique_ptr<loss> loss = make_loss(m_config.loss);
		sgd optimizer(m_config.learning_rate, m_config.momentum);
		rng rng(m_config.seed);

		const std::vector<tensor> parameters = network.parameters();
		const matrix train_targets = one_hot(train_labels, class_count);
		const bool has_validation = validation_inputs.rows() > 0;
		const auto sample_count = static_cast<std::size_t>(train_inputs.rows());
		const std::size_t epoch_width = detail::digits(m_config.epochs);

		history history;
		std::vector<matrix> best_parameters;
		double best_validation_loss = std::numeric_limits<double>::infinity();
		std::size_t epochs_without_improvement = 0;

		for (std::size_t epoch = 0; epoch < m_config.epochs; ++epoch)
		{
			const std::vector<std::size_t> order = rng.permutation(sample_count);

			for (std::size_t start = 0; start < sample_count; start += m_config.batch_size)
			{
				const std::size_t stop = std::min(start + m_config.batch_size, sample_count);
				const std::vector<std::size_t> batch(
					order.begin() + static_cast<std::ptrdiff_t>(start),
					order.begin() + static_cast<std::ptrdiff_t>(stop));

				zero_grad(parameters);

				const tensor predictions =
					network.forward(tensor::constant(train_inputs(batch, Eigen::all)));
				const tensor batch_loss = (*loss)(predictions, train_targets(batch, Eigen::all));

				batch_loss.backward();
				optimizer.step(parameters);
			}

			const evaluation train_metrics =
				evaluate(network, train_inputs, train_labels, m_config.loss, class_count);
			history.train_loss.push_back(train_metrics.loss);
			history.train_accuracy.push_back(train_metrics.accuracy);

			std::string validation_report;
			if (has_validation)
			{
				const evaluation validation_metrics = evaluate(
					network, validation_inputs, validation_labels, m_config.loss, class_count);
				history.validation_loss.push_back(validation_metrics.loss);
				history.validation_accuracy.push_back(validation_metrics.accuracy);
				validation_report =
					fmt::format(" - val_loss: {:.4f} - val_acc: {:.4f}", validation_metrics.loss,
								validation_metrics.accuracy);

				if (m_config.early_stopping_patience > 0)
				{
					if (validation_metrics.loss < best_validation_loss)
					{
						best_validation_loss = validation_metrics.loss;
						best_parameters = detail::snapshot(parameters);
						epochs_without_improvement = 0;
					}
					else
					{
						++epochs_without_improvement;
					}
				}
			}

			if (m_config.verbose)
				fmt::print("epoch {:0{}}/{} - loss: {:.4f} - acc: {:.4f}{}\n", epoch + 1,
						   epoch_width, m_config.epochs, train_metrics.loss, train_metrics.accuracy,
						   validation_report);

			if (m_config.early_stopping_patience > 0 &&
				epochs_without_improvement >= m_config.early_stopping_patience)
			{
				if (m_config.verbose)
					fmt::print("early stopping: no improvement for {} epochs, restoring epoch {}\n",
							   epochs_without_improvement, history.best_epoch() + 1);
				if (!best_parameters.empty())
					detail::restore(parameters, best_parameters);
				break;
			}
		}

		return history;
	}

	[[nodiscard]] const train_config& config() const noexcept { return m_config; }

  private:
	train_config m_config;
};
} // namespace mlp
