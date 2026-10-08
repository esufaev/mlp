#pragma once

#include "mlp/data/csv.h"

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlp
{
struct history
{
	std::vector<double> train_loss;
	std::vector<double> validation_loss;
	std::vector<double> train_accuracy;
	std::vector<double> validation_accuracy;

	[[nodiscard]] std::size_t epochs() const noexcept { return train_loss.size(); }
	[[nodiscard]] bool has_validation() const noexcept { return !validation_loss.empty(); }

	[[nodiscard]] std::size_t best_epoch() const
	{
		const std::vector<double>& series = has_validation() ? validation_loss : train_loss;
		if (series.empty())
			throw std::logic_error("history::best_epoch: no epochs were recorded");
		return static_cast<std::size_t>(
			std::distance(series.begin(), std::min_element(series.begin(), series.end())));
	}

	void save_csv(const std::filesystem::path& path) const
	{
		csv_table table;
		table.reserve(epochs() + 1);
		table.push_back({"epoch", "loss", "val_loss", "acc", "val_acc"});

		for (std::size_t i = 0; i < epochs(); ++i)
			table.push_back(
				{fmt::format("{}", i + 1), fmt::format("{:.6f}", train_loss[i]),
				 has_validation() ? fmt::format("{:.6f}", validation_loss[i]) : std::string(),
				 fmt::format("{:.6f}", train_accuracy[i]),
				 has_validation() ? fmt::format("{:.6f}", validation_accuracy[i]) : std::string()});

		write_csv(path, table);
	}
};
} // namespace mlp
