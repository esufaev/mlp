#pragma once

#include "mlp/core/linalg.h"
#include "mlp/data/csv.h"

#include <fmt/format.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlp
{
struct dataset
{
	matrix inputs;
	std::vector<int> labels;
	std::vector<std::string> ids;
	std::vector<std::string> class_names{"B", "M"};

	[[nodiscard]] std::size_t size() const noexcept
	{ return static_cast<std::size_t>(inputs.rows()); }

	[[nodiscard]] std::size_t feature_count() const noexcept
	{ return static_cast<std::size_t>(inputs.cols()); }
	[[nodiscard]] std::size_t class_count() const noexcept { return class_names.size(); }

	[[nodiscard]] dataset select(const std::vector<std::size_t>& indices) const
	{
		for (const std::size_t index : indices)
			if (index >= size())
				throw std::out_of_range(
					fmt::format("dataset::select: {} is outside of {} samples", index, size()));

		dataset result;
		result.inputs = inputs(indices, Eigen::all);
		result.class_names = class_names;
		result.labels.reserve(indices.size());
		result.ids.reserve(indices.size());

		for (const std::size_t index : indices)
		{
			result.labels.push_back(labels[index]);
			result.ids.push_back(ids[index]);
		}
		return result;
	}

	[[nodiscard]] std::vector<std::size_t> class_counts() const
	{
		std::vector<std::size_t> counts(class_count(), 0);
		for (const int label : labels)
		{
			const auto index = static_cast<std::size_t>(label);
			if (index >= counts.size())
				throw std::out_of_range(
					fmt::format("dataset::class_counts: label {} is out of range", label));
			++counts[index];
		}
		return counts;
	}

	void save(const std::filesystem::path& path) const
	{
		csv_table table;
		table.reserve(size());

		for (std::size_t i = 0; i < size(); ++i)
		{
			csv_row row;
			row.reserve(feature_count() + 2);
			row.push_back(ids.empty() ? std::to_string(i) : ids[i]);
			row.push_back(class_names[static_cast<std::size_t>(labels[i])]);
			for (Eigen::Index j = 0; j < inputs.cols(); ++j)
				row.push_back(fmt::format("{}", inputs(static_cast<Eigen::Index>(i), j)));
			table.push_back(std::move(row));
		}

		write_csv(path, table);
	}
};

namespace detail
{
inline constexpr std::size_t k_wdbc_feature_count = 30;
inline constexpr std::size_t k_wdbc_column_count = k_wdbc_feature_count + 2;

template <typename T>
[[nodiscard]] T cell(const rapidcsv::Document& document, std::size_t column, std::size_t row,
					 const std::filesystem::path& path)
{
	try
	{
		return document.GetCell<T>(column, row);
	}
	catch (const std::exception&)
	{
		throw std::runtime_error(
			fmt::format("load_wdbc: '{}' row {}, column {} is missing or malformed", path.string(),
						row + 1, column + 1));
	}
}
} // namespace detail

[[nodiscard]] inline dataset load_wdbc(const std::filesystem::path& path)
{
	if (!std::filesystem::is_regular_file(path))
		throw std::runtime_error(fmt::format("load_wdbc: cannot open '{}'", path.string()));

	const rapidcsv::Document document(
		path.string(), rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(',', true),
		rapidcsv::ConverterParams(), rapidcsv::LineReaderParams(false, '#', true));

	const std::size_t rows = document.GetRowCount();
	if (rows == 0)
		throw std::runtime_error(fmt::format("load_wdbc: '{}' contains no data", path.string()));
	if (document.GetColumnCount() != detail::k_wdbc_column_count)
		throw std::runtime_error(fmt::format("load_wdbc: '{}' has {} columns, expected {}",
											 path.string(), document.GetColumnCount(),
											 detail::k_wdbc_column_count));

	dataset dataset;
	dataset.inputs = matrix::Zero(static_cast<Eigen::Index>(rows),
								  static_cast<Eigen::Index>(detail::k_wdbc_feature_count));
	dataset.labels.reserve(rows);
	dataset.ids.reserve(rows);

	for (std::size_t i = 0; i < rows; ++i)
	{
		dataset.ids.push_back(detail::cell<std::string>(document, 0, i, path));

		const std::string diagnosis = detail::cell<std::string>(document, 1, i, path);
		if (diagnosis == "M")
			dataset.labels.push_back(1);
		else if (diagnosis == "B")
			dataset.labels.push_back(0);
		else
			throw std::runtime_error(fmt::format(
				"load_wdbc: row {}: '{}' is not a diagnosis (expected M or B)", i + 1, diagnosis));

		for (std::size_t j = 0; j < detail::k_wdbc_feature_count; ++j)
			dataset.inputs(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
				detail::cell<double>(document, j + 2, i, path);
	}

	return dataset;
}
} // namespace mlp
