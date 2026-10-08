#pragma once

#include <rapidcsv.h>

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace mlp
{
using csv_row = std::vector<std::string>;

using csv_table = std::vector<csv_row>;

inline void write_csv(const std::filesystem::path& path, const csv_table& table)
{
	if (path.has_parent_path())
		std::filesystem::create_directories(path.parent_path());

	rapidcsv::Document document(std::string(), rapidcsv::LabelParams(-1, -1),
								rapidcsv::SeparatorParams(','));

	for (std::size_t i = 0; i < table.size(); ++i)
		document.SetRow<std::string>(i, table[i]);

	document.Save(path.string());
}
} // namespace mlp
