#pragma once

#include "mlp/data/preprocessing.h"
#include "mlp/nn/activations.h"
#include "mlp/nn/dense.h"
#include "mlp/nn/module.h"
#include "mlp/nn/network.h"
#include "mlp/nn/residual.h"
#include "mlp/nn/sequential.h"

#include <fmt/format.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <istream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mlp
{
inline constexpr int k_model_format_version = 2;

struct model
{
	network net;
	standard_scaler scaler;
	std::vector<std::string> class_names{"B", "M"};
	std::string loss_name{"binaryCrossentropy"};
	std::uint64_t seed{42};
};

namespace detail
{
inline void expect_keyword(std::istream& stream, const std::string& expected)
{
	std::string keyword;
	if (!(stream >> keyword) || keyword != expected)
		throw std::runtime_error(
			fmt::format("load_model: expected '{}', found '{}'", expected, keyword));
}

template <typename T> [[nodiscard]] T read_value(std::istream& stream, const std::string& keyword)
{
	expect_keyword(stream, keyword);
	T value{};
	if (!(stream >> value))
		throw std::runtime_error(fmt::format("load_model: '{}' has no value", keyword));
	return value;
}

inline void write_module(std::ostream& stream, const module& current)
{
	const std::string config = current.config();
	stream << current.type();
	if (!config.empty())
		stream << ' ' << config;
	stream << '\n';
	current.write_parameters(stream);

	if (const auto* container = dynamic_cast<const sequential*>(&current))
		for (std::size_t i = 0; i < container->size(); ++i)
			write_module(stream, container->at(i));
	else if (const auto* block = dynamic_cast<const residual*>(&current))
		for (std::size_t i = 0; i < block->size(); ++i)
			write_module(stream, block->branch().at(i));
}

[[nodiscard]] inline std::size_t read_child_count(std::istream& stream, const std::string& type)
{
	std::size_t count = 0;
	if (!(stream >> count))
		throw std::runtime_error(fmt::format("load_model: '{}' has no child count", type));
	return count;
}

[[nodiscard]] inline std::unique_ptr<module> read_module(std::istream& stream)
{
	std::string type;
	if (!(stream >> type))
		throw std::runtime_error("load_model: the module tree is truncated");

	if (type == "dense")
	{
		std::size_t input_size = 0;
		std::size_t output_size = 0;
		if (!(stream >> input_size >> output_size))
			throw std::runtime_error("load_model: a dense module has no shape");

		auto result = std::make_unique<dense>(input_size, output_size);
		result->read_parameters(stream);
		return result;
	}

	if (type == "sequential")
	{
		const std::size_t count = read_child_count(stream, type);
		auto result = std::make_unique<sequential>();
		for (std::size_t i = 0; i < count; ++i)
			result->add(read_module(stream));
		return result;
	}

	if (type == "residual")
	{
		const std::size_t count = read_child_count(stream, type);
		auto result = std::make_unique<residual>();
		for (std::size_t i = 0; i < count; ++i)
			result->add(read_module(stream));
		return result;
	}

	if (std::unique_ptr<module> activation = make_activation_by_type(type))
		return activation;

	throw std::runtime_error(fmt::format("load_model: unknown module type '{}'", type));
}
} // namespace detail

inline void save_model(const model& model, const std::filesystem::path& path)
{
	if (path.has_parent_path())
		std::filesystem::create_directories(path.parent_path());

	std::ofstream output(path);
	if (!output)
		throw std::runtime_error(fmt::format("save_model: cannot write to '{}'", path.string()));

	output << "mlp-model " << k_model_format_version << '\n';
	output << "seed " << model.seed << '\n';
	output << "loss " << model.loss_name << '\n';

	output << "classes " << model.class_names.size() << '\n';
	output << "class_names";
	for (const std::string& name : model.class_names)
		output << ' ' << name;
	output << '\n';

	model.scaler.save(output);

	detail::write_module(output, model.net.root());

	if (!output)
		throw std::runtime_error(fmt::format("save_model: writing '{}' failed", path.string()));
}

[[nodiscard]] inline model load_model(const std::filesystem::path& path)
{
	std::ifstream input(path);
	if (!input)
		throw std::runtime_error(fmt::format("load_model: cannot open '{}'", path.string()));

	const auto version = detail::read_value<int>(input, "mlp-model");
	if (version != k_model_format_version)
		throw std::runtime_error(
			fmt::format("load_model: '{}' is format version {}, this build reads {}", path.string(),
						version, k_model_format_version));

	model model;
	model.seed = detail::read_value<std::uint64_t>(input, "seed");
	model.loss_name = detail::read_value<std::string>(input, "loss");

	const auto class_count = detail::read_value<std::size_t>(input, "classes");
	detail::expect_keyword(input, "class_names");
	model.class_names.clear();
	for (std::size_t i = 0; i < class_count; ++i)
	{
		std::string name;
		if (!(input >> name))
			throw std::runtime_error(fmt::format("load_model: truncated class names at {}", i));
		model.class_names.push_back(std::move(name));
	}

	model.scaler.load(input);

	const auto module_count = detail::read_value<std::size_t>(input, "sequential");
	for (std::size_t i = 0; i < module_count; ++i)
		model.net.add(detail::read_module(input));

	return model;
}
} // namespace mlp
