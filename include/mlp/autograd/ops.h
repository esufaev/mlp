#pragma once

#include "mlp/autograd/tensor.h"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mlp
{
namespace detail
{
using gradient_rule =
	std::function<void(const matrix&, tensor_node&, const std::vector<tensor_node*>&)>;

inline void require_defined(const tensor& value, const char* who)
{
	if (!value.defined())
		throw std::logic_error(fmt::format("{}: the operand is empty", who));
}

[[nodiscard]] inline tensor make_node(matrix value, const char* op, std::vector<tensor> inputs,
									  gradient_rule rule)
{
	bool tracked = false;
	if (grad_enabled())
		for (const tensor& input : inputs)
			if (input.requires_grad())
				tracked = true;

	if (!tracked)
		return tensor::constant(std::move(value));

	auto node = std::make_shared<tensor_node>();
	node->grad = matrix::Zero(value.rows(), value.cols());
	node->value = std::move(value);
	node->requires_grad = true;
	node->op = op;

	node->inputs.reserve(inputs.size());
	for (const tensor& input : inputs)
		node->inputs.push_back(input.node());

	std::vector<tensor_node*> raw;
	raw.reserve(node->inputs.size());
	for (const node_pointer& each : node->inputs)
		raw.push_back(each.get());

	node->propagate = [self = node.get(), raw = std::move(raw),
					   rule = std::move(rule)](const matrix& outer) { rule(outer, *self, raw); };

	return tensor(std::move(node));
}
} // namespace detail
[[nodiscard]] inline tensor matmul(const tensor& left, const tensor& right)
{
	detail::require_defined(left, "matmul");
	detail::require_defined(right, "matmul");

	if (left.cols() != right.rows())
		throw std::invalid_argument(fmt::format("matmul: cannot multiply {} by {}",
												shape_string(left.value()),
												shape_string(right.value())));

	return detail::make_node(
		left.value() * right.value(), "matmul", {left, right},
		[](const matrix& outer, tensor_node&, const std::vector<tensor_node*>& inputs)
		{
			if (inputs[0]->requires_grad)
				inputs[0]->grad += outer * inputs[1]->value.transpose();
			if (inputs[1]->requires_grad)
				inputs[1]->grad += inputs[0]->value.transpose() * outer;
		});
}

[[nodiscard]] inline tensor operator+(const tensor& left, const tensor& right)
{
	detail::require_defined(left, "operator+");
	detail::require_defined(right, "operator+");

	const bool broadcast = right.rows() == 1 && left.rows() != 1 && left.cols() == right.cols();

	if (!broadcast && (left.rows() != right.rows() || left.cols() != right.cols()))
		throw std::invalid_argument(fmt::format("operator+: cannot add {} to {}",
												shape_string(right.value()),
												shape_string(left.value())));

	matrix value = broadcast ? matrix(left.value().rowwise() + right.value().row(0))
							 : matrix(left.value() + right.value());

	return detail::make_node(
		std::move(value), "add", {left, right},
		[broadcast](const matrix& outer, tensor_node&, const std::vector<tensor_node*>& inputs)
		{
			if (inputs[0]->requires_grad)
				inputs[0]->grad += outer;
			if (inputs[1]->requires_grad)
			{
				if (broadcast)
					inputs[1]->grad += outer.colwise().sum();
				else
					inputs[1]->grad += outer;
			}
		});
}

[[nodiscard]] inline tensor operator-(const tensor& left, const tensor& right)
{
	detail::require_defined(left, "operator-");
	detail::require_defined(right, "operator-");

	if (left.rows() != right.rows() || left.cols() != right.cols())
		throw std::invalid_argument(fmt::format("operator-: cannot subtract {} from {}",
												shape_string(right.value()),
												shape_string(left.value())));

	return detail::make_node(
		left.value() - right.value(), "subtract", {left, right},
		[](const matrix& outer, tensor_node&, const std::vector<tensor_node*>& inputs)
		{
			if (inputs[0]->requires_grad)
				inputs[0]->grad += outer;
			if (inputs[1]->requires_grad)
				inputs[1]->grad -= outer;
		});
}

[[nodiscard]] inline tensor operator*(const tensor& operand, double scalar)
{
	detail::require_defined(operand, "operator*");

	return detail::make_node(
		operand.value() * scalar, "scale", {operand},
		[scalar](const matrix& outer, tensor_node&, const std::vector<tensor_node*>& inputs)
		{
			if (inputs[0]->requires_grad)
				inputs[0]->grad += outer * scalar;
		});
}

[[nodiscard]] inline tensor operator*(double scalar, const tensor& operand)
{ return operand * scalar; }

[[nodiscard]] inline tensor operator-(const tensor& operand) { return operand * -1.0; }

[[nodiscard]] inline tensor hadamard(const tensor& left, const tensor& right)
{
	detail::require_defined(left, "hadamard");
	detail::require_defined(right, "hadamard");

	if (left.rows() != right.rows() || left.cols() != right.cols())
		throw std::invalid_argument(fmt::format("hadamard: shape mismatch, {} vs {}",
												shape_string(left.value()),
												shape_string(right.value())));

	return detail::make_node(
		left.value().cwiseProduct(right.value()), "hadamard", {left, right},
		[](const matrix& outer, tensor_node&, const std::vector<tensor_node*>& inputs)
		{
			if (inputs[0]->requires_grad)
				inputs[0]->grad += outer.cwiseProduct(inputs[1]->value);
			if (inputs[1]->requires_grad)
				inputs[1]->grad += outer.cwiseProduct(inputs[0]->value);
		});
}

[[nodiscard]] inline tensor sigmoid(const tensor& operand)
{
	detail::require_defined(operand, "sigmoid");

	return detail::make_node(
		1.0 / (1.0 + (-operand.value().array()).exp()), "sigmoid", {operand},
		[](const matrix& outer, tensor_node& self, const std::vector<tensor_node*>& inputs)
		{
			if (inputs[0]->requires_grad)
				inputs[0]->grad +=
					(outer.array() * self.value.array() * (1.0 - self.value.array())).matrix();
		});
}

[[nodiscard]] inline tensor relu(const tensor& operand)
{
	detail::require_defined(operand, "relu");

	return detail::make_node(
		operand.value().array().max(0.0), "relu", {operand},
		[](const matrix& outer, tensor_node&, const std::vector<tensor_node*>& inputs)
		{
			if (inputs[0]->requires_grad)
				inputs[0]->grad +=
					(outer.array() * (inputs[0]->value.array() > 0.0).cast<double>()).matrix();
		});
}

[[nodiscard]] inline tensor tanh(const tensor& operand)
{
	detail::require_defined(operand, "tanh");

	return detail::make_node(
		operand.value().array().tanh(), "tanh", {operand},
		[](const matrix& outer, tensor_node& self, const std::vector<tensor_node*>& inputs)
		{
			if (inputs[0]->requires_grad)
				inputs[0]->grad += (outer.array() * (1.0 - self.value.array().square())).matrix();
		});
}

[[nodiscard]] inline tensor softmax_rows(const tensor& operand)
{
	detail::require_defined(operand, "softmax_rows");

	const matrix& input = operand.value();
	if (input.cols() == 0)
		throw std::invalid_argument("softmax_rows: the input has no columns");

	const matrix exponentials = (input.colwise() - input.rowwise().maxCoeff()).array().exp();
	const matrix totals = exponentials.rowwise().sum();

	return detail::make_node(
		exponentials.array().colwise() / totals.col(0).array(), "softmax", {operand},
		[](const matrix& outer, tensor_node& self, const std::vector<tensor_node*>& inputs)
		{
			if (!inputs[0]->requires_grad)
				return;

			const matrix weighted = outer.cwiseProduct(self.value);
			inputs[0]->grad += (weighted.array() - self.value.array().colwise() *
													   weighted.rowwise().sum().col(0).array())
								   .matrix();
		});
}

[[nodiscard]] inline tensor log(const tensor& operand)
{
	detail::require_defined(operand, "log");

	return detail::make_node(
		operand.value().array().log(), "log", {operand},
		[](const matrix& outer, tensor_node&, const std::vector<tensor_node*>& inputs)
		{
			if (inputs[0]->requires_grad)
				inputs[0]->grad += (outer.array() / inputs[0]->value.array()).matrix();
		});
}

[[nodiscard]] inline tensor clamp(const tensor& operand, double lowest, double highest)
{
	detail::require_defined(operand, "clamp");

	if (lowest > highest)
		throw std::invalid_argument(
			fmt::format("clamp: the range [{}, {}] is empty", lowest, highest));

	return detail::make_node(operand.value().array().max(lowest).min(highest), "clamp", {operand},
							 [lowest, highest](const matrix& outer, tensor_node&,
											   const std::vector<tensor_node*>& inputs)
							 {
								 if (inputs[0]->requires_grad)
									 inputs[0]->grad +=
										 (outer.array() * ((inputs[0]->value.array() >= lowest) &&
														   (inputs[0]->value.array() <= highest))
															  .cast<double>())
											 .matrix();
							 });
}

[[nodiscard]] inline tensor sum(const tensor& operand)
{
	detail::require_defined(operand, "sum");

	return detail::make_node(
		matrix::Constant(1, 1, operand.value().sum()), "sum", {operand},
		[](const matrix& outer, tensor_node&, const std::vector<tensor_node*>& inputs)
		{
			if (inputs[0]->requires_grad)
				inputs[0]->grad.array() += outer(0, 0);
		});
}

[[nodiscard]] inline tensor mean(const tensor& operand)
{
	detail::require_defined(operand, "mean");

	if (operand.value().size() == 0)
		throw std::invalid_argument("mean: the input is empty");

	const auto count = static_cast<double>(operand.value().size());

	return detail::make_node(
		matrix::Constant(1, 1, operand.value().sum() / count), "mean", {operand},
		[count](const matrix& outer, tensor_node&, const std::vector<tensor_node*>& inputs)
		{
			if (inputs[0]->requires_grad)
				inputs[0]->grad.array() += outer(0, 0) / count;
		});
}

[[nodiscard]] inline tensor column(const tensor& operand, std::size_t index)
{
	detail::require_defined(operand, "column");

	if (index >= operand.cols())
		throw std::out_of_range(
			fmt::format("column: {} is outside of {}", index, shape_string(operand.value())));

	const auto target = static_cast<Eigen::Index>(index);

	return detail::make_node(
		operand.value().col(target), "column", {operand},
		[target](const matrix& outer, tensor_node&, const std::vector<tensor_node*>& inputs)
		{
			if (inputs[0]->requires_grad)
				inputs[0]->grad.col(target) += outer.col(0);
		});
}

[[nodiscard]] inline tensor concat_cols(const tensor& left, const tensor& right)
{
	detail::require_defined(left, "concat_cols");
	detail::require_defined(right, "concat_cols");

	if (left.rows() != right.rows())
		throw std::invalid_argument(fmt::format("concat_cols: cannot join {} and {}",
												shape_string(left.value()),
												shape_string(right.value())));

	const Eigen::Index split = left.value().cols();
	matrix value(left.value().rows(), split + right.value().cols());
	value.leftCols(split) = left.value();
	value.rightCols(right.value().cols()) = right.value();

	return detail::make_node(
		std::move(value), "concat", {left, right},
		[split](const matrix& outer, tensor_node&, const std::vector<tensor_node*>& inputs)
		{
			if (inputs[0]->requires_grad)
				inputs[0]->grad += outer.leftCols(split);
			if (inputs[1]->requires_grad)
				inputs[1]->grad += outer.rightCols(outer.cols() - split);
		});
}
} // namespace mlp
