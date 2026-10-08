#pragma once

#include "mlp/autograd/ops.h"
#include "mlp/autograd/tensor.h"
#include "mlp/core/linalg.h"

#include <fmt/format.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace mlp
{
    namespace detail
    {
        inline constexpr double k_loss_epsilon = 1e-12;

        inline void require_matching_shapes(const matrix& predictions, const matrix& targets, const char* who)
        {
            if (predictions.rows() != targets.rows() || predictions.cols() != targets.cols())
                throw std::invalid_argument(fmt::format("{}: predictions {} do not match targets {}", who,
                                                        shape_string(predictions), shape_string(targets)));
            if (predictions.rows() == 0)
                throw std::invalid_argument(fmt::format("{}: empty batch", who));
        }

        inline void require_two_columns(const matrix& predictions, const char* who)
        {
            if (predictions.cols() != 2)
                throw std::invalid_argument(
                    fmt::format("{}: expects a two-class output, got {} columns", who, predictions.cols()));
        }
    }

    class loss
    {
    public:
        loss() = default;
        loss(const loss&) = default;
        loss(loss&&) = default;
        loss& operator=(const loss&) = default;
        loss& operator=(loss&&) = default;
        virtual ~loss() = default;

        [[nodiscard]] virtual tensor operator()(const tensor& predictions, const matrix& targets) const = 0;

        [[nodiscard]] virtual std::string name() const = 0;

        [[nodiscard]] double value(const tensor& predictions, const matrix& targets) const
        {
            const no_grad guard;
            return (*this)(predictions, targets).value()(0, 0);
        }

        [[nodiscard]] double value(const matrix& predictions, const matrix& targets) const
        {
            return value(tensor::constant(predictions), targets);
        }
    };

    class binary_cross_entropy final : public loss
    {
    public:
        [[nodiscard]] tensor operator()(const tensor& predictions, const matrix& targets) const override
        {
            detail::require_matching_shapes(predictions.value(), targets, "binary_cross_entropy");
            detail::require_two_columns(predictions.value(), "binary_cross_entropy");

            const Eigen::Index rows = predictions.value().rows();

            matrix positive = targets.col(1);
            matrix negative = matrix::Constant(rows, 1, 1.0) - positive;

            const tensor probability = clamp(column(predictions, 1), detail::k_loss_epsilon,
                                             1.0 - detail::k_loss_epsilon);
            const tensor complement =
                tensor::constant(matrix::Constant(rows, 1, 1.0)) - probability;

            const tensor total = sum(hadamard(tensor::constant(std::move(positive)), log(probability)) +
                                     hadamard(tensor::constant(std::move(negative)), log(complement)));

            return total * (-1.0 / static_cast<double>(rows));
        }

        [[nodiscard]] std::string name() const override { return "binaryCrossentropy"; }
    };

    class categorical_cross_entropy final : public loss
    {
    public:
        [[nodiscard]] tensor operator()(const tensor& predictions, const matrix& targets) const override
        {
            detail::require_matching_shapes(predictions.value(), targets, "categorical_cross_entropy");

            const tensor probabilities =
                clamp(predictions, detail::k_loss_epsilon, 1.0 - detail::k_loss_epsilon);
            const tensor total = sum(hadamard(tensor::constant(targets), log(probabilities)));

            return total * (-1.0 / static_cast<double>(predictions.rows()));
        }

        [[nodiscard]] std::string name() const override { return "categoricalCrossentropy"; }
    };

    class mean_squared_error final : public loss
    {
    public:
        [[nodiscard]] tensor operator()(const tensor& predictions, const matrix& targets) const override
        {
            detail::require_matching_shapes(predictions.value(), targets, "mean_squared_error");

            const tensor difference = predictions - tensor::constant(targets);

            return sum(hadamard(difference, difference)) * (1.0 / static_cast<double>(predictions.rows()));
        }

        [[nodiscard]] std::string name() const override { return "mse"; }
    };

    [[nodiscard]] inline std::unique_ptr<loss> make_loss(std::string_view name)
    {
        if (name == "binaryCrossentropy" || name == "binary_crossentropy" || name == "bce")
            return std::make_unique<binary_cross_entropy>();
        if (name == "categoricalCrossentropy" || name == "categorical_crossentropy" || name == "cce")
            return std::make_unique<categorical_cross_entropy>();
        if (name == "mse" || name == "meanSquaredError")
            return std::make_unique<mean_squared_error>();

        throw std::invalid_argument(fmt::format(
            "unknown loss '{}' (expected binaryCrossentropy, categoricalCrossentropy or mse)", name));
    }
}
