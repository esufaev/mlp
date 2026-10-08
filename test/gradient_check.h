#pragma once

#include "mlp/mlp.h"

#include <cmath>
#include <cstddef>
#include <functional>

namespace test_support
{

    inline constexpr double k_step = 1e-6;

    using scalar_graph = std::function<mlp::tensor(const mlp::tensor&)>;

    inline mlp::matrix numeric_gradient(const std::function<double(const mlp::matrix&)>& scalar_of,
                                        const mlp::matrix& point, double step = k_step)
    {
        mlp::matrix result = mlp::matrix::Zero(point.rows(), point.cols());
        mlp::matrix probe = point;

        for (Eigen::Index i = 0; i < point.rows(); ++i)
            for (Eigen::Index j = 0; j < point.cols(); ++j)
            {
                const double original = point(i, j);

                probe(i, j) = original + step;
                const double plus = scalar_of(probe);

                probe(i, j) = original - step;
                const double minus = scalar_of(probe);

                probe(i, j) = original;
                result(i, j) = (plus - minus) / (2.0 * step);
            }

        return result;
    }

    inline mlp::matrix autograd_gradient(const scalar_graph& graph, const mlp::matrix& point)
    {
        const mlp::tensor input = mlp::tensor::parameter(point);
        graph(input).backward();
        return input.grad();
    }

    inline mlp::matrix numeric_graph_gradient(const scalar_graph& graph, const mlp::matrix& point)
    {
        return numeric_gradient(
            [&](const mlp::matrix& probe) {
                const mlp::no_grad guard;
                return graph(mlp::tensor::constant(probe)).value()(0, 0);
            },
            point);
    }

    inline mlp::matrix numeric_parameter_gradient(mlp::matrix& parameter,
                                                  const std::function<double()>& scalar_of,
                                                  double step = k_step)
    {
        const mlp::matrix original = parameter;
        mlp::matrix result = mlp::matrix::Zero(parameter.rows(), parameter.cols());

        for (Eigen::Index i = 0; i < parameter.rows(); ++i)
            for (Eigen::Index j = 0; j < parameter.cols(); ++j)
            {
                const double value = original(i, j);

                parameter(i, j) = value + step;
                const double plus = scalar_of();

                parameter(i, j) = value - step;
                const double minus = scalar_of();

                parameter(i, j) = value;
                result(i, j) = (plus - minus) / (2.0 * step);
            }

        parameter = original;
        return result;
    }

    inline double max_difference(const mlp::matrix& left, const mlp::matrix& right)
    {
        return (left - right).cwiseAbs().maxCoeff();
    }

    inline mlp::matrix patterned(std::size_t rows, std::size_t cols, double scale = 1.0)
    {
        mlp::matrix result = mlp::matrix::Zero(static_cast<Eigen::Index>(rows),
                                               static_cast<Eigen::Index>(cols));
        for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t j = 0; j < cols; ++j)
                result(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
                    scale * std::sin(static_cast<double>(3 * i + 7 * j + 1));
        return result;
    }

    inline mlp::matrix positive(std::size_t rows, std::size_t cols, double base = 2.0)
    {
        return patterned(rows, cols).array() + base;
    }
}
