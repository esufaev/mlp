#pragma once

#include <Eigen/Core>

#include <fmt/format.h>

#include <string>

namespace mlp
{
using matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

[[nodiscard]] inline std::string shape_string(const matrix& value)
{ return fmt::format("[{} x {}]", value.rows(), value.cols()); }
} // namespace mlp
