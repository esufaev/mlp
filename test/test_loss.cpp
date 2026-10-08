#include "mlp/mlp.h"

#include "gradient_check.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>
#include <string>

using Catch::Approx;
using mlp::matrix;
using mlp::tensor;
using test_support::max_difference;
using test_support::numeric_gradient;

TEST_CASE("binary cross-entropy matches the hand-computed value", "[loss]")
{

    const matrix predictions = matrix{{0.1, 0.9}, {0.8, 0.2}};
    const matrix targets = matrix{{0.0, 1.0}, {1.0, 0.0}};

    const double expected = -(std::log(0.9) + std::log(0.8)) / 2.0;
    REQUIRE(mlp::binary_cross_entropy{}.value(predictions, targets) == Approx(expected));
}

TEST_CASE("a perfect prediction costs almost nothing and a wrong one costs a lot", "[loss]")
{
    const mlp::binary_cross_entropy loss;
    const matrix targets = matrix{{0.0, 1.0}};

    const double good = loss.value(matrix{{0.001, 0.999}}, targets);
    const double bad = loss.value(matrix{{0.999, 0.001}}, targets);

    REQUIRE(good < 0.01);
    REQUIRE(bad > 6.0);
}

TEST_CASE("a certain and wrong prediction stays finite", "[loss]")
{
    const mlp::binary_cross_entropy loss;
    const matrix targets = matrix{{0.0, 1.0}};
    const tensor predictions = tensor::parameter(matrix{{1.0, 0.0}});

    const tensor value = loss(predictions, targets);
    REQUIRE(std::isfinite(value.value()(0, 0)));

    value.backward();
    REQUIRE(predictions.grad().allFinite());
}

TEST_CASE("with two classes, binary and categorical cross-entropy agree", "[loss]")
{
    const matrix predictions = matrix{{0.3, 0.7}, {0.65, 0.35}, {0.5, 0.5}};
    const matrix targets = matrix{{0.0, 1.0}, {1.0, 0.0}, {0.0, 1.0}};

    REQUIRE(mlp::binary_cross_entropy{}.value(predictions, targets) ==
            Approx(mlp::categorical_cross_entropy{}.value(predictions, targets)));
}

TEST_CASE("every loss gradient matches a numeric gradient", "[loss][gradient]")
{
    const matrix predictions = matrix{{0.3, 0.7}, {0.65, 0.35}, {0.42, 0.58}};
    const matrix targets = matrix{{0.0, 1.0}, {1.0, 0.0}, {0.0, 1.0}};

    for (const std::string name : {"binaryCrossentropy", "categoricalCrossentropy", "mse"})
    {
        INFO("loss: " << name);
        const std::unique_ptr<mlp::loss> loss = mlp::make_loss(name);

        const tensor tracked = tensor::parameter(predictions);
        (*loss)(tracked, targets).backward();

        const matrix numeric = numeric_gradient(
            [&](const matrix& probe) { return loss->value(probe, targets); }, predictions);

        REQUIRE(max_difference(tracked.grad(), numeric) < 1e-6);
    }
}

TEST_CASE("softmax and cross-entropy together give (p - y) / batch", "[loss][gradient]")
{

    const matrix targets = mlp::one_hot({1, 0, 1}, 2);
    const tensor scores = tensor::parameter(matrix{{0.4, -1.2}, {-0.7, 0.9}, {2.0, 0.1}});

    const tensor probabilities = mlp::softmax_rows(scores);
    const mlp::binary_cross_entropy loss;

    loss(probabilities, targets).backward();

    const auto batch = static_cast<double>(scores.rows());
    for (std::size_t i = 0; i < scores.rows(); ++i)
        for (std::size_t j = 0; j < scores.cols(); ++j)
            REQUIRE(scores.grad()(i, j) ==
                    Approx((probabilities.value()(i, j) - targets(i, j)) / batch).margin(1e-9));
}

TEST_CASE("losses validate their input", "[loss]")
{
    const mlp::binary_cross_entropy binary;
    REQUIRE_THROWS_AS(binary.value(matrix::Zero(2, 2), matrix::Zero(3, 2)), std::invalid_argument);
    REQUIRE_THROWS_AS(binary.value(matrix::Zero(2, 3), matrix::Zero(2, 3)), std::invalid_argument);
    REQUIRE_THROWS_AS(binary.value(matrix::Zero(0, 2), matrix::Zero(0, 2)), std::invalid_argument);
}

TEST_CASE("losses are looked up by their command line names", "[loss]")
{
    REQUIRE(mlp::make_loss("binaryCrossentropy")->name() == "binaryCrossentropy");
    REQUIRE(mlp::make_loss("mse")->name() == "mse");
    REQUIRE_THROWS_AS(mlp::make_loss("hinge"), std::invalid_argument);
}

TEST_CASE("a loss is a scalar node in the graph", "[loss]")
{
    const matrix targets = mlp::one_hot({0, 1}, 2);
    const tensor predictions = tensor::parameter(matrix{{0.6, 0.4}, {0.3, 0.7}});

    const tensor value = mlp::binary_cross_entropy{}(predictions, targets);

    REQUIRE(value.rows() == 1);
    REQUIRE(value.cols() == 1);
    REQUIRE(value.requires_grad());
    REQUIRE(value.graph_size() > 1);
}
