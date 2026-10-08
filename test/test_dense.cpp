#include "mlp/mlp.h"

#include "gradient_check.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using Catch::Approx;
using mlp::matrix;
using mlp::tensor;
using test_support::max_difference;
using test_support::patterned;

TEST_CASE("a dense layer computes X W + b", "[dense]")
{
    mlp::dense layer(2, 2);
    layer.weights().mutable_value() = matrix{{1, 2}, {3, 4}};
    layer.bias().mutable_value() = matrix{{10, 20}};

    const matrix output = layer.forward(tensor::constant(matrix{{1, 1}, {2, 0}})).value();

    REQUIRE(output.rows() == 2);
    REQUIRE(output.cols() == 2);
    REQUIRE(output(0, 0) == Approx(1 * 1 + 1 * 3 + 10));
    REQUIRE(output(0, 1) == Approx(1 * 2 + 1 * 4 + 20));
    REQUIRE(output(1, 0) == Approx(2 * 1 + 0 * 3 + 10));
    REQUIRE(output(1, 1) == Approx(2 * 2 + 0 * 4 + 20));
}

TEST_CASE("the bias gradient is the column sum of the incoming gradient", "[dense]")
{
    mlp::dense layer(2, 2);
    const tensor seed = tensor::constant(matrix{{1, 2}, {3, 4}});

    mlp::sum(mlp::hadamard(layer.forward(tensor::constant(matrix{{1, 1}, {2, 0}})), seed)).backward();

    REQUIRE(layer.bias().grad()(0, 0) == Approx(4.0));
    REQUIRE(layer.bias().grad()(0, 1) == Approx(6.0));
}

TEST_CASE("gradients accumulate until they are cleared", "[dense]")
{
    mlp::dense layer(2, 2);
    const tensor input = tensor::constant(matrix{{1, 1}});

    mlp::sum(layer.forward(input)).backward();
    const double once = layer.bias().grad()(0, 0);
    REQUIRE(once == Approx(1.0));

    mlp::sum(layer.forward(input)).backward();
    REQUIRE(layer.bias().grad()(0, 0) == Approx(2.0 * once));

    layer.zero_grad();
    REQUIRE(layer.bias().grad()(0, 0) == Approx(0.0));
}

TEST_CASE("dense gradients match numeric gradients", "[dense][gradient]")
{
    mlp::rng rng(11);
    mlp::dense layer(4, 3, mlp::initializer::he_uniform, rng);

    const matrix input = patterned(5, 4, 1.2);
    const matrix seed = patterned(5, 3, 0.9);

    const auto loss_of = [&](const tensor& probe) {
        return mlp::sum(mlp::hadamard(layer.forward(probe), tensor::constant(seed)));
    };

    const auto scalar = [&]() {
        const mlp::no_grad guard;
        return loss_of(tensor::constant(input)).value()(0, 0);
    };

    const tensor tracked_input = tensor::parameter(input);
    layer.zero_grad();
    loss_of(tracked_input).backward();

    SECTION("with respect to the input")
    {
        REQUIRE(max_difference(tracked_input.grad(),
                               test_support::numeric_graph_gradient(loss_of, input)) < 1e-6);
    }

    SECTION("with respect to the weights")
    {
        const matrix numeric =
            test_support::numeric_parameter_gradient(layer.weights().mutable_value(), scalar);
        REQUIRE(max_difference(layer.weights().grad(), numeric) < 1e-6);
    }

    SECTION("with respect to the bias")
    {
        const matrix numeric = test_support::numeric_parameter_gradient(layer.bias().mutable_value(), scalar);
        REQUIRE(max_difference(layer.bias().grad(), numeric) < 1e-6);
    }
}

TEST_CASE("a dense layer reports its parameters and shape", "[dense]")
{
    mlp::dense layer(30, 24);

    REQUIRE(layer.type() == "dense");
    REQUIRE(layer.config() == "30 24");
    REQUIRE(layer.parameters().size() == 2);
    REQUIRE(layer.parameters()[0].value().size() == 30 * 24);
    REQUIRE(layer.parameters()[1].value().size() == 24);
    REQUIRE(layer.parameter_count() == 30 * 24 + 24);

    for (const tensor& parameter : layer.parameters())
        REQUIRE(parameter.requires_grad());
}

TEST_CASE("a dense layer rejects mismatched shapes", "[dense]")
{
    mlp::dense layer(3, 2);

    REQUIRE_THROWS_AS(layer.forward(tensor::constant(matrix::Zero(4, 5))), std::invalid_argument);
    REQUIRE_THROWS_AS(mlp::dense(0, 2), std::invalid_argument);
    REQUIRE_THROWS_AS(mlp::dense(2, 0), std::invalid_argument);
}

TEST_CASE("initializers produce the expected spread", "[dense][initializers]")
{
    mlp::rng rng(3);

    matrix zeros = matrix::Zero(10, 10);
    mlp::initialize(zeros, mlp::initializer::zeros, rng);
    REQUIRE(zeros.sum() == Approx(0.0));

    matrix he = matrix::Zero(100, 20);
    mlp::initialize(he, mlp::initializer::he_uniform, rng);
    const double limit = std::sqrt(6.0 / 100.0);
    REQUIRE(he.cwiseAbs().maxCoeff() <= limit);

    REQUIRE(mlp::initializer_from_string("heUniform") == mlp::initializer::he_uniform);
    REQUIRE(mlp::to_string(mlp::initializer::glorot_uniform) == "glorotUniform");
    REQUIRE_THROWS_AS(mlp::initializer_from_string("lecun"), std::invalid_argument);
}

TEST_CASE("a dense layer starts with zero bias", "[dense]")
{
    mlp::rng rng(7);
    const mlp::dense layer(5, 4, mlp::initializer::he_normal, rng);

    REQUIRE(layer.bias().value().sum() == Approx(0.0));
    REQUIRE(layer.weights().value().sum() != Approx(0.0));
}
