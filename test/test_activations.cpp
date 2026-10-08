#include "mlp/mlp.h"

#include "gradient_check.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>
#include <vector>

using Catch::Approx;
using mlp::matrix;
using mlp::tensor;
using test_support::max_difference;
using test_support::patterned;

namespace
{
    matrix forward_of(mlp::activation kind, const matrix& input)
    {
        const std::unique_ptr<mlp::module> activation = mlp::make_activation(kind);
        return activation->forward(tensor::constant(input)).value();
    }
}

TEST_CASE("sigmoid matches its definition", "[activations]")
{
    const matrix output = forward_of(mlp::activation::sigmoid,
                                     matrix{{0.0, 1.0, -1.0, 20.0, -20.0}});

    REQUIRE(output(0, 0) == Approx(0.5));
    REQUIRE(output(0, 1) == Approx(1.0 / (1.0 + std::exp(-1.0))));
    REQUIRE(output(0, 2) == Approx(1.0 / (1.0 + std::exp(1.0))));
    REQUIRE(output(0, 3) > 0.999);
    REQUIRE(output(0, 4) < 0.001);
}

TEST_CASE("relu and tanh match their definitions", "[activations]")
{
    const matrix relu_output = forward_of(mlp::activation::relu, matrix{{-2.0, 0.0, 3.0}});
    REQUIRE(relu_output(0, 0) == Approx(0.0));
    REQUIRE(relu_output(0, 1) == Approx(0.0));
    REQUIRE(relu_output(0, 2) == Approx(3.0));

    const matrix tanh_output = forward_of(mlp::activation::tanh, matrix{{-1.0, 0.0, 2.0}});
    REQUIRE(tanh_output(0, 0) == Approx(std::tanh(-1.0)));
    REQUIRE(tanh_output(0, 1) == Approx(0.0));
    REQUIRE(tanh_output(0, 2) == Approx(std::tanh(2.0)));
}

TEST_CASE("softmax rows are probability distributions", "[activations]")
{
    const matrix output =
        forward_of(mlp::activation::softmax, matrix{{1.0, 2.0, 3.0}, {0.0, 0.0, 0.0}});

    for (std::size_t i = 0; i < output.rows(); ++i)
    {
        double total = 0.0;
        for (std::size_t j = 0; j < output.cols(); ++j)
        {
            REQUIRE(output(i, j) > 0.0);
            REQUIRE(output(i, j) < 1.0);
            total += output(i, j);
        }
        REQUIRE(total == Approx(1.0));
    }

    REQUIRE(output(1, 0) == Approx(1.0 / 3.0));

    REQUIRE(output(0, 2) > output(0, 1));
}

TEST_CASE("softmax survives large scores", "[activations]")
{
    const matrix output =
        forward_of(mlp::activation::softmax, matrix{{1000.0, 1001.0}, {-1000.0, -1001.0}});

    REQUIRE(output.allFinite());
    REQUIRE(output(0, 0) + output(0, 1) == Approx(1.0));
    REQUIRE(output(1, 0) > output(1, 1));

    REQUIRE(output(0, 1) == Approx(std::exp(1.0) / (1.0 + std::exp(1.0))));
}

TEST_CASE("every activation's gradient matches a numeric gradient", "[activations][gradient]")
{
    const matrix input = patterned(4, 5, 1.5);
    const matrix seed = patterned(4, 5, 0.7);

    const std::vector<mlp::activation> activations{mlp::activation::sigmoid, mlp::activation::relu,
                                                   mlp::activation::tanh, mlp::activation::softmax};

    for (const mlp::activation activation : activations)
    {
        INFO("activation: " << mlp::to_string(activation));

        const std::unique_ptr<mlp::module> layer = mlp::make_activation(activation);
        const test_support::scalar_graph graph = [&](const tensor& probe) {
            return mlp::sum(mlp::hadamard(layer->forward(probe), tensor::constant(seed)));
        };

        REQUIRE(max_difference(test_support::autograd_gradient(graph, input),
                               test_support::numeric_graph_gradient(graph, input)) < 1e-6);
    }
}

TEST_CASE("an activation module has no parameters of its own", "[activations]")
{
    const std::unique_ptr<mlp::module> layer = mlp::make_activation(mlp::activation::relu);

    REQUIRE(layer->parameters().empty());
    REQUIRE(layer->type() == "relu");
    REQUIRE(layer->config().empty());
}

TEST_CASE("softmax needs at least one column", "[activations]")
{
    const std::unique_ptr<mlp::module> layer = mlp::make_activation(mlp::activation::softmax);
    REQUIRE_THROWS_AS(layer->forward(tensor::constant(matrix::Zero(2, 0))), std::invalid_argument);
}

TEST_CASE("activations are named both ways", "[activations]")
{
    REQUIRE(mlp::activation_from_string("sigmoid") == mlp::activation::sigmoid);
    REQUIRE(mlp::activation_from_string("softmax") == mlp::activation::softmax);
    REQUIRE(mlp::activation_from_string("relu") == mlp::activation::relu);
    REQUIRE(mlp::activation_from_string("ReLU") == mlp::activation::relu);
    REQUIRE(mlp::to_string(mlp::activation::relu) == "relu");
    REQUIRE_THROWS_AS(mlp::activation_from_string("elu"), std::invalid_argument);
    REQUIRE(mlp::make_activation_by_type("tanh") != nullptr);
    REQUIRE(mlp::make_activation_by_type("dense") == nullptr);
}
