#include "mlp/mlp.h"

#include "gradient_check.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

using Catch::Approx;
using mlp::matrix;
using mlp::tensor;
using test_support::max_difference;
using test_support::patterned;
using test_support::positive;

TEST_CASE("a leaf carries its own gradient", "[autograd]")
{
    const tensor x = tensor::parameter(matrix{{1.0, 2.0, 3.0}});

    REQUIRE(x.requires_grad());
    REQUIRE(x.op() == "leaf");
    REQUIRE(x.grad().sum() == Approx(0.0));

    mlp::sum(x).backward();

    REQUIRE((x.grad().array() - 1.0).cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("the chain rule runs backwards through a tiny graph", "[autograd]")
{

    const tensor x = tensor::parameter(matrix::Constant(1, 1, 2.0));
    const tensor w = tensor::parameter(matrix::Constant(1, 1, 3.0));

    const tensor y = mlp::hadamard(w, x) + tensor::constant(matrix::Constant(1, 1, 1.0));
    const tensor loss = mlp::sum(mlp::hadamard(y, y));

    REQUIRE(y.value()(0, 0) == Approx(7.0));
    REQUIRE(loss.value()(0, 0) == Approx(49.0));

    loss.backward();

    REQUIRE(x.grad()(0, 0) == Approx(42.0));
    REQUIRE(w.grad()(0, 0) == Approx(28.0));
}

TEST_CASE("a node used by two branches sums both gradients", "[autograd][skip]")
{

    const matrix point = patterned(3, 4, 1.3);
    const tensor x = tensor::parameter(point);

    const tensor left = mlp::sigmoid(x);
    const tensor right = mlp::relu(x);

    mlp::sum(left + right).backward();

    matrix expected = matrix::Zero(point.rows(), point.cols());
    for (Eigen::Index i = 0; i < point.size(); ++i)
    {
        const double sigmoid = 1.0 / (1.0 + std::exp(-point(i)));
        expected(i) = sigmoid * (1.0 - sigmoid) + (point(i) > 0.0 ? 1.0 : 0.0);
    }

    REQUIRE(max_difference(x.grad(), expected) < 1e-12);

    const matrix numeric = test_support::numeric_graph_gradient(
        [](const tensor& probe) { return mlp::sum(mlp::sigmoid(probe) + mlp::relu(probe)); }, point);
    REQUIRE(max_difference(x.grad(), numeric) < 1e-6);
}

TEST_CASE("squaring a tensor against itself doubles its gradient", "[autograd]")
{
    const matrix point = matrix{{1.5, -2.0}};
    const tensor x = tensor::parameter(point);

    mlp::sum(mlp::hadamard(x, x)).backward();

    REQUIRE(x.grad()(0, 0) == Approx(3.0));
    REQUIRE(x.grad()(0, 1) == Approx(-4.0));
}

TEST_CASE("every operation matches a numeric gradient", "[autograd][gradient]")
{
    const matrix weights = patterned(4, 3, 0.8);
    const matrix row = patterned(1, 4, 0.5);
    const matrix other = patterned(5, 4, 0.6);

    const std::vector<std::pair<std::string, test_support::scalar_graph>> graphs{
        {"matmul", [&](const tensor& x) { return mlp::sum(mlp::matmul(x, tensor::constant(weights))); }},
        {"add", [&](const tensor& x) { return mlp::sum(x + tensor::constant(other)); }},
        {"add_broadcast", [&](const tensor& x) { return mlp::sum(x + tensor::constant(row)); }},
        {"subtract", [&](const tensor& x) { return mlp::sum(x - tensor::constant(other)); }},
        {"scale", [](const tensor& x) { return mlp::sum(x * 2.5); }},
        {"negate", [](const tensor& x) { return mlp::sum(-x); }},
        {"hadamard", [&](const tensor& x) { return mlp::sum(mlp::hadamard(x, tensor::constant(other))); }},
        {"sigmoid", [](const tensor& x) { return mlp::sum(mlp::sigmoid(x)); }},
        {"relu", [](const tensor& x) { return mlp::sum(mlp::relu(x)); }},
        {"tanh", [](const tensor& x) { return mlp::sum(mlp::tanh(x)); }},
        {"softmax", [&](const tensor& x) {
             return mlp::sum(mlp::hadamard(mlp::softmax_rows(x), tensor::constant(other)));
         }},
        {"sum", [](const tensor& x) { return mlp::sum(x); }},
        {"mean", [](const tensor& x) { return mlp::mean(x); }},
        {"column", [](const tensor& x) { return mlp::sum(mlp::column(x, 2)); }},
        {"concat", [](const tensor& x) { return mlp::sum(mlp::concat_cols(x, mlp::relu(x))); }},
        {"composition", [&](const tensor& x) {
             return mlp::mean(mlp::softmax_rows(mlp::tanh(mlp::matmul(x, tensor::constant(weights)))));
         }}};

    const matrix point = patterned(5, 4, 1.1);

    for (const auto& [name, graph] : graphs)
    {
        INFO("operation: " << name);
        REQUIRE(max_difference(test_support::autograd_gradient(graph, point),
                               test_support::numeric_graph_gradient(graph, point)) < 1e-6);
    }
}

TEST_CASE("log and clamp match numeric gradients on positive input", "[autograd][gradient]")
{
    const matrix point = positive(4, 3);

    const std::vector<std::pair<std::string, test_support::scalar_graph>> graphs{
        {"log", [](const tensor& x) { return mlp::sum(mlp::log(x)); }},
        {"clamp", [](const tensor& x) { return mlp::sum(mlp::clamp(x, 0.0, 10.0)); }},
        {"log_of_clamp", [](const tensor& x) { return mlp::sum(mlp::log(mlp::clamp(x, 1e-12, 10.0))); }}};

    for (const auto& [name, graph] : graphs)
    {
        INFO("operation: " << name);
        REQUIRE(max_difference(test_support::autograd_gradient(graph, point),
                               test_support::numeric_graph_gradient(graph, point)) < 1e-6);
    }
}

TEST_CASE("a clamped value stops the gradient", "[autograd]")
{
    const tensor x = tensor::parameter(matrix{{-1.0, 0.5, 4.0}});

    mlp::sum(mlp::clamp(x, 0.0, 1.0)).backward();

    REQUIRE(x.grad()(0, 0) == Approx(0.0));
    REQUIRE(x.grad()(0, 1) == Approx(1.0));
    REQUIRE(x.grad()(0, 2) == Approx(0.0));
}

TEST_CASE("no_grad builds no tape at all", "[autograd]")
{
    const tensor x = tensor::parameter(patterned(3, 3));

    tensor tracked = mlp::sum(mlp::sigmoid(x));
    REQUIRE(tracked.requires_grad());
    REQUIRE(tracked.graph_size() > 1);

    {
        const mlp::no_grad guard;
        REQUIRE(!mlp::grad_enabled());

        const tensor loose = mlp::sum(mlp::sigmoid(x));
        REQUIRE(!loose.requires_grad());
        REQUIRE(loose.graph_size() == 1);
        REQUIRE(loose.value()(0, 0) == Approx(tracked.value()(0, 0)));

        loose.backward();
        REQUIRE(x.grad().sum() == Approx(0.0));
    }

    REQUIRE(mlp::grad_enabled());
}

TEST_CASE("a constant input is left without a gradient", "[autograd]")
{
    const tensor inputs = tensor::constant(patterned(4, 2));
    const tensor weights = tensor::parameter(patterned(2, 3));

    const tensor loss = mlp::sum(mlp::matmul(inputs, weights));
    REQUIRE(loss.requires_grad());

    loss.backward();

    REQUIRE(inputs.grad().sum() == Approx(0.0));
    REQUIRE(weights.grad().sum() != Approx(0.0));
}

TEST_CASE("backward can be repeated and leaves accumulate until cleared", "[autograd]")
{
    const matrix point = patterned(2, 3, 1.4);
    const tensor x = tensor::parameter(point);
    const tensor loss = mlp::sum(mlp::hadamard(mlp::sigmoid(x), mlp::sigmoid(x)));

    loss.backward();
    const matrix once = x.grad();

    loss.backward();
    REQUIRE(max_difference(x.grad(), matrix(once * 2.0)) < 1e-15);

    x.zero_grad();
    REQUIRE(x.grad().sum() == Approx(0.0));

    loss.backward();
    REQUIRE(max_difference(x.grad(), once) < 1e-15);
}

TEST_CASE("the tape is released once the loss goes out of scope", "[autograd]")
{
    const tensor x = tensor::parameter(patterned(3, 3));
    const long baseline = x.node().use_count();

    {
        const tensor loss = mlp::sum(mlp::softmax_rows(mlp::tanh(x)));
        REQUIRE(x.node().use_count() > baseline);
        loss.backward();
    }

    REQUIRE(x.node().use_count() == baseline);
}

TEST_CASE("the engine refuses graphs it cannot differentiate", "[autograd]")
{
    const tensor x = tensor::parameter(patterned(2, 3));

    REQUIRE_THROWS_AS(x.backward(), std::invalid_argument);
    REQUIRE_THROWS_AS(x.backward(matrix::Zero(3, 2)), std::invalid_argument);

    REQUIRE_THROWS_AS(x + tensor::constant(matrix::Zero(4, 4)), std::invalid_argument);
    REQUIRE_THROWS_AS(x - tensor::constant(matrix::Zero(4, 4)), std::invalid_argument);
    REQUIRE_THROWS_AS(mlp::matmul(x, tensor::constant(matrix::Zero(4, 4))), std::invalid_argument);
    REQUIRE_THROWS_AS(mlp::column(x, 7), std::out_of_range);
    REQUIRE_THROWS_AS(mlp::concat_cols(x, tensor::constant(matrix::Zero(4, 1))), std::invalid_argument);

    const tensor empty;
    REQUIRE(!empty.defined());
    REQUIRE_THROWS_AS(mlp::sum(empty), std::logic_error);
    REQUIRE_THROWS_AS(empty.backward(), std::logic_error);
}
