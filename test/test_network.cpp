#include "mlp/mlp.h"

#include "gradient_check.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

using Catch::Approx;
using mlp::matrix;
using mlp::tensor;
using test_support::max_difference;
using test_support::patterned;

namespace
{

    struct synthetic
    {
        matrix inputs;
        std::vector<int> labels;
    };

    synthetic make_synthetic(std::size_t per_class, std::uint64_t seed)
    {
        mlp::rng rng(seed);
        synthetic data;
        data.inputs = matrix(per_class * 2, 2);

        for (std::size_t i = 0; i < per_class * 2; ++i)
        {
            const bool positive = i >= per_class;
            const double centre = positive ? 1.5 : -1.5;
            data.inputs(i, 0) = centre + rng.normal(0.0, 0.4);
            data.inputs(i, 1) = centre + rng.normal(0.0, 0.4);
            data.labels.push_back(positive ? 1 : 0);
        }
        return data;
    }

    mlp::network make_residual_network(mlp::rng& rng, std::size_t width = 8)
    {
        mlp::network net;
        net.add(std::make_unique<mlp::dense>(2, width, mlp::initializer::he_uniform, rng));
        net.add(mlp::make_activation(mlp::activation::tanh));

        auto block = std::make_unique<mlp::residual>();
        block->add(std::make_unique<mlp::dense>(width, width, mlp::initializer::he_uniform, rng));
        block->add(mlp::make_activation(mlp::activation::tanh));
        net.add(std::move(block));

        net.add(std::make_unique<mlp::dense>(width, 2, mlp::initializer::he_uniform, rng));
        net.add(mlp::make_activation(mlp::activation::softmax));
        return net;
    }

    void check_parameter_gradients(mlp::network& network, const matrix& inputs, const matrix& targets)
    {
        const mlp::binary_cross_entropy loss;

        const auto scalar = [&]() {
            const mlp::no_grad guard;
            return loss.value(network.forward(inputs), targets);
        };

        network.zero_grad();
        loss(network.forward(tensor::constant(inputs)), targets).backward();

        const std::vector<tensor> parameters = network.parameters();
        REQUIRE(!parameters.empty());

        for (std::size_t p = 0; p < parameters.size(); ++p)
        {
            INFO("parameter " << p << " of shape " << mlp::shape_string(parameters[p].value()));
            const matrix numeric =
                test_support::numeric_parameter_gradient(parameters[p].mutable_value(), scalar);
            REQUIRE(max_difference(parameters[p].grad(), numeric) < 1e-6);
        }
    }
}

TEST_CASE("build lays out dense and activation modules in order", "[network]")
{
    mlp::architecture architecture;
    architecture.input_size = 30;
    architecture.hidden_sizes = {24, 24, 24};
    architecture.output_size = 2;

    mlp::rng rng(5);
    mlp::network network = mlp::network::build(architecture, rng);

    REQUIRE(network.module_count() == 8);
    REQUIRE(network.module_at(0).type() == "dense");
    REQUIRE(network.module_at(0).config() == "30 24");
    REQUIRE(network.module_at(1).type() == "sigmoid");
    REQUIRE(network.module_at(6).config() == "24 2");
    REQUIRE(network.module_at(7).type() == "softmax");

    REQUIRE(network.parameter_count() == 30 * 24 + 24 + 2 * (24 * 24 + 24) + 24 * 2 + 2);
    REQUIRE(network.summary().find("total trainable parameters") != std::string::npos);
}

TEST_CASE("the default architecture has two hidden layers", "[network]")
{
    const mlp::architecture architecture;
    REQUIRE(architecture.hidden_sizes.size() >= 2);
}

TEST_CASE("build rejects degenerate shapes", "[network]")
{
    mlp::rng rng(1);

    mlp::architecture zero_input;
    zero_input.input_size = 0;
    REQUIRE_THROWS_AS(mlp::network::build(zero_input, rng), std::invalid_argument);

    mlp::architecture zero_hidden;
    zero_hidden.hidden_sizes = {4, 0};
    REQUIRE_THROWS_AS(mlp::network::build(zero_hidden, rng), std::invalid_argument);
}

TEST_CASE("a forward pass produces probability rows", "[network]")
{
    mlp::architecture architecture;
    architecture.input_size = 3;
    architecture.hidden_sizes = {5, 5};
    architecture.output_size = 2;

    mlp::rng rng(13);
    mlp::network network = mlp::network::build(architecture, rng);

    const matrix probabilities = network.forward(patterned(6, 3));
    REQUIRE(probabilities.rows() == 6);
    REQUIRE(probabilities.cols() == 2);

    for (std::size_t i = 0; i < probabilities.rows(); ++i)
        REQUIRE(probabilities(i, 0) + probabilities(i, 1) == Approx(1.0));

    const std::vector<int> classes = network.predict(patterned(6, 3));
    REQUIRE(classes.size() == 6);
    for (const int klass : classes)
        REQUIRE((klass == 0 || klass == 1));
}

TEST_CASE("a forward pass for prediction leaves no tape behind", "[network]")
{
    mlp::architecture architecture;
    architecture.input_size = 3;
    architecture.hidden_sizes = {4, 4};
    architecture.output_size = 2;

    mlp::rng rng(29);
    mlp::network network = mlp::network::build(architecture, rng);

    const std::vector<tensor> parameters = network.parameters();
    std::vector<long> references;
    for (const tensor& parameter : parameters)
        references.push_back(parameter.node().use_count());

    const matrix probabilities = network.forward(patterned(4, 3));
    REQUIRE(probabilities.rows() == 4);

    for (std::size_t i = 0; i < parameters.size(); ++i)
    {
        REQUIRE(parameters[i].node().use_count() == references[i]);
        REQUIRE(parameters[i].grad().sum() == Approx(0.0));
    }
}

TEST_CASE("backpropagation through the whole network matches numeric gradients", "[network][gradient]")
{

    mlp::architecture architecture;
    architecture.input_size = 3;
    architecture.hidden_sizes = {4, 4};
    architecture.output_size = 2;

    mlp::rng rng(17);
    mlp::network network = mlp::network::build(architecture, rng);

    check_parameter_gradients(network, patterned(5, 3, 1.1), mlp::one_hot({0, 1, 1, 0, 1}, 2));
}

TEST_CASE("backpropagation through a skip connection matches numeric gradients", "[network][gradient][skip]")
{

    mlp::rng rng(23);
    mlp::network network = make_residual_network(rng, 5);

    REQUIRE(network.module_count() == 5);
    REQUIRE(network.module_at(2).type() == "residual");

    check_parameter_gradients(network, patterned(4, 2, 1.0), mlp::one_hot({1, 0, 1, 0}, 2));
}

TEST_CASE("a skip connection adds the input back to the branch", "[network][skip]")
{
    mlp::residual block;
    block.add(mlp::make_activation(mlp::activation::relu));

    const matrix input = matrix{{-1.0, 2.0}};
    const matrix output = block.forward(tensor::constant(input)).value();

    REQUIRE(output(0, 0) == Approx(-1.0));
    REQUIRE(output(0, 1) == Approx(4.0));
}

TEST_CASE("a skip connection refuses to change the shape", "[network][skip]")
{
    mlp::rng rng(31);

    mlp::residual block;
    block.add(std::make_unique<mlp::dense>(4, 7, mlp::initializer::he_uniform, rng));

    REQUIRE_THROWS_AS(block.forward(tensor::constant(patterned(3, 4))), std::invalid_argument);

    const mlp::residual empty;
    REQUIRE(empty.size() == 0);
}

TEST_CASE("training drives the loss down and separates two clouds", "[network][training]")
{
    const synthetic train = make_synthetic(80, 1);
    const synthetic validation = make_synthetic(20, 2);

    mlp::architecture architecture;
    architecture.input_size = 2;
    architecture.hidden_sizes = {8, 8};
    architecture.output_size = 2;

    mlp::rng rng(42);
    mlp::network network = mlp::network::build(architecture, rng);

    mlp::train_config config;
    config.epochs = 40;
    config.batch_size = 16;
    config.learning_rate = 0.5;
    config.verbose = false;

    mlp::trainer trainer(config);
    const mlp::history history =
        trainer.fit(network, train.inputs, train.labels, validation.inputs, validation.labels);

    REQUIRE(history.epochs() == 40);
    REQUIRE(history.has_validation());
    REQUIRE(history.train_loss.back() < history.train_loss.front());
    REQUIRE(history.train_accuracy.back() > 0.95);
    REQUIRE(history.validation_accuracy.back() > 0.9);

    for (const double value : history.train_loss)
        REQUIRE(std::isfinite(value));
}

TEST_CASE("a network with a skip connection trains too", "[network][training][skip]")
{
    const synthetic train = make_synthetic(80, 6);

    mlp::rng rng(7);
    mlp::network network = make_residual_network(rng);

    mlp::train_config config;
    config.epochs = 30;
    config.batch_size = 16;
    config.learning_rate = 0.3;
    config.verbose = false;

    const mlp::history history =
        mlp::trainer(config).fit(network, train.inputs, train.labels, matrix{}, {});

    REQUIRE(history.train_loss.back() < history.train_loss.front());
    REQUIRE(history.train_accuracy.back() > 0.95);
}

TEST_CASE("the same seed gives an identical run", "[network][training]")
{
    const synthetic data = make_synthetic(40, 3);

    const auto run = [&]() {
        mlp::architecture architecture;
        architecture.input_size = 2;
        architecture.hidden_sizes = {6, 6};
        architecture.output_size = 2;

        mlp::rng rng(2026);
        mlp::network network = mlp::network::build(architecture, rng);

        mlp::train_config config;
        config.epochs = 10;
        config.seed = 2026;
        config.verbose = false;

        return mlp::trainer(config).fit(network, data.inputs, data.labels, matrix{}, {});
    };

    const mlp::history first = run();
    const mlp::history second = run();
    REQUIRE(first.train_loss == second.train_loss);
    REQUIRE(!first.has_validation());
}

TEST_CASE("early stopping restores the best epoch", "[network][training]")
{
    const synthetic train = make_synthetic(30, 4);
    const synthetic validation = make_synthetic(10, 5);

    mlp::architecture architecture;
    architecture.input_size = 2;
    architecture.hidden_sizes = {4, 4};
    architecture.output_size = 2;

    mlp::rng rng(8);
    mlp::network network = mlp::network::build(architecture, rng);

    mlp::train_config config;
    config.epochs = 200;
    config.learning_rate = 0.4;
    config.early_stopping_patience = 3;
    config.verbose = false;

    const mlp::history history =
        mlp::trainer(config).fit(network, train.inputs, train.labels, validation.inputs, validation.labels);

    REQUIRE(history.epochs() <= 200);
    REQUIRE(history.best_epoch() < history.epochs());
}

TEST_CASE("the trainer validates its input", "[network][training]")
{
    mlp::network network;
    mlp::train_config config;

    REQUIRE_THROWS_AS(mlp::trainer(mlp::train_config{0, 8, 0.1, 0.0, "mse", 1, false, 0}),
                      std::invalid_argument);

    mlp::trainer trainer(config);
    REQUIRE_THROWS_AS(trainer.fit(network, matrix::Zero(3, 2), {0, 1}, matrix{}, {}), std::invalid_argument);
}

TEST_CASE("the optimizer moves every parameter against its gradient", "[network][optim]")
{
    mlp::dense layer(2, 2);
    layer.weights().mutable_value() = matrix{{1.0, 1.0}, {1.0, 1.0}};

    mlp::sum(layer.forward(tensor::constant(matrix{{1.0, 1.0}}))).backward();

    const std::vector<tensor> parameters = layer.parameters();
    mlp::sgd optimizer(0.5);
    optimizer.step(parameters);

    REQUIRE(layer.weights().value()(0, 0) == Approx(0.5));
    REQUIRE(layer.bias().value()(0, 0) == Approx(-0.5));
    REQUIRE(optimizer.name() == "sgd");
    REQUIRE(optimizer.learning_rate() == Approx(0.5));

    mlp::zero_grad(parameters);
    for (const tensor& parameter : parameters)
        REQUIRE(parameter.grad().sum() == Approx(0.0));
}
