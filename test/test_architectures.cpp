#include "mlp/mlp.h"

#include <catch2/catch_test_macros.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace
{
using mlp::matrix;

struct candidate
{
    std::string name;
    std::vector<std::size_t> hidden;
    mlp::activation hidden_activation;
};

struct outcome
{
    std::size_t parameters{0};
    double train_loss{0.0};
    double validation_loss{0.0};
    double validation_accuracy{0.0};
};

// Every run differs from its neighbours in exactly one thing: the architecture. The split is
// fixed, the scaler is fixed, and the seed is passed in so that the same seeds are replayed
// for every candidate.
[[nodiscard]] outcome train_once(const candidate &spec, const matrix &train_inputs,
                                 const std::vector<int> &train_labels,
                                 const matrix &validation_inputs,
                                 const std::vector<int> &validation_labels, std::uint64_t seed)
{
    mlp::architecture architecture;
    architecture.input_size = static_cast<std::size_t>(train_inputs.cols());
    architecture.output_size = 2;
    architecture.hidden_sizes = spec.hidden;
    architecture.hidden_activation = spec.hidden_activation;

    mlp::rng rng(seed);
    mlp::network network = mlp::network::build(architecture, rng);

    mlp::train_config config;
    config.epochs = 84;
    config.batch_size = 8;
    config.learning_rate = 0.0314;
    config.loss = "binaryCrossentropy";
    config.seed = seed;
    config.verbose = false;

    mlp::trainer trainer(config);
    const mlp::history history =
        trainer.fit(network, train_inputs, train_labels, validation_inputs, validation_labels);

    outcome result;
    result.parameters = network.parameter_count();
    result.train_loss = history.train_loss.back();
    result.validation_loss = history.validation_loss.back();
    result.validation_accuracy = history.validation_accuracy.back();
    return result;
}
} // namespace

TEST_CASE("architectures are compared on the assignment's dataset", "[experiment][network]")
{
    const mlp::dataset full = mlp::load_wdbc(std::string(MLP_DATA_DIR) + "/data.csv");
    REQUIRE(full.size() == 569);
    REQUIRE(full.feature_count() == 30);

    // The same stratified 80/20 split the `split` subcommand produces with --seed 42.
    mlp::rng split_rng(42);
    const mlp::split_result split = mlp::train_test_split(full, 0.8, split_rng, true);
    REQUIRE(split.train.size() == 456);
    REQUIRE(split.validation.size() == 113);

    mlp::standard_scaler scaler;
    const matrix train_inputs = scaler.fit_transform(split.train.inputs);
    const matrix validation_inputs = scaler.transform(split.validation.inputs);

    const std::vector<candidate> candidates{
        {"24-24 sigmoid", {24, 24}, mlp::activation::sigmoid},
        {"24-24-24 sigmoid", {24, 24, 24}, mlp::activation::sigmoid},
        {"48-48 sigmoid", {48, 48}, mlp::activation::sigmoid},
        {"8-8 sigmoid", {8, 8}, mlp::activation::sigmoid},
        {"24-24 relu", {24, 24}, mlp::activation::relu},
        {"24-24 tanh", {24, 24}, mlp::activation::tanh},
    };

    // A single seed decides nothing: the spread between seeds is comparable to the spread
    // between architectures, so every candidate is averaged over the same five seeds.
    const std::vector<std::uint64_t> seeds{1, 7, 42, 123, 2026};

    std::vector<outcome> averages;
    averages.reserve(candidates.size());

    fmt::print("\n{:<18}{:>12}{:>12}{:>12}{:>10}{:>14}\n", "architecture", "parameters",
               "train loss", "val loss", "sd", "val accuracy");
    fmt::print("{}\n", std::string(78, '-'));

    for (const candidate &spec : candidates)
    {
        std::vector<double> validation_losses;
        outcome total;
        for (const std::uint64_t seed : seeds)
        {
            const outcome single = train_once(spec, train_inputs, split.train.labels,
                                              validation_inputs, split.validation.labels, seed);
            total.parameters = single.parameters;
            total.train_loss += single.train_loss;
            total.validation_loss += single.validation_loss;
            total.validation_accuracy += single.validation_accuracy;
            validation_losses.push_back(single.validation_loss);
        }

        const auto runs = static_cast<double>(seeds.size());
        total.train_loss /= runs;
        total.validation_loss /= runs;
        total.validation_accuracy /= runs;

        // The spread across seeds says which gaps in the table are real and which are noise.
        double variance = 0.0;
        for (const double value : validation_losses)
            variance += (value - total.validation_loss) * (value - total.validation_loss);
        const double deviation = std::sqrt(variance / runs);

        fmt::print("{:<18}{:>12}{:>12.4f}{:>12.4f}{:>10.4f}{:>14.4f}\n", spec.name,
                   total.parameters, total.train_loss, total.validation_loss, deviation,
                   total.validation_accuracy);
        averages.push_back(total);
    }
    fmt::print("\n");

    // Every candidate has to be a working classifier: the comparison below is only meaningful
    // if none of them simply failed to train.
    for (std::size_t i = 0; i < candidates.size(); ++i)
    {
        INFO("architecture: " << candidates[i].name);
        REQUIRE(averages[i].validation_accuracy > 0.90);
        REQUIRE(averages[i].validation_loss < 0.30);
    }

    std::size_t best = 0;
    for (std::size_t i = 1; i < averages.size(); ++i)
        if (averages[i].validation_loss < averages[best].validation_loss)
            best = i;

    INFO("best by mean validation loss: " << candidates[best].name);
    REQUIRE(candidates[best].name == "24-24 tanh");
}
