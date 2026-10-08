#include "mlp/mlp.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <sstream>

using Catch::Approx;
using mlp::matrix;

namespace
{
    mlp::dataset make_dataset(std::size_t benign, std::size_t malignant)
    {
        mlp::dataset dataset;
        const std::size_t total = benign + malignant;
        dataset.inputs = matrix(total, 3);

        for (std::size_t i = 0; i < total; ++i)
        {
            const bool is_malignant = i >= benign;
            dataset.labels.push_back(is_malignant ? 1 : 0);
            dataset.ids.push_back(std::to_string(i));
            for (std::size_t j = 0; j < 3; ++j)
                dataset.inputs(i, j) = static_cast<double>(i * 3 + j);
        }
        return dataset;
    }
}

TEST_CASE("the scaler centres and rescales every feature", "[preprocessing]")
{

    const matrix inputs = matrix{{1, 10}, {3, 30}, {5, 50}};

    mlp::standard_scaler scaler;
    const matrix scaled = scaler.fit_transform(inputs);

    REQUIRE(scaler.means()[0] == Approx(3.0));
    REQUIRE(scaler.means()[1] == Approx(30.0));
    REQUIRE(scaler.stddevs()[0] == Approx(std::sqrt(8.0 / 3.0)));

    for (std::size_t j = 0; j < scaled.cols(); ++j)
    {
        double mean = 0.0;
        for (std::size_t i = 0; i < scaled.rows(); ++i)
            mean += scaled(i, j);
        mean /= static_cast<double>(scaled.rows());

        double variance = 0.0;
        for (std::size_t i = 0; i < scaled.rows(); ++i)
            variance += (scaled(i, j) - mean) * (scaled(i, j) - mean);
        variance /= static_cast<double>(scaled.rows());

        REQUIRE(mean == Approx(0.0).margin(1e-12));
        REQUIRE(variance == Approx(1.0));
    }
}

TEST_CASE("inverse_transform undoes transform", "[preprocessing]")
{
    const matrix inputs = matrix{{1, 10}, {3, 30}, {5, 50}};

    mlp::standard_scaler scaler;
    scaler.fit(inputs);
    const matrix restored = scaler.inverse_transform(scaler.transform(inputs));

    REQUIRE((restored - inputs).cwiseAbs().maxCoeff() < 1e-9);
}

TEST_CASE("a constant feature does not divide by zero", "[preprocessing]")
{
    const matrix inputs = matrix{{5}, {5}, {5}};

    mlp::standard_scaler scaler;
    const matrix scaled = scaler.fit_transform(inputs);

    REQUIRE(scaled.allFinite());
}

TEST_CASE("an unfitted scaler refuses to transform", "[preprocessing]")
{
    mlp::standard_scaler scaler;
    REQUIRE_THROWS_AS(scaler.transform(matrix::Zero(2, 2)), std::logic_error);

    mlp::standard_scaler fitted;
    fitted.fit(matrix{{1, 2}, {3, 4}});
    REQUIRE_THROWS_AS(fitted.transform(matrix::Zero(2, 5)), std::invalid_argument);
}

TEST_CASE("the scaler survives a save and load", "[preprocessing]")
{
    mlp::standard_scaler original;
    original.fit(matrix{{1, 10}, {3, 30}, {5, 50}});

    std::stringstream stream;
    original.save(stream);

    mlp::standard_scaler restored;
    restored.load(stream);

    REQUIRE(restored.means() == original.means());
    REQUIRE(restored.stddevs() == original.stddevs());
}

TEST_CASE("the split keeps every sample and the requested proportion", "[preprocessing]")
{
    const mlp::dataset dataset = make_dataset(60, 40);

    mlp::rng rng(42);
    const mlp::split_result split = mlp::train_test_split(dataset, 0.8, rng, true);

    REQUIRE(split.train.size() + split.validation.size() == dataset.size());
    REQUIRE(split.train.size() == 80);
    REQUIRE(split.validation.size() == 20);
}

TEST_CASE("a stratified split preserves the class balance", "[preprocessing]")
{
    const mlp::dataset dataset = make_dataset(60, 40);

    mlp::rng rng(42);
    const mlp::split_result split = mlp::train_test_split(dataset, 0.8, rng, true);

    REQUIRE(split.train.class_counts() == std::vector<std::size_t>{48, 32});
    REQUIRE(split.validation.class_counts() == std::vector<std::size_t>{12, 8});
}

TEST_CASE("the same seed produces the same split", "[preprocessing]")
{
    const mlp::dataset dataset = make_dataset(60, 40);

    mlp::rng first(2024);
    mlp::rng second(2024);
    const mlp::split_result left = mlp::train_test_split(dataset, 0.75, first, true);
    const mlp::split_result right = mlp::train_test_split(dataset, 0.75, second, true);

    REQUIRE(left.train.ids == right.train.ids);
    REQUIRE(left.validation.ids == right.validation.ids);

    mlp::rng other(7);
    const mlp::split_result different = mlp::train_test_split(dataset, 0.75, other, true);
    REQUIRE(different.train.ids != left.train.ids);
}

TEST_CASE("an impossible ratio is rejected", "[preprocessing]")
{
    const mlp::dataset dataset = make_dataset(5, 5);
    mlp::rng rng(1);

    REQUIRE_THROWS_AS(mlp::train_test_split(dataset, 0.0, rng), std::invalid_argument);
    REQUIRE_THROWS_AS(mlp::train_test_split(dataset, 1.0, rng), std::invalid_argument);
}

TEST_CASE("one_hot builds indicator rows", "[preprocessing]")
{
    const matrix encoded = mlp::one_hot({0, 1, 1, 0}, 2);

    REQUIRE(encoded.rows() == 4);
    REQUIRE(encoded.cols() == 2);
    REQUIRE(encoded(0, 0) == Approx(1.0));
    REQUIRE(encoded(0, 1) == Approx(0.0));
    REQUIRE(encoded(2, 1) == Approx(1.0));
    REQUIRE(encoded.sum() == Approx(4.0));

    REQUIRE_THROWS_AS(mlp::one_hot({0, 5}, 2), std::out_of_range);
}
