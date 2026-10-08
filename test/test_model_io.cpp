#include "mlp/mlp.h"

#include "gradient_check.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

using Catch::Approx;
using mlp::matrix;

namespace
{
    std::filesystem::path scratch_file(const std::string& name)
    {
        const std::filesystem::path directory = std::filesystem::temp_directory_path() / "mlp-tests";
        std::filesystem::create_directories(directory);
        return directory / name;
    }

    mlp::model make_trained_model()
    {
        mlp::architecture architecture;
        architecture.input_size = 4;
        architecture.hidden_sizes = {6, 5};
        architecture.output_size = 2;

        mlp::rng rng(21);
        mlp::model model;
        model.net = mlp::network::build(architecture, rng);
        model.scaler = mlp::standard_scaler({1.0, 2.0, 3.0, 4.0}, {0.5, 1.5, 2.5, 3.5});
        model.class_names = {"B", "M"};
        model.loss_name = "binaryCrossentropy";
        model.seed = 21;
        return model;
    }
}

TEST_CASE("a saved model predicts exactly as it did before saving", "[model_io]")
{
    mlp::model original = make_trained_model();
    const matrix inputs = test_support::patterned(7, 4, 1.3);
    const matrix before = original.net.forward(inputs);

    const std::filesystem::path path = scratch_file("roundtrip.mlp");
    mlp::save_model(original, path);

    mlp::model reloaded = mlp::load_model(path);
    const matrix after = reloaded.net.forward(inputs);

    REQUIRE(after.rows() == before.rows());
    REQUIRE(after.cols() == before.cols());

    REQUIRE(after.cwiseEqual(before).all());
}

TEST_CASE("the metadata survives the round trip", "[model_io]")
{
    const mlp::model original = make_trained_model();
    const std::filesystem::path path = scratch_file("metadata.mlp");
    mlp::save_model(original, path);

    const mlp::model reloaded = mlp::load_model(path);

    REQUIRE(reloaded.seed == original.seed);
    REQUIRE(reloaded.loss_name == original.loss_name);
    REQUIRE(reloaded.class_names == original.class_names);
    REQUIRE(reloaded.scaler.means() == original.scaler.means());
    REQUIRE(reloaded.scaler.stddevs() == original.scaler.stddevs());
    REQUIRE(reloaded.net.module_count() == original.net.module_count());

    for (std::size_t i = 0; i < original.net.module_count(); ++i)
    {
        REQUIRE(reloaded.net.module_at(i).type() == original.net.module_at(i).type());
        REQUIRE(reloaded.net.module_at(i).config() == original.net.module_at(i).config());
    }
}

TEST_CASE("the model file is readable text", "[model_io]")
{
    const std::filesystem::path path = scratch_file("readable.mlp");
    mlp::save_model(make_trained_model(), path);

    std::ifstream input(path);
    std::string first_line;
    std::getline(input, first_line);
    REQUIRE(first_line == "mlp-model 2");

    const std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    REQUIRE(contents.find("scaler 4") != std::string::npos);
    REQUIRE(contents.find("sequential 6") != std::string::npos);
    REQUIRE(contents.find("dense 4 6") != std::string::npos);
    REQUIRE(contents.find("softmax") != std::string::npos);
}

TEST_CASE("a damaged model file is reported", "[model_io]")
{
    const std::filesystem::path missing = scratch_file("not_written.mlp");
    std::filesystem::remove(missing);
    REQUIRE_THROWS_AS(mlp::load_model(missing), std::runtime_error);

    const std::filesystem::path wrong_version = scratch_file("version.mlp");
    {
        std::ofstream output(wrong_version);
        output << "mlp-model 99\n";
    }
    REQUIRE_THROWS_AS(mlp::load_model(wrong_version), std::runtime_error);

    const std::filesystem::path unknown_layer = scratch_file("unknown_layer.mlp");
    {
        std::ofstream output(unknown_layer);
        output << "mlp-model 2\nseed 1\nloss mse\nclasses 2\nclass_names B M\nscaler 1\n0 1\n"
                  "sequential 1\nConvolution 1 1\n";
    }
    REQUIRE_THROWS_AS(mlp::load_model(unknown_layer), std::runtime_error);

    const std::filesystem::path truncated = scratch_file("truncated.mlp");
    {
        std::ofstream output(truncated);
        output << "mlp-model 2\nseed 1\nloss mse\nclasses 2\nclass_names B M\nscaler 1\n0 1\n"
                  "sequential 1\ndense 2 2\n1 2\n";
    }
    REQUIRE_THROWS_AS(mlp::load_model(truncated), std::runtime_error);
}

namespace
{
    mlp::model make_residual_model()
    {
        mlp::rng rng(5);

        auto block = std::make_unique<mlp::residual>();
        block->add(std::make_unique<mlp::dense>(6, 6, mlp::initializer::he_uniform, rng));
        block->add(mlp::make_activation(mlp::activation::relu));
        block->add(std::make_unique<mlp::dense>(6, 6, mlp::initializer::he_uniform, rng));

        mlp::model model;
        model.net.add(std::make_unique<mlp::dense>(4, 6, mlp::initializer::he_uniform, rng));
        model.net.add(mlp::make_activation(mlp::activation::relu));
        model.net.add(std::move(block));
        model.net.add(std::make_unique<mlp::dense>(6, 2, mlp::initializer::he_uniform, rng));
        model.net.add(mlp::make_activation(mlp::activation::softmax));

        model.scaler = mlp::standard_scaler({1.0, 2.0, 3.0, 4.0}, {0.5, 1.5, 2.5, 3.5});
        model.class_names = {"B", "M"};
        model.loss_name = "binaryCrossentropy";
        model.seed = 5;
        return model;
    }
}

TEST_CASE("a network with a skip connection survives the round trip", "[model_io][skip]")
{
    mlp::model original = make_residual_model();
    const matrix inputs = test_support::patterned(6, 4, 1.1);
    const matrix before = original.net.forward(inputs);

    const std::filesystem::path path = scratch_file("residual.mlp");
    mlp::save_model(original, path);

    mlp::model reloaded = mlp::load_model(path);

    REQUIRE(reloaded.net.module_count() == original.net.module_count());
    REQUIRE(reloaded.net.module_at(2).type() == "residual");
    REQUIRE(reloaded.net.module_at(2).config() == "3");
    REQUIRE(reloaded.net.parameter_count() == original.net.parameter_count());
    REQUIRE(reloaded.net.forward(inputs).cwiseEqual(before).all());
}

TEST_CASE("the nesting is visible in the saved file", "[model_io][skip]")
{
    const std::filesystem::path path = scratch_file("residual_text.mlp");
    mlp::save_model(make_residual_model(), path);

    std::ifstream input(path);
    const std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());

    REQUIRE(contents.find("sequential 5") != std::string::npos);
    REQUIRE(contents.find("residual 3") != std::string::npos);
    REQUIRE(contents.find("dense 6 6") != std::string::npos);
}

TEST_CASE("a model from the previous format version is refused", "[model_io]")
{
    const std::filesystem::path path = scratch_file("version_one.mlp");
    {
        std::ofstream output(path);
        output << "mlp-model 1\nseed 1\nloss mse\nclasses 2\nclass_names B M\nscaler 1\n0 1\n"
                  "layers 1\nsigmoid\n";
    }

    REQUIRE_THROWS_AS(mlp::load_model(path), std::runtime_error);
}
