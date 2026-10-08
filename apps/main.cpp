#include "mlp/mlp.h"

#include <CLI/CLI.hpp>
#include <fmt/format.h>

#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace
{
using namespace mlp;

struct describe_options
{
    std::string dataset;
    std::size_t top{10};
    double correlation{0.9};
};

struct split_options
{
    std::string dataset;
    std::string out_dir;
    double ratio{0.8};
    std::uint64_t seed{42};
    bool no_stratify{false};
};

struct train_options
{
    std::string dataset;
    std::string valid;
    std::vector<std::size_t> layers{24, 24};
    std::string activation{"tanh"};
    std::string initializer{"heUniform"};
    std::size_t epochs{84};
    std::size_t batch_size{8};
    double learning_rate{0.0314};
    double momentum{0.0};
    std::string loss{"binaryCrossentropy"};
    std::uint64_t seed{42};
    std::size_t early_stopping{0};
    std::string out{"model.mlp"};
    std::string plot_dir{"plots"};
    bool show{false};
    bool quiet{false};
};

struct predict_options
{
    std::string model;
    std::string dataset;
    std::string out;
};

void run_describe(const describe_options &options)
{
    const dataset dataset = load_wdbc(options.dataset);

    fmt::print("{}\n", format_class_balance(dataset));
    fmt::print("{}\n", format_summary_table(describe(dataset)));
    fmt::print("{}\n", format_label_correlations(dataset, options.top));
    fmt::print("{}\n", format_correlation_highlights(dataset, options.correlation));
}

void run_split(const split_options &options)
{
    const std::filesystem::path source = options.dataset;
    const dataset dataset = load_wdbc(source);
    const bool stratify = !options.no_stratify;

    rng rng(options.seed);
    const split_result split = train_test_split(dataset, options.ratio, rng, stratify);

    const std::filesystem::path directory =
        options.out_dir.empty() ? source.parent_path() : std::filesystem::path(options.out_dir);

    const std::filesystem::path train_path = directory / "data_training.csv";
    const std::filesystem::path validation_path = directory / "data_validation.csv";
    split.train.save(train_path);
    split.validation.save(validation_path);

    const std::vector<std::size_t> train_counts = split.train.class_counts();
    const std::vector<std::size_t> validation_counts = split.validation.class_counts();

    fmt::print("seed {}, ratio {:g}, {}stratified\n", options.seed, options.ratio,
               stratify ? "" : "not ");
    fmt::print("training   : {:>3} samples (B {}, M {}) -> {}\n", split.train.size(),
               train_counts[0], train_counts[1], train_path.string());
    fmt::print("validation : {:>3} samples (B {}, M {}) -> {}\n", split.validation.size(),
               validation_counts[0], validation_counts[1], validation_path.string());
}

void run_train(const train_options &options)
{
    dataset train_set = load_wdbc(options.dataset);
    dataset validation_set;

    if (!options.valid.empty())
    {
        validation_set = load_wdbc(options.valid);
    }
    else
    {
        rng split_rng(options.seed);
        split_result split = train_test_split(train_set, 0.8, split_rng, true);
        train_set = std::move(split.train);
        validation_set = std::move(split.validation);
        fmt::print("no --valid given, holding out 20% of the training file\n");
    }

    standard_scaler scaler;
    const matrix train_inputs = scaler.fit_transform(train_set.inputs);
    const matrix validation_inputs = scaler.transform(validation_set.inputs);

    architecture architecture;
    architecture.input_size = train_set.feature_count();
    architecture.output_size = train_set.class_count();
    architecture.hidden_sizes = options.layers;
    architecture.hidden_activation = activation_from_string(options.activation);
    architecture.weight_initializer = initializer_from_string(options.initializer);

    if (architecture.hidden_sizes.size() < 2)
        fmt::print(stderr, "warning: {} hidden layer(s); the assignment asks for at least two\n",
                   architecture.hidden_sizes.size());

    rng rng(options.seed);
    network network = network::build(architecture, rng);

    train_config config;
    config.epochs = options.epochs;
    config.batch_size = options.batch_size;
    config.learning_rate = options.learning_rate;
    config.momentum = options.momentum;
    config.loss = options.loss;
    config.seed = options.seed;
    config.verbose = !options.quiet;
    config.early_stopping_patience = options.early_stopping;

    fmt::print("x_train shape : ({}, {})\n", train_inputs.rows(), train_inputs.cols());
    fmt::print("x_valid shape : ({}, {})\n", validation_inputs.rows(), validation_inputs.cols());
    fmt::print("\n{}\n", network.summary());

    trainer trainer(config);
    const history history = trainer.fit(network, train_inputs, train_set.labels, validation_inputs,
                                        validation_set.labels, train_set.class_count());

    const std::size_t best = history.best_epoch();
    fmt::print("best epoch: {} (loss {:.4f}{})\n", best + 1, history.train_loss[best],
               history.has_validation()
                   ? fmt::format(", val_loss {:.4f}", history.validation_loss[best])
                   : "");

    const std::filesystem::path plot_directory = options.plot_dir;
    for (const std::filesystem::path &written : plot_history(history, plot_directory, options.show))
        fmt::print("> wrote '{}'\n", written.string());

    history.save_csv(plot_directory / "history.csv");

    model model;
    model.net = std::move(network);
    model.scaler = std::move(scaler);
    model.class_names = train_set.class_names;
    model.loss_name = config.loss;
    model.seed = options.seed;

    save_model(model, options.out);
    fmt::print("> saving model '{}' to disk...\n", options.out);
}

void run_predict(const predict_options &options)
{
    model model = load_model(options.model);
    const dataset dataset = load_wdbc(options.dataset);

    const matrix inputs = model.scaler.transform(dataset.inputs);
    const matrix probabilities = model.net.forward(inputs);
    const matrix targets = one_hot(dataset.labels, model.class_names.size());

    const std::unique_ptr<loss> loss = make_loss(model.loss_name);
    const std::vector<int> predicted = argmax_rows(probabilities);
    const confusion_matrix matrix =
        make_confusion_matrix(predicted, dataset.labels, model.class_names.size());

    fmt::print("samples        : {}\n", dataset.size());
    fmt::print("{:<15}: {:.4f}\n", model.loss_name, loss->value(probabilities, targets));
    fmt::print("accuracy       : {:.4f}\n", accuracy(predicted, dataset.labels));
    fmt::print("\n{}\n", format_confusion_matrix(matrix, model.class_names));
    fmt::print("{}", format_classification_report(matrix, model.class_names));

    if (!options.out.empty())
    {
        csv_table table;
        table.push_back({"id", "actual", "predicted", "p_benign", "p_malignant"});
        for (std::size_t i = 0; i < dataset.size(); ++i)
            table.push_back({dataset.ids[i],
                             model.class_names[static_cast<std::size_t>(dataset.labels[i])],
                             model.class_names[static_cast<std::size_t>(predicted[i])],
                             fmt::format("{:.6f}", probabilities(static_cast<Eigen::Index>(i), 0)),
                             fmt::format("{:.6f}",
                                         probabilities(static_cast<Eigen::Index>(i), 1))});

        write_csv(options.out, table);
        fmt::print("\n> wrote '{}'\n", options.out);
    }
}
}

int main(int argc, char **argv)
{
    CLI::App app{"mlp - a multilayer perceptron, from scratch", "mlp"};
    app.require_subcommand(0, 1);

    describe_options describe_opts;
    CLI::App *describe = app.add_subcommand("describe", "statistics and correlations of a dataset");
    describe->add_option("--dataset", describe_opts.dataset, "dataset in WDBC csv format")
        ->required()
        ->check(CLI::ExistingFile);
    describe->add_option("--top", describe_opts.top, "features to list by correlation with the label")
        ->capture_default_str();
    describe->add_option("--correlation", describe_opts.correlation, "report feature pairs above this |r|")
        ->capture_default_str();
    describe->callback([&] { run_describe(describe_opts); });

    split_options split_opts;
    CLI::App *split = app.add_subcommand("split", "split a dataset into a training and a validation file");
    split->add_option("--dataset", split_opts.dataset, "dataset in WDBC csv format")
        ->required()
        ->check(CLI::ExistingFile);
    split->add_option("--out-dir", split_opts.out_dir, "where to write the two files")
        ->default_str("next to the input");
    split->add_option("--ratio", split_opts.ratio, "share of samples used for training")
        ->capture_default_str();
    split->add_option("--seed", split_opts.seed, "random seed")->capture_default_str();
    split->add_flag("--no-stratify", split_opts.no_stratify,
                    "do not preserve the class balance across the split");
    split->callback([&] { run_split(split_opts); });

    train_options train_opts;
    CLI::App *train = app.add_subcommand("train", "train a network and save it");
    train->add_option("--dataset", train_opts.dataset, "training set in WDBC csv format")
        ->required()
        ->check(CLI::ExistingFile);
    train->add_option("--valid", train_opts.valid,
                      "validation set; without it the training file is split")
        ->check(CLI::ExistingFile);
    train->add_option("--layer", train_opts.layers, "hidden layer widths")->capture_default_str();
    train->add_option("--activation", train_opts.activation, "hidden activation: sigmoid, relu, tanh")
        ->capture_default_str();
    train->add_option("--initializer", train_opts.initializer,
                      "heUniform, heNormal, glorotUniform, zeros")
        ->capture_default_str();
    train->add_option("--epochs", train_opts.epochs, "number of passes over the training set")
        ->capture_default_str();
    train->add_option("--batch_size", train_opts.batch_size, "samples per gradient step")
        ->capture_default_str();
    train->add_option("--learning_rate", train_opts.learning_rate, "step size")->capture_default_str();
    train->add_option("--momentum", train_opts.momentum, "0 means plain gradient descent")
        ->capture_default_str();
    train->add_option("--loss", train_opts.loss,
                      "binaryCrossentropy, categoricalCrossentropy, mse")
        ->capture_default_str();
    train->add_option("--seed", train_opts.seed, "random seed")->capture_default_str();
    train->add_option("--early-stopping", train_opts.early_stopping,
                      "stop after n epochs without improvement, 0 is off")
        ->capture_default_str();
    train->add_option("--out", train_opts.out, "where to save the model")->capture_default_str();
    train->add_option("--plot-dir", train_opts.plot_dir, "where to write the learning curves")
        ->capture_default_str();
    train->add_flag("--show", train_opts.show, "open the plot windows as well");
    train->add_flag("--quiet", train_opts.quiet, "do not print per-epoch metrics");
    train->callback([&] { run_train(train_opts); });

    predict_options predict_opts;
    CLI::App *predict = app.add_subcommand("predict", "load a saved network and evaluate it on a dataset");
    predict->add_option("--model", predict_opts.model, "model saved by train")
        ->required()
        ->check(CLI::ExistingFile);
    predict->add_option("--dataset", predict_opts.dataset, "dataset in WDBC csv format")
        ->required()
        ->check(CLI::ExistingFile);
    predict->add_option("--out", predict_opts.out, "also write per-sample predictions to this file");
    predict->callback([&] { run_predict(predict_opts); });

    CLI::App *help_command = app.add_subcommand("help", "show this message");

    const std::string usage = app.help("", CLI::AppFormatMode::All);

    help_command->callback([&] { fmt::print("{}", usage); });

    if (argc < 2)
    {
        fmt::print("{}", usage);
        return 1;
    }

    try
    {
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError &error)
    {
        return app.exit(error) == 0 ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        fmt::print(stderr, "error: {}\n", error.what());
        return 1;
    }

    if (app.get_subcommands().empty())
    {
        fmt::print("{}", usage);
        return 1;
    }

    return 0;
}
