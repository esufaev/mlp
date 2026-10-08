#include "mlp/mlp.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;

TEST_CASE("accuracy counts exact matches", "[metrics]")
{
    REQUIRE(mlp::accuracy({0, 1, 1, 0}, {0, 1, 0, 0}) == Approx(0.75));
    REQUIRE(mlp::accuracy({1, 1}, {1, 1}) == Approx(1.0));
    REQUIRE(mlp::accuracy({0, 0}, {1, 1}) == Approx(0.0));

    REQUIRE_THROWS_AS(mlp::accuracy({0}, {0, 1}), std::invalid_argument);
    REQUIRE_THROWS_AS(mlp::accuracy({}, {}), std::invalid_argument);
}

TEST_CASE("the confusion matrix is indexed [actual][predicted]", "[metrics]")
{

    const std::vector<int> actual{0, 0, 0, 1, 1, 1, 1};
    const std::vector<int> predicted{0, 0, 1, 0, 1, 1, 1};

    const mlp::confusion_matrix matrix = mlp::make_confusion_matrix(predicted, actual, 2);

    REQUIRE(matrix.counts[0][0] == 2);
    REQUIRE(matrix.counts[0][1] == 1);
    REQUIRE(matrix.counts[1][0] == 1);
    REQUIRE(matrix.counts[1][1] == 3);
    REQUIRE(matrix.total() == 7);
    REQUIRE(matrix.correct() == 5);

    REQUIRE(mlp::precision(matrix, 1) == Approx(0.75));
    REQUIRE(mlp::recall(matrix, 1) == Approx(0.75));
    REQUIRE(mlp::f1_score(matrix, 1) == Approx(0.75));

    REQUIRE_THROWS_AS(mlp::precision(matrix, 9), std::out_of_range);
}

TEST_CASE("a class nobody predicts scores zero instead of dividing by zero", "[metrics]")
{
    const mlp::confusion_matrix matrix = mlp::make_confusion_matrix({0, 0, 0}, {0, 1, 1}, 2);

    REQUIRE(mlp::precision(matrix, 1) == Approx(0.0));
    REQUIRE(mlp::recall(matrix, 1) == Approx(0.0));
    REQUIRE(mlp::f1_score(matrix, 1) == Approx(0.0));
}

TEST_CASE("a label outside the class range is rejected", "[metrics]")
{
    REQUIRE_THROWS_AS(mlp::make_confusion_matrix({0, 2}, {0, 1}, 2), std::out_of_range);
    REQUIRE_THROWS_AS(mlp::make_confusion_matrix({0}, {0, 1}, 2), std::invalid_argument);
}

TEST_CASE("argmax_rows picks the most likely class", "[metrics]")
{
    const mlp::matrix probabilities = mlp::matrix{{0.2, 0.8}, {0.9, 0.1}, {0.5, 0.5}};
    REQUIRE(mlp::argmax_rows(probabilities) == std::vector<int>{1, 0, 0});
}

TEST_CASE("the reports render every class", "[metrics]")
{
    const mlp::confusion_matrix matrix = mlp::make_confusion_matrix({0, 1}, {0, 1}, 2);
    const std::vector<std::string> names{"B", "M"};

    const std::string table = mlp::format_confusion_matrix(matrix, names);
    REQUIRE(table.find('B') != std::string::npos);
    REQUIRE(table.find('M') != std::string::npos);

    const std::string report = mlp::format_classification_report(matrix, names);
    REQUIRE(report.find("precision") != std::string::npos);
    REQUIRE(report.find("f1-score") != std::string::npos);
}
