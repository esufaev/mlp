#include "mlp/mlp.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using Catch::Approx;

namespace
{
    std::filesystem::path scratch_file(const std::string& name)
    {
        const std::filesystem::path directory = std::filesystem::temp_directory_path() / "mlp-tests";
        std::filesystem::create_directories(directory);
        return directory / name;
    }

    void write_text(const std::filesystem::path& path, const std::string& text)
    {
        std::ofstream output(path);
        output << text;
    }

    std::filesystem::path write_tiny_wdbc(const std::string& name)
    {
        std::string text;
        for (int row = 0; row < 2; ++row)
        {
            text += row == 0 ? "1001,M" : "1002,B";
            for (int feature = 0; feature < 30; ++feature)
                text += "," + std::to_string(row * 100 + feature);
            text += "\n";
        }

        const std::filesystem::path path = scratch_file(name);
        write_text(path, text);
        return path;
    }
}

TEST_CASE("a written table is exactly the text we expect", "[csv]")
{
    const std::filesystem::path path = scratch_file("roundtrip.csv");
    const mlp::csv_table original{{"a", "b", "c"}, {"1", "2", "3"}, {"4", "5", "6"}};

    mlp::write_csv(path, original);

    std::ifstream input(path);
    const std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    REQUIRE(contents == "a,b,c\n1,2,3\n4,5,6\n");
}

TEST_CASE("an empty cell is written as an empty field", "[csv]")
{
    const std::filesystem::path path = scratch_file("empty_cell.csv");
    mlp::write_csv(path, {{"epoch", "loss", "val_loss"}, {"1", "0.5", ""}});

    std::ifstream input(path);
    const std::string contents((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    REQUIRE(contents == "epoch,loss,val_loss\n1,0.5,\n");
}

TEST_CASE("write_csv creates the directories it needs", "[csv]")
{
    const std::filesystem::path nested = scratch_file("made/up/path/table.csv");
    std::filesystem::remove_all(scratch_file("made"));

    mlp::write_csv(nested, {{"1", "2"}});
    REQUIRE(std::filesystem::exists(nested));
}

TEST_CASE("fields are trimmed on the way in", "[dataset]")
{
    const std::filesystem::path path = scratch_file("spaces.csv");
    std::string text = " 842302 , M ";
    for (int feature = 0; feature < 30; ++feature)
        text += " , 1.5 ";
    write_text(path, text + "\n");

    const mlp::dataset dataset = mlp::load_wdbc(path);
    REQUIRE(dataset.ids[0] == "842302");
    REQUIRE(dataset.labels == std::vector<int>{1});
    REQUIRE(dataset.inputs(0, 0) == Approx(1.5));
}

TEST_CASE("a ragged row is reported with its position", "[dataset]")
{
    const std::filesystem::path path = scratch_file("ragged.csv");

    std::string text;
    for (int row = 0; row < 3; ++row)
    {
        text += row == 0 ? "1001,M" : "1002,B";
        const int fields = row == 2 ? 4 : 30;
        for (int feature = 0; feature < fields; ++feature)
            text += ",1";
        text += "\n";
    }
    write_text(path, text);

    REQUIRE_THROWS_WITH(mlp::load_wdbc(path), Catch::Matchers::ContainsSubstring("row 3"));
}

TEST_CASE("a non-numeric feature is reported with its position", "[dataset]")
{
    const std::filesystem::path path = scratch_file("not_a_number.csv");
    std::string text = "1001,M";
    for (int feature = 0; feature < 30; ++feature)
        text += feature == 2 ? ",oops" : ",1";
    write_text(path, text + "\n");

    REQUIRE_THROWS_WITH(mlp::load_wdbc(path), Catch::Matchers::ContainsSubstring("column 5"));
}

TEST_CASE("blank lines are skipped and an empty file is an error", "[dataset]")
{
    const std::filesystem::path path = scratch_file("blanks.csv");

    std::string row = "1001,M";
    for (int feature = 0; feature < 30; ++feature)
        row += ",1";
    write_text(path, row + "\n\n" + row + "\n\n");
    REQUIRE(mlp::load_wdbc(path).size() == 2);

    const std::filesystem::path empty = scratch_file("empty.csv");
    write_text(empty, "\n\n");
    REQUIRE_THROWS_AS(mlp::load_wdbc(empty), std::runtime_error);
}

TEST_CASE("a wrong number of columns is an error", "[dataset]")
{
    const std::filesystem::path path = scratch_file("too_few_columns.csv");
    write_text(path, "1001,M,1,2,3\n");

    REQUIRE_THROWS_WITH(mlp::load_wdbc(path), Catch::Matchers::ContainsSubstring("expected 32"));
}

TEST_CASE("a missing file is reported, not crashed on", "[dataset]")
{
    REQUIRE_THROWS_AS(mlp::load_wdbc("/definitely/not/here.csv"), std::runtime_error);
}

TEST_CASE("the wdbc loader maps diagnoses to class indices", "[dataset]")
{
    const mlp::dataset dataset = mlp::load_wdbc(write_tiny_wdbc("tiny_wdbc.csv"));
    REQUIRE(dataset.size() == 2);
    REQUIRE(dataset.feature_count() == 30);
    REQUIRE(dataset.labels == std::vector<int>{1, 0});
    REQUIRE(dataset.ids[0] == "1001");
    REQUIRE(dataset.inputs(1, 0) == Approx(100.0));

    const std::vector<std::size_t> counts = dataset.class_counts();
    REQUIRE(counts[0] == 1);
    REQUIRE(counts[1] == 1);
}

TEST_CASE("an unexpected diagnosis is rejected", "[dataset]")
{
    const std::filesystem::path path = scratch_file("bad_label.csv");
    std::string text = "1001,X";
    for (int feature = 0; feature < 30; ++feature)
        text += ",1";
    write_text(path, text + "\n");

    REQUIRE_THROWS_AS(mlp::load_wdbc(path), std::runtime_error);
}

TEST_CASE("a dataset round-trips through its own file format", "[dataset]")
{
    const mlp::dataset original = mlp::load_wdbc(write_tiny_wdbc("tiny_wdbc_source.csv"));

    const std::filesystem::path copy = scratch_file("tiny_wdbc_copy.csv");
    original.save(copy);
    const mlp::dataset reloaded = mlp::load_wdbc(copy);

    REQUIRE(reloaded.labels == original.labels);
    REQUIRE(reloaded.ids == original.ids);
    REQUIRE(reloaded.inputs.cwiseEqual(original.inputs).all());
}
