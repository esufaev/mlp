#include "mlp/mlp.h"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <set>

TEST_CASE("the same seed replays the same sequence", "[random]")
{
    mlp::rng first(1234);
    mlp::rng second(1234);

    for (int i = 0; i < 20; ++i)
        REQUIRE(first.uniform(-1.0, 1.0) == second.uniform(-1.0, 1.0));

    REQUIRE(first.permutation(50) == second.permutation(50));
}

TEST_CASE("a different seed gives a different sequence", "[random]")
{
    mlp::rng first(1);
    mlp::rng second(2);
    REQUIRE(first.permutation(100) != second.permutation(100));
}

TEST_CASE("a permutation contains every index exactly once", "[random]")
{
    mlp::rng rng(7);
    const std::vector<std::size_t> order = rng.permutation(64);

    REQUIRE(order.size() == 64);
    const std::set<std::size_t> unique(order.begin(), order.end());
    REQUIRE(unique.size() == 64);
    REQUIRE(*unique.rbegin() == 63);
}

TEST_CASE("uniform stays inside its bounds", "[random]")
{
    mlp::rng rng(99);
    for (int i = 0; i < 500; ++i)
    {
        const double value = rng.uniform(-0.25, 0.75);
        REQUIRE(value >= -0.25);
        REQUIRE(value <= 0.75);
    }
}
