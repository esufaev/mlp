#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace mlp
{
    class rng
    {
    public:
        explicit rng(std::uint64_t seed = 42) : m_engine(seed), m_seed(seed) {}

        [[nodiscard]] std::uint64_t seed() const noexcept { return m_seed; }

        [[nodiscard]] double uniform(double low, double high)
        {
            std::uniform_real_distribution<double> distribution(low, high);
            return distribution(m_engine);
        }

        [[nodiscard]] double normal(double mean, double stddev)
        {
            std::normal_distribution<double> distribution(mean, stddev);
            return distribution(m_engine);
        }

        template <typename T>
        void shuffle(std::vector<T>& values)
        {
            std::shuffle(values.begin(), values.end(), m_engine);
        }


        [[nodiscard]] std::vector<std::size_t> permutation(std::size_t count)
        {
            std::vector<std::size_t> indices(count);
            std::iota(indices.begin(), indices.end(), std::size_t{0});
            shuffle(indices);
            return indices;
        }

        [[nodiscard]] std::mt19937_64& engine() noexcept { return m_engine; }

    private:
        std::mt19937_64 m_engine;
        std::uint64_t m_seed;
    };
}
