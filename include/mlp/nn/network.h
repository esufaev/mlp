#pragma once

#include "mlp/nn/activations.h"
#include "mlp/nn/dense.h"
#include "mlp/nn/initializers.h"
#include "mlp/nn/module.h"
#include "mlp/nn/residual.h"
#include "mlp/nn/sequential.h"

#include <fmt/format.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mlp
{
    struct architecture
    {
        std::size_t input_size{30};
        std::vector<std::size_t> hidden_sizes{24, 24};
        std::size_t output_size{2};
        activation hidden_activation{activation::sigmoid};
        activation output_activation{activation::softmax};
        initializer weight_initializer{initializer::he_uniform};
    };

    class network
    {
    public:
        network() = default;

        void add(std::unique_ptr<module> child) { m_root.add(std::move(child)); }

        tensor forward(const tensor& input) { return m_root.forward(input); }

        [[nodiscard]] matrix forward(const matrix& input)
        {
            const no_grad guard;
            return m_root.forward(tensor::constant(input)).value();
        }

        [[nodiscard]] matrix predict_proba(const matrix& input) { return forward(input); }

        [[nodiscard]] std::vector<int> predict(const matrix& input)
        {
            const matrix probabilities = forward(input);

            std::vector<int> classes(static_cast<std::size_t>(probabilities.rows()));
            for (Eigen::Index i = 0; i < probabilities.rows(); ++i)
            {
                Eigen::Index best = 0;
                probabilities.row(i).maxCoeff(&best);
                classes[static_cast<std::size_t>(i)] = static_cast<int>(best);
            }
            return classes;
        }

        [[nodiscard]] std::vector<tensor> parameters() { return m_root.parameters(); }

        void zero_grad() { m_root.zero_grad(); }

        [[nodiscard]] std::size_t module_count() const noexcept { return m_root.size(); }

        [[nodiscard]] module& module_at(std::size_t index) { return m_root.at(index); }

        [[nodiscard]] const module& module_at(std::size_t index) const { return m_root.at(index); }

        [[nodiscard]] sequential& root() noexcept { return m_root; }

        [[nodiscard]] const sequential& root() const noexcept { return m_root; }

        [[nodiscard]] std::size_t parameter_count() { return m_root.parameter_count(); }

        [[nodiscard]] std::string summary()
        {
            std::string output = fmt::format("{:<6}{:<12}{:<16}{:>12}\n", "#", "layer", "shape", "parameters");
            output += std::string(46, '-') + '\n';

            for (std::size_t i = 0; i < m_root.size(); ++i)
            {
                module& current = m_root.at(i);

                std::string shape = current.config();
                if (!shape.empty())
                    shape = fmt::format("[{}]", shape);

                output +=
                    fmt::format("{:<6}{:<12}{:<16}{:>12}\n", i, current.type(), shape, current.parameter_count());
            }

            output += std::string(46, '-') + '\n';
            output += fmt::format("{:<34}{:>12}\n", "total trainable parameters", parameter_count());
            return output;
        }

        [[nodiscard]] static network build(const architecture& architecture, rng& rng)
        {
            if (architecture.input_size == 0 || architecture.output_size == 0)
                throw std::invalid_argument("network::build: input and output sizes must be positive");

            network net;
            std::size_t previous = architecture.input_size;

            for (const std::size_t width : architecture.hidden_sizes)
            {
                if (width == 0)
                    throw std::invalid_argument("network::build: a hidden layer cannot have zero units");
                net.add(std::make_unique<dense>(previous, width, architecture.weight_initializer, rng));
                net.add(make_activation(architecture.hidden_activation));
                previous = width;
            }

            net.add(
                std::make_unique<dense>(previous, architecture.output_size, architecture.weight_initializer, rng));
            net.add(make_activation(architecture.output_activation));
            return net;
        }

    private:
        sequential m_root;
    };
}
