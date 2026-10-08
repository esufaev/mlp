#pragma once

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
    class residual final : public module
    {
    public:
        residual() = default;

        void add(std::unique_ptr<module> child) { m_inner.add(std::move(child)); }

        tensor forward(const tensor& input) override
        {
            const tensor inner = m_inner.forward(input);

            if (inner.rows() != input.rows() || inner.cols() != input.cols())
                throw std::invalid_argument(
                    fmt::format("residual::forward: the branch returns {} but the input is {}",
                                shape_string(inner.value()), shape_string(input.value())));

            return inner + input;
        }

        std::vector<tensor> parameters() override { return m_inner.parameters(); }

        [[nodiscard]] std::string type() const override { return "residual"; }

        [[nodiscard]] std::string config() const override { return m_inner.config(); }

        [[nodiscard]] std::size_t size() const noexcept { return m_inner.size(); }

        [[nodiscard]] sequential& branch() noexcept { return m_inner; }
        [[nodiscard]] const sequential& branch() const noexcept { return m_inner; }

    private:
        sequential m_inner;
    };
}
