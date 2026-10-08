#pragma once

#include "mlp/nn/module.h"

#include <fmt/format.h>

#include <cstddef>
#include <istream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mlp
{
    class sequential final : public module
    {
    public:
        sequential() = default;

        void add(std::unique_ptr<module> child)
        {
            if (child == nullptr)
                throw std::invalid_argument("sequential::add: null module");
            m_children.push_back(std::move(child));
        }

        tensor forward(const tensor& input) override
        {
            if (m_children.empty())
                throw std::logic_error("sequential::forward: the container is empty");

            tensor current = input;
            for (std::unique_ptr<module>& child : m_children)
                current = child->forward(current);
            return current;
        }

        std::vector<tensor> parameters() override
        {
            std::vector<tensor> all;
            for (std::unique_ptr<module>& child : m_children)
                for (const tensor& parameter : child->parameters())
                    all.push_back(parameter);
            return all;
        }

        [[nodiscard]] std::string type() const override { return "sequential"; }

        [[nodiscard]] std::string config() const override { return fmt::format("{}", m_children.size()); }

        [[nodiscard]] std::size_t size() const noexcept { return m_children.size(); }

        [[nodiscard]] module& at(std::size_t index)
        {
            require_inside(index);
            return *m_children[index];
        }

        [[nodiscard]] const module& at(std::size_t index) const
        {
            require_inside(index);
            return *m_children[index];
        }

    private:
        void require_inside(std::size_t index) const
        {
            if (index >= m_children.size())
                throw std::out_of_range(
                    fmt::format("sequential::at: {} is outside of {} modules", index, m_children.size()));
        }

        std::vector<std::unique_ptr<module>> m_children;
    };
}
