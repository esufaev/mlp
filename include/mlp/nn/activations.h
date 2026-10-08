#pragma once

#include "mlp/nn/module.h"

#include <fmt/format.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mlp
{
    enum class activation
    {
        sigmoid,
        relu,
        tanh,
        softmax
    };

    [[nodiscard]] inline activation activation_from_string(std::string_view name)
    {
        if (name == "sigmoid")
            return activation::sigmoid;
        if (name == "relu" || name == "ReLU")
            return activation::relu;
        if (name == "tanh")
            return activation::tanh;
        if (name == "softmax")
            return activation::softmax;

        throw std::invalid_argument(
            fmt::format("unknown activation '{}' (expected sigmoid, relu, tanh or softmax)", name));
    }

    [[nodiscard]] inline std::string to_string(activation activation)
    {
        switch (activation)
        {
        case activation::sigmoid:
            return "sigmoid";
        case activation::relu:
            return "relu";
        case activation::tanh:
            return "tanh";
        case activation::softmax:
            return "softmax";
        }
        return "sigmoid";
    }

    class activation_module final : public module
    {
    public:
        explicit activation_module(activation kind) : m_kind(kind) {}

        tensor forward(const tensor& input) override
        {
            switch (m_kind)
            {
            case activation::sigmoid:
                return sigmoid(input);
            case activation::relu:
                return relu(input);
            case activation::tanh:
                return tanh(input);
            case activation::softmax:
                return softmax_rows(input);
            }
            throw std::logic_error("activation_module::forward: unreachable activation");
        }

        [[nodiscard]] std::string type() const override { return to_string(m_kind); }

        [[nodiscard]] activation kind() const noexcept { return m_kind; }

    private:
        activation m_kind;
    };

    [[nodiscard]] inline std::unique_ptr<module> make_activation(activation activation)
    {
        return std::make_unique<activation_module>(activation);
    }

    [[nodiscard]] inline std::unique_ptr<module> make_activation_by_type(std::string_view type)
    {
        if (type == "sigmoid")
            return make_activation(activation::sigmoid);
        if (type == "relu")
            return make_activation(activation::relu);
        if (type == "tanh")
            return make_activation(activation::tanh);
        if (type == "softmax")
            return make_activation(activation::softmax);
        return nullptr;
    }
}
