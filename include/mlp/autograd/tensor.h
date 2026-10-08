#pragma once

#include "mlp/core/linalg.h"

#include <fmt/format.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mlp
{
struct tensor_node;

using node_pointer = std::shared_ptr<tensor_node>;

struct tensor_node
{
    matrix value;
    matrix grad;
    bool requires_grad{false};
    std::string op{"leaf"};
    std::vector<node_pointer> inputs;
    std::function<void(const matrix &)> propagate;
};

namespace detail
{
[[nodiscard]] inline bool &grad_mode()
{
    static thread_local bool enabled = true;
    return enabled;
}
} // namespace detail

[[nodiscard]] inline bool grad_enabled() { return detail::grad_mode(); }

class no_grad
{
  public:
    no_grad() : m_previous(detail::grad_mode()) { detail::grad_mode() = false; }

    no_grad(const no_grad &) = delete;
    no_grad(no_grad &&) = delete;
    no_grad &operator=(const no_grad &) = delete;
    no_grad &operator=(no_grad &&) = delete;

    ~no_grad() { detail::grad_mode() = m_previous; }

  private:
    bool m_previous;
};

class tensor
{
  public:
    tensor() = default;

    explicit tensor(node_pointer node) : m_node(std::move(node)) {}

    [[nodiscard]] static tensor constant(matrix value)
    {
        auto node = std::make_shared<tensor_node>();
        node->grad = matrix::Zero(value.rows(), value.cols());
        node->value = std::move(value);
        return tensor(std::move(node));
    }

    [[nodiscard]] static tensor parameter(matrix value)
    {
        tensor result = constant(std::move(value));
        result.m_node->requires_grad = true;
        return result;
    }

    [[nodiscard]] bool defined() const noexcept { return m_node != nullptr; }

    [[nodiscard]] const node_pointer &node() const noexcept { return m_node; }

    [[nodiscard]] const matrix &value() const
    {
        require_defined("value");
        return m_node->value;
    }

    [[nodiscard]] const matrix &grad() const
    {
        require_defined("grad");
        return m_node->grad;
    }

    [[nodiscard]] matrix &mutable_value() const
    {
        require_defined("mutable_value");
        return m_node->value;
    }

    [[nodiscard]] matrix &mutable_grad() const
    {
        require_defined("mutable_grad");
        return m_node->grad;
    }

    [[nodiscard]] bool requires_grad() const noexcept
    {
        return m_node != nullptr && m_node->requires_grad;
    }

    [[nodiscard]] const std::string &op() const
    {
        require_defined("op");
        return m_node->op;
    }

    [[nodiscard]] std::size_t rows() const { return static_cast<std::size_t>(value().rows()); }
    [[nodiscard]] std::size_t cols() const { return static_cast<std::size_t>(value().cols()); }

    void zero_grad() const
    {
        require_defined("zero_grad");
        m_node->grad.fill(0.0);
    }

    void backward() const
    {
        require_defined("backward");
        if (m_node->value.rows() != 1 || m_node->value.cols() != 1)
            throw std::invalid_argument(
                fmt::format("tensor::backward: the root is {}, only a [1 x 1] root seeds itself",
                            shape_string(m_node->value)));
        backward(matrix::Constant(1, 1, 1.0));
    }

    void backward(const matrix &seed) const
    {
        require_defined("backward");
        if (seed.rows() != m_node->value.rows() || seed.cols() != m_node->value.cols())
            throw std::invalid_argument(
                fmt::format("tensor::backward: seed {} does not match the root {}",
                            shape_string(seed), shape_string(m_node->value)));

        const std::vector<tensor_node *> order = reverse_topological_order(m_node.get());

        for (tensor_node *each : order)
            if (!each->inputs.empty())
                each->grad.fill(0.0);

        m_node->grad += seed;

        for (tensor_node *each : order)
            if (each->propagate)
                each->propagate(each->grad);
    }

    [[nodiscard]] std::size_t graph_size() const
    {
        require_defined("graph_size");
        return reverse_topological_order(m_node.get()).size();
    }

  private:
    void require_defined(const char *who) const
    {
        if (m_node == nullptr)
            throw std::logic_error(fmt::format("tensor::{}: the tensor is empty", who));
    }

    [[nodiscard]] static std::vector<tensor_node *> reverse_topological_order(tensor_node *root)
    {
        std::vector<tensor_node *> order;
        std::unordered_set<tensor_node *> visited;
        std::vector<std::pair<tensor_node *, std::size_t>> stack;

        visited.insert(root);
        stack.emplace_back(root, 0);

        while (!stack.empty())
        {
            tensor_node *node = stack.back().first;
            const std::size_t index = stack.back().second;

            if (index < node->inputs.size())
            {
                stack.back().second = index + 1;
                tensor_node *next = node->inputs[index].get();
                if (visited.insert(next).second)
                    stack.emplace_back(next, 0);
            }
            else
            {
                order.push_back(node);
                stack.pop_back();
            }
        }

        std::vector<tensor_node *> reversed;
        reversed.reserve(order.size());
        for (std::size_t i = order.size(); i-- > 0;)
            reversed.push_back(order[i]);
        return reversed;
    }

    node_pointer m_node;
};
} // namespace mlp
