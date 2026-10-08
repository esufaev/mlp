#pragma once

#include "mlp/autograd/tensor.h"

#include <string>
#include <vector>

namespace mlp
{
class optimizer
{
  public:
    optimizer() = default;
    optimizer(const optimizer &) = default;
    optimizer(optimizer &&) = default;
    optimizer &operator=(const optimizer &) = default;
    optimizer &operator=(optimizer &&) = default;
    virtual ~optimizer() = default;

    virtual void step(const std::vector<tensor> &parameters) = 0;

    [[nodiscard]] virtual std::string name() const = 0;
    [[nodiscard]] virtual double learning_rate() const = 0;
};

inline void zero_grad(const std::vector<tensor> &parameters)
{
    for (const tensor &parameter : parameters)
        parameter.zero_grad();
}
} // namespace mlp
