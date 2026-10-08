#pragma once

#include "mlp/autograd/ops.h"
#include "mlp/autograd/tensor.h"

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace mlp
{
class module
{
  public:
    module() = default;
    module(const module &) = default;
    module(module &&) = default;
    module &operator=(const module &) = default;
    module &operator=(module &&) = default;
    virtual ~module() = default;

    virtual tensor forward(const tensor &input) = 0;

    tensor operator()(const tensor &input) { return forward(input); }

    virtual std::vector<tensor> parameters() { return {}; }

    [[nodiscard]] virtual std::string type() const = 0;

    [[nodiscard]] virtual std::string config() const { return {}; }

    virtual void write_parameters(std::ostream &stream) const { (void)stream; }
    virtual void read_parameters(std::istream &stream) { (void)stream; }

    void zero_grad()
    {
        for (const tensor &parameter : parameters())
            parameter.zero_grad();
    }

    [[nodiscard]] std::size_t parameter_count()
    {
        std::size_t total = 0;
        for (const tensor &parameter : parameters())
            total += static_cast<std::size_t>(parameter.value().size());
        return total;
    }
};
} // namespace mlp
