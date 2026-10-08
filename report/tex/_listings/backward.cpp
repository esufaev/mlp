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
