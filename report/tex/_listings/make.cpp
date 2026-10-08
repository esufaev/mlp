[[nodiscard]] inline tensor make_node(matrix value, const char* op, std::vector<tensor> inputs,
                                      gradient_rule rule)
{
    bool tracked = false;
    if (grad_enabled())
        for (const tensor& input : inputs)
            if (input.requires_grad())
                tracked = true;

    if (!tracked)
        return tensor::constant(std::move(value));

    auto node = std::make_shared<tensor_node>();
    node->grad = matrix::Zero(value.rows(), value.cols());
    node->value = std::move(value);
    node->requires_grad = true;
    node->op = op;

    node->inputs.reserve(inputs.size());
    for (const tensor& input : inputs)
        node->inputs.push_back(input.node());

    std::vector<tensor_node*> raw;
    raw.reserve(node->inputs.size());
    for (const node_pointer& each : node->inputs)
        raw.push_back(each.get());

    node->propagate = [self = node.get(), raw = std::move(raw),
                       rule = std::move(rule)](const matrix& outer) { rule(outer, *self, raw); };

    return tensor(std::move(node));
}
