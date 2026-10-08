[[nodiscard]] inline tensor matmul(const tensor& left, const tensor& right)
{
    detail::require_defined(left, "matmul");
    detail::require_defined(right, "matmul");

    if (left.cols() != right.rows())
        throw std::invalid_argument(fmt::format("matmul: cannot multiply {} by {}",
                                                shape_string(left.value()),
                                                shape_string(right.value())));

    return detail::make_node(
        left.value() * right.value(), "matmul", {left, right},
        [](const matrix& outer, tensor_node&, const std::vector<tensor_node*>& inputs)
        {
            if (inputs[0]->requires_grad)
                inputs[0]->grad += outer * inputs[1]->value.transpose();
            if (inputs[1]->requires_grad)
                inputs[1]->grad += inputs[0]->value.transpose() * outer;
        });
}
