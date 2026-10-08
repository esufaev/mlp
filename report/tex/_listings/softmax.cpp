[[nodiscard]] inline tensor softmax_rows(const tensor& operand)
{
    detail::require_defined(operand, "softmax_rows");

    const matrix& input = operand.value();
    if (input.cols() == 0)
        throw std::invalid_argument("softmax_rows: the input has no columns");

    const matrix exponentials = (input.colwise() - input.rowwise().maxCoeff()).array().exp();
    const matrix totals = exponentials.rowwise().sum();

    return detail::make_node(
        exponentials.array().colwise() / totals.col(0).array(), "softmax", {operand},
        [](const matrix& outer, tensor_node& self, const std::vector<tensor_node*>& inputs)
        {
            if (!inputs[0]->requires_grad)
                return;

            const matrix weighted = outer.cwiseProduct(self.value);
            inputs[0]->grad += (weighted.array() - self.value.array().colwise() *
                                                       weighted.rowwise().sum().col(0).array())
                                   .matrix();
        });
}
