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
