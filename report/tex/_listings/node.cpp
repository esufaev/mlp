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
