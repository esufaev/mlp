void step(const std::vector<tensor>& parameters) override
{
    const bool use_momentum = m_momentum > 0.0;
    if (use_momentum && m_velocities.size() != parameters.size())
    {
        m_velocities.clear();
        m_velocities.reserve(parameters.size());
        for (const tensor& parameter : parameters)
            m_velocities.push_back(
                matrix::Zero(parameter.value().rows(), parameter.value().cols()));
    }

    for (std::size_t i = 0; i < parameters.size(); ++i)
    {
        matrix& value = parameters[i].mutable_value();
        const matrix& gradient = parameters[i].grad();

        if (value.rows() != gradient.rows() || value.cols() != gradient.cols())
            throw std::invalid_argument(
                fmt::format("sgd::step: parameter {} is {} but its gradient is {}", i,
                            shape_string(value), shape_string(gradient)));

        if (use_momentum)
        {
            matrix& velocity = m_velocities[i];
            velocity = m_momentum * velocity - m_learning_rate * gradient;
            value += velocity;
        }
        else
        {
            value -= m_learning_rate * gradient;
        }
    }
}
