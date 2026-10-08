for (std::size_t epoch = 0; epoch < m_config.epochs; ++epoch)
{
    const std::vector<std::size_t> order = rng.permutation(sample_count);

    for (std::size_t start = 0; start < sample_count; start += m_config.batch_size)
    {
        const std::size_t stop = std::min(start + m_config.batch_size, sample_count);
        const std::vector<std::size_t> batch(
            order.begin() + static_cast<std::ptrdiff_t>(start),
            order.begin() + static_cast<std::ptrdiff_t>(stop));

        zero_grad(parameters);

        const tensor predictions =
            network.forward(tensor::constant(train_inputs(batch, Eigen::all)));
        const tensor batch_loss = (*loss)(predictions, train_targets(batch, Eigen::all));

        batch_loss.backward();
        optimizer.step(parameters);
    }

}
