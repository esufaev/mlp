#pragma once

#include "mlp/train/history.h"

#include <matplot/matplot.h>

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace mlp
{
inline std::vector<std::filesystem::path>
plot_history(const history& history, const std::filesystem::path& directory, bool show = false)
{
	std::vector<std::filesystem::path> written;
	if (history.epochs() == 0)
		return written;

	std::filesystem::create_directories(directory);

	std::vector<double> epochs(history.epochs());
	for (std::size_t i = 0; i < epochs.size(); ++i)
		epochs[i] = static_cast<double>(i + 1);

	const auto draw = [&](const std::string& title, const std::string& y_label,
						  const std::vector<double>& train, const std::vector<double>& validation,
						  const std::filesystem::path& path)
	{
		auto figure = matplot::figure(true);
		figure->quiet_mode(!show);

		auto axes = figure->current_axes();
		axes->hold(true);
		axes->plot(epochs, train)->line_width(2);
		if (!validation.empty())
			axes->plot(epochs, validation)->line_width(2);
		axes->hold(false);

		axes->title(title);
		axes->xlabel("epoch");
		axes->ylabel(y_label);

		if (validation.empty())
			axes->legend(std::vector<std::string>{"training"});
		else
			axes->legend(std::vector<std::string>{"training", "validation"});

		figure->save(path.string());
		if (show)
			figure->show();
	};

	const std::filesystem::path loss_path = directory / "learning_loss.png";
	const std::filesystem::path accuracy_path = directory / "learning_accuracy.png";

	draw("Learning curves - loss", "loss", history.train_loss, history.validation_loss, loss_path);
	draw("Learning curves - accuracy", "accuracy", history.train_accuracy,
		 history.validation_accuracy, accuracy_path);

	written.push_back(loss_path);
	written.push_back(accuracy_path);
	return written;
}
} // namespace mlp
