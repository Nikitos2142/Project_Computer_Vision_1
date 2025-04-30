import matplotlib.pyplot as plt
import os

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)

class PlotModelResults:
    def __init__(self, training_results):
        self.training_results = training_results

    # Retrieve training results.
    def retrive_TR(self):
        train_loss = self.training_results.history["loss"]
        train_acc = self.training_results.history["accuracy"]
        valid_loss = self.training_results.history["val_loss"]
        valid_acc = self.training_results.history["val_accuracy"]
        return train_loss, train_acc, valid_loss, valid_acc

    # plot "Training Loss vs Validation Loss" and "Training accuracy vs Validation accuracy"
    def plot_results(self, metrics, title=None, ylabel=None,
                     ylim=None, xlabel=None, xlim=None,
                     metric_name=None, color=None,
                     output_dir=None, filename=None, save=False):

        fig, ax = plt.subplots(figsize=(15, 4))

        if not (isinstance(metric_name, list) or isinstance(metric_name, tuple)):
            metrics = [metrics, ]
            metric_name = [metric_name, ]

        for idx, metric in enumerate(metrics):
            ax.plot(metric, color=color[idx])

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        #plt.xlim([0, 20])
        #plt.ylim(ylim)
        # Tailor x-axis tick marks
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        plt.grid(True)
        #saving to the project directory
        if save:
            if output_dir is None:
                output_dir = "./Targil_1/Configurations_plots"
            os.makedirs(output_dir, exist_ok=True)

            if filename is None and title is not None:
                filename = title.replace(" ", "_").replace(":", "") + ".png"
        path = os.path.join(output_dir, filename)
        plt.savefig(path)
        print(f"Plot saved to '{path}'")

        plt.legend(metric_name)
        plt.show()
        plt.close()


