import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import csv

class EvaluationModel:

    def best_model(self, dict_of_plots, output_dir=None, filename="validation_summary.csv"):
        best_model_name = None
        best_val_acc = 0.0
        best_val_loss = float('inf')
        summary_data = []

        for model_name, (train_loss, train_acc, val_loss, val_acc) in dict_of_plots.items():
            final_val_acc = val_acc[-1]  # accuracy at final epoch
            final_val_loss = val_loss[-1]  # loss at final epoch
            summary_data.append((model_name, final_val_acc, final_val_loss))
            if final_val_acc > best_val_acc:
                best_val_acc = final_val_acc
                best_val_loss = final_val_loss
                best_model_name = model_name
            elif final_val_acc == best_val_acc and final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_model_name = model_name
        #Saving in csv
        if output_dir is None:
            output_dir = "./Targil_1/Training_results"
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir,filename)
        with open (csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Model Name", "Final Validation Accuracy", "Final Validation Loss"])
            for model_name, final_val_acc, final_val_loss in summary_data:
                writer.writerow([model_name, f"{final_val_acc:.4f}", f"{final_val_loss:.4f}"])

        print(f"Validation summary saved to '{csv_path}'")
        print(f"Best model is: '{best_model_name}' with final validation accuracy: "
              f" {best_val_acc:.4f}, and final validation loss: {best_val_loss:.4f}")

        return best_model_name

    def confusion_matrix(self,
                         y_test,
                         y_predict,
                         alphabet,
                         output_dir=None,
                         filename="matrix_confusion.csv",
                         filename_picture="Confusion_matrix.png",
                         title = "Confusion matrix",
                         save=False):
        conf_matrix_results = tf.math.confusion_matrix(y_test, y_predict)
        if output_dir is None:
            output_dir = "./Targil_1/Training_results"
        os.makedirs(output_dir, exist_ok=True)
        # Drawing confusion matrix as a heatmap.
        plt.figure(figsize=[15, 8])
        sn.heatmap(conf_matrix_results, annot=True, fmt="d", annot_kws={"size": 14},
                   xticklabels=alphabet, yticklabels=alphabet)
        plt.xlabel("Predicted")
        plt.ylabel("Truth")
        plt.title(title)
        plt.tight_layout()
        if save:
            cm_plot_path = os.path.join(output_dir, filename_picture)
            plt.savefig(cm_plot_path)
            print(f"Confusion matrix plot saved to '{cm_plot_path}'")
            plt.show()

        #saving to CSV
        cm_csv_path = os.path.join(output_dir, filename)
        df_confusion_matrix = pd.DataFrame(conf_matrix_results.numpy(), index=alphabet, columns=alphabet)
        df_confusion_matrix.to_csv(cm_csv_path)
        print(f"Confusion matrix is exported to {cm_csv_path}")
        return cm_csv_path


