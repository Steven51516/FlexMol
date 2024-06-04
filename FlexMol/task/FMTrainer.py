from sklearn.metrics import roc_curve
from .base import BaseTrainer
import torch.nn as nn
import torch
import numpy as np


class BinaryTrainer(BaseTrainer):
    def __init__(self, BE, task, device='cpu', epochs=100, lr=0.001, batch_size=256,
                 num_workers=0, scheduler=None, early_stopping=False, patience=10, checkpoint_dir=None, metrics_dir=None, auto_threshold= None, test_metrics=None):
        super().__init__(BE, task, device, epochs, lr, batch_size, num_workers, scheduler, early_stopping, patience, checkpoint_dir, metrics_dir, test_metrics)
        self.criterion = nn.BCEWithLogitsLoss()
        self.auto_thresold = auto_threshold

    def post_process_output(self, output):
        return torch.sigmoid(output)
        
    def get_auto_threshold_optional(self, labels, predictions, metrics):
        if any(metric in ["precision", "recall", "f1", "accuracy"] for metric in metrics):
            if(self.auto_thresold == "max-f1"):
                all_labels = np.array(labels)
                all_predictions = np.array(predictions)
                fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
                precision = tpr / (tpr + fpr + 0.00001)  # Avoid division by zero
                f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
                try :
                    optimal_idx = np.argmax(f1[5:]) + 5  # Skipping the first 5 values
                except:
                    optimal_idx = np.argmax(f1)
                return thresholds[optimal_idx]
            else:
                return 0.5
        return None


    def get_default_metrics(self):
        return ["roc-auc", "accuracy", "precision", "recall", "f1"]
    

class RegressionTrainer(BaseTrainer):
    def __init__(self, BE, task, device='cpu', epochs=100, lr=0.001, batch_size=256,
                 num_workers=0, scheduler=None, early_stopping=False, patience=10, checkpoint_dir=None, metrics_dir=None):
        super().__init__(BE, task, device, epochs, lr, batch_size, num_workers, scheduler, early_stopping, patience, checkpoint_dir, metrics_dir)
        self.criterion = nn.MSELoss()

    def post_process_output(self, output):
        return output

    def get_default_metrics(self):
        return ["mse", "mae", "r2", "rmse", "pcc"]


class MultiClassTrainer(BaseTrainer):
    def __init__(self, BE, task, device='cpu', epochs=100, lr=0.001, batch_size=256,
                 num_workers=0, scheduler=None, early_stopping=False, patience=10, checkpoint_dir=None, metrics_dir=None):
        super().__init__(BE, task, device, epochs, lr, batch_size, num_workers, scheduler, early_stopping, patience, checkpoint_dir, metrics_dir)
        self.criterion = nn.CrossEntropyLoss()

    def post_process_output(self, output):
        return torch.softmax(output, dim=1)

    def get_default_metrics(self):
        return ["accuracy", "micro-f1", "macro-f1", "kappa"]
