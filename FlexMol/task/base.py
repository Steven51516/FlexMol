import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd
import os
from abc import ABC, abstractmethod
from functools import partial

from FlexMol.util.metrics import Evaluator
from .FMLoader import FMDataset
from FlexMol.util.data import get_collate
import numpy as np
from FlexMol.encoder import FlexMol

from tqdm import tqdm

class BaseTrainer(ABC):
    def __init__(self, FM, task, device='cpu', epochs=100, lr=0.001, batch_size=256,
                 num_workers=0, scheduler=None, early_stopping=None, patience=10, checkpoint_dir=None, metrics_dir=None, test_metrics=None, weight_decay=0.0, val_metrics = None):
        """
        Initializes the BaseTrainer class.

        Args:
            BE: BioEncoder object containing the model and encoders.
            device: Device to use for training (e.g., 'cpu' or 'cuda').
            epochs: Number of epochs to train.
            lr: Learning rate.
            batch_size: Batch size.
            num_workers: Number of workers for data loading.
            scheduler: Learning rate scheduler.
            early_stopping: Metric to use for early stopping.
            patience: Number of epochs with no improvement after which training will be stopped.
            checkpoint_dir: Directory to save the model checkpoints.
            metrics_dir: Directory to save the evaluation metrics.
            test_metrics: List of metrics to evaluate during testing.
        """
        self.device = device
        FM.set_device(self.device)
        self.model = FM.get_model().to(self.device)

        if device == 'cuda' and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, dim=0)

        self.encoders = FM.get_encoders()
        self.early_stopping = early_stopping
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.best_model_path = None

        self.evaluator = Evaluator()
        self.test_metrics = test_metrics if test_metrics else []
        self.metrics_dir = metrics_dir

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.scheduler = scheduler
        
        self.task = task
        self.encoder_input = None

        if val_metrics and not early_stopping:
            self.early_stopping = val_metrics
            self.patience = float('inf')

    @abstractmethod
    def post_process_output(self, output):
        """
        Post-processes the output of the model.

        Args:
            output: Output of the model.

        Returns:
            Processed output.
        """
        pass

    @abstractmethod
    def get_default_metrics(self):
        """
        Returns the default metrics for evaluation.

        Returns:
            List of default metrics.
        """
        pass


    def get_auto_threshold_optional(self, labels, predictions, metrics):
        """
        Determines the optimal threshold automatically if applicable.

        Args:
            labels (list or numpy array): The true labels.
            predictions (list or numpy array): The predicted values from the model.
            metrics (list): List of metrics to check if threshold calculation is needed.

        Returns:
            float: The optimal threshold value if applicable, otherwise None.
        """
        return None


    def prepare_datasets(self, train_df, val_df, test_df):
        """
        Prepares the datasets for training, validation, and testing.

        Args:
            train_df: DataFrame for training data.
            val_df: DataFrame for validation data.
            test_df: DataFrame for testing data.

        Returns:
            Tuple of transformed datasets.
        """
        combined_df = self.combine_datasets(train_df, val_df, test_df)
        transformed_combined_df = self.transform_dataset(combined_df)
        return self.separate_datasets(transformed_combined_df)
     

    def combine_datasets(self, train_df, val_df, test_df):
        """
        Combines the training, validation, and testing DataFrames into a single DataFrame.

        Args:
            train_df: DataFrame for training data.
            val_df: DataFrame for validation data.
            test_df: DataFrame for testing data.

        Returns:
            Combined DataFrame.
        """
        train_df['dataset'] = 'train'
        val_df['dataset'] = 'val'
        test_df['dataset'] = 'test'
        return pd.concat([train_df, val_df, test_df], ignore_index=True)


    def separate_datasets(self, combined_df):
        """
        Separates the combined DataFrame back into training, validation, and testing DataFrames.

        Args:
            combined_df: Transformed combined DataFrame.

        Returns:
            Tuple of separated DataFrames.
        """
        train_df = combined_df[combined_df['dataset'] == 'train'].drop(columns=['dataset']).reset_index(drop=True)
        val_df = combined_df[combined_df['dataset'] == 'val'].drop(columns=['dataset']).reset_index(drop=True)
        test_df = combined_df[combined_df['dataset'] == 'test'].drop(columns=['dataset']).reset_index(drop=True)
        return train_df, val_df, test_df

    def transform_dataset(self, df):
        """
        Transforms the dataset by applying the encoders.

        Args:
            df: DataFrame to transform.

        Returns:
            Transformed DataFrame.
        """
        data_dict = {}
        if self.task == "DTI":
            for idx, encoder in enumerate(self.encoders):
                input_type = encoder.input_type
                assert input_type in df.columns, f"Column for encoder's input type {input_type} not found in dataframe"
                if encoder.model.training_setup()["loadtime_transform"]:
                    processed_data = encoder.transform(list(df[input_type]), mode="initial")
                else:
                    processed_data = encoder.transform(list(df[input_type]))
                data_dict[f'Input_{idx}'] = processed_data
        else:
            if self.encoder_input is None:
                if len(self.encoders) != 2:
                    raise ValueError("Please assign encoder_input when number of encoders > 2.")
            
                input_map = {
                    FlexMol.DRUG: ["Drug1", "Drug2"],
                    FlexMol.PROT_3D: ["Protein1_ID", "Protein2_ID"],
                    FlexMol.PROT_SEQ: ["Protein1", "Protein2"]
                }

                self.encoder_input = []
                for idx, encoder in enumerate(self.encoders):
                    self.encoder_input.append(input_map[encoder.input_type][idx])

            for idx, encoder in enumerate(self.encoders):
                input_type = self.encoder_input[idx]
                assert input_type in df.columns, f"Column for encoder's input type {input_type} not found in dataframe"
                if encoder.model_training_setup["loadtime_transform"]:
                    processed_data = encoder.transform(list(df[input_type]), mode="initial")
                else:
                    processed_data = encoder.transform(list(df[input_type]))
                data_dict[f'Input_{idx}'] = processed_data

        data_dict['Y'] = df['Y']
        data_dict['dataset'] = df['dataset']
        return pd.DataFrame(data_dict)

    def to_device(self, *inputs_and_label):
        """
        Moves inputs and labels to the specified device.

        Args:
            inputs_and_label: Inputs and labels to move.

        Returns:
            Tuple of inputs and labels moved to the device.
        """
        inputs_and_label = list(inputs_and_label)
        for idx, encoder in enumerate(self.encoders):
            if not encoder.model.training_setup()["to_device_in_model"]:
                inputs_and_label[idx] = inputs_and_label[idx].float().to(self.device)
        inputs_and_label[-1] = torch.from_numpy(np.array(inputs_and_label[-1])).float().to(self.device)
        return tuple(inputs_and_label)

    def create_loader(self, dataframe, shuffle=True):
        """
        Creates a DataLoader for the dataset.

        Args:
            dataframe: DataFrame to load.
            shuffle: Whether to shuffle the data.

        Returns:
            DataLoader for the dataset.
        """
        data = FMDataset(dataframe, self.encoders)
        params = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False
        }
        collate_funcs = [encoder.model.training_setup()["collate_func"] for encoder in self.encoders]
        if collate_funcs:
            params['collate_fn'] = partial(get_collate, collate_func=collate_funcs)
        return DataLoader(data, shuffle=shuffle, **params)

    def train(self, train_df, val_df=None, threshold = None):
        """
        Trains the model.

        Args:
            train_df: DataFrame for training data.
            val_df: DataFrame for validation data.
        """

        early_stopping_metric = self.early_stopping
        print("Start training...")
        train_loader = self.create_loader(train_df)
        val_loader = self.create_loader(val_df) if val_df is not None else None
        best_val_metric = float('-inf') if Evaluator.get_mode(early_stopping_metric) == 'max' else float('inf')
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            self.train_one_epoch(train_loader, epoch)
            if val_loader:
                val_loss, val_labels, val_predictions = self.inference(val_loader)
                print(f'Epoch: {epoch} \tValidation Loss: {val_loss:.6f}')
                if early_stopping_metric:
                    if(early_stopping_metric == "loss"):
                        val_metric = val_loss
                    else:
                        if threshold is None:
                            threshold = self.get_auto_threshold_optional(val_labels, val_predictions, metrics = [early_stopping_metric])
                        val_metric = self.evaluator(early_stopping_metric, val_labels, val_predictions, threshold)
                        print(f'Epoch: {epoch} \tValidation {early_stopping_metric}: {val_metric:.4f}')
                    improved = (val_metric > best_val_metric) if Evaluator.get_mode(early_stopping_metric) == 'max' else (val_metric < best_val_metric)
                    if improved:
                        best_val_metric = val_metric
                        epochs_no_improve = 0
                        if self.checkpoint_dir:
                            if not os.path.exists(self.checkpoint_dir):
                                os.makedirs(self.checkpoint_dir)
                            self.best_model_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.pt')
                            torch.save(self.model.state_dict(), self.best_model_path)
                    else:
                        epochs_no_improve += 1
                        if self.early_stopping and epochs_no_improve == self.patience:
                            print(f'Early stopping triggered after {epoch} epochs.')
                            break

            if self.scheduler is not None:
                self.scheduler.step()


    def train_one_epoch(self, train_loader, epoch):
        """
        Trains the model for one epoch.

        Args:
            train_loader: DataLoader for training data.
            epoch: Current epoch number.
        """
        self.model.train()
        train_loss = 0
        total_samples = 0
        

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}', unit='batch')
        
        for batch_idx, (*inputs, label) in progress_bar:
            inputs_and_label = self.to_device(*inputs, label)
            self.optimizer.zero_grad()
            output = self.model(*inputs_and_label[:-1]).float().squeeze(1)
            loss = self.criterion(output, inputs_and_label[-1])
            loss.backward()
            self.optimizer.step()
            batch_size = inputs_and_label[-1].size(0)
            train_loss += loss.item() * batch_size
            total_samples += batch_size
            progress_bar.set_postfix({'loss': train_loss / total_samples})
        
        train_loss /= total_samples
        print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f}')
        return train_loss


    def inference(self, loader):
        """
        Performs inference on the data.

        Args:
            loader: DataLoader for the data.

        Returns:
            Tuple of total loss, all labels, and all predictions.
        """
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for *inputs, label in loader:
                inputs_and_label = self.to_device(*inputs, label)
                output = self.model(*inputs_and_label[:-1]).float().squeeze(1)
                loss = self.criterion(output, inputs_and_label[-1])
                total_loss += loss.item() * inputs_and_label[-1].size(0)
                predictions = self.post_process_output(output)
                all_labels.extend(label.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
        total_loss /= len(loader.dataset)
        return total_loss, all_labels, all_predictions

    def test(self, test_df, threshold = None):
        """
        Tests the model.

        Args:
            test_df: DataFrame for testing data.
        """
        test_loader = self.create_loader(test_df)
        if self.best_model_path:
            self.model.load_state_dict(torch.load(self.best_model_path))
        print("Start testing...")
        test_loss, all_labels, all_predictions = self.inference(test_loader)
        metrics_to_evaluate = self.test_metrics if self.test_metrics else self.get_default_metrics()
        if threshold is None:
            threshold = self.get_auto_threshold_optional(all_labels, all_predictions, metrics = metrics_to_evaluate)
        test_metrics = {"Test Loss": test_loss}
        for metric in metrics_to_evaluate:
            test_metrics[metric] = self.evaluator(metric, all_labels, all_predictions, threshold)
        self.save_or_print_metrics(test_metrics)

    def save_or_print_metrics(self, metrics):
        """
        Saves or prints the evaluation metrics.

        Args:
            metrics: Dictionary of evaluation metrics.
        """
        if self.metrics_dir:
            if not os.path.exists(self.metrics_dir):
                os.makedirs(self.metrics_dir)
            with open(os.path.join(self.metrics_dir, 'test_metrics.txt'), 'w') as f:
                for metric, value in metrics.items():
                    f.write(f'{metric}: {value:.6f}\n')
        else:
            for metric, value in metrics.items():
                print(f'{metric}: {value:.6f}')

    def save_model(self, path):
        """
        Saves the model to the specified path.

        Args:
            path (str): The path where the model will be saved.
        """
        torch.save(self.model.state_dict(), path)
