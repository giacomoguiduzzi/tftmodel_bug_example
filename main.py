import sys
from typing import Tuple, Optional, Union
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import TFTModel
from darts.utils.data import MixedCovariatesSequentialDataset, MixedCovariatesInferenceDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error as mse


class CustomDataset:
    """
    A custom dataset class that contains common attributes and methods for training and inference datasets.
    """
    def __init__(self):
        # Initialize all attributes to None
        self.labels_timestamps = None
        self.labels = None
        self.timestamps = None
        self.data = None
        self.features = None
        self.target = None
        self.data_windows = None

    def check_unique_timestamps(self):
        """
        Check the uniqueness of timestamps in the data windows.
        If timestamps are not unique, print the count of each timestamp and raise a ValueError.
        """
        timestamps_unique = np.unique(self.timestamps, axis=0)
        if timestamps_unique.shape != self.timestamps.shape:
            for timestamp, count in zip(
                    *np.unique(self.timestamps, return_counts=True, axis=0)
            ):
                if count > 1:
                    print(f"{int(timestamp)}: {count}.")

            raise ValueError("Timestamps in the data windows are not unique.")

    def init_common_attributes(self, **kwargs):
        """
        Initialize common attributes for the dataset.
        """
        self.data_windows = kwargs["data_windows"]
        self.target = kwargs["target_feature"]
        self.features = len(self.data_windows["data"][0][0])
        self.data = self.data_windows["data"]
        self.timestamps = self.data_windows["timestamps"]
        self.labels = self.data_windows["labels"]
        self.labels_timestamps = self.data_windows["labels_timestamps"]

    def common_init(self):
        """
        This method should be implemented in subclasses.
        """
        raise NotImplementedError


class CustomTrainingDataset(CustomDataset):
    def __init__(self):
        super().__init__()
        self.output_chunk_length = None
        self.input_chunk_length = None

    def common_init(self, **kwargs):
        """
        Initialize common attributes for the training dataset.
        """
        self.input_chunk_length = kwargs["input_chunk_length"]
        self.output_chunk_length = kwargs["output_chunk_length"]

        self.init_common_attributes(**kwargs)
        self.check_unique_timestamps()

        print(f"Initialized dataset with {np.unique(self.timestamps).shape[0]} timestamps.", flush=True)


class CustomInferenceDataset(CustomDataset):

    def __init__(self):
        super().__init__()

    def _common_init(self, **kwargs):
        """
        Initialize common attributes for the inference dataset.
        """
        self.input_time_series = kwargs["target_series"]
        target_series = kwargs["target_series"]
        target_feature = kwargs["target_feature"]
        if isinstance(target_feature, str):
            print(
                "Target feature passed as string. Getting index of the feature from the TimeSeries object.",
                file=sys.stderr,
            )
            target = target_series.features.index(target_feature)
        else:
            target = target_feature

        kwargs["target_feature"] = target

        self.init_common_attributes(
            **kwargs
        )
        self.check_unique_timestamps()

        print(f"Initialized dataset with {np.unique(self.timestamps).shape[0]} timestamps.", flush=True)


class CustomMixedCovariatesSequentialDataset(CustomTrainingDataset, MixedCovariatesSequentialDataset):
    def __init__(
            self,
            target_series: TimeSeries,
            covariates: TimeSeries,
            target_feature: int,
            input_chunk_length: int,
            output_chunk_length: int,
            data_windows: dict,
    ):
        CustomTrainingDataset.__init__(self)
        MixedCovariatesSequentialDataset.__init__(
            self,
            target_series=target_series,
            past_covariates=covariates,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
        )

        self.common_init(
            target_series=target_series,
            target_feature=target_feature,
            data_windows=data_windows,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length
        )

    def __len__(self):
        return len(self.data_windows["data"])

    def __getitem__(
            self, item
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        np.ndarray,
    ]:
        past_target = self.data[item][:, [self.target]]
        past_covariate = self.data[item][
                         :, [i for i in range(self.features) if i != self.target]
                         ]
        historic_future_covariate = None
        future_covariate = None
        static_covariate = None
        future_target = self.labels[item].reshape(1, 1)

        # copied these names from MixedCovariatesSequentialDataset as a reference
        return (
            past_target,
            past_covariate,
            historic_future_covariate,
            future_covariate,
            static_covariate,
            future_target,
        )


class CustomMixedCovariatesInferenceDataset(CustomInferenceDataset, MixedCovariatesInferenceDataset):
    def __init__(
            self,
            target_series: TimeSeries,
            covariates: TimeSeries,
            target_feature: int,
            input_chunk_length: int,
            output_chunk_length: int,
            data_windows: dict,
    ):
        CustomInferenceDataset.__init__(self)
        MixedCovariatesInferenceDataset.__init__(
            self,
            target_series=target_series,
            past_covariates=covariates,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
        )

        self._common_init(
            target_series=target_series,
            target_feature=target_feature,
            data_windows=data_windows,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length
        )

    def __len__(self):
        return len(self.data_windows["data"])

    def __getitem__(
            self, item
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        TimeSeries,
        Union[pd.Timestamp, int],
    ]:
        # extract only target feature
        # darts sanity checking wants the predict sample to be the same shape as the train sample
        past_target = self.data[item][:, [self.target]]
        # extract all features except target
        past_covariate = self.data[item][
                         :, [i for i in range(self.features) if i != self.target]
                         ]
        historic_future_covariate = None
        future_covariate = None
        future_past_covariate = None
        static_covariate = None
        ts_target = self.input_time_series
        if isinstance(ts_target.pd_dataframe().index[0], pd.Timestamp):
            pred_point = pd.to_datetime(self.labels_timestamps[item][0])
        else:
            pred_point = self.labels_timestamps[item].astype(int)[0]

        return (
            past_target,
            past_covariate,
            historic_future_covariate,
            future_covariate,
            future_past_covariate,
            static_covariate,
            ts_target,
            pred_point,
        )


def generate_dataset(data: pd.DataFrame) -> dict:
    """
        This function generates a dataset for training, validation and testing.
        The dataset is a dictionary containing data, timestamps, labels and labels_timestamps.

        Args:
            data (pd.DataFrame): The input data as a pandas DataFrame.

        Returns:
            dict: The generated dataset.
        """
    ...  # data is used to generate the sliding window datasets
    return {
        "data": np.array(
            [
                [[1, 6, 11], [2, 7, 12], [3, 8, 13], [4, 9, 14], [5, 10, 15]],
                [[2, 7, 12], [3, 8, 13], [4, 9, 14], [5, 10, 15], [6, 11, 16]],
                [[3, 8, 13], [4, 9, 14], [5, 10, 15], [6, 11, 16], [7, 12, 17]],
                [[4, 9, 14], [5, 10, 15], [6, 11, 16], [7, 12, 17], [8, 13, 18]],
                [[5, 10, 15], [6, 11, 16], [7, 12, 17], [8, 13, 18], [9, 14, 19]],
                [[6, 11, 16], [7, 12, 17], [8, 13, 18], [9, 14, 19], [10, 15, 20]],
                [[7, 12, 17], [8, 13, 18], [9, 14, 19], [10, 15, 20], [11, 16, 21]],
                [[8, 13, 18], [9, 14, 19], [10, 15, 20], [11, 16, 21], [12, 17, 22]],
                [[9, 14, 19], [10, 15, 20], [11, 16, 21], [12, 17, 22], [13, 18, 23]],
                [[10, 15, 20], [11, 16, 21], [12, 17, 22], [13, 18, 23], [14, 19, 24]],
                [[11, 16, 21], [12, 17, 22], [13, 18, 23], [14, 19, 24], [15, 20, 25]],
                [[12, 17, 22], [13, 18, 23], [14, 19, 24], [15, 20, 25], [16, 21, 26]],
                [[13, 18, 23], [14, 19, 24], [15, 20, 25], [16, 21, 26], [17, 22, 27]],
                [[14, 19, 24], [15, 20, 25], [16, 21, 26], [17, 22, 27], [18, 23, 28]],
                [[15, 20, 25], [16, 21, 26], [17, 22, 27], [18, 23, 28], [19, 24, 29]],
            ],
            dtype=np.float32,
        ),
        "timestamps": np.array(
            [
                [0, 1, 2, 3, 4],
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7],
                [4, 5, 6, 7, 8],
                [5, 6, 7, 8, 9],
                [6, 7, 8, 9, 10],
                [7, 8, 9, 10, 11],
                [8, 9, 10, 11, 12],
                [9, 10, 11, 12, 13],
                [10, 11, 12, 13, 14],
                [11, 12, 13, 14, 15],
                [12, 13, 14, 15, 16],
                [13, 14, 15, 16, 17],
                [14, 15, 16, 17, 18],
            ],
            dtype=np.float32,
        ),
        "labels": np.array(
            [
                [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20]
            ], dtype=np.float32
        ),
        "labels_timestamps": np.array(
            [
                [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19]
            ], dtype=np.float32
        ),
    }


def main():
    """
    The main function of the script. It prepares the data, trains the model, and makes predictions.
    """
    input_chunk_length = 5
    output_chunk_length = 1
    model = TFTModel(input_chunk_length=input_chunk_length, output_chunk_length=output_chunk_length)
    target_feature = 0
    target_feature_name = 'feature1'
    data = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "feature2": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        "feature3": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    }, index=range(20))

    # split train, val and test data. Not necessary for this example as we are using a fixed dataset
    # train_data, val_test_data = target_series.split_after(0.6)  # up to [6, 11, 16], 6 elements
    # val_data, test_data = val_test_data.split_after(0.5)  # up to [8, 13, 18] and [10, 15, 20], 2 elements each

    data_windows = generate_dataset(data)
    train_data_windows = {
        "data": data_windows["data"][:6], "timestamps": data_windows["timestamps"][:6],
        "labels": data_windows["labels"][:6], "labels_timestamps": data_windows["labels_timestamps"][:6]
    }
    val_data_windows = {
        "data": data_windows["data"][6:8], "timestamps": data_windows["timestamps"][6:8],
        "labels": data_windows["labels"][6:8], "labels_timestamps": data_windows["labels_timestamps"][6:8]
    }

    test_data_windows = {
        "data": data_windows["data"][8:], "timestamps": data_windows["timestamps"][8:],
        "labels": data_windows["labels"][8:], "labels_timestamps": data_windows["labels_timestamps"][8:]
    }

    train_target_series = TimeSeries.from_dataframe(
        data.iloc[:6][[target_feature_name]]
    )
    val_target_series = TimeSeries.from_dataframe(
        data.iloc[6:8][[target_feature_name]]
    )
    test_target_series = TimeSeries.from_dataframe(
        data.iloc[8:][[target_feature_name]]
    )

    train_covariates = TimeSeries.from_dataframe(
        data.iloc[:6][['feature2', 'feature3']]
    )
    val_covariates = TimeSeries.from_dataframe(
        data.iloc[6:8][['feature2', 'feature3']]
    )
    test_covariates = TimeSeries.from_dataframe(
        data.iloc[8:][['feature2', 'feature3']]
    )

    train_dataset = CustomMixedCovariatesSequentialDataset(
        target_series=train_target_series,
        covariates=train_covariates,
        target_feature=target_feature,
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        data_windows=train_data_windows,
    )

    val_dataset = CustomMixedCovariatesSequentialDataset(
        target_series=val_target_series,
        covariates=val_covariates,
        target_feature=target_feature,
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        data_windows=val_data_windows,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=0.05,
        verbose=True,
        mode="min",
    )

    # Can't use EarlyStopping with TFTModel as it raises the following exception:
    # RuntimeError: Early stopping conditioned on metric `val_loss` which is not available.
    # Pass in or modify your `EarlyStopping` callback to use any of the following: ``

    trainer = Trainer(
        accelerator='auto',
        devices=1,
        # callbacks=[early_stopping],
        max_epochs=100,
        log_every_n_steps=1  # used to print the loss at each step, but it is not printed as the epochs are not run.
    )

    model.fit_from_dataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        trainer=trainer,
        verbose=True,
    )

    inference_dataset = CustomMixedCovariatesInferenceDataset(
        target_series=test_target_series,
        covariates=test_covariates,
        target_feature=target_feature,
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        data_windows=test_data_windows,
    )

    tester = Trainer(accelerator='auto', devices=1)

    forecast_horizon = 1
    y_pred: list[TimeSeries] = model.predict_from_dataset(
        n=forecast_horizon,
        input_series_dataset=inference_dataset,
        trainer=tester,
        verbose=True,
    )

    y_pred = np.concatenate([[ts.values()[0]] for ts in y_pred])
    y_true = test_data_windows["labels"].flatten()
    print("MSE:", mse(y_true, y_pred))


if __name__ == '__main__':
    main()
