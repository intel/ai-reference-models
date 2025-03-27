import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from keras.layers import (
    Input,
    Embedding,
    Dense,
    Flatten,
    Dropout,
    SpatialDropout1D,
    Activation,
    concatenate,
)
from keras.optimizers import Adam, SGD
from keras.layers import ReLU, PReLU, LeakyReLU, ELU
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
import time
from tensorflow.keras.models import load_model
import os
from argparse import ArgumentParser

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income_bracket",
]

LABEL_COLUMN = "label"

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "native_country",
]

CONTINUOUS_COLUMNS = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]


class Wide_and_Deep:
    def __init__(self, mode="wide and deep"):
        arg_parser = ArgumentParser(description="Parse args")
        arg_parser.add_argument(
            "-b",
            "--batch-size",
            help="Specify the batch size. If this "
            "parameter is not specified, then "
            "it will run with batch size of 1 ",
            dest="batch_size",
            type=int,
            default=1,
        )
        arg_parser.add_argument(
            "-p",
            "--precision",
            help="Specify the model precision to use: fp32",
            required=True,
            choices=["fp32"],
            dest="precision",
        )
        arg_parser.add_argument(
            "-mt",
            "--model_type",
            help="Specify the model type to use: wide, deep or wide_deep",
            choices=["wide", "deep", "wide_deep"],
            dest="precision",
            type=str,
            default="wide_deep",
        )

        arg_parser.add_argument(
            "-e",
            "--num-inter-threads",
            help="The number of inter-thread.",
            dest="num_inter_threads",
            type=int,
            default=0,
        )

        arg_parser.add_argument(
            "-a",
            "--num-intra-threads",
            help="The number of intra-thread.",
            dest="num_intra_threads",
            type=int,
            default=0,
        )

        arg_parser.add_argument(
            "-m",
            "--pretrained-model",
            help="Specify the path to the pretrained model",
            dest="pretrained_model",
        )

        arg_parser.add_argument(
            "-d",
            "--data-location",
            help="Specify the location of the data. ",
            dest="data_location",
        )
        arg_parser.add_argument(
            "-r",
            "--accuracy-only",
            help="For accuracy measurement only.",
            dest="accuracy_only",
            action="store_true",
        )

        # parse the arguments
        self.args = arg_parser.parse_args()

        def preprocessing():
            train_file = os.path.join(self.args.data_location, "adult.data")
            test_file = os.path.join(self.args.data_location, "adult.test")
            train_data = pd.read_csv(train_file, names=COLUMNS)
            train_data.dropna(how="any", axis=0)
            test_data = pd.read_csv(test_file, skiprows=1, names=COLUMNS)
            test_data.dropna(how="any", axis=0)
            all_data = pd.concat([train_data, test_data])
            all_data[LABEL_COLUMN] = (
                all_data["income_bracket"].apply(lambda x: ">50K" in x).astype(int)
            )
            all_data.pop("income_bracket")
            y = all_data[LABEL_COLUMN].values
            all_data.pop(LABEL_COLUMN)
            for c in CATEGORICAL_COLUMNS:
                le = LabelEncoder()
                all_data[c] = le.fit_transform(all_data[c])
            train_size = len(train_data)
            x_train = all_data.iloc[:train_size]
            y_train = y[:train_size]
            x_test = all_data.iloc[train_size:]
            y_test = y[train_size:]
            x_train_categ = np.array(x_train[CATEGORICAL_COLUMNS])
            x_test_categ = np.array(x_test[CATEGORICAL_COLUMNS])
            x_train_conti = np.array(x_train[CONTINUOUS_COLUMNS], dtype="float64")
            x_test_conti = np.array(x_test[CONTINUOUS_COLUMNS], dtype="float64")
            scaler = StandardScaler()
            x_train_conti = scaler.fit_transform(x_train_conti)
            x_test_conti = scaler.transform(x_test_conti)
            return [
                x_train,
                y_train,
                x_test,
                y_test,
                x_train_categ,
                x_test_categ,
                x_train_conti,
                x_test_conti,
                all_data,
            ]

        self.mode = mode
        (
            x_train,
            y_train,
            x_test,
            y_test,
            x_train_categ,
            x_test_categ,
            x_train_conti,
            x_test_conti,
            all_data,
        ) = preprocessing()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_train_categ = x_train_categ
        self.x_test_categ = x_test_categ
        self.x_train_conti = x_train_conti
        self.x_test_conti = x_test_conti
        self.all_data = all_data
        self.poly = PolynomialFeatures(degree=2, interaction_only=True)
        # cross product
        self.x_train_categ_poly = self.poly.fit_transform(x_train_categ)
        self.x_test_categ_poly = self.poly.transform(x_test_categ)
        self.categ_inputs = None
        self.conti_input = None
        self.deep_component_outlayer = None
        self.logistic_input = None
        self.model = None

    def evaluate_model(self):
        if not self.args.accuracy_only:
            # if self.args.batch_size==1:
            print("Benchmark")
            # batch_size = 1024
            input_data = (
                [self.x_test_conti]
                + [self.x_test_categ[:, i] for i in range(self.x_test_categ.shape[1])]
                + [self.x_test_categ_poly]
            )
            loaded_model = load_model(
                os.path.join(self.args.pretrained_model, "wide_and_deep.h5")
            )
            # Warmup run
            result = loaded_model.predict(input_data, batch_size=self.args.batch_size)
            test_file = os.path.join(self.args.data_location, "adult.test")
            num_records = sum(1 for line in open(test_file))
            # Benchmark run
            inference_start = time.time()
            result = loaded_model.predict(input_data, batch_size=self.args.batch_size)
            main_end = time.time()
            E2Eduration = main_end - main_start
            print("End-to-End duration is", E2Eduration)
            evaluate_duration = main_end - inference_start
            print("Evaluation duration is", evaluate_duration)

            if self.args.batch_size == 1:
                latency = (E2Eduration / num_records) * 1000
                print(f"Latency is: {latency:.4f} ms")
            print("Throughput is:", num_records / evaluate_duration)
        else:
            print("Accuracy ")
            loaded_model = load_model(
                os.path.join(self.args.pretrained_model, "wide_and_deep.h5")
            )
            self.model = loaded_model
            if self.mode == "wide and deep":
                input_data = (
                    [self.x_test_conti]
                    + [
                        self.x_test_categ[:, i]
                        for i in range(self.x_test_categ.shape[1])
                    ]
                    + [self.x_test_categ_poly]
                )
            elif self.mode == "deep":
                input_data = [self.x_test_conti] + [
                    self.x_test_categ[:, i] for i in range(self.x_test_categ.shape[1])
                ]
            else:
                print("wrong mode")
                return

            loss, acc = self.model.evaluate(input_data, self.y_test)
            print(f"Test Accuracy: {acc}")


if __name__ == "__main__":
    main_start = time.time()
    wide_deep_net = Wide_and_Deep()
    wide_deep_net.evaluate_model()
