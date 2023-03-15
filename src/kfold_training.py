from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from metadata_configs import get_training_data_col_names
import shap
from model import MenMLModel, WomenMLModel
import sys


class KFoldFit:
    def __init__(self, season, season_info, x_data, y_data, epochs, batch_size, is_men, colnames=None):
        self.season = season
        self.season_info = season_info
        self.x_data = x_data
        self.y_data = y_data
        self.colnames = colnames
        self.epochs = epochs
        self.batch_size = batch_size
        self.is_men = is_men
        if self.is_men:
            self.model, self.callbacks = MenMLModel().run(input_shape=self.x_data.shape[1])
        else:
            self.model, self.callbacks = WomenMLModel().run(input_shape=self.x_data.shape[1])

    def run(self):
        self.build_tvt_kfold()
        self.apply_scaling()
        self.fit_kfold()
        self.get_test_metrics()
        return self.metrics

    def build_tvt_kfold(self):
        x_train_ind = np.where(self.season_info != self.season)
        x_test_ind = np.where(self.season_info == self.season)
        x_train = self.x_data[x_train_ind]
        y_train = self.y_data[x_train_ind]
        self.x_test = self.x_data[x_test_ind]
        self.y_test = self.y_data[x_test_ind]
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=3604
        )

    def apply_scaling(self):
        self.scaler = StandardScaler()
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_val = self.scaler.transform(self.x_val)
        self.x_test = self.scaler.transform(self.x_test)

    def fit_kfold(self):
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(self.x_val, self.y_val),
            callbacks=self.callbacks,
        )

    def explain_model(self):
        explainer = shap.DeepExplainer(self.model, self.x_train)
        shap_values = explainer.shap_values(self.x_test)
        shap.summary_plot(shap_values[0], plot_type="bar", feature_names=self.colnames)

    def get_test_metrics(self):
        self.train_metrics = self.model.evaluate(self.x_train, self.y_train)
        self.val_metrics = self.model.evaluate(self.x_val, self.y_val)
        self.test_metrics = self.model.evaluate(self.x_test, self.y_test)
        print(f"Year {self.season} mse {self.test_metrics}")
        self.metrics = [self.season, self.train_metrics, self.val_metrics, self.test_metrics]

    def get_model_scaler(self):
        return self.model, self.scaler


if __name__ == "__main__":
    a = np.load("./training_data/women_tourney/all_games.npy")
    colnames = get_training_data_col_names()
    df = pd.DataFrame(a, columns=colnames)

    season_info = df.iloc[:, 2].values
    overall_x = df.iloc[:, 3:-1].values
    overall_y = df.iloc[:, -1].values
    is_men = False

    # scaler = StandardScaler()
    # scaled_overall_x = scaler.fit_transform(overall_x)
    # pca = PCA(n_components=0.95)
    # scaled_pca_overall_x = pca.fit_transform(scaled_overall_x)

    epochs = 1000
    batch_size = 2048
    metrics_list = []
    for year in range(2016, 2023):
        if year == 2020:
            continue
        kfold_model = KFoldFit(
            season=year,
            season_info=season_info,
            x_data=overall_x,
            y_data=overall_y,
            colnames=colnames,
            epochs=epochs,
            batch_size=batch_size,
            is_men=is_men,
        )
        year_metrics = kfold_model.run()
        metrics_list.append(year_metrics)
    print(metrics_list)
    print(f"Average Training Error is {np.mean([a[1] for a in metrics_list])}")
    print(f"Average Validation Error is {np.mean([a[2] for a in metrics_list])}")
    print(f"Average Test Error is {np.mean([a[3] for a in metrics_list])}")

    # Next steps
    # Create a make submission class
    # Integrate with MLFlow to track metrics (or save in dict)
    # Hyperparameter tuning for number of layers, using dropout, and size of hidden layers
