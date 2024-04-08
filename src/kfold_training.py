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
    def __init__(
        self,
        season,
        season_info,
        x_data,
        y_data,
        epochs,
        batch_size,
        is_men,
        node_mult,
        dropout_pct,
        layers,
        colnames=None,
    ):
        self.season = season
        self.season_info = season_info
        self.x_data = x_data
        self.y_data = y_data
        self.colnames = colnames
        self.epochs = epochs
        self.batch_size = batch_size
        self.is_men = is_men
        self.node_mult = node_mult
        self.dropout_pct = dropout_pct
        self.layers = layers
        if self.is_men:
            self.model, self.callbacks = MenMLModel().run(
                self.x_data.shape[1], self.node_mult, self.dropout_pct, self.layers
            )
        else:
            self.model, self.callbacks = WomenMLModel().run(
                self.x_data.shape[1], self.node_mult, self.dropout_pct, self.layers
            )

    def run(self):
        self.build_tvt_kfold()
        self.apply_scaling()
        self.fit_kfold()
        self.explain_model()
        self.get_test_metrics()
        return self.metrics

    def build_tvt_kfold(self):
        x_train_ind = np.where(self.season_info != self.season)
        self.x_test_ind = np.where(self.season_info == self.season)
        x_train = self.x_data[x_train_ind]
        df = pd.DataFrame(x_train)
        y_train = self.y_data[x_train_ind]
        if len(self.x_test_ind[0]) > 1:
            self.x_test = self.x_data[self.x_test_ind]
            self.y_test = self.y_data[self.x_test_ind]
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_train, y_train, test_size=0.1, random_state=3604
        )

    def apply_scaling(self):
        self.scaler = StandardScaler()
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_val = self.scaler.transform(self.x_val)
        if len(self.x_test_ind[0]) > 1:
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
        shap.summary_plot(shap_values[0], plot_type="bar", feature_names=self.colnames[3:-1])

    def get_test_metrics(self):
        self.train_metrics = self.model.evaluate(self.x_train, self.y_train)
        self.val_metrics = self.model.evaluate(self.x_val, self.y_val)
        if len(self.x_test_ind[0]) > 1:
            self.test_metrics = self.model.evaluate(self.x_test, self.y_test)
            print(f"Year {self.season} mse {self.test_metrics}")
            self.metrics = [self.season, self.train_metrics, self.val_metrics, self.test_metrics]
        else:
            print(f"Year {self.season} mse {self.val_metrics}")
            self.metrics = [self.season, self.train_metrics, self.val_metrics]

    def get_model_scaler(self):
        return self.model, self.scaler


def get_feature_indices(colnames):
    list_of_features_to_keep = ['Wins', 'Losses', 'Team_Rank', 'Opp_Rank', 'Points_Scored', 'FGM', 'FGA', 'FGM3', 'OR', 'DR', 'Seed']
    pass

if __name__ == "__main__":
    is_men = False
    if is_men:
        a = np.load("./training_data/men_tourney/all_games.npy")
        identifier = "men"
    else:
        a = np.load("./training_data/women_tourney/all_games.npy")
        identifier = "women"
    colnames = get_training_data_col_names()
    print(type(colnames))
    # important_cols = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 23, 24, 35, 36, 37, 48, 49, 50, 51, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 81, 82, 83, 94, 95, 96, 107, 108, 109, 110, 120]
    df = pd.DataFrame(a, columns=colnames)

    season_info = df.iloc[:, 2].values
    overall_x = df.iloc[:, 3:-1].values
    overall_y = df.iloc[:, -1].values
    
    epochs = 1000
    batch_size = 2048
    node_mult_list = [0.7]
    dropout_pct_list = [0.7]
    num_layers = [3]
    num_vals = len(node_mult_list) * len(dropout_pct_list) * len(num_layers) + 1
    df_hyper_metrics = pd.DataFrame(
        index=range(num_vals), columns=["Node_Mult", "Dropout", "Num_Layers", "Train", "Validation", "Test"]
    )
    ctr = 0
    for layers in num_layers:
        for node_mult in node_mult_list:
            for dropout_pct in dropout_pct_list:
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
                        node_mult=node_mult,
                        dropout_pct=dropout_pct,
                        layers=layers,
                    )
                    year_metrics = kfold_model.run()
                    metrics_list.append(year_metrics)
                print(metrics_list)
                training_error = np.mean([a[1] for a in metrics_list])
                validation_error = np.mean([a[2] for a in metrics_list])
                testing_error = np.mean([a[3] for a in metrics_list])
                print(f"Average Training Error is {training_error}")
                print(f"Average Validation Error is {validation_error}")
                print(f"Average Test Error is {testing_error}")
                df_hyper_metrics.iloc[ctr, :] = [
                    node_mult,
                    dropout_pct,
                    layers,
                    training_error,
                    validation_error,
                    testing_error,
                ]
                ctr += 1
                print("\n")
                print(f"Done with {ctr}")
                print("\n")
                df_hyper_metrics.to_csv(f"Hyper_metrics_{identifier}_2_val_reduced_features.csv")