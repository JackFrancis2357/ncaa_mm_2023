import numpy as np
import pandas as pd
from metadata_configs import get_metadata, get_training_data_col_names
from kfold_training import KFoldFit
from sklearn.model_selection import train_test_split


class SubmissionFit(KFoldFit):
    def build_tvt_kfold(self):
        x_train_ind = np.where(self.season_info < self.season)
        self.x_test_ind = np.where(self.season_info >= self.season)
        x_train = self.x_data[x_train_ind]
        y_train = self.y_data[x_train_ind]
        if len(self.x_test_ind[0]) > 1:
            self.x_test = self.x_data[self.x_test_ind]
            self.y_test = self.y_data[self.x_test_ind]
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=3604
        )
    
    def explain_model(self):
        pass


class MakeWarmupSubmission:
    def __init__(self):
        self.submission_file = pd.read_csv("./data/SampleSubmissionWarmup.csv", index_col=0)
        self.submission_filename = 'test_sub.csv'

    def load_data(self):
        if self.is_men:
            data = np.load(f"./training_data/{self.file_save_id}/all_games.npy")
            self.identifier = 'men'
        else:
            data = np.load(f"./training_data/{self.file_save_id}/all_games.npy")
            self.identifier = 'women'
            
        self.season_info = data[:, 2]
        self.important_cols = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 23, 24, 35, 36, 37, 48, 49, 50, 51, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 81, 82, 83, 94, 95, 96, 107, 108, 109, 110, 120]
        self.overall_x = data[:, self.important_cols]
        self.overall_y = data[:, -1]

    def get_model_scaler_metadata(self):
        kfold_model = SubmissionFit(
            season=self.cutoff_year,
            season_info=self.season_info,
            x_data=self.overall_x,
            y_data=self.overall_y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            is_men=self.is_men,
            node_mult=0,
            dropout_pct=0,
            layers=0
        )
        self.year_metrics = kfold_model.run()
        self.model, self.scaler = kfold_model.get_model_scaler()

    def populate_submission_file(self):
        for year in range(self.cutoff_year, self.max_year):
            if year == 2020:
                continue
            else:
                year_data = np.load(f"./training_data/{self.file_save_id}/tourney_{year}.npy")
                year_data_x = year_data[:, self.important_cols]
                year_data_x_scaled = self.scaler.transform(year_data_x)
                y_pred = self.model.predict(year_data_x_scaled, batch_size=self.batch_size)
                num_games = y_pred.shape[0] // 2
                for i in range(num_games):
                    team_1 = int(year_data[2 * i, 0])
                    team_2 = int(year_data[2 * i, 1])
                    team_1_prob = (y_pred[2 * i] + (1 - y_pred[2 * i + 1])) / 2
                    team_2_prob = ((1 - y_pred[2 * i]) + y_pred[2 * i + 1]) / 2
                    if team_1 < team_2:
                        sub_id = f"{year}_{team_1}_{team_2}"
                        self.submission_file.loc[sub_id, :] = team_1_prob[0]
                    else:
                        sub_id = f"{year}_{team_2}_{team_1}"
                        self.submission_file.loc[sub_id, :] = team_2_prob[0]

    def run(self, cutoff_year=2018, epochs=1000, batch_size=2048):
        self.cutoff_year = cutoff_year
        self.epochs = epochs
        self.batch_size = batch_size
        for self.is_men in [True, False]:
            self.file_save_id, _, self.max_year = get_metadata(self.is_men, regular_season=False)
            self.load_data()
            self.get_model_scaler_metadata()
            self.populate_submission_file()
        self.submission_file.to_csv(self.submission_filename)


if __name__ == "__main__":
    MakeWarmupSubmission().run()
