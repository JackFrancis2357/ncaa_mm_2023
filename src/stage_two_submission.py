import itertools
import numpy as np
from prep_training_data import CreateTourneyTrainingData
from make_warmup_submission import MakeWarmupSubmission
import pandas as pd
from bracketeer import build_bracket


class CreateCurrentYearTourneyTrainingData(CreateTourneyTrainingData):
    def create_training_data(self):
        all_teams = self.seeding_data["TeamID"].values
        team_pairs = [*itertools.combinations(all_teams, 2)]
        self.season_games = []
        for team_1, team_2 in team_pairs:
            # team_1 = self.season_data.loc[team_1]
            # team_2 = self.season_data.loc[team_2]
            team_1_stats = self.df[self.df["TeamID"] == team_1].values[0][2:]
            team_2_stats = self.df[self.df["TeamID"] == team_2].values[0][2:]
            team_1_seed = self.get_team_seed(team_1)
            team_2_seed = self.get_team_seed(team_2)

            # Build 2 training examples
            example_one = np.array(
                [team_1, team_2, self.season, *team_1_stats, team_1_seed, *team_2_stats, team_2_seed]
            )
            example_two = np.array(
                [team_2, team_1, self.season, *team_2_stats, team_2_seed, *team_1_stats, team_1_seed]
            )
            self.season_games.append(example_one)
            self.season_games.append(example_two)


class MakeFinalSubmission(MakeWarmupSubmission):
    def __init__(self):
        self.submission_file = pd.read_csv("./data/SampleSubmission2023.csv", index_col=0)
        self.submission_filename = 'final_submission.csv'
        self.df = pd.DataFrame(index=range(3000), columns=['ID', 'Pred'])
    
    def populate_submission_file(self):
        year = self.max_year
        year_data = np.load(f"./training_data/{self.file_save_id}/tourney_{year}.npy")
        year_data_x = year_data[:, 3:]
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
                self.df.iloc[i, :] = sub_id, team_1_prob[0]
            else:
                sub_id = f"{year}_{team_2}_{team_1}"
                self.submission_file.loc[sub_id, :] = team_2_prob[0]
                self.df.iloc[i, :] = sub_id, team_2_prob[0]
        self.df.to_csv(f'old_kaggle_submission_format_{self.identifier}.csv')

def bracketeer_png_men():
    b = build_bracket(
    outputPath='output_men.png',
    teamsPath='./data/MTeams.csv',
    seedsPath='./data/MNCAATourneySeeds.csv',
    submissionPath='old_kaggle_submission_format_men.csv',
    slotsPath='./data/MNCAATourneySlots.csv',
    year=2023
)

def bracketeer_png_women():
    b = build_bracket(
    outputPath='output_women.png',
    teamsPath='./data/WTeams.csv',
    seedsPath='./data/WNCAATourneySeeds.csv',
    submissionPath='old_kaggle_submission_format_women.csv',
    slotsPath='./data/WNCAATourneySlots.csv',
    year=2023
)

MakeFinalSubmission().run(cutoff_year=2023)
bracketeer_png_men()
bracketeer_png_women()
