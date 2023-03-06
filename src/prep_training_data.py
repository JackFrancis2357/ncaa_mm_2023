# Columns
# Team ID, Season, Wins, Losses, Team Rank, Team Rank Std
# WFGM, FGA, FGM3, FGA3, FTM, FTA, OR, DR, Ast, TO, Stl, Blk

import numpy as np
import pandas as pd

from metadata_configs import get_col_names


class SeedingStats:
    def __init__(self, men=True):
        if men:
            self.data = pd.read_csv(f"./data/MNCAATourneySeeds.csv")
        else:
            self.data = pd.read_csv(f"./data/WNCAATourneySeeds.csv")

    def get_season(self, season):
        self.season = season
        self.season_data = self.data[self.data["Season"] == self.season]
        # Get rid of "a" and "b" from first 4
        self.season_data["Seed"] = self.season_data["Seed"].str[1:3].astype(int)
        return self.season_data


class CreateTourneyTrainingData:
    def __init__(self, is_men, season_data, season, seeding_data, file_save_id):
        self.is_men = is_men
        self.season_data = season_data
        self.season = season
        self.seeding_data = seeding_data
        self.file_save_id = file_save_id
        self.load_team_season_data()

    def run(self):
        self.create_training_data()
        self.save_training_data()
        return self.season_games

    def load_team_season_data(self):
        if self.is_men:
            self.data = np.load(f"./generated_data/men_regular_season/season_{self.season}_games.npy")
        else:
            self.data = np.load(f"./generated_data/women_regular_season/season_{self.season}_games.npy")
        col_names = get_col_names()
        self.df = pd.DataFrame(self.data, columns=col_names)

    def get_team_seed(self, team_id):
        seed_info = self.seeding_data[self.seeding_data["TeamID"] == team_id]
        seed = seed_info["Seed"].values[0]
        return seed

    def create_training_data(self):
        self.season_games = []
        for row in range(self.season_data.shape[0]):
            team_1 = self.season_data.loc[row, "WTeamID"]
            team_2 = self.season_data.loc[row, "LTeamID"]
            team_1_stats = self.df[self.df["TeamID"] == team_1].values[0][2:]
            team_2_stats = self.df[self.df["TeamID"] == team_2].values[0][2:]
            team_1_seed = self.get_team_seed(team_1)
            team_2_seed = self.get_team_seed(team_2)

            # Build 2 training examples
            example_one = np.array(
                [team_1, team_2, self.season, *team_1_stats, team_1_seed, *team_2_stats, team_2_seed, 1]
            )
            example_two = np.array(
                [team_2, team_1, self.season, *team_2_stats, team_2_seed, *team_1_stats, team_1_seed, 0]
            )
            self.season_games.append(example_one)
            self.season_games.append(example_two)

    def save_training_data(self):
        training_data = np.vstack(self.season_games)
        np.save(f"./training_data/{self.file_save_id}/tourney_{self.season}.npy", training_data)
