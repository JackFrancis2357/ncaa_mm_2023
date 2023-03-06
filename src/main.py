from get_season_stats import SeasonStats, OrdinalsYearEnd, TeamSeasonStats
from prep_training_data import SeedingStats, CreateTourneyTrainingData
from metadata_configs import get_metadata
import numpy as np


class NCAAKaggle:
    def __init__(self, is_men):
        self.is_men = is_men

    def run(self):
        self.get_team_season_stats()
        self.generate_tourney_training_data()

    def get_team_season_stats(self):
        self.is_regular_season = True
        self.file_save_id, self.min_year, self.max_year = get_metadata(self.is_men, self.is_regular_season)
        season_stats = SeasonStats(men=self.is_men, regular_season=self.is_regular_season)
        ordinals = OrdinalsYearEnd(men=self.is_men)
        season_list = range(self.min_year, self.max_year + 1)
        all_games = []
        for season in season_list:
            season_games = []
            season_data = season_stats.get_season(season)
            ordinal_data = ordinals.update_current_season(season)
            team_id_list = season_stats.get_teams()
            for team_id in team_id_list:
                team_stats = TeamSeasonStats(season_data, ordinal_data, team_id, season, self.is_men).run()
                all_games.append(team_stats)
                season_games.append(team_stats)
            season_games_npy = np.vstack(season_games)
            np.save(f"./generated_data/{self.file_save_id}/season_{season}_games.npy", season_games_npy)
            print(f"Finished with season {season}")
        all_games_npy = np.vstack(all_games)
        np.save(f"./generated_data/{self.file_save_id}/all_games.npy", all_games_npy)
        print(f"Finished generating team season stats for all seasons")

    def generate_tourney_training_data(self):
        self.is_regular_season = False
        self.file_save_id, self.min_year, self.max_year = get_metadata(self.is_men, self.is_regular_season)
        season_stats = SeasonStats(men=self.is_men, regular_season=self.is_regular_season)
        seeding_stats = SeedingStats(men=self.is_men)
        season_list = range(self.min_year, self.max_year + 1)
        all_games = []
        for season in season_list:
            if season == 2020 or season == 2023:
                continue
            season_data = season_stats.get_season(season)
            seeding_data = seeding_stats.get_season(season)
            tourney_training_data = CreateTourneyTrainingData(
                is_men, season_data, season, seeding_data, self.file_save_id
            ).run()
            all_games.append(tourney_training_data)
            print(f"Finished with season {season}")
        all_games_npy = np.vstack(all_games)
        np.save(f"./training_data/{self.file_save_id}/all_games.npy", all_games_npy)
        print(f"Finished generating tourney data for all seasons")


if __name__ == "__main__":
    is_men = False
    is_regular_season = True
    ncaa_model = NCAAKaggle(is_men=is_men)
    ncaa_model.run()
