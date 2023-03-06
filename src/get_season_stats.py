import pandas as pd
import numpy as np


class SeasonStats:
    def __init__(self, men=True, regular_season=True):
        if men:
            self.identifier = "M"
        else:
            self.identifier = "W"
        if regular_season:
            self.data = pd.read_csv(f"./data/{self.identifier}RegularSeasonDetailedResults.csv")
        else:
            self.data = pd.read_csv(f"./data/{self.identifier}NCAATourneyCompactResults.csv")

    def get_season(self, season):
        self.season = season
        self.season_data = self.data[self.data["Season"] == self.season].reset_index(drop=True)
        return self.season_data

    def get_teams(self):
        winning_teams = self.season_data["WTeamID"].tolist()
        losing_teams = self.season_data["LTeamID"].tolist()
        all_teams = list(set(winning_teams + losing_teams))
        return all_teams


class OrdinalsYearEnd:
    def __init__(self, men=True):
        self.season = None
        if men:
            self.identifier = "M"
            self.data = pd.read_csv("./data/MMasseyOrdinals.csv")
        else:
            self.identifier = "W"
            self.data = pd.read_csv("./data/wncaa_espn.csv")

    def update_current_season(self, season):
        if self.identifier == "M":
            self.update_mens_current_season(season)
        else:
            self.update_womens_current_season(season)
        return self.ordinal_data

    def update_mens_current_season(self, season):
        if self.season != season:
            self.season = season
            self.ordinal_data = self.data[self.data["Season"] == self.season]
            self.max_rank_day = max(self.ordinal_data["RankingDayNum"])
            self.ordinal_data = self.ordinal_data[self.ordinal_data["RankingDayNum"] == self.max_rank_day]
        return self

    def update_womens_current_season(self, season):
        if self.season != season:
            self.season = season
            self.ordinal_data = self.data[self.data["Season"] == season].sort_values(by="Votes", ascending=False)
            self.ordinal_data = self.ordinal_data.reset_index()
            self.ordinal_data = self.ordinal_data.rename(columns={"index": "OrdinalRank"})
            self.ordinal_data = self.ordinal_data.sort_values(by="OrdinalRank")
        return self


class TeamSeasonStats:
    def __init__(self, season_data, ordinal_data, team_id, season, is_men):
        self.team_id = team_id
        self.season = season
        self.is_men = is_men
        self.season_data = season_data
        self.ordinal_data = ordinal_data
        self.team_data = self.season_data.loc[
            (self.season_data["WTeamID"] == self.team_id) | (self.season_data["LTeamID"] == self.team_id)
        ]

    def get_wins_losses(self):
        self.win_loss_array = np.array(self.team_data["WTeamID"] == self.team_id)
        self.wins = sum(self.win_loss_array)
        self.games = len(self.win_loss_array)
        self.losses = self.games - self.wins

    def get_team_game_stats(self):
        team_stats = np.zeros((self.games, 13))
        opponent_stats = np.zeros((self.games, 13))
        for i in range(self.team_data.shape[0]):
            if self.win_loss_array[i]:
                # Location of WTeam Score
                team_stats[i, 0] = self.team_data.iloc[i, 3]
                # Location of WTeam Stats
                team_stats[i, 1:] = self.team_data.iloc[i, 8:20]
                # Location of Opponent Team Score
                opponent_stats[i, 0] = self.team_data.iloc[i, 5]
                # Location of Opponent Team Stats
                opponent_stats[i, 1:] = self.team_data.iloc[i, 21:33]
            else:
                # Location of LTeam Score
                team_stats[i, 0] = self.team_data.iloc[i, 5]
                # Location of LTeam Stats
                team_stats[i, 1:] = self.team_data.iloc[i, 21:33]
                # Location of Opponent Team Score
                opponent_stats[i, 0] = self.team_data.iloc[i, 3]
                # Location of Opponent Team Stats
                opponent_stats[i, 1:] = self.team_data.iloc[i, 8:20]
        self.average_stats = np.mean(team_stats, axis=0)
        self.std_stats = np.std(team_stats, axis=0)
        self.opponent_average_stats = np.mean(opponent_stats, axis=0)
        self.opponent_std_stats = np.std(opponent_stats, axis=0)

    def get_year_end_ranking(self):
        team_ordinal_data = self.ordinal_data[self.ordinal_data["TeamID"] == self.team_id]
        self.team_rank = np.mean(team_ordinal_data["OrdinalRank"])
        self.team_rank_std = np.std(team_ordinal_data["OrdinalRank"])
        self.get_opponent_year_end_rankings()

    def get_womens_year_end_ranking(self):
        team_ordinal_data = self.ordinal_data[self.ordinal_data["TeamID"] == self.team_id]
        if team_ordinal_data.shape[0] == 0:
            self.team_rank = 999
        else:
            self.team_rank = np.mean(team_ordinal_data["OrdinalRank"])
        self.team_rank_std = 0
        self.get_opponent_women_year_end_rankings()

    def get_opponent_year_end_rankings(self):
        winning_teams = self.team_data["WTeamID"].tolist()
        losing_teams = self.team_data["LTeamID"].tolist()
        all_teams = list(set(winning_teams + losing_teams))
        all_teams.remove(self.team_id)
        team_rank_list = []
        for team in all_teams:
            team_rank_data = self.ordinal_data[self.ordinal_data["TeamID"] == team]
            if team_rank_data.shape[0] == 0:
                continue
            team_rank_list.append(np.mean(team_rank_data["OrdinalRank"]))
        self.opponent_rank = np.mean(team_rank_list)
        self.opponent_rank_std = np.std(team_rank_list)

    def get_opponent_women_year_end_rankings(self):
        winning_teams = self.team_data["WTeamID"].tolist()
        losing_teams = self.team_data["LTeamID"].tolist()
        all_teams = list(set(winning_teams + losing_teams))
        all_teams.remove(self.team_id)
        team_rank_list = []
        for team in all_teams:
            team_rank_data = self.ordinal_data[self.ordinal_data["TeamID"] == team]
            if team_rank_data.shape[0] == 0:
                team_rank_list.append(999)
            else:
                team_rank_list.append(np.mean(team_rank_data["OrdinalRank"]))
        self.opponent_rank = np.mean(team_rank_list)
        self.opponent_rank_std = np.std(team_rank_list)

    def build_team_stats(self):
        self.team_yearly_stats = np.array(
            [
                self.team_id,
                self.season,
                self.wins,
                self.losses,
                self.team_rank,
                self.team_rank_std,
                self.opponent_rank,
                self.opponent_rank_std,
                *self.average_stats,
                *self.std_stats,
                *self.opponent_average_stats,
                *self.opponent_std_stats,
            ]
        )

    def run(self):
        self.get_wins_losses()
        self.get_team_game_stats()
        if self.is_men:
            self.get_year_end_ranking()
        else:
            self.get_womens_year_end_ranking()
        self.build_team_stats()
        return self.team_yearly_stats
