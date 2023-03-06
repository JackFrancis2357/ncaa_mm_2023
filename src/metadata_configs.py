def get_metadata(men, regular_season):
    if men:
        id_one = "men_"
        min_year, max_year = 2003, 2023
    else:
        id_one = "women_"
        min_year, max_year = 2016, 2023
    if regular_season:
        id_two = "regular_season"
    else:
        id_two = "tourney"
    return id_one + id_two, min_year, max_year


def get_col_names():
    df_cols = [
        "TeamID",
        "Season",
        "Wins",
        "Losses",
        "Team_Rank",
        "Team_Rank_Std",
        "Opp_Rank",
        "Opp_Rank_Std",
        "Points_Scored",
        "FGM",
        "FGA",
        "FGM3",
        "FGA3",
        "FTM",
        "FTA",
        "OR",
        "DR",
        "Ast",
        "TO",
        "Stl",
        "Blk",
        "Points_Scored_std",
        "FGM_std",
        "FGA_std",
        "FGM3_std",
        "FGA3_std",
        "FTM_std",
        "FTA_std",
        "OR_std",
        "DR_std",
        "Ast_std",
        "TO_std",
        "Stl_std",
        "Blk_std",
        "Opp_Points_Scored",
        "Opp_FGM",
        "Opp_FGA",
        "Opp_FGM3",
        "Opp_FGA3",
        "Opp_FTM",
        "Opp_FTA",
        "Opp_OR",
        "Opp_DR",
        "Opp_Ast",
        "Opp_TO",
        "Opp_Stl",
        "Opp_Blk",
        "Opp_Points_Scored_std",
        "Opp_FGM_std",
        "Opp_FGA_std",
        "Opp_FGM3_std",
        "Opp_FGA3_std",
        "Opp_FTM_std",
        "Opp_FTA_std",
        "Opp_OR_std",
        "Opp_DR_std",
        "Opp_Ast_std",
        "Opp_TO_std",
        "Opp_Stl_std",
        "Opp_Blk_std",
    ]
    return df_cols
