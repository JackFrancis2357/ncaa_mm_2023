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


def get_training_data_col_names():
    df_cols = [
        "Team_1_TeamID",
        "Team_2_TeamID",
        "Season",
        "Team_1_Wins",
        "Team_1_Losses",
        "Team_1_Team_Rank",
        "Team_1_Team_Rank_Std",
        "Team_1_Opp_Rank",
        "Team_1_Opp_Rank_Std",
        "Team_1_Points_Scored",
        "Team_1_FGM",
        "Team_1_FGA",
        "Team_1_FGM3",
        "Team_1_FGA3",
        "Team_1_FTM",
        "Team_1_FTA",
        "Team_1_OR",
        "Team_1_DR",
        "Team_1_Ast",
        "Team_1_TO",
        "Team_1_Stl",
        "Team_1_Blk",
        "Team_1_Points_Scored_std",
        "Team_1_FGM_std",
        "Team_1_FGA_std",
        "Team_1_FGM3_std",
        "Team_1_FGA3_std",
        "Team_1_FTM_std",
        "Team_1_FTA_std",
        "Team_1_OR_std",
        "Team_1_DR_std",
        "Team_1_Ast_std",
        "Team_1_TO_std",
        "Team_1_Stl_std",
        "Team_1_Blk_std",
        "Team_1_Opp_Points_Scored",
        "Team_1_Opp_FGM",
        "Team_1_Opp_FGA",
        "Team_1_Opp_FGM3",
        "Team_1_Opp_FGA3",
        "Team_1_Opp_FTM",
        "Team_1_Opp_FTA",
        "Team_1_Opp_OR",
        "Team_1_Opp_DR",
        "Team_1_Opp_Ast",
        "Team_1_Opp_TO",
        "Team_1_Opp_Stl",
        "Team_1_Opp_Blk",
        "Team_1_Opp_Points_Scored_std",
        "Team_1_Opp_FGM_std",
        "Team_1_Opp_FGA_std",
        "Team_1_Opp_FGM3_std",
        "Team_1_Opp_FGA3_std",
        "Team_1_Opp_FTM_std",
        "Team_1_Opp_FTA_std",
        "Team_1_Opp_OR_std",
        "Team_1_Opp_DR_std",
        "Team_1_Opp_Ast_std",
        "Team_1_Opp_TO_std",
        "Team_1_Opp_Stl_std",
        "Team_1_Opp_Blk_std",
        "Team_1_Seed",
        "Team_2_Wins",
        "Team_2_Losses",
        "Team_2_Team_Rank",
        "Team_2_Team_Rank_Std",
        "Team_2_Opp_Rank",
        "Team_2_Opp_Rank_Std",
        "Team_2_Points_Scored",
        "Team_2_FGM",
        "Team_2_FGA",
        "Team_2_FGM3",
        "Team_2_FGA3",
        "Team_2_FTM",
        "Team_2_FTA",
        "Team_2_OR",
        "Team_2_DR",
        "Team_2_Ast",
        "Team_2_TO",
        "Team_2_Stl",
        "Team_2_Blk",
        "Team_2_Points_Scored_std",
        "Team_2_FGM_std",
        "Team_2_FGA_std",
        "Team_2_FGM3_std",
        "Team_2_FGA3_std",
        "Team_2_FTM_std",
        "Team_2_FTA_std",
        "Team_2_OR_std",
        "Team_2_DR_std",
        "Team_2_Ast_std",
        "Team_2_TO_std",
        "Team_2_Stl_std",
        "Team_2_Blk_std",
        "Team_2_Opp_Points_Scored",
        "Team_2_Opp_FGM",
        "Team_2_Opp_FGA",
        "Team_2_Opp_FGM3",
        "Team_2_Opp_FGA3",
        "Team_2_Opp_FTM",
        "Team_2_Opp_FTA",
        "Team_2_Opp_OR",
        "Team_2_Opp_DR",
        "Team_2_Opp_Ast",
        "Team_2_Opp_TO",
        "Team_2_Opp_Stl",
        "Team_2_Opp_Blk",
        "Team_2_Opp_Points_Scored_std",
        "Team_2_Opp_FGM_std",
        "Team_2_Opp_FGA_std",
        "Team_2_Opp_FGM3_std",
        "Team_2_Opp_FGA3_std",
        "Team_2_Opp_FTM_std",
        "Team_2_Opp_FTA_std",
        "Team_2_Opp_OR_std",
        "Team_2_Opp_DR_std",
        "Team_2_Opp_Ast_std",
        "Team_2_Opp_TO_std",
        "Team_2_Opp_Stl_std",
        "Team_2_Opp_Blk_std",
        "Team_2_Seed",
        "Result"
    ]
    print(len(df_cols))
    return df_cols
