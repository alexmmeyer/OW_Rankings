import pandas as pd
import os
from datetime import datetime as dt
from itertools import combinations
import variables
import networkx as nx
import matplotlib.pyplot as plt

DIRECTORY = variables.DIRECTORY
RANKING_FILE_NAME = variables.RANKING_FILE_NAME
RANKING_AS_OF = variables.RANKING_AS_OF
DEPRECIATION_PERIOD = variables.DEPRECIATION_PERIOD
LAMBDA = variables.LAMBDA
event_type_weights = variables.event_points


def age_weight_exp(date_as_text):
    race_date = dt.strptime(date_as_text, "%m/%d/%Y")
    rank_date = dt.strptime(RANKING_AS_OF, "%m/%d/%Y")
    if (rank_date.date() - race_date.date()).days > DEPRECIATION_PERIOD or rank_date.date() < race_date.date():
        return 0
    else:
        days_old = (rank_date.date() - race_date.date()).days
        years_old = days_old / 365
        weight = 2.71828 ** (LAMBDA * years_old)
        return weight


def age_weight_linear(date_as_text):
    race_date = dt.strptime(date_as_text, "%m/%d/%Y")
    rank_date = dt.strptime(RANKING_AS_OF, "%m/%d/%Y")
    if (rank_date.date() - race_date.date()).days > DEPRECIATION_PERIOD or rank_date.date() < race_date.date():
        return 0
    else:
        days_old = (rank_date.date() - race_date.date()).days
        weight = (DEPRECIATION_PERIOD - days_old) / DEPRECIATION_PERIOD
        return weight


def comp_level(race_result_file):
    race_data = pd.read_csv(race_result_file)
    event_type = race_data.event[0]
    weight = float(event_type_weights.weight[event_type_weights.event == event_type])
    return weight


def update_rankings(race_result_file):
    """
    :param race_result_file: csv file with results from a single race
    :return: adds nodes and edges from race_result_file to existing graph
    """
    race_data = pd.read_csv(race_result_file)
    name_list = race_data.athlete_name.tolist()
    name_list = [name.title() for name in name_list]
    combos = list(combinations(name_list, 2))
    combos = [tuple(reversed(combo)) for combo in combos]
    age_weight = age_weight_linear(race_data.date[0])
    comp_weight = comp_level(race_result_file)
    distance_weight = 1
    total_weight = age_weight * comp_weight * distance_weight
    print(f"Loading {race_result_file}")


    for combo in combos:
        G.add_edge(*combo, weight=total_weight)
        # is this overriding (bad), or adding weight to (good), existing edges for subsequent matchups between athletes?

    pr_dict = nx.pagerank(G)

    ranking_dict = {
        "name": list(pr_dict.keys()),
        "pagerank": list(pr_dict.values())
    }

    ranking_df = pd.DataFrame(ranking_dict)
    ranking_df = ranking_df.sort_values(by="pagerank", ascending=False).reset_index(drop=True)
    ranking_df["rank"] = range(1, len(pr_dict) + 1)
    ranking_df.to_csv(RANKING_FILE_NAME)


def test_predictability(race_result_file):
    """
    :param race_result_file: a new race result csv file to compare against the ranking at that point in time
    :return: adds to correct_predictions (if applicable) and total_matchups running count
    """

    global correct_predictions
    global total_matchups

    ranking_data = pd.read_csv(RANKING_FILE_NAME)
    race_data = pd.read_csv(race_result_file)
    name_list = race_data.athlete_name.tolist()
    combos = list(combinations(name_list, 2))

    for matchup in combos:
        winner_name = matchup[0].title()
        loser_name = matchup[1].title()
        if winner_name in list(ranking_data.name) and loser_name in list(ranking_data.name):
            winner_rank = int(ranking_data["rank"][ranking_data.name == winner_name])
            loser_rank = int(ranking_data["rank"][ranking_data.name == loser_name])
            total_matchups += 1
            if winner_rank < loser_rank:
                correct_predictions += 1

G = nx.DiGraph()

correct_predictions = 0
total_matchups = 0

if os.path.exists(RANKING_FILE_NAME):
    os.remove(RANKING_FILE_NAME)

for file in os.listdir(DIRECTORY):
    results_file_path = os.path.join(DIRECTORY, file)
    race_data = pd.read_csv(results_file_path)
    race_date = dt.strptime(race_data.date[0], "%m/%d/%Y")
    rank_date = dt.strptime(RANKING_AS_OF, "%m/%d/%Y")
    if (rank_date.date() - race_date.date()).days > DEPRECIATION_PERIOD or rank_date.date() < race_date.date():
        print(f"Excluding {file}, not in date range.")
    elif os.path.exists(RANKING_FILE_NAME):
        test_predictability(results_file_path)
        update_rankings(results_file_path)
    else:
        update_rankings(results_file_path)

predictability = correct_predictions / total_matchups
print(correct_predictions)
print(total_matchups)
print(predictability)

pr_dict = nx.pagerank(G)

ranking_dict = {
    "name": list(pr_dict.keys()),
    "pagerank": list(pr_dict.values())
}

ranking_df = pd.DataFrame(ranking_dict)
ranking_df = ranking_df.sort_values(by="pagerank", ascending=False).reset_index(drop=True)
ranking_df["rank"] = range(1, len(pr_dict) + 1)
print(ranking_df[ranking_df["rank"] < 26])
