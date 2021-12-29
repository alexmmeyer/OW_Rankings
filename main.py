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


def age_weight_exp(race_date_text):
    """
    :param race_date_text: date in MM/DD/YYYY text format
    :return: weight based on RANKING_AS_OF, DEPRECIATION_PERIOD, and LAMBDA (exponential decay)
    """
    race_date = dt.strptime(race_date_text, "%m/%d/%Y")
    rank_date = dt.strptime(RANKING_AS_OF, "%m/%d/%Y")
    days_old = (rank_date.date() - race_date.date()).days
    years_old = days_old / 365
    weight = 2.71828 ** (LAMBDA * years_old)
    return weight


def age_weight_linear(race_date_text):
    """
    :param race_date_text: date in MM/DD/YYYY text format
    :return: weight based on RANKING_AS_OF and DEPRECIATION_PERIOD (linear decay)
    """
    race_date = dt.strptime(race_date_text, "%m/%d/%Y")
    rank_date = dt.strptime(RANKING_AS_OF, "%m/%d/%Y")
    days_old = (rank_date.date() - race_date.date()).days
    weight = (DEPRECIATION_PERIOD - days_old) / DEPRECIATION_PERIOD
    return weight


def comp_level(event_type):
    """
    :param event_type: event type as text, ie: "FINA World Cup"
    :return: weight as a float
    """
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
    age_weight = age_weight_exp(race_data.date[0])
    comp_weight = comp_level(race_data.event[0])
    distance_weight = 1
    total_weight = age_weight * comp_weight * distance_weight
    print(f"Loading {race_result_file}")

    for combo in combos:
        if combo in G.edges:
            current_weight = G[combo[0]][combo[1]]["weight"]
            new_weight = current_weight + total_weight
            G[combo[0]][combo[1]]["weight"] = new_weight
        else:
            G.add_edge(*combo, weight=total_weight)

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
        print(f"Excluding {file}, race is not in date range.")
    elif os.path.exists(RANKING_FILE_NAME):
        test_predictability(results_file_path)
        update_rankings(results_file_path)
    else:
        update_rankings(results_file_path)

predictability = correct_predictions / total_matchups
print(f"Predictability: {predictability}")

pr_dict = nx.pagerank(G)

ranking_dict = {
    "name": list(pr_dict.keys()),
    "pagerank": list(pr_dict.values())
}

ranking_df = pd.DataFrame(ranking_dict)
ranking_df = ranking_df.sort_values(by="pagerank", ascending=False).reset_index(drop=True)
ranking_df["rank"] = range(1, len(pr_dict) + 1)
print(ranking_df[ranking_df["rank"] < 26])

# Visualization
num_of_athletes = 10
top_athletes = list(ranking_df.name[ranking_df["rank"] < num_of_athletes + 1])
G = G.subgraph(top_athletes)

size_map = []
thicknesses = []
for name in G.nodes:
    size_map.append(float(ranking_df.pagerank[ranking_df.name == name] * 10000))
for edge in G.edges:
    thicknesses.append(G[edge[0]][edge[1]]["weight"] * 2)

nx.draw_networkx(G, node_size=size_map, width=thicknesses, pos=nx.spring_layout(G))
plt.show()
