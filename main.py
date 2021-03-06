import pandas as pd
import os
from datetime import datetime as dt
from datetime import timedelta
from itertools import combinations
import variables
import networkx as nx
import matplotlib.pyplot as plt

RESULTS_DIRECTORY = variables.RESULTS_DIRECTORY
RANKING_FILE_NAME = variables.RANKING_FILE_NAME
DEPRECIATION_PERIOD = variables.DEPRECIATION_PERIOD
LAMBDA = variables.LAMBDA
event_type_weights = variables.event_points
RANK_DIST = variables.RANK_DIST


def age_weight_exp(race_date_text, ranking_date):
    """
    :param ranking_date:
    :param race_date_text: date in MM/DD/YYYY text format
    :return: weight based on RANKING_AS_OF, DEPRECIATION_PERIOD, and LAMBDA (exponential decay)
    """
    race_date = dt.strptime(race_date_text, "%m/%d/%Y")
    rank_date = dt.strptime(ranking_date, "%m/%d/%Y")
    days_old = (rank_date.date() - race_date.date()).days
    years_old = days_old / 365
    weight = 2.71828 ** (LAMBDA * years_old)
    return weight


def age_weight_linear(race_date_text, ranking_date):
    """
    :param ranking_date:
    :param race_date_text: date in MM/DD/YYYY text format
    :return: weight based on RANKING_AS_OF and DEPRECIATION_PERIOD (linear decay)
    """
    race_date = dt.strptime(race_date_text, "%m/%d/%Y")
    rank_date = dt.strptime(ranking_date, "%m/%d/%Y")
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


def get_distance_weight(race_dist, units="km"):
    """
    :param race_dist: distance of the race (default is km)
    :param units: 'km' (default) or 'mi'
    :return: weight as a float
    """
    if RANK_DIST == 0:
        weight = 1
    else:
        if units == "mi":
            race_dist = race_dist * 1.60934
        weight = min(race_dist, RANK_DIST) / max(race_dist, RANK_DIST)
    return weight


def label(race_result_file, *args):
    race_data = pd.read_csv(race_result_file)
    race_label = ""
    for i in args:
        race_label = race_label + str(race_data[i][0]) + " "
    return race_label.strip()


def update_rankings(race_result_file, ranking_date):
    """
    :param ranking_date:
    :param race_result_file: csv file with results from a single race
    :return: adds nodes and edges from race_result_file to existing graph
    """
    race_data = pd.read_csv(race_result_file)
    name_list = [name.title() for name in race_data.athlete_name.tolist()]
    age_weight = age_weight_exp(race_data.date[0], ranking_date)
    comp_weight = comp_level(race_data.event[0])
    dist_weight = get_distance_weight(race_data.distance[0])
    total_weight = age_weight * comp_weight * dist_weight
    race_label = label(race_result_file, "event", "location", "distance", "date")

    combos = list(combinations(name_list, 2))
    combos = [tuple(reversed(combo)) for combo in combos]

    for combo in combos:
        if combo in G.edges:
            current_weight = G[combo[0]][combo[1]]["weight"]
            new_weight = current_weight + total_weight
            G[combo[0]][combo[1]]["weight"] = new_weight
            G[combo[0]][combo[1]]["race_weights"][race_label] = total_weight

        else:
            label_dict = {
                race_label: total_weight
            }
            G.add_edge(*combo, weight=total_weight, race_weights=label_dict)


def test_predictability(race_result_file):
    """
    :param race_result_file: a new race result csv file to compare against the ranking at that point in time
    :return: adds to correct_predictions (if applicable) and total_matchups running count
    """

    global correct_predictions
    global total_tests

    instance_correct_predictions = 0
    instance_total_tests = 0
    race_label = label(race_result_file, "event", "location", "date")

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
            total_tests += 1
            instance_total_tests += 1
            if winner_rank < loser_rank:
                correct_predictions += 1
                instance_correct_predictions += 1

    instance_predictability = instance_correct_predictions / instance_total_tests
    instance_predictability = "{:.0%}".format(instance_predictability)
    print(f"Ranking predictability at {race_label}: {instance_predictability}")


def create_ranking(ranking_date, test=False, comment=False, display_list=0, vis=0):

    global correct_predictions
    global total_tests

    if os.path.exists(RANKING_FILE_NAME):
        os.remove(RANKING_FILE_NAME)

    for file in os.listdir(RESULTS_DIRECTORY):
        results_file_path = os.path.join(RESULTS_DIRECTORY, file)
        race_data = pd.read_csv(results_file_path)
        race_date = dt.strptime(race_data.date[0], "%m/%d/%Y")
        rank_date = dt.strptime(ranking_date, "%m/%d/%Y")
        if (rank_date.date() - race_date.date()).days > DEPRECIATION_PERIOD or rank_date.date() < race_date.date():
            if comment:
                print(f"Excluding {file}, race is not in date range.")
            else:
                pass
        elif os.path.exists(RANKING_FILE_NAME):
            if test:
                test_predictability(results_file_path)
            update_rankings(results_file_path, ranking_date)
        else:
            if comment:
                print(f"Loading {file}")
            update_rankings(results_file_path, ranking_date)

    if test:
        predictability = correct_predictions / total_tests
        predictability = "{:.0%}".format(predictability)
        print(f"Predictability: {predictability}")

    if display_list > 0:
        ranking_data = pd.read_csv(RANKING_FILE_NAME)
        print(ranking_data[ranking_data["rank"] < display_list + 1])

    # if vis > 0:
    #
    #     pr_dict = nx.pagerank(G)
    #
    #     ranking_dict = {
    #         "name": list(pr_dict.keys()),
    #         "pagerank": list(pr_dict.values())
    #     }
    #
    #     ranking_df = pd.DataFrame(ranking_dict)
    #     ranking_df = ranking_df.sort_values(by="pagerank", ascending=False).reset_index(drop=True)
    #     ranking_df["rank"] = range(1, len(pr_dict) + 1)
    #     print(ranking_df[ranking_df["rank"] < 26])
    #
    #     num_of_athletes = vis
    #     top_athletes = list(ranking_df.name[ranking_df["rank"] < num_of_athletes + 1])
    #     G = G.subgraph(top_athletes)
    #
    #     size_map = []
    #     thicknesses = []
    #     for name in G.nodes:
    #         size_map.append(float(ranking_df.pagerank[ranking_df.name == name] * 10000))
    #     for edge in G.edges:
    #         thicknesses.append(G[edge[0]][edge[1]]["weight"] * 2)
    #
    #     nx.draw_networkx(G, node_size=size_map, width=thicknesses, pos=nx.spring_layout(G))
    #     plt.show()

    pr_dict = nx.pagerank(G)

    ranking_dict = {
        "name": list(pr_dict.keys()),
        "pagerank": list(pr_dict.values())
    }

    ranking_df = pd.DataFrame(ranking_dict)
    ranking_df = ranking_df.sort_values(by="pagerank", ascending=False).reset_index(drop=True)
    ranking_df["rank"] = range(1, len(pr_dict) + 1)
    ranking_df.to_csv(RANKING_FILE_NAME, index=False)


def archive_ranking(ranking_date):

    for file in os.listdir(RESULTS_DIRECTORY):
        results_file_path = os.path.join(RESULTS_DIRECTORY, file)
        race_data = pd.read_csv(results_file_path)
        race_date = dt.strptime(race_data.date[0], "%m/%d/%Y")
        rank_date = dt.strptime(ranking_date, "%m/%d/%Y")
        if (rank_date.date() - race_date.date()).days > DEPRECIATION_PERIOD or rank_date.date() < race_date.date():
            pass
        else:
            update_rankings(results_file_path, ranking_date)

    pr_dict = nx.pagerank(G)

    ranking_dict = {
        "name": list(pr_dict.keys()),
        "pagerank": list(pr_dict.values())
    }

    ranking_df = pd.DataFrame(ranking_dict)
    ranking_df = ranking_df.sort_values(by="pagerank", ascending=False).reset_index(drop=True)
    ranking_df["rank"] = range(1, len(pr_dict) + 1)
    file_name = ranking_date.replace("/", "_")
    ranking_df.to_csv(f"rankings_archive/{file_name}.csv", index=False)


def archive_rankings_range(start_date, end_date, increment=1):
    start_date = dt.strptime(start_date, "%m/%d/%Y")
    end_date = dt.strptime(end_date, "%m/%d/%Y")
    rank_dates = [(start_date + timedelta(days=i)).strftime("%m/%d/%Y") for i in range((end_date - start_date).days + 1)
                  if i % increment == 0]

    files_created = 0
    total_files = len(rank_dates)

    for date in rank_dates:
        archive_ranking(date)
        files_created += 1
        progress = files_created / total_files
        print("{:.0%}".format(progress))



def ranking_progression(athlete_name, start_date, end_date, increment=7):
    """
    :param athlete_name:
    :param start_date:
    :param end_date:
    :return: graph showing athlete's ranking on every day between (inclusive) start_date and end_date
    :param increment:
    """

    global G

    # Get a list of dates called date_range within the start and end range
    start_date = dt.strptime(start_date, "%m/%d/%Y")
    end_date = dt.strptime(end_date, "%m/%d/%Y")
    athlete_name = athlete_name.title()
    date_range = [(start_date + timedelta(days=i)).strftime("%m/%d/%Y") for i in range((end_date - start_date).days + 1)
                  if i % increment == 0]

    # Loop through each of the dates in date_range and create a ranking. Add the date to one list and add the
    # athlete's rank to a separate list. Count loops to track progress.
    dates = []
    ranks = []
    loop_count = 0

    for date in date_range:
        G = nx.DiGraph()
        create_ranking(date)
        ranking_data = pd.read_csv(RANKING_FILE_NAME)
        ranked_athletes = list(ranking_data.name)
        if athlete_name in ranked_athletes:
            dates.append(date)
            rank_on_date = int(ranking_data["rank"][ranking_data.name == athlete_name])
            ranks.append(rank_on_date)
        progress = loop_count / len(date_range)
        loop_count += 1
        print("{:.0%}".format(progress))

    # Create a list of all dates that the athlete raced, a list of the athlete's rank on that date, and a list of the
    # race names to be used as labels in the graph
    all_race_dates = list(get_results(athlete_name).date)
    race_dates = [date for date in all_race_dates if start_date <= dt.strptime(date, "%m/%d/%Y") <= end_date]

    all_events = list(get_results(athlete_name).event)
    all_locations = list(get_results(athlete_name).location)
    all_race_labels = [all_events[i] + " " + all_locations[i] for i in range(len(all_events))]
    race_labels = []

    # Build the list of dates and labels to be used in the graph
    for date in all_race_dates:
        if start_date <= dt.strptime(date, "%m/%d/%Y") <= end_date:
            race_labels.append(all_race_labels[race_dates.index(date)])

    # Build the list of ranks to be used in the graph
    race_date_ranks = []

    for rd in race_dates:
        G = nx.DiGraph()
        create_ranking(rd)
        ranking_data = pd.read_csv(RANKING_FILE_NAME)
        rank_on_date = int(ranking_data["rank"][ranking_data.name == athlete_name])
        race_date_ranks.append(rank_on_date)
        progress = len(race_date_ranks) / len(race_dates)
        print("{:.0%}".format(progress))

    print("Progression dates and ranks:")
    print(dates)
    print(ranks)
    print("Race dates, ranks, and race labels:")
    print(race_dates)
    print(race_date_ranks)
    print(race_labels)

    # dict = {
    #     "dates": dates,
    #     "ranks": ranks
    # }
    # df = pd.DataFrame(dict)
    # df.to_csv("progression1.csv")

    # Plot progression dates and ranks in a step chart, plot races on top of that as singular scatter points.
    dates = [dt.strptime(date, "%m/%d/%Y") for date in dates]
    race_dates = [dt.strptime(date, "%m/%d/%Y") for date in race_dates]

    plt.step(dates, ranks, where="post")
    plt.plot(race_dates, race_date_ranks, "o")
    for i, label in enumerate(race_labels):
        plt.text(race_dates[i], race_date_ranks[i], label, rotation=45, fontsize="xx-small")
    plt.ylim(ymin=0)
    plt.gca().invert_yaxis()
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("World Ranking")
    plt.title(f"World Ranking Progression: {athlete_name}\n")
    plt.show()


def ranking_progression_from_archive(athlete_name, start_date, end_date, increment=7):
    """
    :param athlete_name:
    :param start_date:
    :param end_date:
    :return: graph showing athlete's ranking on every day between (inclusive) start_date and end_date
    :param increment:
    """

    # Get a list of dates called date_range within the start and end range
    start_date = dt.strptime(start_date, "%m/%d/%Y")
    end_date = dt.strptime(end_date, "%m/%d/%Y")
    athlete_name = athlete_name.title()
    date_range = [(start_date + timedelta(days=i)).strftime("%m/%d/%Y") for i in range((end_date - start_date).days + 1)
                  if i % increment == 0]

    # Loop through each of the dates in date_range and create a ranking. Add the date to one list and add the
    # athlete's rank to a separate list. Count loops to track progress.
    dates = []
    ranks = []
    loop_count = 0

    for date in date_range:
        file_name = date.replace("/", "_")
        ranking_data = pd.read_csv(f"rankings_archive/{file_name}.csv")
        ranked_athletes = list(ranking_data.name)
        if athlete_name in ranked_athletes:
            dates.append(date)
            rank_on_date = int(ranking_data["rank"][ranking_data.name == athlete_name])
            ranks.append(rank_on_date)
        progress = loop_count / len(date_range)
        loop_count += 1
        print("{:.0%}".format(progress))

    # Create a list of all dates within the date range that the athlete raced, a list of the athlete's rank on that
    # date, and a list of the race names to be used as labels in the graph
    all_race_dates = list(get_results(athlete_name).date)
    race_dates = [date for date in all_race_dates if start_date <= dt.strptime(date, "%m/%d/%Y") <= end_date]

    all_events = list(get_results(athlete_name).event)
    all_locations = list(get_results(athlete_name).location)
    all_race_labels = [all_events[i] + " " + all_locations[i] for i in range(len(all_events))]
    race_labels = []

    # Build the list of dates and labels to be used in the graph
    for date in all_race_dates:
        if start_date <= dt.strptime(date, "%m/%d/%Y") <= end_date:
            race_labels.append(all_race_labels[all_race_dates.index(date)])

    # Build the list of ranks to be used in the graph
    race_date_ranks = []

    print(all_race_dates)
    print(all_race_labels)
    print(race_dates)
    print(race_labels)

    for rd in race_dates:
        file_name = rd.replace("/", "_")
        ranking_data = pd.read_csv(f"rankings_archive/{file_name}.csv")
        rank_on_date = int(ranking_data["rank"][ranking_data.name == athlete_name])
        race_date_ranks.append(rank_on_date)
        progress = len(race_date_ranks) / len(race_dates)
        print("{:.0%}".format(progress))

    print("Progression dates and ranks:")
    print(dates)
    print(ranks)
    print("Race dates, ranks, and race labels:")
    print(race_dates)
    print(race_date_ranks)
    print(race_labels)

    # dict = {
    #     "dates": dates,
    #     "ranks": ranks
    # }
    # df = pd.DataFrame(dict)
    # df.to_csv(f"{athlete_name}progression.csv")


    # Plot progression dates and ranks in a step chart, plot races on top of that as singular scatter points.
    dates = [dt.strptime(date, "%m/%d/%Y") for date in dates]
    race_dates = [dt.strptime(date, "%m/%d/%Y") for date in race_dates]

    plt.step(dates, ranks, where="post")
    plt.plot(race_dates, race_date_ranks, "o")
    for i, label in enumerate(race_labels):
        plt.text(race_dates[i], race_date_ranks[i], label, rotation=45, fontsize="xx-small")
    plt.ylim(ymin=0)
    plt.gca().invert_yaxis()
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("World Ranking")
    start_date = dt.strftime(start_date, "%m/%d/%Y")
    end_date = dt.strftime(end_date, "%m/%d/%Y")
    plt.title(f"World Ranking Progression {start_date} to {end_date}: {athlete_name}\n")
    plt.show()


def ranking_progression2(athlete_name, start_date, end_date):
    """

    :param athlete_name:
    :param start_date:
    :param end_date:
    :return: graph showing athlete's ranking after every race they finished between (inclusive) start_date and end_date
    """

    global G
    G = nx.DiGraph()

    dates = []
    races = []
    ranks = []
    athlete_name = athlete_name.title()

    create_ranking(start_date)
    ranking_data = pd.read_csv(RANKING_FILE_NAME)
    ranked_athletes = list(ranking_data.name)
    if athlete_name in ranked_athletes:
        dates.append(dt.strptime(start_date, "%m/%d/%Y"))
        rank_on_date = int(ranking_data["rank"][ranking_data.name == athlete_name])
        ranks.append(rank_on_date)
        races.append("start")

    start_date = dt.strptime(start_date, "%m/%d/%Y")
    end_date = dt.strptime(end_date, "%m/%d/%Y")

    for file in os.listdir(RESULTS_DIRECTORY):
        G = nx.DiGraph()
        results_file_path = os.path.join(RESULTS_DIRECTORY, file)
        race_data = pd.read_csv(results_file_path)
        race_date = dt.strptime(race_data.date[0], "%m/%d/%Y")
        names_list = [name.title() for name in race_data.athlete_name]
        if race_date < start_date or race_date > end_date or athlete_name not in names_list:
            pass
        else:
            create_ranking(race_data.date[0])
            # print(f"ranking as of {race_data.date[0]} below:")
            # print(pd.read_csv("PageRanking.csv"))
            ranking_data = pd.read_csv(RANKING_FILE_NAME)
            rank_on_date = int(ranking_data["rank"][ranking_data.name == athlete_name])
            # if len(ranks) == 0 or rank_on_date != ranks[-1] or rank_on_date == end_date:
            dates.append(race_date)
            races.append(f"{int(race_data.distance[0])}km {race_data.event[0]} ({race_data.location[0]})")
            ranks.append(rank_on_date)

    dates.append(end_date)
    create_ranking(dt.strftime(end_date, "%m/%d/%Y"))
    ranking_data = pd.read_csv(RANKING_FILE_NAME)
    rank_on_date = int(ranking_data["rank"][ranking_data.name == athlete_name])
    ranks.append(rank_on_date)
    races.append("end")

    dates = [date.strftime("%m/%d/%Y") for date in dates]
    print(dates)
    print(races)
    print(ranks)
    dict = {
        "dates": dates,
        "ranks": ranks
    }
    df = pd.DataFrame(dict)
    df.to_csv("progression2.csv")
    plt.plot(dates, ranks, "o")
    plt.gca().invert_yaxis()
    plt.xlabel("Date")
    plt.ylabel("World Ranking")
    plt.title(f"World Ranking Progression: {athlete_name}")
    plt.show()


def show_results(athlete_name):

    rows = []

    for file in os.listdir(RESULTS_DIRECTORY):
        results_file_path = os.path.join(RESULTS_DIRECTORY, file)
        race_data = pd.read_csv(results_file_path)
        names_list = list(race_data.athlete_name)
        names_list = [name.title() for name in names_list]
        race_data.athlete_name = names_list
        if athlete_name.title() in names_list:
            row = race_data[race_data.athlete_name == athlete_name.title()]
            rows.append(row)

    print(pd.concat(rows, ignore_index=True))


def get_results(athlete_name):

    rows = []

    for file in os.listdir(RESULTS_DIRECTORY):
        results_file_path = os.path.join(RESULTS_DIRECTORY, file)
        race_data = pd.read_csv(results_file_path)
        names_list = list(race_data.athlete_name)
        names_list = [name.title() for name in names_list]
        race_data.athlete_name = names_list
        if athlete_name.title() in names_list:
            row = race_data[race_data.athlete_name == athlete_name.title()]
            rows.append(row)

    return pd.concat(rows, ignore_index=True)


G = nx.DiGraph()
correct_predictions = 0
total_tests = 0

ranking_progression_from_archive("Ferry Weertman", "01/01/2017", "07/30/2018")

