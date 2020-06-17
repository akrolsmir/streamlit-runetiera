import json
from collections import defaultdict
import streamlit as st

import pandas as pd
import numpy as np

from surprise import SVD
from surprise import NormalPredictor
from surprise import KNNBaseline
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate


# @st.cache(suppress_st_warning=True)
def loadJson(filename):
    with open(filename, encoding='utf-8') as json_file:
        return json.load(json_file)


def writeJson(filename, data):
    with open(filename, 'w') as out_file:
        json.dump(data, out_file)


def processRuns():
    """Read from all users, and output just the runs."""
    runs = []
    users = loadJson('data/users.json')
    for user in users[0:200]:
        for run in user['runs']:
            # TODO: Filter by 3 or more wins?
            # if run['Wins'] >= 3:
            runs.append(run)
    writeJson('data/runs.json', runs)
# processRuns()


def init_counts(run):
    """run is either the output of expeditions-state.json, or array of card IDs"""
    # If card id array, just initialize to all 0s
    if 'Deck' not in run:
        return defaultdict(lambda: 0)

    # Otherwise, specifically initialize seen but unplayed cards to 0.
    counts = {}
    # Initialize picked cards
    for card in run['Deck']:
        counts[card] = 0

    # Initialize all picked and swapped cards
    for pick in run['DraftPicks']:
        if pick['IsSwap']:
            card = pick['SwappedIn'][0]
            counts[card] = 0
        else:
            for card in pick['Picks']:
                counts[card] = 0

    # Initialize offered cards
    if 'offered' in run:
        for offer in run['offered']:
            for i in ['1', '2', '3']:
                for card in offer[i]:
                    counts[card] = 0
    return counts


def format_surprise_data(runs, new_deck=[], new_id='dummy'):
    # Format data for surprise's collaborative filtering algorithms
    # https://surprise.readthedocs.io/en/stable/getting_started.html#load-dom-dataframe-py
    ratings_dict = {
        'deck': [],  # user: RuneTiera Deck ID
        'card': [],  # item: LoR Card ID
        'count': []  # rating: count of cards in the deck
    }

    # NOTE: very hacky, where we merge jsons and a cardlist together to form data set
    for run in (runs + [new_deck]):
        counts = init_counts(run)

        decklist = run['Deck'] if 'Deck' in run else run
        # Count cards used in the final decklist
        for card in decklist:
            counts[card] = counts[card] + 1

        for card, count in counts.items():
            ratings_dict['deck'].append(run['id'] if 'id' in run else new_id)
            ratings_dict['card'].append(card)
            ratings_dict['count'].append(count)

    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(0, 5))
    # The columns must correspond to user id, item id and ratings (in that order).
    return Dataset.load_from_df(df[['deck', 'card', 'count']], reader)


runs = loadJson('data/runs.json')
data = format_surprise_data(runs)


def predict_with_knn(data):
    # From https://surprise.readthedocs.io/en/stable/getting_started.html#train-on-a-whole-trainset-and-the-predict-method
    # Retrieve the trainset.
    trainset = data.build_full_trainset()

    # Build an algorithm, and train it.
    algo = KNNBasic()
    # Alternatively, item-based collaborative filtering:
    # algo = KNNBasic(sim_options={'user_based': False})
    algo.fit(trainset)

    # Check how likely this deck would want Back to Back (actually had 3)
    # Deck: https://runetiera.com/draft-viewer?run=8Y8ZjejfT
    pred = algo.predict('8Y8ZjejfT', '01DE041', r_ui=3, verbose=True)
    # They Who Endure
    pred = algo.predict('8Y8ZjejfT', '01FR034', r_ui=0, verbose=True)
    # Now for https://runetiera.com/draft-viewer?run=JAbG8Neme
    pred = algo.predict('JAbG8Neme', '01DE041', r_ui=2, verbose=True)
    pred = algo.predict('JAbG8Neme', '01FR034', r_ui=1, verbose=True)


def benchmark(data):
    benchmark = []
    # Iterate over all algorithms
    for algorithm in [KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore()]:
        # Perform cross validation
        results = cross_validate(algorithm, data, measures=[
            'RMSE'], cv=3, verbose=False)

        # Get results & append algorithm name
        tmp = pd.DataFrame.from_dict(results).mean(axis=0)
        tmp = tmp.append(pd.Series([str(algorithm).split(
            ' ')[0].split('.')[-1]], index=['Algorithm']))
        benchmark.append(tmp)

    final_df = pd.DataFrame(benchmark).set_index(
        'Algorithm').sort_values('test_rmse')
    final_df
# benchmark(data)

# From https://surprise.readthedocs.io/en/stable/FAQ.html#top-n-recommendations-py


def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def predict_best_cards(data, new_id):
    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)

    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)

    top_n = get_top_n(predictions, n=10)

    # Print the recommended items for each user
    for uid, user_ratings in top_n.items():
        # Basically assumes the best cards for the new deck
        if uid == new_id:
            print('##   Recommended: ', uid, user_ratings)
            return user_ratings


test_deck = [
    "02BW026",
    "02BW057",
    "02BW037",
    "02BW046",
    "02BW061",
    "02BW024",
    "01IO049",
    "01IO010",
    "02IO009",
    "01IO018",
    "01IO037",
    "02BW042",
    "02BW049",
    "02BW058",
    "01IO037",
    "02BW048",
    "02BW006",
    "02BW061",
    "02BW005",
    "02BW016",
    "02BW037",
    "02BW009",
    "01IO021",
    "02IO006",
    "02IO003",
    "02IO003",
    "02BW014",
    "02IO006",
    "02BW024",
    "02BW014",
    "02BW011",
    "01IO048",
    "02IO009",
    "02BW024",
    "02BW039",
    "02IO009"
]

data = format_surprise_data(runs, test_deck, 'new_id')
predict_best_cards(data, 'new_id')
