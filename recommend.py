import json
import streamlit as st

import pandas as pd
import numpy as np

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
    st.write('Cache miss')
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
    """Initialized all offered cards to be 0 count."""
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


def format_surprise_deck(cards, id):
    ratings_dict = {
        'deck': [],  # user: RuneTiera Deck ID
        'card': [],  # item: LoR Card ID
        'count': []  # rating: count of cards in the deck
    }

    # Count cards used in the final decklist
    counts = {}
    for card in cards:
        counts[card] = counts[card] + 1 if card in counts else 1

    for card, count in counts.items():
        ratings_dict['deck'].append(id)
        ratings_dict['card'].append(card)
        ratings_dict['count'].append(count)

    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(0, 5))
    # The columns must correspond to user id, item id and ratings (in that order).
    return Dataset.load_from_df(df[['deck', 'card', 'count']], reader)


def format_surprise_data(runs):
    # Format data for surprise's collaborative filtering algorithms
    # https://surprise.readthedocs.io/en/stable/getting_started.html#load-dom-dataframe-py
    ratings_dict = {
        'deck': [],  # user: RuneTiera Deck ID
        'card': [],  # item: LoR Card ID
        'count': []  # rating: count of cards in the deck
    }

    for run in runs:
        counts = init_counts(run)

        # Count cards used in the final decklist
        for card in run['Deck']:
            counts[card] = counts[card] + 1

        for card, count in counts.items():
            ratings_dict['deck'].append(run['id'])
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


predict_with_knn(data)


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
