from surprise import KNNBasic
import json
import streamlit as st

import pandas as pd
import numpy as np

from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate


@st.cache(suppress_st_warning=True)
def loadJson(filename):
    st.write('Cache miss')
    with open(filename, encoding='utf-8') as json_file:
        return json.load(json_file)


def writeJson(filename, data):
    with open(filename, 'w') as out_file:
        json.dump(data, out_file)


# TODO: Filter by 3 or more wins. Also: strip buckets and swaps?
# Read from all users, and output to just runs.
# runs = []
# users = loadJson('data/users.json')
# for user in users[0:50]:
#     for run in user['runs']:
#         st.write('run')
#         runs.append(run)
# writeJson('data/runs.json', runs)

# Format data for surprise's collaborative filtering algorithms
# https://surprise.readthedocs.io/en/stable/getting_started.html#load-dom-dataframe-py
ratings_dict = {
    'card': [],  # item: LoR Card ID
    'deck': [],  # user: RuneTiera Deck ID
    'count': []  # rating: count of cards in the deck
}

runs = loadJson('data/runs.json')
for run in runs:
    counts = {}
    for card in run['Deck']:
        counts[card] = counts[card] + 1 if card in counts else 1

    for card, count in counts.items():
        ratings_dict['card'].append(card)
        ratings_dict['deck'].append(run['id'])
        ratings_dict['count'].append(count)

df = pd.DataFrame(ratings_dict)
df

# A reader is still needed but only the rating_scale param is requiered.
# TODO: Update to be e.g. 1 to 3? It's rare to have 5 of a single card
reader = Reader(rating_scale=(1, 5))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df[['card', 'deck', 'count']], reader)

# We can now use this dataset as we please, e.g. calling cross_validate
cross_validate(NormalPredictor(), data, cv=2, verbose=True)
