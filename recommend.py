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
    'deck': [],  # user: RuneTiera Deck ID
    'card': [],  # item: LoR Card ID
    'count': []  # rating: count of cards in the deck
}

runs = loadJson('data/runs.json')
for run in runs:
    counts = {}
    for card in run['Deck']:
        counts[card] = counts[card] + 1 if card in counts else 1

    for card, count in counts.items():
        ratings_dict['deck'].append(run['id'])
        ratings_dict['card'].append(card)
        ratings_dict['count'].append(count)

df = pd.DataFrame(ratings_dict)
df

# A reader is still needed but only the rating_scale param is required.
# TODO: Update to be e.g. 0 to 3? It's rare to have 5 of a single card
reader = Reader(rating_scale=(0, 5))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df[['deck', 'card', 'count']], reader)

# Now try predicting a rating
# From https://surprise.readthedocs.io/en/stable/getting_started.html#train-on-a-whole-trainset-and-the-predict-method

# Retrieve the trainset.
trainset = data.build_full_trainset()
trainset

# Build an algorithm, and train it.
algo = KNNBasic()
algo.fit(trainset)

# Check how likely this deck would want Back to Back (actually had 3)
# Deck: https://runetiera.com/draft-viewer?run=8Y8ZjejfT
pred = algo.predict('8Y8ZjejfT', '01DE041', r_ui=3, verbose=True)
# They Who Endure
pred = algo.predict('8Y8ZjejfT', '01FR034', r_ui=0, verbose=True)
# Now for https://runetiera.com/draft-viewer?run=JAbG8Neme
pred = algo.predict('JAbG8Neme', '01DE041', r_ui=2, verbose=True)
pred = algo.predict('JAbG8Neme', '01FR034', r_ui=1, verbose=True)
