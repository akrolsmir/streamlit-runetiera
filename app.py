import streamlit as st
import urllib.request
import json

from recommend import runs
from recommend import build_algo
from recommend import format_surprise_data
from recommend import predict_best_cards

RUN_URL = 'https://firestore.googleapis.com/v1/projects/runetiera/databases/(default)/documents/runs/'


def extract_cards(json_data):
    cards = []
    for value in json_data['fields']['Deck']['arrayValue']['values']:
        cards.append(value['stringValue'])
    return cards


def chunker(seq, size):
    """From https://stackoverflow.com/a/434328"""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def card_url(card):
    return f'http://dd.b.pvp.net/latest/set{card[1]}/en_us/img/cards/{card}.png'


# Read user's decklist
st.write("# RuneTiera Expedition Recommender")
st.write("Discover the best cards to add to your deck, powered by MACHINE LEARNING!!!")
# Twisted Fate banner image
st.image(card_url('02BW026-full'), use_column_width=True)
deck_url = st.text_input(
    'Paste a RuneTiera decklist e.g. `https://runetiera.com/draft-viewer?run=JAbG8Neme`', 'https://runetiera.com/draft-viewer?run=JAbG8Neme')

# Extract the cards from the Firebase API
cards = []
deck_id = deck_url.split('run=')[1]
with urllib.request.urlopen(RUN_URL + deck_id) as url:
    response = json.loads(url.read().decode())
    cards = extract_cards(response)

# Print best cards as predicted by recommender
data = format_surprise_data(runs)
algo = build_algo(data)
predictions = predict_best_cards(algo, runs, 'new_id', cards)
st.write("## Best cards to add")
for (card, score) in predictions[0:5]:
    st.write(f'#### Score: {score:.1f}')
    st.image(card_url(card), width=200)

# Output card images
st.write("## Your Decklist")
for group in chunker(cards, 3):
    images = [card_url(card) for card in group]
    st.image(images, width=200)
