import streamlit as st
import urllib.request
import json

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
st.write("# RuneTiera's Card Recommender")
deck_url = st.text_input(
    'Paste your decklist e.g. `https://runetiera.com/draft-viewer?run=JAbG8Neme`', 'https://runetiera.com/draft-viewer?run=JAbG8Neme')

# Extract the cards from the Firebase API
cards = []
deck_id = deck_url.split('run=')[1]
with urllib.request.urlopen(RUN_URL + deck_id) as url:
    data = json.loads(url.read().decode())
    cards = extract_cards(data)

# Output card images
st.write("## Cards in your deck")
for group in chunker(cards, 3):
    images = [card_url(card) for card in group]
    st.image(images, width=200)
