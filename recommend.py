import json
import streamlit as st

with open('data/users.json', encoding='utf-8') as json_file:
    users = json.load(json_file)
    for user in users[0:10]:
        st.write(user)
