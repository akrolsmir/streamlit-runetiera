import json
import streamlit as st

import pandas as pd
import numpy as np


# @st.cache(suppress_st_warning=True)
def loadJson(filename):
    st.write('Cache miss')
    with open(filename, encoding='utf-8') as json_file:
        return json.load(json_file)


def writeJson(filename, data):
    with open(filename, 'w') as out_file:
        json.dump(data, out_file)


# Read from all users, and output to just runs.
# runs = []
# users = loadJson('data/users.json')
# for user in users[0:50]:
#     for run in user['runs']:
#         st.write('run')
#         runs.append(run)
# writeJson('data/runs.json', runs)

runs = loadJson('data/runs.json')
for run in runs:
    st.write(run['id'])
