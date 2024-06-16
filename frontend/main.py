import sys

sys.path.append('')

import streamlit as st
from datetime import datetime, timedelta
import json
import argparse
import requests


@st.cache_data
def make_request(address, query):
    data = requests.post(f"http://{address}/query", json={"query": query})

    res = data.json()

    return res


def plot_main_page(address):
    st.title('Yappy search')

    DEFAUL_QUERY = ""

    if 'query' not in st.session_state:
        st.session_state.query = DEFAUL_QUERY

    def click_button():
        st.session_state.clicked = True

    query = st.text_input('Enter query', "")
    st.session_state.query = query

    res = make_request(address, query)

    for link in res:
        container = st.container(border=True)
        container.video(link, autoplay=True, muted=True)

    # st.button('Generate trail', on_click=click_button)

    # st.write('Start date is:', st.session_state.start_date)
    # st.write('End date is:', st.session_state.end_date)
    # st.write('The current location is', destination_city, destination_country)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='Yappy search')
    parser.add_argument('--address', default="127.0.0.1:8000")

    args = parser.parse_args()

    plot_main_page(args.address)
