from datetime import timedelta, date
import pandas as pd
from GoogleNews import GoogleNews
from datetime import datetime
import streamlit as st
from time import sleep
from google.cloud import bigquery
from google.oauth2 import service_account
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(
    page_title="Análise ações Petrobrás",
	layout="wide",
    #initial_sidebar_state="collapsed",
)

m = st.markdown("""
<style>
div.stButton > button:first-child{
    width: 100%;

}
</style>""", unsafe_allow_html=True)

@st.cache(show_spinner=True, ttl=3600)
def raw_petro() -> pd.DataFrame:
    df = pd.read_csv('raw_petro.csv')
    return df


@st.cache(show_spinner=True, ttl=3600)
def raw_gnews() -> pd.DataFrame:
    df = pd.read_csv('raw_gnews.csv')
    return df


@st.cache(show_spinner=True, ttl=3600)
def final_df() -> pd.DataFrame:
    df = pd.read_csv('df_final.csv')
    return df


def dashboard(data_inicial, data_final):
    df_raw_petro = raw_petro()
    st.write(df_raw_petro)

    df_raw_gnews = raw_gnews()
    st.write(df_raw_gnews)

    df_final = final_df()
    st.write(df_final)





if __name__ == '__main__':

    logo = Image.open('sunrise.jpg')
    st.title.image(logo, use_column_width=True)
    st.title.sidebar('Equipe MinIO')
    col1, col2 = st.columns([8, 2])

    st.sidebar.subheader('Menu')
    paginas = ['Análise ações Petrobrás', 'Notícias']
    pagina = st.sidebar.button('Selecione uma análise', paginas)
    data_inicial = st.sidebar.date_input('Data inicial', datetime.now() - timedelta(days=1))
    data_final = st.sidebar.date_input('Data final', datetime.now() - timedelta(days=30))

    if pagina == 'Análise ações Petrobrás':
        dashboard(data_inicial, data_final)
            
    # with col1:
    #     if select_bq:
    #         df = query_stackoverflow()
    #         st.write(df)
    #         fig = go.Figure(data=[go.Candlestick(x=df['Date'],
    #                     open=df['Open'],
    #                     high=df['High'],
    #                     low=df['Low'],
    #                     close=df['Close'])])
    #         st.write(fig)


    
