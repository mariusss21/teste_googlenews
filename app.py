from datetime import timedelta, date
import pandas as pd
from GoogleNews import GoogleNews
from datetime import datetime
import streamlit as st
from time import sleep
from google.cloud import bigquery
from google.oauth2 import service_account
import plotly.graph_objects as go

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
    df.Date = pd.to_datetime(df.Date).dt.date
    return df


@st.cache(show_spinner=True, ttl=3600)
def raw_gnews() -> pd.DataFrame:
    df = pd.read_csv('raw_gnews.csv')
    df.date = pd.to_datetime(df.date).dt.date
    return df


@st.cache(show_spinner=True, ttl=3600)
def final_df() -> pd.DataFrame:
    df = pd.read_csv('df_final.csv', sep='|')
    df.Date = pd.to_datetime(df.Date).dt.date
    return df


def petro_graph(df_raw_petro_date):
    fig = go.Figure(data=[go.Candlestick(x=df_raw_petro_date['Date'],
                open=df_raw_petro_date['Open'],
                high=df_raw_petro_date['High'],
                low=df_raw_petro_date['Low'],
                close=df_raw_petro_date['Close'])])

    fig.update_layout(
		height=300,
		margin=dict(b=5,	t=0,	l=0,	r=0),
        font=dict(size=15))

    #fig.update_traces(textposition='inside', textfont_color='rgb(255,255,255)', textfont_size=20)
    st.plotly_chart(fig, use_container_width=True)

def dashboard(data_inicial, data_final):
    #coletando os dados
    df_raw_petro = raw_petro()
    df_raw_petro_date = df_raw_petro.loc[(df_raw_petro['Date'] >= data_inicial) & (df_raw_petro['Date'] <= data_final)]

    df_raw_gnews = raw_gnews()
    df_raw_gnews_date = df_raw_gnews.loc[(df_raw_gnews['date'] >= data_inicial) & (df_raw_gnews['date'] <= data_final)]

    df_final = final_df()
    df_final_date = df_final.loc[(df_final['Date'] >= data_inicial) & (df_final['Date'] <= data_final)]

    st.title('Dashboard Petrobrás')

    col1, col2 = st.columns([8, 2])

    with col1:
        petro_graph(df_raw_petro_date)

    with col2:
        pass


if __name__ == '__main__':

    st.sidebar.image("logo_stack.png", use_column_width=True)
    st.sidebar.title('Equipe MinIO')
    st.sidebar.write('StackLabs finanças')
    col1, col2 = st.columns([8, 2])

    paginas = ['Análise ações Petrobrás', 'Notícias', 'Equipe MinIO']
    pagina = st.sidebar.radio('Menu', paginas)

    data_inicial = st.sidebar.date_input('Data inicial', datetime.now() - timedelta(days=30))
    data_final = st.sidebar.date_input('Data final', datetime.now() - timedelta(days=1))
    
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


    
