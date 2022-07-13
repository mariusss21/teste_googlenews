from datetime import timedelta, date
import pandas as pd
from GoogleNews import GoogleNews
from datetime import datetime, date
import streamlit as st
from time import sleep
from google.cloud import bigquery
from google.oauth2 import service_account
import plotly.graph_objects as go
import streamlit.components.v1 as components
import yfinance as yf

st.set_page_config(
    page_title="Análise ações Petrobrás",
	layout="wide",
    initial_sidebar_state="expanded",
)

m = st.markdown("""
<style>
div.stButton > button:first-child{
    width: 100%;
}
div.block-container{
    padding-top: 1rem;
}
</style>""", unsafe_allow_html=True)
    

@st.cache(show_spinner=True, ttl=3600)
def raw_petro() -> pd.DataFrame:
    df = pd.read_csv('raw_petro.csv')
    df.Date = pd.to_datetime(df.Date).dt.date
    return df


@st.cache(show_spinner=True, ttl=3600)
def raw_gnews() -> pd.DataFrame:
    df = pd.read_csv('raw_gnews.csv', index_col=0)
    df.date = pd.to_datetime(df.date).dt.date
    return df


@st.cache(show_spinner=True, ttl=3600)
def final_df() -> pd.DataFrame:
    df = pd.read_csv('df_final.csv', sep='|')
    df.Date = pd.to_datetime(df.Date).dt.date
    return df


def petro_chart(df_raw_petro_date):
    fig = go.Figure(data=[go.Candlestick(x=df_raw_petro_date['Date'],
                open=df_raw_petro_date['Open'],
                high=df_raw_petro_date['High'],
                low=df_raw_petro_date['Low'],
                close=df_raw_petro_date['Close'])])

    fig.update_layout(
		height=300,
		margin=dict(b=5,	t=0,	l=0,	r=0),
        font=dict(size=15),
        xaxis_rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)


def latest_news(df):
    st.write(df.tail(10))


def news_sentiment(df_final_date):
    df_final_date['label'] = df_final_date['score'].apply(lambda x: 'green' if x > 0 else 'red')

    label = list(df_final_date['label'])
    fig = go.Figure(data=[go.Bar(x=df_final_date.Date,
                        y=df_final_date.score,
                        marker={'color': label})])
    
    fig.update_layout(
		height=300,
		margin=dict(b=5,	t=0,	l=0,	r=0),
        font=dict(size=15),
        xaxis_rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)


def live_values(df_petr4, df_ibov):
    st.write(df_ibov)
    st.write(df_petr4)


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
    #st.write(df_final_date)
    #st.write(df_raw_gnews_date)

    tipo_cotacao = st.sidebar.radio('Cotação', ['Histórica', 'Dia'])

    date_ = date.today() - timedelta(days=1)
    df_petr4 = yf.download('PETR4.SA', start=date_, interval = "1m")
    df_petr4.reset_index(inplace=True)
    df_petr4.rename(columns={'Datetime': 'Date'}, inplace=True)

    df_ibov = yf.download('IBOV', start=date_, interval = "1m")
    df_petr4.reset_index(inplace=True)
    df_petr4.rename(columns={'Datetime': 'Date'}, inplace=True)

    with col1:
        if tipo_cotacao == 'Histórica':
            st.subheader('Cotação histórica')
            petro_chart(df_raw_petro_date)
        if tipo_cotacao == 'Dia':
            st.subheader('Cotação dia')
            petro_chart(df_petr4)
        #news_sentiment(df_final_date)
        #latest_news(df_raw_gnews)

    with col2:
        live_values(df_petr4, df_ibov)



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


    
