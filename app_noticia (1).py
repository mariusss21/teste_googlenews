# pip install streamlit
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# importar os pacotes necessários
import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import streamlit.components.v1 as components  # Import Streamlit

####################################################################
# Configura o StreamLit 👋
st.set_page_config(
    page_title="Projeto NVL",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)


#########################################################
# SIDEBAR
#st.sidebar.success("Select a demo above.")
side = st.sidebar
side.title('Home')
side.subheader('Home Broker')
side.subheader('Notícias')
side.subheader('About')
tela = side.radio("",['Broker', 'Notícias', 'About'])


#q_mes = side.radio("Selecione um mês:", lista_mes)
#q_tela = side.selectbox('Selecione uma tela:', ['Quantidade','Despesa'])


####################################################################
# Propriedade
st.markdown("""
""", unsafe_allow_html=True)

####################################################################
# Carrega Base de Dados
arq_raw = './dataset/link_01_GoogleNews_Petr_Jan-2022.csv'
df_news = pd.read_csv(arq_raw, sep='|')

# Carrega listas
lista_midias = list(df_news['media'].unique())
lista_midias.insert(0,'Todas')
lista_datas = list(df_news['date'].unique())
lista_datas.insert(0,'Todas')

# Render the h1 block, contained in a frame of size 200x200.
components.html("""
<html><body>
<!-- TradingView Widget BEGIN -->
<div class="tradingview-widget-container">
  <div class="tradingview-widget-container__widget"></div>
  <div class="tradingview-widget-copyright">
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
  {
  "symbols": [
    {
      "description": "Petro4",
      "proName": "BMFBOVESPA:PETR4"
    },
    {
      "description": "Petro3",
      "proName": "BMFBOVESPA:PETR3"
    },
    {
      "description": "Euro",
      "proName": "FX_IDC:EURBRL"
    },
    {
      "description": "IBOV",
      "proName": "BMFBOVESPA:IBOV11"
    },
    {
      "description": "Dolar",
      "proName": "FX_IDC:USDBRL"
    }
  ],
  "showSymbolLogo": true,
  "colorTheme": "dark",
  "isTransparent": true,
  "displayMode": "adaptive",
  "locale": "br"
}
  </script>
</div>
<!-- TradingView Widget END -->
</body></html>
""", height=80)

#########################################################
# COLUNAS DE CONSULTA
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.write("## Notícias")

with col2:
    pass

with col3:
    q_midia = st.selectbox('Selecione a Mídia:', lista_midias)

with col4:
    q_data = st.selectbox('Selecione a Data:', lista_datas)




#########################################################
# AGRUPAMENTO DE NOTÍCIAS
if (q_midia != 'Todas') and (q_data != 'Todas'):
    df_noticias_view = df_news.loc[( df_news['media'] == q_midia ) & ( df_news['date'] == q_data )]
elif (q_midia == 'Todas') and (q_data != 'Todas'):
    df_noticias_view = df_news.loc[( df_news['date'] == q_data )]
elif (q_midia != 'Todas') and (q_data == 'Todas'):
    df_noticias_view = df_news.loc[( df_news['media'] == q_midia )]
else:
    df_noticias_view = df_news.copy()


#########################################################
# Area de Trabalho
st.write("###### Notícias encontradas: (" + str(df_noticias_view.shape[0]) + ")")

col1_noticias, col2_nuvem = st.columns(2)

with col1_noticias:
    for i in range(df_noticias_view.shape[0]):
        row = df_noticias_view.iloc[i]
        st.markdown("---")
        st.markdown(f"""
            ##### {i+1}. {row['media']} - **{row['title']}**
            {row['desc']}\n
            ![LER NOTÍCIA]({row['img']})
            *{row['date']}* - **[LER NOTÍCIA]({row['link']})**
            """)


with col2_nuvem:
    # Especificar a coluna de titulo do DataFrame
    #summary = df_news['title']
    titulos = " ".join(s for s in df_news['title'])
    descricao = " ".join(s for s in df_news['desc'])
    all_summary = titulos + " " + descricao

    # Concatenar as palavras e remover de String
    #all_summary = " ".join(s for s in summary)
    all_summary = all_summary.replace("\n", "")
    all_summary = all_summary.replace(".", "")
    all_summary = all_summary.replace(",", "")
    all_summary = all_summary.replace("?", "")
    all_summary = all_summary.replace(",", "")
    all_summary = all_summary.replace("|", "")
    all_summary = all_summary.replace("(", "")
    all_summary = all_summary.replace(")", "")

    # Lista de stopword
    stopwords = set(STOPWORDS)
    stopwords.update(["da", "meu", "em", "você", "de", "ao", "os", "mês", "ano", "neste", "podem", "pelo"])

    # Gerar uma wordcloud
    wordcloud = WordCloud(stopwords=stopwords,
                          background_color="black",
                          width=1600, height=800).generate(all_summary)
    # Mostrar a imagem final
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_axis_off()
    plt.imshow(wordcloud)

    # Carrega Imagem
    wordcloud.to_file("sumario_wordcloud.png")
    st.image("sumario_wordcloud.png")

    ##################################################
    st.write(df_news['media'].value_counts())
    st.bar_chart({"data": df_news['media'].value_counts()})
