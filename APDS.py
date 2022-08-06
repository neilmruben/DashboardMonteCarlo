##########################################################################################################
########             LIBRAIRIES                                                                   ########
##########################################################################################################

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import yfinance as yf # https://pypi.org/project/yfinance/
import seaborn as sns

import plotly.express as px
import plotly.graph_objs as go
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.figure_factory as ff

from PIL import Image

##########################################################################################################
########             PARAMETRES DU DASHBOARD                                                      ########
##########################################################################################################



image = Image.open("logo.png")

st.image(image, caption='Projet Applied Data science in Python')



st.write('Entrez les ticker de votre portefeuille comme ceci : AAPL, TSLA, AMZN, ... puis appuyez sur ENTER')
ticker =  st.text_input('')
st.write('Ton portefeuille est composé des actifs suivants :', ticker)

j = st.slider('Combien de jour à simuler ?', min_value=0, max_value=360, step=20)
n = st.slider('Combien de simulation ?', min_value=10, max_value=150, step=25)
st.write("On a ", n," simulations pour ", j, " jours.")


today = datetime.date.today()
before = today - datetime.timedelta(days=360)
start_date = st.sidebar.date_input('Start date', before)
end_date = st.sidebar.date_input('End date', today)


if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.sidebar.error('Erreur : La date de fin doit tomber après la date de début.')
   
    
##########################################################################################################
########             FONCTIONS                                                                    ########
##########################################################################################################    
    
def getdata(ticker):
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")["Adj Close"]
    data = pd.DataFrame(data).dropna() # on converti nos données en dataframe et un dropna pour égaliser le nombre de donnée et de jours
    data['portefeuille'] = data.sum(axis=1) # somme des colonnes qui est le portefeuille
    return data

######################################################################################################################################################################################

def getcamembert(data):
    
    # partie Camembert / Pizza : 
    pondList = [1/len(data.loc[:, data.columns != "portefeuille"].columns)] * len(data.loc[:, data.columns != "portefeuille"].columns) # afficher les pondérations
    arr = np.array(pondList) * 100 # converti en matrice
    camembert = pd.DataFrame(arr,data.loc[:, data.columns != "portefeuille"].columns) # on crée un dataframe spécialemment pour notre camembert
    labels = list(data.loc[:, data.columns != "portefeuille"].columns) # collecte les noms des actifs
    
    return arr, labels

######################################################################################################################################################################################

def getparameters(data, n, j):
    
    
    meanVal = data['portefeuille'].pct_change().mean()
    stdVal = data['portefeuille'].pct_change().std()
    lastvalue  = data['portefeuille'][-1]
    defsize1 = j
    df = pd.DataFrame(np.array([])) 

    # on va déclarer une matrice nulle que l'on va convertir en dataframe pour stocker les scénarios générés
    
    for i in range(n): 
        np.random.seed(i) # les scénarios varient en fonction de la "graine" aléatoire de numpy, ici il y en a "i"
   
        df[i] = pd.DataFrame(np.cumprod(np.random.normal(loc=meanVal*1.1, scale=stdVal*1.1, size=defsize1)+1) * lastvalue)
        # on itère les "i" scénarios dans les colonnes du dataframe initial 
        
    df.columns = ['scénario ' + str(col)  for col in df.columns ] # on nomme chaque colonne : df["scénario :" i]

    
    return df

######################################################################################################################################################################################

def getformat(data, df):
    
    data = data.reset_index()
    date = data['Date']
    df['date'] = pd.date_range(data['Date'].max()+pd.Timedelta(1,unit='d'),periods=len(df))
    alldate = date.append(df['date'])
    Simulation = pd.DataFrame(np.array([]))

    for column in df:
        Simulation[column] = data['portefeuille'].append(df[column])
        
    Simulation['Date'] = alldate
    Simulation = Simulation.set_index(Simulation['Date'])
    del Simulation['Date'], Simulation['date']
    del Simulation[0]
    data = data.set_index(data['Date'])
    del data['Date']

    return Simulation

######################################################################################################################################################################################

if len(ticker)!=0: # si l'utilisateur entre une série de ticker alors l'interface du dashboard se complète
    data = getdata(ticker)
    arr , labels = getcamembert(data)    
    df = getparameters(data, n, j)
    Simulation = getformat(data, df) 

    df2 = pd.DataFrame([]) # on initialise un dataframe vierge
    
    for i in Simulation: # pour chaque colonne de data on calcule le rendement de chaque valeur dans le df vierge
        df2[i] = Simulation[i].pct_change().dropna() # on stock les rendements dans le df vierge 
        
##########################################################################################################
########             VISUALISATION                                                                ########
##########################################################################################################

    st.write('Scénarios de MonteCarlo sur le portefeuille') 
    st.line_chart(Simulation)   
    progress_bar = st.progress(0)

    st.write('Distribution de rendement des scénarios : ')
    fig2 = ff.create_distplot([df2[c] for c in df2.columns], df2.columns,       bin_size=0.01,show_rug=False,show_hist=False) # on affiche les #distributions des rendements des actifs pour les comparer avec notre portefeuille
    fig2.update_layout(xaxis_range=[-0.10,0.10])
    st.plotly_chart(fig2)
    progress_bar = st.progress(0)

    st.write('Composition de votre portefeuille : ')
    
    fig = go.Figure(
        go.Pie(
        labels = labels,
        values = arr,
        hoverinfo = "label+percent",
        textinfo = "value"))
    st.plotly_chart(fig)
    progress_bar = st.progress(0)
    
    st.write('Matrice de corrélation des actifs du portefeuille : ') 
    fig3 = px.imshow(data.pct_change().corr(), text_auto=True, aspect="auto",labels=dict(x="Actifs", color="Dégradé de corrélation"))
    st.plotly_chart(fig3)
    progress_bar = st.progress(0)
        
    st.write('Notre portefeuille :') 
    st.area_chart(data) 
    progress_bar = st.progress(0)
    
    with st.expander("Voir le dataframe des simulations"):
         st.write('Données récentes :', n, 'scénarios pour', j,  'jours')
         st.dataframe(Simulation.head(20))
        
else: # si pas de ticker alors le dashboard ne se display pas 
    print("Pas de ticker")