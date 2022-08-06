import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objs as go
import plotly.graph_objs as go
import plotly.offline as pyo
import ffn 
import plotly.figure_factory as ff



def getdata():

    print("Entrer le ticker de l'action (exemple : nom du ticker => ^FCHI ) :")

    try:
        mon_portefeuille = [] # on stock les tickers dans cette liste

        while True:
            mon_portefeuille.append(str(input()).upper()) # l'utilisateur rentre les tickers (en mininuscule ou majuscule grâce à upper)

    except:  
        print(mon_portefeuille) 
            #return mon_portefeuille
            #ticker = str(input('')) # l'utilisateur rentre les tickers  
            #data = yf.download(ticker, period="max", interval="1d")["Adj Close"]
            
    data = yf.download(mon_portefeuille, period="max", interval="1d")["Adj Close"]
    data = pd.DataFrame(data).dropna()

    return data


def PortfolioWeight(data):

    global pondList

    n = len(data.columns)

    pondList = []  

    print("rentrez les pondérations de vos portefeuilles la somme doit être égale à 100, exemple : 24.5 50 25.5")

    for i in range(n):

        pondération = float(input())

        pondList.append (pondération)

    if sum(pondList) > 100.05: # on tolère une marge (au cas où l'individu rentre une valeur proche mais différente de 100) 
        print("Erreur, rentrer un montant égal à 100 %, le total de votre portefeuille fait :",sum(pondList)," %")
        raise ValueError
    elif sum(pondList) < 99.95:
        print("Erreur, rentrer un montant égal à 100 %, le total de votre portefeuille fait :",sum(pondList)," %")
        raise ValueError
    else:
        print("Pondération de votre portefeuille :", pondList)
        valeurs = np.array(pondList)/100
        data["Portefeuille"]  = (data * valeurs).sum(axis=1) # on crée le portefeuille pondéré 

    return data

def Parameters(data):

    n = [] # correspond au nombre de scénario 
    j = [] # correspond au nombre de jours simulés

    print("Il faut entrer le nombre de scénario (entre 1 et 99) que vous voulez générez :")
    n = int(input())
    print("pour combien de jours ? ")
    j = int(input())
    if n < 0: 
        print("Erreur, il faut entrer une valeur positive, vous avez entrez:",n, "scénarios")
        raise ValueError
    elif n > 0 and n < 100:
        print("La simulation va générer :",n, "scénarios", " sur ", j, " jours")
    else:
        print(repr("Pas assez de scénario :( :", n, "scénario"))
        raise ValueError

    return n, j 


def PortfolioAnalysis(data):
    
            # camembert du portefeuille 
    n = len(data.columns) - 1 # on veut la taille du portefeuille 
    arr = np.array(pondList)  # on veut les pondérations du portefeuilles
            #numpy_array = arr
    df = pd.DataFrame(arr,data.columns[:n])  # on crée le df qui permettra de créer le camembert de notre portefeuille
    labels = list(data.columns[:n]) # on veut les labels (noms des actifs sauf pour la dernière colonne qui est "portefeuille")
            #values = numpy_array

    fig = px.pie(df, values=arr, names=labels, color_discrete_sequence=px.colors.sequential.RdBu,     title='Pondération de votre portefeuille')
    fig.show()

            # distribution des rendements
    df2 = pd.DataFrame([]) # on initialise un dataframe vierge
    for i in data: # pour chaque colonne de data on calcule le rendement de chaque valeur dans le df vierge
        df2[i] = data[i].pct_change().dropna() # on stock les rendements dans le df vierge 
    fig = ff.create_distplot([df2[c] for c in df2.columns], df2.columns, bin_size=0.01,show_rug=False) # on affiche les distributions des rendements des actifs pour les comparer avec notre portefeuille
    fig.update_layout(xaxis_range=[-0.15,0.15])
    fig.show()

    stats = data["Portefeuille"].calc_stats()
    stats.display()


def RandomWalkparameters(dataframe, j, n): # la fonction demande le dataframe et la colonne (!) pour laquelle 
    # exemple d'utilisation => def RandomWalkparameters(tesla['Adj Close'], 360, 120): dataframe tesla pour 360 jours
    # et pour 120 scénarios. On calcule les paramètres => Moyenne annuelle des rendements + Volatilité annuelle + La dernière valeur observé du dataframe

        # on déclare nos variables pour pouvoir les appeler en dehors de la fonction 
        # sans cette commande, si on appel l'une des variables déclarés, elle ne sera pas définit (erreur)

    global df, DataParams, meanVal, stdVal, lastvalue, defsize1

    meanVal = dataframe.pct_change().mean()
    stdVal = dataframe.pct_change().std()
    lastvalue  = dataframe[-1]
    defsize1 = j

    df = pd.DataFrame(np.array([])) 

            # on va déclarer une matrice nulle que l'on va convertir en dataframe pour stocker les scénarios générés

    for i in range(n): 
        np.random.seed(i) # les scénarios varient en fonction de la "graine" aléatoire de numpy, ici il y en a "i"

        df[i] = pd.DataFrame(np.cumprod(np.random.normal(loc=meanVal*1.1, scale=stdVal*1.1, size=defsize1)+1) * lastvalue)
                # on itère les "i" scénarios dans les colonnes du dataframe initial 

    df.columns = ['scénario ' + str(col)  for col in df.columns ] # on nomme chaque colonne : df["scénario :" i]

    #la fonction retourne le dataframe crée à partir des i scénarios 
    return df

def OperationOnDF(data, df): 

    data = data.reset_index()
    date = data['Date']

    df['date'] = pd.date_range(data['Date'].max()+pd.Timedelta(1,unit='d'),periods=len(df))
    alldate = date.append(df['date'])
    data2 = pd.DataFrame(np.array([]))

    del data2[0]
    
    for column in df:
        data2[column] = data['Portefeuille'].append(df[column])

    data2['Date'] = alldate
    data2 = data2.set_index(data2['Date'])
    del data2['Date'], data2['date']


    return data2



def low_volatility_scenario(n):
    ############calcule la MAD des différentes actions du portefeuille (sans inclure les valeurs manquante du dataset)##################
    print("Combien de scénario qui minimise la Mean Absolute Deviation voulez - vous affichez:")
    i = int(input())
    
    while i > n: ####n = nombre de scénario prédit précedement
        print("Erreur : Vous n'avez pas prédit autant de scénario. Veuillez entrer un nombre inférieur au nombre de scénario prédit ")
        i = int(input())
    else:
        low_vol = df.mad(axis=0, skipna=True).nsmallest(i)
    return low_vol

    
