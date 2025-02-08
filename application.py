"""
📝 **Instructions** :
- Installez toutes les bibliothèques nécessaires en fonction des imports présents dans le code, utilisez la commande suivante :conda create -n projet python pandas numpy seaborn streamlit plotly 
cd d/ H:\BUT3\SAE601
streamlit run application.py
- Complétez les sections en écrivant votre code où c’est indiqué.
- Ajoutez des commentaires clairs pour expliquer vos choix.
- Utilisez des emoji avec windows + ;
- Interprétez les résultats de vos visualisations (quelques phrases).
"""
#Mathis Boutin et Romain Jegoux


### 1. Importation des librairies et chargement des données
import os
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import plotly.express as px

# Chargement des données
df = pd.read_csv("ds_salaries.csv")
st.set_page_config(layout="wide")

### 2. Exploration visuelle des données
st.title("📊 Visualisation des Salaires en Data Science")
st.markdown("Explorez les tendances des salaires à travers différentes visualisations interactives.")


if st.checkbox("Afficher un aperçu des données"):
    st.dataframe(df.head(10), width=10000000, hide_index=True) 


#Statistique générales avec describe pandas 
st.subheader("📌 Statistiques générales")

col1, col2, col3 = st.columns(3) 
with col1:
    #Statistique descriptive
    st.dataframe(df.describe().round(2), width=10000000, hide_index=False) 
with col2:
    #boxplot des salaires par année
    boxplot_salaire = px.box(df, x='work_year', y='salary_in_usd', color='work_year',
                  title=f"Distribution des salaires selon l'année",
                  labels={'work_year' : 'Year', 'salary_in_usd': 'Salaire'})
    st.plotly_chart(boxplot_salaire)
    st.markdown("Sur ce graphique on remarque que le salaire augmente au fur et mesure des années avec une médiane de 2020 de 73k contre une médiane de 145k en 2024")
with col3:
    #boxplot du télétravail par année
    boxplot_remote = px.box(df, x='work_year', y='remote_ratio', color='work_year',
                  title=f"Distribution du TW selon l'année",
                  labels={'work_year' : 'Year', 'remote_ratio': 'TW'})
    st.plotly_chart(boxplot_remote)
    st.markdown("Sur ce graphique on remarque une normalisation du télétravail en 2023 et 2024")


### 3. Distribution des salaires en France par rôle et niveau d'expérience, uilisant px.box et st.plotly_chart
st.subheader("Q3 📈 Distribution des salaires en France")
#on filtre les données pour la france et on selection les variables
df_fr = df[df["employee_residence"] == "FR"]
critere = st.selectbox("Choisir : ", ['job_title', 'experience_level'])
# Création du boxplot
boxplot_salaire_france = px.box(df_fr, x=critere, y='salary_in_usd', color=critere,
                  title=f"Distribution des salaires en France du/d' {critere}",
                  labels={critere : 'Critere', 'salary_in_usd': 'Salaire'})
st.plotly_chart(boxplot_salaire_france)
st.markdown("Sur ce graphique on remarque que en France les salaires des personnes travaillant dans le machine learning sont mieux pays que le reste. Les data analyst on le salaire les plus faibles. Pour l'experience le personne avec le plus d'experinece sont les mieux payés")
# Création de l'histogramme de la distribution des salaires en France
_, col5, _ = st.columns(3) 
with col5:
    st.markdown(" Distribution des salaires en France en USD")
    plot1 = sns.histplot(df_fr["salary_in_usd"], kde = True, bins = 10)
    st.pyplot(plot1.get_figure())
    st.markdown("Sur ce graphique on remarque que le salaire suit une loi normale avec une moyenne à 60 000")


### 4. Analyse des tendances de salaires :
#### Salaire moyen par catégorie : en choisisant une des : ['experience_level', 'employment_type', 'job_title', 'company_location'], utilisant px.bar et st.selectbox 
#Permet de selectionner la variable souhaité
st.subheader('Q4 Analyse des tendances de salaires')
critere1 = st.selectbox("Choisir :", ['experience_level', 'employment_type', 'job_title', 'company_location'])

df_q4 = df.groupby(critere1)["salary_in_usd"].median().reset_index()
# Affichage sous forme de bar chart
st.markdown("Distribution des salaires en France en USD")
st.bar_chart(df_q4, x=critere1, y="salary_in_usd")


### 5. Corrélation entre variables
# Sélectionner uniquement les colonnes numériques pour la corrélation
# Affichage du heatmap avec sns.heatmap
st.subheader("Q5🔗 Corrélations entre variables numériques")
_, col4, _ = st.columns(3) 
with col4:
    numeric_df = df.select_dtypes(include=[np.number]) 
    plot3 = sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plot3.get_figure())
    st.markdown("Sur ce graphique on remarque que les variables sont assez peu corrélé voir pas du tout. Ce qui signifie qu'il n'y a que tres peu de lien entre elle.")


### 6. Analyse interactive des variations de salaire
# Une évolution des salaires pour les 10 postes les plus courants
# count of job titles pour selectionner les postes

# Sélection des 10 postes les plus fréquents
job_counts = df['job_title'].value_counts()
top_10 = job_counts.nlargest(10).index
df_q6 = df[df['job_title'].isin(top_10)]
# calcule du salaire moyen par an
df_q6 = (df_q6.groupby(['work_year', 'job_title'])['salary_in_usd'].mean().reset_index())
# Création du graphique d'évolution des salaires
fig = px.line(df_q6, x='work_year', y='salary_in_usd', color='job_title',
              title='Évolution des salaires pour les 10 postes les plus courants',
              labels={'salary_in_usd': 'Salaire moyen', 'work_year': 'Année', 'job_title': 'Poste'})

st.subheader('Q6 Evolution des salaires')
st.plotly_chart(fig)
st.markdown("Sur ce graphique on remarque que la majorité des emplois on connu une baisse des salaires en 2021 mais depuis il y a une hausse des salaires. ")


### 7. Salaire médian par expérience et taille d'entreprise
# utilisez median(), px.bar

#calcul du salaire median
salary_median = df.groupby(["company_size", "experience_level"])[["salary_in_usd"]].median().reset_index()

#création du graphique en barre
fig = px.bar(
    salary_median, 
    x="company_size",  
    y="salary_in_usd", 
    color="experience_level",  
    labels={'salary_in_usd': 'Salaire médian en USD'},
    title="Salaire Médian par Expérience et Taille d'Entreprise"
)
st.subheader('Q7 Salaire médian par expérience et taille d entreprise')
st.plotly_chart(fig)
st.markdown("Ce graphique nous permet de voir que les entreprises de taille moyenne ont un salaire médian élevé comparé au grande et petite entreprise qui ont un salaire similaire avec iun leger avantage pour les grandes entreprises")


### 8. Ajout de filtres dynamiques
#Filtrer les données par salaire utilisant st.slider pour selectionner les plages

# Ajout d'un filtre permettant de sélectionner une plage de salaires avec un curseur
st.subheader("Q8") 
values_salaire = st.slider("Select a range of values", min(df.salary_in_usd), max(df.salary_in_usd), (min(df.salary_in_usd), max(df.salary_in_usd)))

### 9.  Impact du télétravail sur le salaire selon le pays

#création du boxplot
fig = px.box(df, 
            x="employee_residence", 
            y="salary_in_usd", 
            color="remote_ratio", 
            title="Impact du Télétravail sur le Salaire Selon le Pays",
            labels={"salary_in_usd": "Salaire en USD", "employee_residence": "Pays", "remote_ratio": "Télétravail"})


st.subheader("Q9 Impact du Télétravail sur les Salaires")
st.plotly_chart(fig)

st.markdown("Ce graphique est un boxplot mais sur l'axe x, il y a beaucoup trop de valeurs ce qui le rend illisible. Pour le rendre plus lisible il aurait fallu par exemple regroupé les pays en continent.")




### 10. Filtrage avancé des données avec deux st.multiselect, un qui indique "Sélectionnez le niveau d'expérience" et l'autre "Sélectionnez la taille d'entreprise"
st.subheader("Q10")
col6, col7 = st.columns(2)

#création du filtre
with col6:
    exp = st.multiselect("Niveau d'experience : ", df.experience_level.unique(), df.experience_level[:].unique())
with col7:
    taille = st.multiselect("Sélectionnez la taille d'entreprise : ", df.company_size.unique(), df.company_size[:].unique())
df_q10 = df[
    (df["experience_level"].isin(exp)) & 
    (df["company_size"].isin(taille)) & 
    (df["salary_in_usd"].between(values_salaire[0], values_salaire[1]))
]
st.dataframe(df_q10, width=10000000, hide_index=True)
