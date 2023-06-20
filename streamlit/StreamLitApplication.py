# Stream Lit application
# https://share.streamlit.io/
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random
import spacy_streamlit
import joblib
from streamlit_shap import st_shap
import shap
import spacy
from streamlit_chat import message
import emoji
import re
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

st.set_page_config(
    page_title='VerbaSatisPyzer',
    page_icon='analysis.png',
    layout='wide')
st.markdown(
    """
    <style>
        /* Style pour centrer le texte */
        .center {
            text-align: center;
            margin-left: auto;
            margin-right: auto;
        }
       .block-container{
            width:80%;
            margin-left: 7.4%;
            margin-right: 7.4%;
            padding:5%;
            text-align: justify;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown('<h2 class="center">VerbaSatisPyzer</h2>',
            unsafe_allow_html=True)
st.markdown('<h4 class="center">Analyse des verbatims pour la satisfaction client</h4>',
            unsafe_allow_html=True)
page = "Accueil"
page = option_menu(None, ["Accueil","Données","Analyse exploratoire", "Modélisation", "Chatbot","Conclusion et Perspectives"],
                   icons=['house-fill',"database-fill", "bar-chart-fill",
                          "pc-display", "chat-left-dots","shield-check"],
                   menu_icon="cast", default_index=0, orientation="horizontal", styles={
    "nav-link": {"--hover-color": "#FFBCBC"},
    "container": {"padding": "0!important"}})

footer_html = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    opacity: 0.5;
    width: 100%;
    height: 30px;
    background-color: #000000;
    color: #cbcbcb;
    text-align: center;
    z-index: 9999;
}

.separator {
    display: inline-block;
    margin: 0 5px;
    font-size: 20px;
    color: #cbcbcb;
}
.footer a {
    color:  #cbcbcb;
    text-decoration: none;
}
.footer a:hover {
    color:  #cbcbcb;
    text-decoration: underline;
}
</style>
<div class="footer">
    <p>
    VerbaSatisPyzer
    <span class="separator">&#8226;</span>
    Analyse des verbatims pour la satisfaction client
    <span class="separator">&#8226;</span>
    <a href="https://www.linkedin.com/in/daguima/">Daniela Guisao Marin</a>
    <span class="separator">&#8226;</span>
    <a href="https://formation.datascientest.com/data-scientist-landing-page?utm_term=data%20scientist%20formation%20continue&utm_campaign=%5Bsearch%5D+data+scientist&utm_source=adwords&utm_medium=ppc&hsa_acc=9618047041&hsa_cam=15509646166&hsa_grp=130979844436&hsa_ad=568081578908&hsa_src=g&hsa_tgt=kwd-314862478488&hsa_kw=data%20scientist%20formation%20continue&hsa_mt=e&hsa_net=adwords&hsa_ver=3&gclid=CjwKCAjwm4ukBhAuEiwA0zQxk4kbRxVPy93ZRGk_bXloREijHxs7_bC2i3K8GeYkgO8vBdvDy-bh3hoC-40QAvD_BwE">
    Formation Continue Data Scientist</a> 
    <span class="separator">&#8226;</span>
    Promotion Septembre 2022
    </p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)

# with subtab_overview:
if page == "Accueil":
    st.runtime.legacy_caching.clear_cache()
    st.markdown("<h3 class='center'>Quelle est l'utilité de recueillir et analyser les verbatims des clients ?</h3>",unsafe_allow_html=True)
    text = "L’analyse des verbatims clients offre un aperçu précieux de la voix des consommateurs. Elle permet aux entreprises de mieux comprendre leur clientèle, d'améliorer leur offre, de renforcer leur relation client et de rester compétitives sur le marché. En exploitant ces informations riches et qualitatives, les entreprises peuvent prendre des décisions éclairées et orienter leur stratégie vers une satisfaction client optimale."
    text2="Une méthode courante pour recueillir et analyser les verbatims clients est le web scraping des avis sur des sites d'avis en ligne tels que Trustpilot. Cette technique automatisée permet de collecter rapidement une grande quantité de données provenant des avis clients et des commentaires, offrant ainsi une source riche et variée de verbatims." 
    text3="Une fois collectés, ces verbatims peuvent être analysés à l'aide d'outils d'analyse de texte et de traitement du langage naturel, tels que les bibliothèques de NLP (Natural Language Processing) disponibles en Python. Ces outils permettent d'extraire des informations pertinentes des verbatims, comme les thèmes récurrents, les sentiments exprimés et les problèmes spécifiques soulevés par les clients. Grâce à cette analyse approfondie, les entreprises peuvent prendre des décisions stratégiques éclairées et mettre en place des actions d'amélioration adaptées à leurs besoins."
    image_path = "emoji.png"
    
    col1, col2 =  st.columns([3, 1])

    with col1:
        st.markdown("<p>"+text+"</p>"+"<p>"+text2+"</p>"+"<p>"+text3+"</p>",unsafe_allow_html=True)
    with col2:
        st.image(image_path,use_column_width=True)

    st.markdown("<h3 class='center'>Objectif du projet</h3>",unsafe_allow_html=True)
    text4="Ce projet vise à créer un système d'extraction d'informations à partir de commentaires. L'objectif principal consiste à prédire la satisfaction des clients, c'est-à-dire déterminer s'ils sont satisfaits ou insatisfaits, en utilisant une approche supervisée." 
    text5="Un deuxième objectif est d'identifier le sujet abordé dans les commentaires (comme un problème de livraison ou un article défectueux) en utilisant une approche non supervisée. "
    text6="Enfin, le but est de développer un chatbot capable d'analyser les commentaires des clients, d'évaluer leur niveau de satisfaction, de comprendre le sujet discuté et de proposer une solution appropriée en conséquence."
    st.markdown("<p>"+text4+"</p>"+"<p>"+text5+"</p>"+"<p>"+text6+"</p>",unsafe_allow_html=True)
    
elif page == "Données":
    st.runtime.legacy_caching.clear_cache()
    st.markdown("<h3 class='center'>Collecte des données</h3>",unsafe_allow_html=True)
    st.write("Pour collecter les verbatims des avis d'utilisateurs de plusieurs sites d'e-commerce tels qu'Amazon, Rakuten, Cdiscount et Wish ont été collectés à partir de la plateforme Trustpilot en utilisant des techniques de web scraping. ")
    image_path = "GetImage.png"
    col11,col21,col31=st.columns([1,3,1])
    with col21:
        st.image(image_path,use_column_width=False)
    with col31:
        st.write(" ")
    with col11:
        st.write(" ")
    st.write("Le web scraping est une technique automatisée permettant d'extraire des données structurées à partir de pages web non structurées. Elle est utilisée dans différents domaines tels que le marketing et l'analyse de données.  ")
    st.write("Dans le cadre de ce projet, la bibliothèque Python BeautifulSoup a été utilisée pour extraire les données en analysant la structure HTML de Trustpilot.")
    image_path = "GetImage1.png"
    col12,col22,col32=st.columns([1,3,1])
    with col22:
        st.image(image_path,use_column_width=False)
    with col32:
        st.write(" ")
    with col12:
        st.write(" ")
    
    st.write("La base de données ainsi obtenue est composée de 41 536 entrées et contient des variables explicatives telles que le nom de l'utilisateur, le pays où l'avis a été publié, la date de publication de l'avis, la note attribuée, le titre de l'avis, le contenu de l'avis et la date de l'expérience de l'utilisateur.")
    df=pd.read_csv("Base.csv")
    df = pd.read_csv("Base.csv")

    # Renommer les colonnes
    df = df.rename(columns={"customer_name": "Nom de l'utilisateur",
                            "pays": "Pays",
                            "date_commentaire":"Date de publication avis",
                            "note":"Note attribuée",
                            "titre_commentaire":"Titre de l'avis",
                            "commentaire":"Verbatim",
                            "date_experience":"Date de l'experience",
                            "Site":"Site e-commerce",
                            })
    st.dataframe(df.sample(100))
    st.markdown("<h3 class='center'>Nettoyage des données </h3>",unsafe_allow_html=True)
    st.write("Avant de procéder à l'analyse exploratoire et à la modélisation, il est essentiel de nettoyer la base de données. Pour ce faire, plusieurs étapes ont été entreprises, notamment la détection de la langue de chaque avis pour éliminer les avis qui ne sont pas en français, la transformation des dates pour uniformiser les modalités, même celles qui ne contiennent pas de dates (par exemple 'Actualisé il y a 21 heures') et la suppression des lignes générées par des bots qui contiennent généralement des codes de réduction sur le site. ")
    df=pd.read_csv("Ouais.csv")
    df = df.rename(columns={"customer_name": "Nom de l'utilisateur",
                            "pays": "Pays",
                            "date_commentaire":"Date de publication avis",
                            "note":"Note attribuée",
                            "titre_commentaire":"Titre de l'avis",
                            "commentaire":"Verbatim",
                            "date_experience":"Date de l'experience",
                            "Site":"Site e-commerce",
                            })
    st.dataframe(df.sample(100))
    st.write("La base de données nettoyée contient désormais 41482 lignes et 8 colonnes. ")
    st.markdown("<h3 class='center'>Ajout de nouvelles variables </h3>",unsafe_allow_html=True)
    st.write("Dans le cadre de l'analyse des verbatims pour mesurer la satisfaction client, il est essentiel d'ajouter de nouvelles variables pour enrichir la base de données et améliorer la précision des résultats obtenus. Ces variables supplémentaires fournissent des informations contextuelles et des nuances qui permettent une analyse plus approfondie et précise des commentaires des clients. ")
    df=pd.read_csv("Train.csv")
    df = df.rename(columns={"note": "Note attribuée",
                            "commentaire":"Verbatim",
                            "type":"Type avis",
                            "nb_mots_comm":"Nb mots verbatim",
                            "expression_count_comm":"Nb signes expression",
                            "annee_comm":"Année de publication",
                            "contains_ellipsis_comm":"Points suspension",
                            "commentaireAmeliore":"Verbatim corrigé"
                            })
    st.dataframe(df.sample(100))
    st.write("Les variables collectées lors de l'étape précédente ont été utilisées pour créer les variables suivantes : ")
    st.markdown("""
    1. **Type** : Il s'agit d'une variable binaire qui prend la valeur 0 lorsque la note attribuée par le client est inférieure ou égale à trois (clients insatisfaits) et la valeur 1 lorsque cette note est supérieure ou égale à quatre (clients satisfaits). Cette variable représente la variable cible pour le problème de classification utilisant une approche supervisée. 
    2. **Date** : Les variables jour, mois, année et jour de la semaine ont été créées à partir des dates de publication des avis et des dates d'expérience des utilisateurs. Elles permettent de prendre en compte le facteur temporel dans l'analyse. 
    3. **Variables liées au titre et au contenu de l'avis** : Diverses variables ont été créées, notamment le nombre de signes de ponctuation standard (virgule, point-virgule et deux points), le nombre de signes de ponctuation expressifs (point d'interrogation, point d'exclamation), la présence de points de suspension (une variable binaire qui indique s'ils sont présents), le nombre d'emojis, le nombre de mots, le nombre de mots en majuscule et la longueur totale du texte en nombre de caractères utilisés. 
    4. **Bigrammes** : Les bigrammes correspondent à des séquences de deux mots consécutifs dans un texte. Dans le cadre de ce projet, des variables indicatrices ont été créées pour identifier la présence de certains bigrammes dans le texte. Cela permet de capturer des relations spécifiques entre les mots et d'extraire des informations supplémentaires. 
    5. **NER et POS** : Les variables sont créées à partir de deux tâches du traitement automatique du langage naturel. NER (Named Entity Recognition) vise à identifier et à classifier les entités nommées dans un texte, tandis que POS (Part-of-Speech) attribue des étiquettes grammaticales aux mots dans une phrase. Ces deux tâches sont essentielles pour comprendre le contenu textuel, extraire des informations spécifiques et analyser la structure grammaticale des phrases. 
    """)
    st.write("Exemple NER: ")
    # streamlit_app.py

    nlp = spacy.load("fr_core_news_sm")
    doc = nlp("Lors de sa visite à Paris, Marie, une célèbre écrivaine, a été invitée à une réunion de l'UNESCO, une organisation internationale basée à Paris, pour discuter de la préservation du patrimoine culturel mondial.")
    spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe("ner").labels,show_table=False)
    
    st.write("Exemple POS: ")
    spacy_streamlit.visualize_parser(doc)

    st.write("En intégrant ces nouvelles variables dans l'analyse des verbatims, il est possible d'obtenir des insights plus précis sur la satisfaction client, en tenant compte du contexte, de la syntaxe, des entités nommées et d'autres aspects pertinents.")


elif page == "Analyse exploratoire":
    st.runtime.legacy_caching.clear_cache()
    st.markdown("<h3 class='center'>Introduction</h3>",unsafe_allow_html=True)
    st.write("L'analyse exploratoire revêt une importance cruciale dans tout projet, car elle offre une compréhension initiale des données, ouvre la voie à des pistes de recherche supplémentaires, favorise des décisions plus éclairées et révèle des informations précieuses dissimulées au sein des données brutes. ")
    st.write("Cette section se consacrera à l'analyse exploratoire qui sera effectuée au cours des deux étapes suivantes : ")
    st.markdown("""
    1. **Exploration visuelle et statistiques descriptives** : Quelle est la répartition des avis par site e-commerce ? Comment les notes attribuées se répartissent-elles ? Existe-t-il des variations de satisfaction client entre les différents sites ? Quelles variables semblent avoir une influence sur la satisfaction client ?  
    2. **Résumé et interprétation** : Quelles variables ont été sélectionnées pour la modélisation et quel est leur impact sur les notes attribuées ? Quelles améliorations peuvent être apportées pour optimiser davantage la modélisation ? """)
    st.markdown("<h3 class='center'>Exploration visuelle et statistiques descriptives</h3>",unsafe_allow_html=True)
    st.write("Grâce au web scraping effectué sur la plateforme Trustpilot, 41 482 avis ont été collectés. Parmi ces données, Rakuten occupe la première place en termes de représentation avec 32,3% des avis collectés, suivi par Cdiscount, Wish et enfin Amazon, qui représente 18,5% des avis.")
    df = pd.read_csv(r"Base.csv")
    # Créer un nouveau DataFrame qui compte le nombre d'occurrences de chaque site
    site_counts = pd.DataFrame({"Site":
        df['Site'].value_counts(normalize=True) * 100})
    # Créer un diagramme à barres à partir des données du nouveau DataFrame
    fig3 = go.Figure(data=[go.Bar(x=site_counts.index, y=site_counts['Site'], marker_color=["#16A82F", "#E42D3A", "#E47C2D",
                     "#E4D82D"], text=site_counts['Site'].round(2).astype(str) + '%', textposition='outside', textfont=dict(size=14))])
    # Définir le titre et les étiquettes des axes
    fig3.update_layout(title='Répartition des sites', xaxis_title='Site',
                       yaxis_title='Pourcentage de commentaires', yaxis_range=[0, max(site_counts['Site']) + 10])
    st.plotly_chart(fig3, use_container_width=True)
    st.write("Les clients de Rakuten, Wish et Cdiscount semblent globalement satisfaits de leur expérience. La note la plus fréquemment attribuée est 5, suivie de 1, 4, 3 et 2. En revanche, les clients d'Amazon semblent extrêmement mécontents, puisque la note la plus couramment attribuée est de 1.")
    st.write("Il est également remarqué que la plupart des utilisateurs ont tendance à donner des notes extrêmes (1 ou 5), ce qui rend plus difficile la classification des notes intermédiaires.")
    col2, col3 = st.columns([6,1])
    with col3:
        datasel=st.radio(
            "Selection des données",
            key="Données",
            options=["Global", "Rakuten", "Cdiscount","Wish","Amazon"]
        )
        
        dfbase=df
        diccolor={"Global":"#4C7CDB", "Rakuten":"#16A82F", "Cdiscount":"#E42D3A","Wish":"#E47C2D","Amazon":"#E4D82D"}
        colorsgraph=diccolor[datasel]
        if datasel!="Global":
            dfbase = df.loc[df.Site==datasel]
    with col2:
        # Créer un nouveau DataFrame qui compte le nombre d'occurrences de chaque note
        note_counts = pd.DataFrame(
            {'note': dfbase['note'].value_counts(normalize=True) * 100})
        # Créer un diagramme à barres à partir des données du nouveau DataFrame
        fig1 = go.Figure(data=[go.Bar(x=note_counts.index, y=note_counts['note'], marker_color=colorsgraph,
                         text=note_counts['note'].round(2).astype(str) + '%', textposition='outside', textfont=dict(size=14))])
        # Définir le titre et les étiquettes des axes
        fig1.update_layout(title='Nombre d\'occurrences de chaque note '+str(datasel), xaxis_title='Note attribuée',
                           yaxis_title='Pourcentage d\'occurrences', yaxis_range=[0, max(note_counts['note']) + 10])
        st.plotly_chart(fig1, use_container_width=True)  
    col12,col22 = st.columns([5,3])
    TabSites=["Rakuten","Wish","Cdiscount","Amazon"]
    with col12:
        TabColors=["#16A82F","#E47C2D","#E42D3A","#E4D82D"]
        fig4 = go.Figure()
        fig4.add_trace(go.Box(
                y=df.note,
                name="Global",
                jitter=0.3,
                marker_color='#4C7CDB',
                line_color='#4C7CDB',
                boxmean=True
            ))
        for j,i in enumerate(TabSites):
            fig4.add_trace(go.Box(
                y=df.loc[df.Site==i].note,
                name=i,
                jitter=0.3,
                marker_color=TabColors[j],
                line_color=TabColors[j],
                boxmean=True
            ))
        fig4.update_layout(title='Diagrammes en boîte sur la répartition des notes', xaxis_title='Site e-commerce',
                           yaxis_title='Notes')
        st.plotly_chart(fig4, use_container_width=True)
    with col22:
            table_data = []
            row=["Global"] +[np.mean(df.note)]+[np.median(df.note)]+[np.std(df.note)]
            table_data.append(row)
            for i in TabSites:
                np.mean(df.loc[df.Site==i].note)
                row = [i] +[np.mean(df.loc[df.Site==i].note)]+[np.median(df.loc[df.Site==i].note)]+[np.std(df.loc[df.Site==i].note)]
                table_data.append(row)
            st.write("Tableau récapitulatif des données")
            st.table(pd.DataFrame(table_data, columns=["Site","Moyenne","Medianne","Ecart type"]))



    st.write("Les avis collectés couvrent une période allant de 2018 à 2023. La répartition des avis par année est relativement homogène pour chaque site e-commerce, évitant ainsi une sous-représentation ou une surreprésentation excessive d'un site en particulier. Les années les plus représentées sont 2020 et 2021, correspondant à la période de la pandémie de Covid-19, où les ventes en ligne ont connu une forte augmentation. ")
    st.write("La satisfaction des clients semble avoir connu une progression au fil du temps, atteignant un pic avant de diminuer légèrement à partir de 2022. Cependant, il convient de prendre cette conclusion avec prudence, car la répartition des avis n'est pas uniforme pour toutes les années de l'étude. ")
    dfData = pd.read_csv(r"Data.csv")
    datasel=st.selectbox(
            "Veuillez séléctionner un site e-commerce",
            key="DonnéesBis",
            options=["Global", "Rakuten", "Cdiscount","Wish","Amazon"]
        )
        
    if datasel!="Global":
        dfData = dfData.loc[dfData.Site==datasel]
    col13, col23 = st.columns(2)
    with col13:
        # Créer un nouveau DataFrame qui compte le nombre d'occurrences de chaque site
        comm_counts = pd.DataFrame({"Annee Commentaire":
            dfData['annee_comm'].value_counts(normalize=True) * 100})
        exp_counts = pd.DataFrame({"Annee Experience":
            dfData['annee_exp'].value_counts(normalize=True) * 100})
        # Créer un diagramme à barres à partir des données du nouveau DataFrame
        fig3 = go.Figure(data=[go.Bar(name="Année du commentaire",x=comm_counts.index, y=comm_counts['Annee Commentaire'], marker_color="#44D4A8", text=comm_counts['Annee Commentaire'].round(2).astype(str) + '%', textposition='outside', textfont=dict(size=14)),
                              go.Bar(name="Année de l'expérience",x=exp_counts.index, y=exp_counts['Annee Experience'], marker_color="#D44470", text=exp_counts['Annee Experience'].round(2).astype(str) + '%', textposition='outside', textfont=dict(size=14))])
        # Définir le titre et les étiquettes des axes
        fig3.update_layout(title='Répartition des avis utilisateurs par année', xaxis_title='Année ',
                           yaxis_title='Pourcentage des avis par année', yaxis_range=[0, max(comm_counts['Annee Commentaire']) + 10])
        st.plotly_chart(fig3, use_container_width=True)
    with col23:
        # Groupement des données par année de l'avis et calcul de la moyenne du nombre d'étoiles
        stars_by_year_comm = dfData.groupby('annee_comm')['note'].mean()
        # Groupement des données par année de l'expérience et calcul de la moyenne du nombre d'étoiles
        stars_by_year_exp = dfData.groupby('annee_exp')['note'].mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stars_by_year_comm.index, y=stars_by_year_comm,
                            mode='lines',
                            name='Année du commentaire',marker_color="#44D4A8"))
        fig.add_trace(go.Scatter(x=stars_by_year_exp.index, y=stars_by_year_exp,
                            mode='lines',
                            name="Année de l'expérience",marker_color="#D44470"))
        fig.update_layout(title="Note moyenne par année", xaxis_title='Année',
                           yaxis_title="Note moyenne",yaxis_range=[0, 5.5])
        st.plotly_chart(fig, use_container_width=True)
    st.write("Les utilisateurs ont tendance à laisser davantage de commentaires en début de semaine et légèrement moins les week-ends, bien que la répartition des jours de la semaine auxquels les avis sont publiés soit très homogène.")
    st.write("Cependant, il est intéressant de noter que les utilisateurs qui postent leurs avis vers la fin de la semaine semblent être plus satisfaits que ceux qui les postent en début de semaine.")
    col14, col24 = st.columns(2)
    with col14:
        count_by_day = dfData.groupby('jour_semaine_comm')['note'].count()
        fig = go.Figure(data=[go.Pie(labels=count_by_day.index, values=count_by_day,textinfo='label+percent')])
        fig.update_layout(title="Nombre d'avis par jour de la semaine")
        st.plotly_chart(fig, use_container_width=True)
    with col24:
        jours_semaine_ordre = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dfData['jour_semaine_comm'] = pd.Categorical(dfData['jour_semaine_comm'], categories=jours_semaine_ordre, ordered=True)
        dfData['jour_semaine_exp'] = pd.Categorical(dfData['jour_semaine_exp'], categories=jours_semaine_ordre, ordered=True)
        
        # Groupement des données par jour de la semaine du commentaire et calcul de la moyenne du nombre d'étoiles
        stars_by_day_comm = dfData.groupby('jour_semaine_comm')['note'].mean()
        
        # Groupement des données par jour de la semaine de l'expérience et calcul de la moyenne du nombre d'étoiles
        stars_by_day_exp = dfData.groupby('jour_semaine_exp')['note'].mean()
        
        fig = go.Figure()
        fig = go.Figure(data=[
            go.Bar(name="Jour du commentaire", x=stars_by_day_comm.index, y=stars_by_day_comm, marker_color="#44D4A8"),
            go.Bar(name="Jour de l'expérience", x=stars_by_day_exp.index, y=stars_by_day_exp, marker_color="#D44470")
        ])
        fig.update_layout(
            title="Note moyenne par jour de la semaine",
            xaxis_title='Jour de la semaine',
            yaxis_title="Note moyenne",
            yaxis_range=[0, 5.5]
        )
        st.plotly_chart(fig, use_container_width=True)
    st.write("La majorité des avis sont très concis, contenant moins de 10 mots, tandis qu'il y a peu de personnes qui laissent des avis de plus de 50 mots. ")
    st.write("En analysant la note moyenne en fonction du nombre de mots dans l'avis, il est possible de constater que les clients les plus satisfaits sont ceux qui rédigent des commentaires courts, tandis que ceux qui expriment leur mécontentement sont enclins à laisser des commentaires plus longs.")
    col15, col25 = st.columns(2)
    with col15:
        ordre=["0-10 mots","10-20 mots","20-30 mots","30-40 mots","40-50 mots","50-60 mots","60-70 mots","70-80 mots","80-90 mots","90-100 mots","100-150 mots","Plus de 150 mots"]
        count_by_length = dfData["nb_mots_comm_disc"].value_counts().reindex(ordre)
        # Créer un diagramme à barres à partir des données du nouveau DataFrame
        fig3 = go.Figure(data=[go.Bar(x=count_by_length.index, y=count_by_length, marker_color="#F2BB00", text=count_by_length.round(2).astype(str), textposition='outside', textfont=dict(size=14))])
        # Définir le titre et les étiquettes des axes
        fig3.update_layout(title='Nombre de mots dans le commentaire', xaxis_title='Nombre de mots',
                           yaxis_title='Nombre de commentaires',yaxis_range=[0, max(count_by_length)+1000])
        st.plotly_chart(fig3, use_container_width=True)
    with col25:
        ordre=["0-10 mots","10-20 mots","20-30 mots","30-40 mots","40-50 mots","50-60 mots","60-70 mots","70-80 mots","80-90 mots","90-100 mots","100-150 mots","Plus de 150 mots"]
        # Groupement des données par année de l'avis et calcul de la moyenne du nombre d'étoiles
        stars_by_comm_length = dfData.groupby('nb_mots_comm_disc')['note'].mean().reindex(ordre)
        fig = go.Figure()
        fig = go.Figure(data=go.Bar(name="Commentaire",x=stars_by_comm_length.index, y=stars_by_comm_length, marker_color="#FD2F22"))

        fig.update_layout(title="Note moyenne par nombre de mots dans le commentaire", xaxis_title='Nombre de mots',
                           yaxis_title="Note moyenne",yaxis_range=[0, 5.5])
        st.plotly_chart(fig, use_container_width=True)
    st.write("L'analyse du nombre de signes de ponctuation dans les commentaires peut fournir des informations intéressantes sur le degré de satisfaction des utilisateurs. Il est remarquable qu'un nombre très limité d'utilisateurs utilise des signes de ponctuation. Cependant, la note moyenne en fonction du nombre de signes de ponctuation révèle une tendance selon laquelle plus un utilisateur utilise de signes de ponctuation, moins il semble être satisfait.")
    st.write("Il convient d'être attentif à ces deux variables, à savoir le nombre de points d'exclamation et d'interrogation, ainsi que le nombre de signes de ponctuation standard, lors de la modélisation, car elles peuvent être corrélées avec le nombre de mots présents dans le commentaire.")
    col16, col26 = st.columns(2)
    with col16:
        ordre=["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15",">15"]
        count_excla = dfData["expression_count_comm2"].value_counts().reindex(ordre)
        count_ponc = dfData["standard_count_comm2"].value_counts().reindex(ordre)
        # Créer un diagramme à barres à partir des données du nouveau DataFrame
        fig3 = go.Figure(data=[go.Bar(name="! ?",x=count_ponc.index, y=count_ponc, marker_color="#EA3D03"),
                              go.Bar(name=", ; :",x=count_excla.index, y=count_excla, marker_color="#23AB32")])
        # Définir le titre et les étiquettes des axes
        fig3.update_layout(title='Nombre de signes de ponctuation dans les commentaires', xaxis_title='Nombre de signes de ponctuation',
                           yaxis_title='Nombre de commentaires')
        st.plotly_chart(fig3, use_container_width=True)
    with col26:
        ordre=["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15",">15"]
        # Groupement des données par année de l'avis et calcul de la moyenne du nombre d'étoiles
        stars_by_comm_ponct = dfData.groupby('standard_count_comm2')['note'].mean().reindex(ordre)
        stars_by_excla_ponct = dfData.groupby('expression_count_comm2')['note'].mean().reindex(ordre)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ordre, y=stars_by_comm_ponct,
                            mode='lines',
                            name="! ?", marker_color="#EA3D03"))
        fig.add_trace(go.Scatter(x=ordre, y=stars_by_excla_ponct,
                            mode='lines',
                            name=", ; :", marker_color="#23AB32"))

        fig.update_layout(title="Note moyenne par nombre de signes de ponctuation", xaxis_title='Nombre de signes de ponctuation',
                           yaxis_title="Note moyenne")
        st.plotly_chart(fig, use_container_width=True)
    st.write("L'utilisation de mots en majuscules peut intensifier l'aspect émotionnel d'un avis, ce qui peut influencer la note attribuée en amplifiant le sentiment positif ou négatif exprimé. Les données collectées suggèrent une corrélation entre l'utilisation de mots en majuscules et le degré d'insatisfaction de l'utilisateur. Cela peut être une variable intéressante à prendre en compte lors de la modélisation. ")
    ordre=["0","1","2","3","4","5",">5"]
    stars_by_up=dfData.groupby('nb_mots_maj_comm2')['note'].mean().reindex(ordre)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ordre, y=stars_by_up, mode='lines',name="Nombre de mots en majuscule",marker_color="#7E18A2"))
    fig.update_layout(title="Note moyenne par nombre de mots en majuscule", xaxis_title='Nombre de mots en majuscule', yaxis_title="Note moyenne")
    st.plotly_chart(fig, use_container_width=True)
    st.write("La matrice de corrélation révèle une faible corrélation entre les variables explicatives, réduisant ainsi le risque de multicollinéarité. De même, les variables explicatives présentent une faible corrélation avec la note attribuée par l'utilisateur, ce qui suggère qu'elles ne seront probablement pas suffisantes pour expliquer la note. Par conséquent, il est nécessaire d'ajouter d'autres variables au modèle.")
    st.write("Il est important de noter que la variable 'nombre de signes de ponctuation standard' présente une forte corrélation avec le nombre de mots du commentaire, tandis que sa corrélation avec la variable cible est plus faible. Pour cette raison, cette variable sera exclue du modèle. ")
    cor = pd.read_csv(r"cor.csv")
     # Couleurs pour chaque valeur dans la matrice de confusion
    colors = [[0.0, '#F79999'], [0.1, '#E28F9F'], [0.2, '#CE86A4'],
            [0.3, '#B97CAA'], [0.4, '#A473AF '], [0.5, '#9069B5'],
            [0.6, '#7B5FBA'], [0.7, '#6656C0'], [0.8, '#514CC5'],
              [0.9, '#3D43CB '],[1.0, '#2839D0']]
    # Créer une figure plotly avec une carte de chaleur
    fig1 = go.Figure(data=go.Heatmap(z=cor, colorscale=colors,
                    hoverongaps=False,x=["Note","Année du commentaire","Nb signes expressifs","Nb signes standard","Points suspension","Nb mots","Nb mots majuscule"],
                                     y=["Note","Année du commentaire","Nb signes expressifs","Nb signes standard","Points suspension","Nb mots","Nb mots majuscule"]))
    for i in range(len(cor)):
        for j in range(len(cor.columns)):
            fig1.add_annotation(x=j, y=i, text=str(np.round(cor.values[i, j], 2)),
                                showarrow=False, font=dict(color='white'))
    # Personnaliser l'apparence de la figure
    fig1.update_layout(title='Matrice de corrélation')
    st.plotly_chart(fig1, use_container_width=True)
    st.write("Le diagramme en barres suivant permet d'observer les mots les plus fréquemment utilisés par les clients insatisfaits, ainsi que par les clients les plus satisfaits. ")
    df_good=pd.read_csv(r"good_words.csv")
    df_bad=pd.read_csv(r"bad_words.csv")
    col17, col27= st.columns(2)
    with col17:
        x_value=df_good.frequency[:10]
        y_index=df_good.word[:10]
        fig3 = go.Figure(data=go.Bar(name="Good words",x=x_value, y=y_index, marker_color="#54AC52",orientation='h'))
        # Définir le titre et les étiquettes des axes
        fig3.update_layout(title='Clients satisfaits', xaxis_title='Fréquence',
                           yaxis_title='Mots')
        st.plotly_chart(fig3, use_container_width=True)
    with col27:
        x_value=df_bad.frequency[:10]
        y_index=df_bad.word[:10]
        fig3 = go.Figure(data=go.Bar(name="Bad words",x=x_value, y=y_index, marker_color="#E61B1B",orientation='h'))
        # Définir le titre et les étiquettes des axes
        fig3.update_layout(title='Clients insatisfaits', xaxis_title='Fréquence',
                           yaxis_title='Mots')
        st.plotly_chart(fig3, use_container_width=True)
    st.write("Il est également possible de les représenter sous forme d'un nuage de mots. ")
    def color_func_red(word, font_size, position, orientation, random_state=None, **kwargs):
        colors = ["#E61B1B", "#EC5454", "#F6AAAA"]
        return random.choice(colors)
    def color_func_green(word, font_size, position, orientation, random_state=None, **kwargs):
        colors = ["#54AC52", "#84C483", "#B5DBB5"]
        return random.choice(colors)
    def generate_wordcloud(dataframe,r):
        # Création d'un dictionnaire de mots et de fréquences à partir du dataframe
        word_freq = dataframe.set_index('word')['frequency'].to_dict()
        # Création du WordCloud
        if r:
            wordcloud = WordCloud(width=800, height=400,margin=0, background_color='black',color_func=color_func_red).generate_from_frequencies(word_freq)
        else:
            wordcloud = WordCloud(width=800, height=400,margin=0, background_color='black',color_func=color_func_green).generate_from_frequencies(word_freq)
        # Affichage du WordCloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        return plt.gcf()
    col18, col28 = st.columns(2)
    with col18:
        st.markdown("<h5>Avis positifs</h5>",unsafe_allow_html=True)
        fig = generate_wordcloud(df_good,False)
        st.pyplot(fig)
    with col28:
        st.markdown("<h5>Avis négatifs</h5>",unsafe_allow_html=True)
        fig = generate_wordcloud(df_bad,True)
        st.pyplot(fig)
    st.write("Pour améliorer le modèle, il est encore plus intéressant d'observer les bigrammes les plus fréquents dans les avis, car ils permettent de capturer les associations de mots consécutifs.")
    st.write("En analysant les paires de mots qui apparaissent régulièrement ensemble, on peut comprendre les expressions et les nuances spécifiques utilisées par les clients pour exprimer leur niveau de satisfaction.")
    st.write("Les bigrammes fournissent des informations contextuelles plus riches et permettent une analyse plus approfondie des sentiments et des expériences des clients. ")
    col19, col29 = st.columns(2)
    with col19:
        x_value=[1515,1123,966,508,487,453,368,365,353,343]
        y_index=["('service', 'client')","('livraison', 'rapide')","('bon', 'site')","('qualité', 'prix')","('délai', 'livraison')","('bon', 'produit')","('bon', 'service')","('rapport', 'qualité')","('bon', 'qualité')","('site', 'sérieux')"]
        fig3 = go.Figure(data=go.Bar(name="Good bigrammes",x=x_value, y=y_index, marker_color="#54AC52",orientation='h'))
        # Définir le titre et les étiquettes des axes
        fig3.update_layout(title='Clients satisfaits', xaxis_title='Fréquence',
                           yaxis_title='Bigrammes')
        st.plotly_chart(fig3, use_container_width=True)
    with col29:
        x_value=[3393,384, 383,365,363, 357, 317,312,302,273]
        y_index=["('service', 'client')","('annuler', 'commande')","('recevoir', 'colis')","('recevoir', 'mail')","('contacter', 'service')","('colis', 'livrer')","('délai', 'livraison')","('commander', 'site')","('commander', 'article')","('recevoir', 'commande')"]
        fig3 = go.Figure(data=go.Bar(name="Bad bigrammes",x=x_value, y=y_index, marker_color="#E61B1B",orientation='h'))
        # Définir le titre et les étiquettes des axes
        fig3.update_layout(title='Clients insatisfaits', xaxis_title='Fréquence',
                           yaxis_title='Bigrammes')
        st.plotly_chart(fig3, use_container_width=True)
    df = pd.read_csv("NER_POS.csv")
    fig = go.Figure()
    keys=["SPACE", "X","VERB","SYM","SCONJ","PUNCT","PROPN","PRON","PART","NUM","NOUN","INTJ","DET","CCONJ","AUX","ADV","ADP","ADJ","PER","ORG","MISC","LOC"]
    selected_options = st.multiselect("Sélectionnez l'option (les options)", keys,['NOUN'])
    if len(selected_options)!=0:
        for i in selected_options: 
            fig.add_trace(go.Scatter(x=df.ordre, y=df[i],
                            mode='lines',
                            name=i))

        fig.update_layout(title="Note moyenne par entité nommée ou étiquette gramaticale", xaxis_title="Pourcentage entité nommée ou étiquette gramaticale",
                       yaxis_title="Note moyenne")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("<h3 class='center'>Résumé et interpretation</h3>",unsafe_allow_html=True)
    st.write("L'analyse exploratoire nous a permis de mieux comprendre les données et d'identifier les variables pouvant expliquer les facteurs déterminant la satisfaction des clients.")
    st.write("Certaines variables collectées et créées ne semblent pas avoir un impact direct sur la satisfaction des clients, donc elles ont été exclues de l'étape de modélisation. ")
    st.write("Les variables explicatives retenues sont les suivantes : ")
    st.markdown("""
        * Année de publication de l'avis 
        * Jour de la semaine de publication de l'avis 
        * Nombre de signes d'interrogation et de ponctuation 
        * Présence des points de suspension dans l'avis 
        * Nombre de mots dans l'avis 
        * Nombre de mots en majuscule dans l'avis 
        * Les bigrammes les plus fréquents dans les avis (positifs et négatifs, soit un total de 18) 
        * Les variables créées à partir de deux tâches de traitement automatique du langage naturel : NER (Reconnaissance d'Entités Nommées) et POS (Part-of-Speech, ou analyse morphosyntaxique). 
        """)
    st.write("Les variables mentionnées précédemment constituent une partie des éléments permettant d'expliquer la satisfaction du client à travers son avis. Toutefois, elles ne suffisent pas à elles seules, car l'analyse de l'avis recèle encore de nombreuses informations exploitables. Afin d'améliorer la modélisation, il est nécessaire d'explorer des techniques plus avancées en traitement automatique du langage naturel, lesquelles seront présentées dans l'étape de modélisation ultérieure. ")
elif page == "Modélisation":
    st.runtime.legacy_caching.clear_cache()
    subtab_h, subtab_mod,subtab_id = st.tabs(["**Accueil**","**Approche supervisée**","**Approche non supervisée**"])
    with subtab_h:
        st.markdown("<h3 class='center'>Stratégies de modélisation pour atteindre les objectifs du projet</h3>",unsafe_allow_html=True)
        st.write("Ce projet vise principalement à atteindre deux objectifs majeurs : tout d'abord, prédire le niveau de satisfaction des clients en les classant comme satisfaits ou insatisfaits grâce à une approche supervisée. Ensuite, il s'agit d'identifier les sujets abordés dans les commentaires sans avoir de pré-étiquetage spécifique, en utilisant une approche non supervisée.")
        st.write("Pour atteindre le premier objectif, plusieurs approches ont été envisagées :")
        st.markdown("""
            1. Une approche basée uniquement sur les variables collectées grâce au web scraping, ainsi que sur les variables créées à partir de ces données en utilisant des modèles de machine learning.
            2. Une approche axée exclusivement sur l'analyse des commentaires en utilisant des techniques couramment utilisées dans le traitement du langage naturel, telles que le count vectorizer et le TF-IDF. Ces techniques permettent de convertir les textes en données exploitables par des algorithmes d'apprentissage automatique et de deep learning comme les réseaux de neurones. 
            3. Une approche combinant les deux approches précédentes, en intégrant à la fois l'analyse des commentaires et les variables collectées et créées. 
            4. Une approche basée sur l'analyse des notes attribuées plutôt que sur la satisfaction ou l'insatisfaction du client.
            """)
        st.write("Pour atteindre le deuxième objectif, une approche non supervisée a été adoptée. Initialement, il a été essentiel de différencier les commentaires positifs des commentaires négatifs. Par la suite, il a été nécessaire de déterminer le nombre optimal de clusters et de sélectionner le modèle le plus performant pour prédire le cluster auquel chaque avis appartient. Ensuite, les commentaires ont été regroupés par cluster afin d'identifier les trigrammes les plus représentatifs, permettant ainsi de déduire le sujet commun associé à chaque cluster.")
    with subtab_mod:
        st.markdown("<h3 class='center'>Modélisation de la satisfaction du client : approche supervisée</h3>",unsafe_allow_html=True)
        st.markdown("<h5>Première approche : basée uniquement sur les variables collectées et créées </h5>",unsafe_allow_html=True)
        st.write("Le problème de modélisation vise à déterminer si la satisfaction du client peut être prédite à partir des variables recueillies avec l'avis ainsi que des variables créées ultérieurement. Ce problème est une tâche de classification binaire, où l'objectif est de classifier les clients en deux catégories distinctes : satisfaits et insatisfaits. ")
        st.write("La variable cible, appelée 'type', est définie comme suit :")
        st.markdown("""
        * **0**: Si la note est inférieure ou égale à 3 (clients insatisfaits) 
        * **1**: Si la note est supérieure ou égale à 4 (clients satisfaits)
        """)
        st.write("Les variables explicatives correspondent à celles sélectionnées lors de l'analyse exploratoire. ")
        st.write("Le graphique ci-dessous met en évidence le déséquilibre de la variable cible, avec un nombre plus élevé de clients satisfaits que d'insatisfaits. Pour remédier à cette situation,  la technique d'oversampling a été appliquée afin de prévenir la sous-représentation des clients insatisfaits. Cette approche permet de fournir un équilibre plus adéquat entre les deux classes dans l'ensemble de données. ")
        df = pd.read_csv(r"Train.csv")
        # Créer un nouveau DataFrame qui compte le nombre d'occurrences de chaque note
        type_counts = pd.DataFrame({'type': df['type'].value_counts(normalize=True) * 100})
        # Créer un diagramme à barres à partir des données du nouveau DataFrame
        fig1 = go.Figure(data=[go.Bar(x=type_counts.index, y=type_counts['type'], marker_color="#6d9e32",
                         text=type_counts['type'].round(2).astype(str) + '%', textposition='outside', textfont=dict(size=14))])
        # Définir le titre et les étiquettes des axes
        fig1.update_layout(title='Répartition des avis positifs et négatifs', xaxis_title='Client insatisfait (0) / client satisfait (1)',
                           yaxis_title='Pourcentage',yaxis_range=[0, max(type_counts['type']) + 5])
        st.plotly_chart(fig1, use_container_width=True)

        st.write("La modélisation a été effectuée en utilisant la technique de recherche sur grille (grid search) pour determiner les meilleurs hyperparamètres de chaque modèle évalué. Les modèles testés comprenaient la régression logistique, le réseau de neurones, la méthode des k plus proches voisins(KNN), les machines à vecteurs de support (SVM), les forêts aléatoires (random forest) et les arbres de décision.")
        st.write("Les résultats obtenus pour chaque modèle sont les suivants:")
        # Options
        options = ['Régression logistique', 'Réseau de neuronnes', 'KNN', 'SVM','Forêt aléatoire',"Arbre de décisions"]
        def Data(option):
            if option =='Régression logistique':
                report_data = {'Précision': [0.85, 0.80], 'Rappel': [0.78, 0.86], 'F1-score': [0.81, 0.83], 'Support': [5204, 5205]}
                best_score = 0.8144981634516892
                test_score = 0.8193870688826976
            if option =='Réseau de neuronnes':
                report_data = {'Précision': [0.84, 0.83], 'Rappel': [0.82, 0.84], 'F1-score': [0.83, 0.83], 'Support': [5204, 5205]}
                best_score = 0.8254989795126834
                test_score = 0.8313959073878374
            if option =='KNN':
                report_data = {'Précision': [0.79, 0.91], 'Rappel': [0.93, 0.75], 'F1-score': [0.85, 0.82], 'Support': [5204, 5205]}
                best_score = 0.8307593261630701
                test_score = 0.8386972811989625
            if option =='SVM':
                report_data = {'Précision': [0.84, 0.84], 'Rappel': [0.84, 0.84], 'F1-score': [0.84, 0.84], 'Support': [5204, 5205]}
                best_score = 0.8280450846222278
                test_score = 0.8377365741185513
            if option =='Forêt aléatoire':
                report_data = {'Précision': [0.87, 0.91], 'Rappel': [0.92, 0.87], 'F1-score': [0.89, 0.89], 'Support': [5204, 5205]}
                best_score = 0.8770928081657218
                test_score = 0.8914400999135363
            if option =="Arbre de décisions":
                report_data = {'Précision': [0.81, 0.86], 'Rappel': [0.87, 0.80], 'F1-score': [0.84, 0.83], 'Support': [5204, 5205]}
                best_score = 0.8141379156061561
                test_score = 0.8337976750888654
            classification_report = pd.DataFrame(report_data, index=[0, 1])
            st.write("Best Score:", best_score)
            st.write("Test Score:", test_score)
            st.write("Rapport de classification:")
            st.table(classification_report)
        # Sélection des options
        selected_options = st.multiselect("Sélectionnez l'option (les options)", options,['Régression logistique'])

        # Affichage du texte en fonction des options sélectionnées
        num_options = len(selected_options)
        if num_options == 1:
            st.write('Vous avez sélectionné l\'option suivante:')
            st.write('- ', selected_options[0])
            Data(selected_options[0])
        elif num_options >= 2:
            st.write('Vous avez sélectionné les options suivantes:')

            num_per_column = (num_options + 1) // 2

            col1,col2 = st.columns(2)
            for i in range(num_options):
                if i < num_per_column:
                    with col1:
                        st.write('- ', selected_options[i])
                        Data(selected_options[i])
                else:
                    with col2:
                        st.write('- ', selected_options[i])
                        Data(selected_options[i])
        st.write("Les résultats obtenus mettent en évidence que le modèle de forêts aléatoires affiche la plus haute performance avec 87.71% pour les données d'entraînement et 89.14% pour les données de test. De plus, le F1-score, qui offre une mesure équilibrée entre la précision et le rappel, présente également les meilleurs résultats avec 89% pour les deux classes(clients insatisfait, satisfaits).")
        st.write("Il est tout aussi pertinent de noter les variables ayant une influence significative sur les résultats de classification. On constate que le nombre de mots utilisés dans l'avis joue un rôle prépondérant dans la prédiction, de même que les variables issues des NER(Named Entity Recognition) et POS (Part-of-Speech). En revanche, le jour de la semaine où l'avis a été publié ainsi que les bigrammes semblent avoir une influence moindre sur la prédiction.")
        df = pd.read_csv(r"feature_importances.csv")
        fig1 = go.Figure(data=[go.Bar(x=df.Feature, y=df.Importance, marker_color="#E61B1B")])
        # Définir le titre et les étiquettes des axes
        fig1.update_layout(title='Variables plus significatives', xaxis_title='',yaxis_title='Importance')
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("<h5>Deuxieme approche : basée uniquement sur l'analyse des avis </h5>",unsafe_allow_html=True)
        st.write("Afin de permettre aux algorithmes de machine learning et de deep learning d'interpréter les commentaires, il est essentiel de les convertir en vecteurs numériques représentant les informations textuelles. Deux techniques de vectorisation de texte sont couramment utilisées dans l'analyse de commentaires : le TF-IDF(Term Frequency-Inverse Document Frequency) et le Count Vectorizer.")
        st.write("Le TF-IDF tient compte de la fréquence et de l'importance des mots dans un document et dans le corpus, tandis que le Count Vectorizer se contente de compter le nombre d'occurences des mots dans chaque document.")
        
        # Titre de l'application
        st.write("Calcul du TF-IDF et du Count Vectorizer")
        texte = """
        Au voleur ! au voleur ! à l'assassin ! au meurtrier ! Justice, juste Ciel ! je suis perdu, je suis assassiné, on m'a coupé la gorge, on m'a dérobé mon argent. Qui peut-ce être ? Qu'est-il devenu ? Où est-il ? Où se cache-t-il ? Que ferai-je pour le trouver? Où courir? Où ne pas courir? N'est-il point là ? N'est-il point ici ? Qui est-ce ? Arrête. Rends-moi mon argent, coquin. 
        """
        vecteurizer_choice = st.selectbox("Choisissez un vectorisateur", ("Count Vectorizer", "TF-IDF"))
        min_df = 1
        max_features = st.slider("Max_features (nombre maximal de mots)", 10, 200, 5)
        if vecteurizer_choice == "Count Vectorizer":
            vecteurizer = CountVectorizer(min_df=min_df, max_features=max_features)
        else:
            vecteurizer = TfidfVectorizer(min_df=min_df, max_features=max_features)
        vecteurs = vecteurizer.fit_transform([texte])
        st.write("Texte d'origine :")
        st.write(texte)
        colvoc,colvec=st.columns([1,3])
        with colvoc:
            st.write("Vocabulaire :", pd.DataFrame({"Mots":vecteurizer.get_feature_names_out()}))
        with colvec:
            st.write("Matrice de vecteurs :")
            st.write(vecteurs.toarray())
            if vecteurizer_choice != "Count Vectorizer":
                # Explication de TF-IDF
                st.subheader("TF-IDF (Term Frequency-Inverse Document Frequency)")
                
                st.latex(r'''TF-IDF(t, d, D) = TF(t, d) \times IDF(t, D)''')
                st.write("avec")
                st.markdown("**TF (Term Frequency)** mesure le nombre de fois qu'un terme `t` apparaît dans un document `d`. ")
                st.latex(r'''TF(t, d) = \frac{{\text{{nombre d'occurrences de }} t \text{{ dans }} d}}{{\text{{nombre total de termes dans }} d}}''')
                st.write('et')
                st.markdown("**IDF (Inverse Document Frequency)** mesure l'importance d'un terme `t` dans une collection de documents `D`. ")
                st.latex(r'''IDF(t, D) = \log\left(\frac{{\text{{nombre total de documents dans }} D}}{{\text{{nombre de documents contenant }} t}}\right)''')
        
        
        
        st.write(" ")
        st.write(" ")
        st.write("Les performances des différents modèles sont récapitulées dans le tableau ci-dessous:")
        report_data = {'Modèle':["Multinomial NB","Multinomial NB","Forêt aléatoire","Forêt aléatoire","K-means","K-means","Régression logistique","Régression logistique","XGBOOST","XGBOOST"],
                       'Type de vectorisation':["TF-IDF","Count Vectorizer","TF-IDF","Count Vectorizer","TF-IDF","Count Vectorizer","TF-IDF","Count Vectorizer","TF-IDF","Count Vectorizer"],
                       'Précision sur validation':["90.7%","90.3%","88.9%","89.2%","55.1%","70.2%","91.4%","90.9%","89.9%","90.6%"]}
        st.table(report_data)
        st.write("Le graphique ci-dessous permet de comparer visuellement les performances des différents modèles et d'identifier les modèles ayant les précisions les plus élevées. Chaque boîte représente la dispersion des précisions des différents plis de la validation croisée pour un modèle spécifique. Les lignes à l'intèrieur de chaque boîte représentent la médiane et les moustaches indiquent la plage des précisions.")
        TabColors=["#E42D95","#E47C2D","#E42D3A","#E4D82D"]
        res = [[0.89679072, 0.90191352, 0.89528401, 0.90387223, 0.90206419],
               [0.88744915, 0.89181859, 0.89001055, 0.89001055, 0.8958867],
               [0.90598162, 0.91230978, 0.90959771, 0.91246045, 0.9117071],
               [0.9014615, 0.90628296, 0.90221486, 0.90311888, 0.90537894]]
        name=["Multinomial NB","Forêt aléatoire","Régression logistique","XGBOOST"]
        traces = []
        for i, data in enumerate(res):
            traces.append(go.Box(y=data,name=name[i],jitter=0.3,boxmean=True))
        layout = go.Layout(title="Boîte à moustaches",yaxis=dict(title="Précision"))
        fig = go.Figure(data=traces, layout=layout)
        colbox1,colbox2,colbox3=st.columns(3)
        with colbox1:
            st.write(" ")
        with colbox2:
            st.plotly_chart(fig)
        with colbox3:
            st.write(" ")
        st.write("En analysant le tableau récapitulatif et le graphique de boîtes, il est evident que le modèle de régression logistique avec le type de vectorisation TF-IDF affiche les performances les plus élevées. Ces résultats concluants mettent en évidence la partinence de ce modèle pour notre analyse.")
        st.markdown("<h5>Analyse des résultats de la régression logistique</h5>",unsafe_allow_html=True)
        st.write("En analysant la matrice de confusion, nous pouvons constater que le nombre de faux positifs (échantillons prédits à tort comme \"satisfaits\" alors qu'ils sont réellement \"insatisfaits\") et de faux négatifs (échantillons prédits à tort comme \"insatisfaits\" alors qu'ils sont réellement \"satisfaits\") est relativement faible par rapport aux vrais positifs et aux vrais négatifs. Ces résultats expliquent pourquoi ce modèle a obtenu des bonnes perfomances en termes d'éxactitude.")
        st.write("De plus, le modèle a également démontré une précision élevée, indiquant qu'il a une faible tendance à prédire les échantillons comme \"satisfaits\" ou \"insatisfaits\", ainsi qu'un rappel élevé, indiquant qu'il a une faible tendance à manquer les échantillons \"satisfaits\" ou \"insatisfaits\".")
        df = pd.read_csv(r"result_RL.csv")
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import roc_curve
        # Obtenez la matrice de confusion
        cm = confusion_matrix(df["Real"], df["Predict"])
        fig = go.Figure(data=go.Heatmap(z=cm,
                               x=['Classe 0', 'Classe 1'],  # Remplacez par les noms des classes
                               y=['Classe 0', 'Classe 1'],  # Remplacez par les noms des classes
                               colorscale='Viridis'))
        # Ajoutez des annotations pour afficher les valeurs dans les cellules
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                fig.add_annotation(x=j, y=i, text=str(cm[i][j]),
                                   showarrow=False, font=dict(color='white'))

        # Ajoutez des titres et des étiquettes d'axes
        fig.update_layout(title="Matrice de confusion",
                          xaxis_title="Réalité",
                          yaxis_title="Prédiction")
        st.plotly_chart(fig, use_container_width=True)
        st.write("La courbe ROC qui se rapproche du coin supérieur gauche du graphique témoigne d'une capactité remarquable du modèle à distinguer avec précision les échantillons positifs des échantillons négatifs, avec un taux élevé de vrais positifs et un taux faible de faux positifs. L'AUC de 96% confirme et renforce cette conclusion, soulignant l'excellente performance du modèle.")
        fpr, tpr, _ = roc_curve(df["Real"], df["Predict_proba1"])
        diagonal = go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Diagonale')
        fig = go.Figure(data=[go.Scatter(x=fpr, y=tpr, mode='lines', name='Courbe ROC'), diagonal])
        fig.update_layout(
            title='Courbe ROC (Receiver Operating Characteristic)',
            xaxis=dict(title='Taux de faux positifs'),
            yaxis=dict(title='Taux de vrais positifs'),
        )
        
        colroc1,colroc2,colroc3=st.columns(3)
        with colroc1:
            st.write(" ")
        with colroc2:
            st.plotly_chart(fig)
        with colroc3:
            st.write(" ")
        st.write("Le graphique ci-dessous permet de visualiser la distribution des scores de probabilité pour chaque classe et d'évaluer la séparation entre les deux classes. Il nous aide a déterminer s'il y a un chevauchement entre les distributions des scores de probabilité des deux classes ou s'il existe une séparation claire entre elles.")
        st.write("Cette distinction claire indique une forte capacité de prédiction du modèle, car il est capable de différencier efficacement les échantillons positifs des échantillons négatifs.")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df["Predict_proba0"], histnorm='density', name='Classe 0'))
        fig.add_trace(go.Histogram(x=df["Predict_proba1"], histnorm='density', name='Classe 1'))
        fig.update_layout(
            title='Distribution des scores de probabilité pour chaque classe',
            xaxis_title='Score de probabilité',
            yaxis_title='Densité'
        )
        
        coldis1,coldis2,coldis3=st.columns(3)
        with coldis1:
            st.write(" ")
        with coldis2:
            st.plotly_chart(fig)
        with coldis3:
            st.write(" ")
            
            
        st.markdown("<h5>Interprétabilité du modèle de régression logistique</h5>",unsafe_allow_html=True)
        st.write("De la même manière que dans l'approche précédente, il est possible d'identifier les mots qui ont exercé une influence plus importante sur les prédictions.")
        col12,col22 =st.columns(2)
        st.write("Le graphique SHAP (SHapley Additive exPlanations) présenté ci-dessous offre une représentation concise de la contribution des caractéristiques dans le modèle de régression logistique. Il permet d'identifier les caractéristiques les plus influentes et d'observer leur impact sur les prédictions des classes positives et négatives. La variation de couleur des points reflète la valeur respective de chaque caracteristique, où le bleu indique une valeur faible et le rouge indique une valeur élevée. Cette visualisation facilite ainsi l'interpretation de l'importance relative des caractéristiques dans le modèle.")
      
        # Chargement des modèles
        RLsave = joblib.load("RL.sav")
        TRLsave = joblib.load("TRL.sav")
        def compute_shap_values(model, X):
            explainer = shap.Explainer(model, X,feature_names=TRLsave.get_feature_names_out())
            return explainer(X)
        df = pd.read_csv("Train.csv")
        df = df.dropna()
        df = df.sample(500)
        df.reset_index(inplace=True)
        y = df.type
        X = TRLsave.transform(df.commentaireAmeliore).toarray()

        X_display, y_display = X, y

        model = RLsave
        shap_values = compute_shap_values(model, X)
        col1sh,col2sh=st.columns(2)
        with col1sh:
            st_shap(shap.plots.waterfall(shap_values[0]), height=300)
        with col2sh:
            st_shap(shap.plots.beeswarm(shap_values), height=300)   
        # Création des dictionnaires de mots et de poids pour chaque classe
        
        word_dict_bad = dict(zip(TRLsave.get_feature_names_out(), -1 * (RLsave.coef_[0])))
        word_dict_good = dict(zip(TRLsave.get_feature_names_out(), RLsave.coef_[0]))

        def color_func_red(word, font_size, position, orientation, random_state=None, **kwargs):
            colors = ["#E61B1B", "#EC5454", "#F6AAAA"]
            return random.choice(colors)

        def color_func_green(word, font_size, position, orientation, random_state=None, **kwargs):
            colors = ["#54AC52", "#84C483", "#B5DBB5"]
            return random.choice(colors)

        def generate_wordcloud(data_dict, r):
            # Création du WordCloud à partir du dictionnaire de mots et de poids
            if r:
                wordcloud = WordCloud(width=800, height=400, margin=0, background_color='black', color_func=color_func_red).generate_from_frequencies(data_dict)
            else:
                wordcloud = WordCloud(width=800, height=400, margin=0, background_color='black', color_func=color_func_green).generate_from_frequencies(data_dict)
            # Affichage du WordCloud
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            return plt.gcf()
        
        with col22:
            # Création du nuage de mots pour les caractéristiques les plus importantes côté positif
            st.markdown("<h5>Mots positifs plus influents dans le modèle</h5>", unsafe_allow_html=True)
            fig3 = generate_wordcloud(word_dict_good, False)
            st.pyplot(fig3)
        with col12:
            # Création du nuage de mots pour les caractéristiques les plus importantes côté négatif
            st.markdown("<h5>Mots négatifs plus influents dans le modèle</h5>", unsafe_allow_html=True)
            fig2 = generate_wordcloud(word_dict_bad, True)
            st.pyplot(fig2)
        
        
        st.write("Le graphique ci-dessous présente l'influence de chaque caractéristique sur la prédiction du modèle pour un exemple spécifique. Les barres verticales indiquent l'importance de chaque caractéristique, avec du bleu pour une contribution négative et du rouge pour une contribution positive. La valeur de base du modèle est représentée par le point central, et les valeurs SHAP sont ajoutées ou soustraites pour obtenir la prédiction finale.")
        masker = shap.maskers.Independent(X)
        st.markdown("**Commentaire:**")
        st.write(df.commentaire[0])
        explainer = shap.LinearExplainer(model,masker,feature_names=TRLsave.get_feature_names_out())
        shap_values = explainer.shap_values(X)
        st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X_display[0],feature_names=TRLsave.get_feature_names_out()), height=200, width=1000)
        st.write("Il est également possible d'utiliser d'autres packages d'interprétabilité locale tels que ELI5(Explain Like I'm 5) pour comprendre pourquoi un avis spécifique a été classé comme positif ou négatif.")
        st.write("L'exemple ci-dessous présente les informations relatives à la contribution des caractéristiques spécifiques dans la prédiction d'un exemple particulier. Dans cet exemple, la valeur cible (y) est égale à 1 avec une probabilité de 0.999 et un score de 6,647.")
        st.write("La colonne \"contribution\" indique l'impact relatif de chaque caractéristique sur la prédiction finale. Une valeur positive indique une contribution positive, ce qui signifie que la présence ou l'occurence de cette caractéristique a eu un effet positif sur la prédiction de la classe cible. Une valeur négative indiquerait une contribution négative.")
        text="""
<style>
    table.eli5-weights tr:hover {
        filter: brightness(85%);
        color:#000000;
    }
    table.eli5-weights tbody{
        color:#000000;
    }
</style>
<p style="margin-bottom: 0.5em; margin-top: 0em">
    <b>
        y=1
    </b>
    (probability <b>0.999</b>, score <b>6.647</b>)
    top features
</p>
<table class="eli5-weights"
style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
<thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
            Contribution<sup>?</sup>
        </th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
</thead>
<tbody>
    <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.284
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
    </tr>
    <tr style="background-color: hsl(120, 100.00%, 85.51%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.810
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            top
        </td>
    </tr>
    <tr style="background-color: hsl(120, 100.00%, 87.02%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.693
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            facile
        </td>
    </tr>
    <tr style="background-color: hsl(120, 100.00%, 87.65%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.644
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            recommande
        </td>
    </tr>
    <tr style="background-color: hsl(120, 100.00%, 87.68%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.642
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            bémol
        </td>
    </tr>
    <tr style="background-color: hsl(120, 100.00%, 89.77%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.493
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            génial
        </td>
    </tr>
    <tr style="background-color: hsl(120, 100.00%, 89.95%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.480
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            agréable
        </td>
    </tr>
    <tr style="background-color: hsl(120, 100.00%, 91.04%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.407
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            trouver
        </td>
    </tr>
    <tr style="background-color: hsl(120, 100.00%, 91.88%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.354
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            site agréable
        </td>
    </tr>
    <tr style="background-color: hsl(120, 100.00%, 92.17%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.336
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            occasion
        </td>
    </tr>
    <tr style="background-color: hsl(120, 100.00%, 92.87%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.294
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            facile utiliser
        </td>
    </tr>
    <tr style="background-color: hsl(120, 100.00%, 94.32%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.212
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            visiter
        </td>
    </tr>
    <tr style="background-color: hsl(120, 100.00%, 96.14%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.122
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            cumuler
        </td>
    </tr>
    <tr style="background-color: hsl(120, 100.00%, 96.59%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.103
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            programme fidélité
        </td>
    </tr>
    <tr style="background-color: hsl(120, 100.00%, 97.16%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.079
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            utiliser
        </td>
    </tr>
    <tr style="background-color: hsl(120, 100.00%, 97.36%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.071
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            agréable visiter
        </td>
    </tr>
    <tr style="background-color: hsl(120, 100.00%, 97.44%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.068
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            fidélité
        </td>
    </tr>
    <tr style="background-color: hsl(120, 100.00%, 97.49%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.066
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            produit occasion
        </td>
    </tr>
    <tr style="background-color: hsl(120, 100.00%, 98.57%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.030
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            point
        </td>
    </tr>
    <tr style="background-color: hsl(120, 100.00%, 99.57%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.005
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            trouver produit
        </td>
    </tr>
    <tr style="background-color: hsl(120, 100.00%, 99.76%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +0.002
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            programme
        </td>
    </tr>
    <tr style="background-color: hsl(0, 100.00%, 97.43%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.069
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            produit
        </td>
    </tr>
    <tr style="background-color: hsl(0, 100.00%, 97.37%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.071
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            site
        </td>
    </tr>
    <tr style="background-color: hsl(0, 100.00%, 96.06%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.126
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            pouvoir trouver
        </td>
    </tr>
    <tr style="background-color: hsl(0, 100.00%, 95.79%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.139
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            essayer
        </td>
    </tr>
    <tr style="background-color: hsl(0, 100.00%, 95.63%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            -0.146
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            pouvoir
        </td>
    </tr>
</tbody>
</table>
<p style="margin-bottom: 0.5em; margin-top: 0em">
    <b>
        y=1
    </b>
    (probability <b>0.999</b>, score <b>6.647</b>)
    top features
</p>
<table class="eli5-weights"
style="border-collapse: collapse; border: none; margin-top: 0em; table-layout: auto; margin-bottom: 2em;">
<thead>
    <tr style="border: none;">
        <th style="padding: 0 1em 0 0.5em; text-align: right; border: none;" title="Feature contribution already accounts for the feature value (for linear models, contribution = weight * feature value), and the sum of feature contributions is equal to the score or, for some classifiers, to the probability. Feature values are shown if &quot;show_feature_values&quot; is True.">
            Contribution<sup>?</sup>
        </th>
        <th style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">Feature</th>
    </tr>
</thead>
<tbody>
    <tr style="background-color: hsl(120, 100.00%, 80.00%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +5.364
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            Highlighted in text (sum)
        </td>
    </tr>
    <tr style="background-color: hsl(120, 100.00%, 92.65%); border: none;">
        <td style="padding: 0 1em 0 0.5em; text-align: right; border: none;">
            +1.284
        </td>
        <td style="padding: 0 0.5em 0 0.5em; text-align: left; border: none;">
            &lt;BIAS&gt;
        </td>
    </tr> 
</tbody>
</table>
<p style="margin-bottom: 2.5em; margin-top:-0.5em; color:#000000;">
    <span style="background-color: hsl(120, 100.00%, 83.29%); opacity: 0.86" title="0.284">site</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 62.34%); opacity: 0.98" title="0.906">agréable</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.30%); opacity: 0.86" title="0.284">visiter</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 60.00%); opacity: 1.00" title="0.987">facile</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 79.74%); opacity: 0.88" title="0.374">utiliser</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 75.41%); opacity: 0.90" title="0.493">génial</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 86.95%); opacity: 0.84" title="-0.199">pouvoir</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 83.16%); opacity: 0.86" title="0.287">trouver</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 99.34%); opacity: 0.80" title="0.003">produit</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 78.66%); opacity: 0.88" title="0.402">occasion</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 96.57%); opacity: 0.81" title="0.030">point</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 65.16%); opacity: 0.96" title="0.810">top</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 93.53%); opacity: 0.81" title="-0.073">pouvoir</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 90.72%); opacity: 0.82" title="0.122">cumuler</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 91.66%); opacity: 0.82" title="0.105">programme</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 88.28%); opacity: 0.83" title="0.171">fidélité</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 70.32%); opacity: 0.93" title="0.644">recommande</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(0, 100.00%, 89.87%); opacity: 0.83" title="-0.139">essayer</span><span style="opacity: 0.80"> </span><span style="background-color: hsl(120, 100.00%, 70.38%); opacity: 0.93" title="0.642">bémol</span>
</p>
        """
        st.markdown(text, unsafe_allow_html=True)
        st.markdown("<h5>Et si on cherchait à améliorer les résultats avec un modèle de Deep Learning?</h5>", unsafe_allow_html=True)
        st.markdown("<h6>Modèle pré-entraîné : Distilcamembert Base Sentiment</h6>", unsafe_allow_html=True)
        st.markdown("Distilcamembert Base Sentiment (<a href=\"https://huggingface.co/cmarkea/distilcamembert-base-sentiment\">lien</a>) est le nom d'un modèle pré-entraîné pour l'analyse de sentiments en utilisant le langage naturel. Ce modèle est basé sur la variante \"DistilCamembert\" qui est une version compacte et légère du modèle \"Camembert\". Il a été entraîné sur une large quantité de données textuelles pour prédire les sentiments associés à des commentaires ou des phrases. En utilisant ce modèle pré-entraîné, il est possible de bénéficier des avantages de la représentation du langage apprise à partir de grandes quantités de données, sans avoir à entraîner un modèle à partir de zéro.", unsafe_allow_html=True)
        col1dis,col2dis=st.columns(2)
        with col1dis:
            # Obtenez la matrice de confusion
            cm=np.array([[2976, 116], [650, 4555]])
            fig = go.Figure(data=go.Heatmap(z=cm,
                                   x=['Classe 0', 'Classe 1'],  # Remplacez par les noms des classes
                                   y=['Classe 0', 'Classe 1'],  # Remplacez par les noms des classes
                                   colorscale='Viridis'))
            # Ajoutez des annotations pour afficher les valeurs dans les cellules
            for i in range(len(cm)):
                for j in range(len(cm[i])):
                    fig.add_annotation(x=j, y=i, text=str(cm[i][j]),
                                       showarrow=False, font=dict(color='white'))

            # Ajoutez des titres et des étiquettes d'axes
            fig.update_layout(title="Matrice de confusion",
                              xaxis_title="Réalité",
                              yaxis_title="Prédiction")
            st.plotly_chart(fig, use_container_width=True)
        with col2dis:
            report_data = {'Precision': [0.82, 0.98], 'Recall': [0.96, 0.85], 'F1-score': [0.89, 0.92], 'Support': [3092, 5205]}
            st.write("Accuracy:", 0.91)
            st.write("Rapport de classification:")
            st.table(report_data)
          
            
        st.markdown("<h6>Modèle de réseau de neurones : Multilayer Perceptron </h6>", unsafe_allow_html=True)
        st.write("L'utilisation d'un Multilayer Perceptron (MLP) dans l'analyse de texte offre plusieurs avantages. Il est capable de capturer des relations complexes et non linéaires dans le texte, permettant ainsi une compréhension plus précise de la sémantique et du sens des phrases. De plus, le MLP peut être entraîné sur de grandes quantités de données, ce qui lui permet d'acquérir une connaissance approfondie des modèles linguistiques et de généraliser efficacement à des nouveaux exemples. Sa flexibilité en termes de taille et de complexité du modèle permet de trouver un équilibre optimal entre la capacité à capturer des informations complexes et la généralisation.")
        report_data = {'Nombre d\'époques': [4, "En limitant le nombre d\'époques à 4, on prévient l'apparition du surapprentissage qui tend à se manifester au-delà de 4 ou 5 épochs."],
                       'Batchsize': [30, "Étant donné que le modèle est complexe et que le jeu de données est volumineux, il est essentiel de trouver un compromis optimal pour le batch size. Une valeur de 30 semble être un bon choix, car elle permet de capturer un échantillon représentatif de l'ensemble des données disponibles, sans être trop grand ni trop petit."],
                       'Optimiseur': ["Adam", "L'algorithme Adam est réputé pour faciliter l'ajustement des hyperparamètres et améliorer la précision des modèles MLP. Il s'agit d'un choix couramment utilisé dans la construction de ces modèles, pouvant être considéré comme une option fondamentale."],
                       'Fonction de perte': ["Binary Cross-Entropy", " Etant donné que le modèle doit prédire deux classes distinctes, il est logique d'opter pour une classification binaire. L'objectif est de pouvoir différencier efficacement les deux classes et réduire au maximum les erreurs de classification."],
                      "Métrique":["Accuracy","L'objectif principal est d'améliorer la précision globale (accuracy) quelle que soit la sélection des modèles. Cela est également indirectement lié à la fonction de perte utilisée. Par la suite,  les résultats obtenus pour d'autres mesures telles que le F1-score ou le rappel seront examinés et pris en compte pour évaluer les performances globales."],
                       "Dimension d'embedding":[16,"Une exploration empirique a permis d'approximer de manière fiable le meilleur paramètre sans compromettre les résultats, évitant ainsi une charge excessive en termes de temps et de mémoire."]}
        st.write("Le modèle MLP optimisé comprend les hyperparamètres suivants : ")
        st.table(report_data)
        st.write("Ce modèle utilise une couche d'embedding pour représenter les mots de manière dense, suivie d'une couche de pooling globale en moyenne pour réduire la dimensionnalité des données. Ensuite, une couche dense avec une activation ReLU est ajoutée pour capturer des relations complexes. Enfin, une dernière couche dense avec une activation sigmoïde est utilisée pour effectuer une classification binaire. Ce modèle permet de transformer les mots en vecteurs significatifs, de réduire la complexité du modèle, d'apprendre des représentations complexes et de prédire la classe d'un texte. ")
        def get_square_size(param_count):
            if param_count > 1000:
                return 150
            elif param_count > 20:
                return 100
            elif param_count > 10:
                return 60
            else:
                return 30

        def plot_model_architecture():
            fig = go.Figure()

            # Define the layers
            layers = [
                ("Embedding", 128000),
                ("GlobalAveragePooling1D", 16),
                ("Dense", 24),
                ("Dense", 1)
            ]

            # Add nodes for each layer
            for i, (layer_name, param_count) in enumerate(layers):
                square_size = get_square_size(param_count)

                # Add a node for the layer
                fig.add_trace(go.Scatter(
                    x=[i],
                    y=[1],
                    mode="markers",
                    name=layer_name,
                    marker=dict(size=[square_size], symbol='square'),
                    hovertext=f"Layer: {layer_name}<br>Params: {param_count}",
                ))

                # Add a connection between layers
                if i < len(layers) - 1:
                    fig.add_trace(go.Scatter(
                        x=[i, i+1],
                        y=[1, 1],
                        mode="lines",
                        line=dict(color="black", width=2),
                    ))

            # Set the layout
            fig.update_layout(
                title="Architecture du modèle",
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
                showlegend=False,
                height=400,
                width=800
            )

            # Display the figure
            st.plotly_chart(fig)

        


        plot_model_architecture()


        st.write("La matrice de confusion obtenue ainsi que le rapport de classification confirment la capacité du modèle à effectuer une classification précise des classes. ")
        col1disBIS,col2disBIS=st.columns(2)
        with col1disBIS:
            # Obtenez la matrice de confusion
            cm=np.array([[2681, 411], [288 , 4917]])
            fig = go.Figure(data=go.Heatmap(z=cm,
                                   x=['Classe 0', 'Classe 1'],  # Remplacez par les noms des classes
                                   y=['Classe 0', 'Classe 1'],  # Remplacez par les noms des classes
                                   colorscale='Viridis'))
            # Ajoutez des annotations pour afficher les valeurs dans les cellules
            for i in range(len(cm)):
                for j in range(len(cm[i])):
                    fig.add_annotation(x=j, y=i, text=str(cm[i][j]),
                                       showarrow=False, font=dict(color='white'))

            # Ajoutez des titres et des étiquettes d'axes
            fig.update_layout(title="Matrice de confusion",
                              xaxis_title="Réalité",
                              yaxis_title="Prédiction")
            st.plotly_chart(fig, use_container_width=True)
        with col2disBIS:
            report_data = {'Precision': [0.90, 0.92], 'Recall': [0.87, 0.94], 'F1-score': [0.88, 0.93], 'Support': [3092, 5205]}
            st.write("Accuracy:", 0.91)
            st.write("Rapport de classification:")
            st.table(report_data)
        st.markdown("<h5>Troisième approche: intégrant à la fois l'analyse des commentaires et les variables collectées et créées (Bagging) </h5>", unsafe_allow_html=True)
        st.write("Le modèle final choisi pour répondre à l'objectif de classification de la satisfaction client utilise la technique d'ensemble appelée bagging. Cette approche combine efficacement trois modèles distincts : un MultiLayer Perceptron (MLP), une régression logistique et une forêt aléatoire.  ")
        col15,col25=st.columns([5,1])
        with col15:
            st.write("Le bagging (Bootstrap Aggregating) est une technique d'ensemble utilisée en apprentissage automatique. Elle consiste à créer plusieurs échantillons d'entraînement à partir du jeu de données original en utilisant le bootstrap (échantillonnage avec remplacement). **Sur chaque échantillon, un modèle d'apprentissage est entraîné de manière indépendante. Les prédictions des modèles individuels sont ensuite combinées pour obtenir une prédiction finale.** Cela peut se faire en utilisant une règle de vote majoritaire pour les problèmes de classification. Le bagging vise à réduire la variance et le surapprentissage en favorisant la diversité des modèles et en combinant leurs prédictions. Cela permet d'obtenir une prédiction plus robuste et généralement de meilleurs résultats en comparaison à l'utilisation d'un seul modèle. ")
        with col25:
            st.image("unnamed.png",use_column_width=True)
        st.write("Les différentes métriques du modèle confirment qu'il s'agit du meilleur modèle obtenu à ce jour. L'utilisation du bagging a considérablement renforcé les prédictions par rapport aux résultats des modèles individuels. Toutefois, il est important de noter que le modèle rencontre certaines difficultés lorsqu'il s'agit de classer précisément les clients insatisfaits (classe 0), étant donné leur représentation moins fréquente dans les données collectées. ")
        report_data = {'Precision': [0.87, 0.94], 'Recall': [0.90, 0.93], 'F1-score': [0.89, 0.93], 'Support': [3001, 5296]}
        st.write("Accuracy:", 0.916)
        st.write("Rapport de classification:")
        st.table(report_data)
        st.markdown("<h5>Quatrième approche :  basée sur la prédiction des notes attribuées </h5>", unsafe_allow_html=True)
        st.write("Afin de prédire la note attribuée par les clients en fonction de leurs avis, un modèle de réseau de neurones artificiels a été entraîné. Pour ce faire, une étape préliminaire de traitement du texte a été effectuée, comprenant une tokenisation suivie d'un remplacement par des embeddings. ")
        st.write("Pour remédier au déséquilibre des classes, la technique de RandomOverSampler a été utilisée pour obtenir un nombre égal d'échantillons pour chaque note. Cette approche a été motivée par l'observation d'une préférence des utilisateurs pour des notes extrêmes plutôt que des notes intermédiaires. ")
        st.write("Le modèle lui-même est composé de plusieurs couches : ")
        st.markdown("""
        1. La couche d'embedding est utilisée pour transformer les séquences d'entiers en vecteurs d'embeddings.  
        2. La couche de pooling permet d'agréger les informations des embeddings.  
        3. La couche cachée est une couche dense avec une fonction d'activation relu.  
        4. La couche de dropout est utilisée pour régulariser le modèle et éviter le surapprentissage.  
        5. La couche de sortie est une couche dense avec une fonction d'activation softmax pour prédire les probabilités des différentes classes de notes. 
        """)
        st.write("Le modèle est compilé avec l'optimiseur Adam, la perte est définie comme la \"sparse_categorical_crossentropy\" (utilisée pour les problèmes de classification avec des étiquettes entières) et la métrique d'évaluation est la performance (accuracy). ")
        st.write("Le rapport de classification fournit les résultats suivants: ")
        report_data = {'Note':[1,2,3,4,5],'Precision': [0.95,0.96,0.89,0.64,0.78],
                       'Recall': [0.90,0.98,0.91,0.83,0.55],
                       'F1-score': [0.92,0.97,0.90,0.72,0.64],
                       'Support': [3988,4008,3945,4007,4101]}
        st.write("Accuracy:", 0.83)
        st.write("Rapport de classification:")
        st.table(report_data)
        st.write("Les résultats du modèle sont particulièrement médiocres pour la classe 5, ce qui peut être attribué au fait que les utilisateurs qui donnent cette note tendent à laisser des commentaires plus courts par rapport aux clients insatisfaits. Le prétraitement du texte, y compris la suppression des mots vides (stop words), peut pénaliser les commentaires courts, tels que ceux contenant simplement \"très bien\". Cette suppression peut entraîner une perte d'informations précieuses dans les commentaires courts et affecter négativement les performances du modèle pour la classe 5. ")
    with subtab_id :
        st.markdown("<h3 class='center'>Identification du sujet abordé dans les commentaires : approche non supervisée</h3>",unsafe_allow_html=True)
        st.write("Afin d'identifier les principaux sujets abordés dans les avis des utilisateurs, l'algorithme de clustering K-Means a été appliqué. La démarche du K-Means comprend plusieurs étapes, comme illustré dans la figure ci-dessous : ")
        st.image("schéma.png",use_column_width=True)
        st.write("Cette approche permet de détecter les principaux problèmes qui contribuent à l'insatisfaction des utilisateurs sur les quatre plateformes de commerce électronique étudiées. En regroupant les avis similaires en clusters, nous pouvons identifier les sujets récurrents et les problèmes les plus fréquemment mentionnés par les utilisateurs mécontents. Cela permettra aux entreprises d'identifier les domaines à améliorer et de prendre des mesures pour résoudre ces problèmes spécifiques, afin d'améliorer l'expérience utilisateur et la satisfaction globale.")
        st.write("Le tableau présenté ci-dessous résume les problèmes identifiés pour les utilisateurs mécontents, ainsi que les motifs de satisfaction pour les clients satisfaits :")
        report_data = {
            'Type d\'avis': ['Négatifs', 'Négatifs', 'Négatifs', 'Négatifs', 'Positifs', 'Positifs'],
            'Numéro de cluster': [0, 1, 2, 3, 0, 1],
            'Problème identifié': [
                "Contact difficile service client",
                "Mauvaise qualité service client",
                "Insatisfaction Service Après-Vente, Mauvaise qualité produit",
                "Délai de livraison, Point de livraison",
                "Rapidité service de livraison",
                "Bonne qualité service client, Bon rapport qualité prix"
            ]
        }

        df = pd.DataFrame(report_data)
        styled_df = df.style.apply(lambda x: ['background-color: lightcoral' if v == "Négatifs" else 'background-color: lightgreen' if v == "Positifs" else '' for v in x.values], axis=1)
        st.dataframe(styled_df)
elif page == "Chatbot":
    st.runtime.legacy_caching.clear_cache()
    st.markdown("<h4 class='center'>Cas pratique de l’analyse des verbatims : création d’un chatbot</h4>",unsafe_allow_html=True)
    st.write("L'objectif principal du chatbot est d'utiliser les résultats du modèle de bagging pour analyser la réponse du client, classifier son commentaire comme étant positif ou négatif et identifier le cluster auquel il appartient.")
    st.write("Par exemple, si le commentaire est identifié comme étant négatif et appartenant au cluster 0, le chatbot fournira une réponse spécifique préalablement préparée pour répondre au client de manière systématique.") 
    st.write("Veuillez cliquer sur le lien ci-dessous pour utiliser le chatbot:")
    st.markdown(
        """
        <style>
        .button-64 {
          align-items: center;
          background-image: linear-gradient(144deg,#FEFEFE, #E37878 50%,#F42D2D);
          border: 0;
          border-radius: 8px;
          box-shadow: rgba(151, 65, 252, 0.2) 0 15px 30px -5px;
          box-sizing: border-box;
          color: #FFFFFF;
          display: flex;
          font-family: Phantomsans, sans-serif;
          font-size: 20px;
          justify-content: center;
          line-height: 1em;
          max-width: 100%;
          min-width: 140px;
          padding: 3px;
          text-decoration: none;
          user-select: none;
          -webkit-user-select: none;
          touch-action: manipulation;
          white-space: nowrap;
          cursor: pointer;
        }
        .button-64:active,
        .button-64:hover {
          outline: 0;
        }
        .button-64 span {
          background-color: rgb(5, 6, 45);
          padding: 16px 24px;
          border-radius: 6px;
          width: 100%;
          height: 100%;
          transition: 300ms;
        }
        .button-64:hover span {
          background: none;
        }
        
         
        
        @media (min-width: 768px) {
          .button-64 {
            font-size: 24px;
            min-width: 196px;
          }
        }
        a { text-decoration: none; }
        a:hover { text-decoration: none; }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<a href='https://verbasatispyzer-botty.streamlit.app/'><button class='button-64 center' role='button'><span class='text'>Lien ChatBot</span></button></a>", unsafe_allow_html=True)

elif page == "Conclusion et Perspectives":
    st.runtime.legacy_caching.clear_cache()
    st.markdown("<h4 class='center'>Conclusion</h4>",unsafe_allow_html=True)
    col1,col2=st.columns([6,1])
    with col1:
        st.write("Dans le cadre de ce projet, deux approches d'analyse ont été utilisées pour atteindre les objectifs fixés : l'analyse quantitative et l'analyse qualitative. L'analyse quantitative a été employée pour classifier les clients en tant que satisfaits ou insatisfaits en utilisant la notation attribuée comme mesure de satisfaction quantitative. En revanche, l'analyse qualitative a permis d'identifier les principaux sujets abordés dans les verbatims, mettant en lumière les problèmes spécifiques mentionnés par les clients ainsi que les éléments contribuant à leur satisfaction.")
        st.write("Pour atteindre le premier objectif de classification de la satisfaction des clients, un modèle de bagging a été choisi, combinant à la fois les variables collectées via le web scraping et l'analyse des verbatims. ")
        st.write("Le deuxième objectif, qui visait à identifier les sujets principaux abordés par les utilisateurs, a été réalisé en utilisant un modèle de clustering k-means. Ce modèle a permis de regrouper les avis des clients en différents clusters, ce qui a facilité l'identification des sujets abordés dans chaque cluster. ")
        st.write("Enfin, pour mettre en pratique les deux modèles entraînés, un chatbot a été développé afin d'évaluer concrètement l'utilité de ces modèles. Le chatbot permet aux clients d'interagir en temps réel et d'utiliser les modèles d'analyse de la satisfaction client pour obtenir des réponses personnalisées. ")    
    with col2:
        st.image("chatbot.png",use_column_width=True)
    st.markdown("<h4 class='center'>Perspectives</h4>",unsafe_allow_html=True)
    st.write("Pour aller plus loin dans ce projet, plusieurs pistes d'amélioration peuvent être envisagées : ")
    st.markdown("""
    1. Améliorer l'identification des principaux sujets abordés par les clients. 
    2. Améliorer le modèle de prédiction de la note attribuée par le client. 
    3. Étendre le projet à plusieurs langues.
    4. Mettre en place un système de collecte des commentaires réalisés par les clients dans le chatbot pour constituer une base de données incluant les commentaires et les prédictions de celui-ci, permettant ainsi d'améliorer progressivement le modèle et les réponses fournies aux clients.
    """)
    st.write("En poursuivant ces pistes d'amélioration, le projet pourrait aboutir à des résultats plus précis et pertinents, offrant ainsi une meilleure compréhension de la satisfaction client dans différentes langues et permettant de prendre des décisions plus éclairées pour améliorer l'expérience client.")

    
    
    
    
    
    
    
    
    
    
    
    
    
