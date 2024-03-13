import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px
import seaborn as sns

# Charger votre ensemble de données (remplacez cela par le chargement de vos données réelles)
data = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
data.drop(columns=['ID', 'ZIP Code'], inplace=True)

# Diviser les données en caractéristiques (X) et la variable cible (y)
X = data.drop('Personal Loan', axis=1)
y = data['Personal Loan']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle de Random Forest (remplacez cela par votre modèle)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Fonction pour prédire si un client souscrira ou non à un prêt personnel
def predict_loan_approval(features):
    # Prétraiter les données saisies par l'utilisateur (adapté en fonction de vos données)
    input_data = pd.DataFrame([features], columns=X.columns)  # X est le DataFrame utilisé pour l'entraînement
    input_data['Personal Loan'] = 0  # La colonne 'Personal Loan' est souvent celle que vous souhaitez prédire
    
    # Faire la prédiction
    prediction = model.predict(input_data.drop('Personal Loan', axis=1))  # Ne pas inclure la colonne 'Personal Loan' dans les fonctionnalités

    return prediction[0]


# Configuration de la page
st.set_page_config(page_title="Application Streamlit", layout="wide")

# Sidebar pour la navigation entre les pages
page = st.sidebar.selectbox("Navigation", ["Accueil", "Visualisation des distributions", "Visualisation des relations"])

# Page d'accueil
if page == "Accueil":
    st.title('Prédiction d\'approbation de prêt personnel')
    st.subheader('Paramètres utilisateur')
    st.markdown("Réalisé par Olive")

    # Créer des champs de formulaire pour saisir les informations utilisateur
    age = st.sidebar.slider('Âge', min_value=18, max_value=100, value=25)
    experience = st.sidebar.slider('Expérience', min_value=0, max_value=50, value=5)
    income = st.sidebar.slider('Revenu annuel', min_value=0, max_value=200, value=50)
    family = st.sidebar.slider('Nombre de membres de la famille', min_value=1, max_value=10, value=4)
    ccavg = st.sidebar.slider('Dépenses mensuelles par carte de crédit (CCAvg)', min_value=0.0, max_value=10.0, value=1.0)
    education = st.sidebar.selectbox('Niveau d\'éducation', options=[1, 2, 3], index=0)
    mortgage = st.sidebar.slider('Montant du prêt hypothécaire', min_value=0, max_value=500, value=0)
    securities_account = st.sidebar.checkbox('Possède un compte de titres')
    cd_account = st.sidebar.checkbox('Possède un compte à terme')
    online = st.sidebar.checkbox('Utilise les services en ligne')
    credit_card = st.sidebar.checkbox('Possède une carte de crédit')

    # ... le reste du code pour la page d'accueil ...
# ... (le code précédent)

    # ... le reste du code pour la page d'accueil ...
    # Bouton pour faire la prédiction
    # Modifier cette partie de votre code
    if st.sidebar.button('Prédire'):
        # Organisez les données sous forme de liste pour la prédiction
        user_input = [age, experience, income, family, ccavg, education, mortgage, securities_account, cd_account, online, credit_card]
        
        # Prédire avec le modèle
        prediction = predict_loan_approval(user_input)

        # Afficher le résultat de la prédiction
        st.write('### Résultat de la prédiction')
        st.write('Le client', 'souscrira à un prêt personnel.' if prediction == 1 else 'ne souscrira pas à un prêt personnel.')

    # Informations supplémentaires (vous pouvez personnaliser cela en fonction de vos besoins)
    st.sidebar.markdown('---')
    st.sidebar.header('À propos de ce modèle')
    st.sidebar.info('Ce modèle de prédiction d\'approbation de prêt personnel utilise un algorithme de Random Forest.')
    
    # Option pour afficher les données (à des fins de débogage)
    if st.checkbox('Afficher les données'):
        st.write('### Données saisies pour la prédiction')
        st.write(pd.DataFrame({
            'Âge': [age],
            'Expérience': [experience],
            'Revenu annuel': [income],
            'Nombre de membres de la famille': [family],
            'CCAvg': [ccavg],
            'Niveau d\'éducation': [education],
            'Montant du prêt hypothécaire': [mortgage],
            'Possède un compte de titres': [securities_account],
            'Possède un compte à terme': [cd_account],
            'Utilise les services en ligne': [online],
            'Possède une carte de crédit': [credit_card]}))

# Page de visualisation des distributions
elif page == "Visualisation des distributions":
    st.title('Visualisation des distributions  des variables')

    # ... (le reste du code pour la visualisation des distributions) ...
    # Sélection du type de graphique
    chart_type = st.selectbox('Sélectionnez le type de graphique', ['Countplot', 'Boxplot', 'Violinplot', 'Scatterplot', 'Pairplot', 'Histogram', 'Pie Chart', 'Bar Chart', 'Line Chart', 'Area Chart'])

    # Sélection des variables
    selected_variable = st.selectbox('Sélectionnez la variable à visualiser', data.columns)

    # Bouton pour afficher le graphique
    if st.button('Afficher le graphique'):
        try:
            # Filtrer le DataFrame avec la variable sélectionnée
            data_for_plot = data[[selected_variable, 'Personal Loan']]
            # Afficher le graphique en fonction du type sélectionné
            if chart_type == 'Countplot':
                fig = px.histogram(data_for_plot, x=selected_variable, color='Personal Loan', barmode='overlay')
                st.plotly_chart(fig)
            
            elif chart_type == 'Boxplot':
                fig = px.box(data_for_plot, x='Personal Loan', y=selected_variable, points="all")
                st.plotly_chart(fig)

            elif chart_type == 'Violinplot':
                fig = px.violin(data_for_plot, x='Personal Loan', y=selected_variable, box=True, points="all")
                st.plotly_chart(fig)

            elif chart_type == 'Scatterplot':
                fig = px.scatter(data_for_plot, x=selected_variable, color='Personal Loan')
                st.plotly_chart(fig)

            elif chart_type == 'Pairplot':
                if data_for_plot[selected_variable].dtype.name == 'category':
                    pair_plot = sns.pairplot(data_for_plot, hue='Personal Loan')
                st.pyplot(pair_plot.fig)

            elif chart_type == 'Histogram':
                fig = px.histogram(data_for_plot, x=selected_variable, color='Personal Loan', barmode='overlay')
                st.plotly_chart(fig)

            elif chart_type == 'Pie Chart':
                fig = px.pie(data_for_plot, names=selected_variable, title=f'Distribution de {selected_variable}', hole=0.3)
                st.plotly_chart(fig)

            elif chart_type == 'Bar Chart':
               fig = px.bar(data_for_plot, x=selected_variable, color='Personal Loan')
               st.plotly_chart(fig)

            elif chart_type == 'Line Chart':
               fig = px.line(data_for_plot, x=data_for_plot.index, y=selected_variable, color='Personal Loan')
               st.plotly_chart(fig)

            elif chart_type == 'Area Chart':
              fig = px.area(data_for_plot, x=data_for_plot.index, y=selected_variable, color='Personal Loan')
              st.plotly_chart(fig)

        except Exception as e:
          st.error(f"Une erreur s'est produite : {str(e)}")
            

# Page de visualisation des relations entre les variables
elif page == "Visualisation des relations":
    st.title('Visualisation des relations entre les variables')

    # ... (le reste du code pour la visualisation des relations) ...
    # Convertir certaines colonnes en variables catégorielles
    categorical_columns = ['Education', 'Securities Account', 'CD Account', 'Online', 'CreditCard']
    data[categorical_columns] = data[categorical_columns].astype('category')

    # Sélection des variables pour le graphique
    variable_x = st.selectbox('Sélectionnez la variable pour l\'axe X', data.columns[:-1])  # Exclure 'Personal Loan'
    variable_y = st.selectbox('Sélectionnez la variable pour l\'axe Y', data.columns[:-1])  # Exclure 'Personal Loan'

    # Sélection du type de graphique
    graph_type = st.selectbox('Sélectionnez le type de graphique', ['Scatter Plot', 'Box Plot', 'Violin Plot', 'Count Plot', 'Pie Chart'])

    # Bouton pour afficher les graphiques
    retry_limit = 3
    retry_count = 0
    while retry_count < retry_limit:
        try:
            if st.button('Afficher les graphiques'):
                # Filtrer le DataFrame avec les variables sélectionnées
                if graph_type in ['Violin Plot', 'Count Plot']:
                    data_for_plot = data[[variable_x, variable_y, 'Personal Loan']]
                else:
                    data_for_plot = data[[variable_x, variable_y]]

                # ... (ajoutez le code pour les différents types de graphiques) ...
                    # Scatter Plot
            if graph_type == 'Scatter Plot':
                fig = px.scatter(data_for_plot, x=variable_x, y=variable_y)
                st.plotly_chart(fig)

            # Box Plot
            elif graph_type == 'Box Plot':
                fig = px.box(data_for_plot, x=variable_x, y=variable_y, points="all")
                st.plotly_chart(fig)

            # Violin Plot
            elif graph_type == 'Violin Plot':
                fig = px.violin(data_for_plot, x=variable_x, y=variable_y, box=True, points="all", color='Personal Loan')
                st.plotly_chart(fig)

            # Count Plot
            elif graph_type == 'Count Plot':
                fig = px.histogram(data_for_plot, x=variable_x, color='Personal Loan', barmode='overlay')
                st.plotly_chart(fig)

            # Pie Chart
            elif graph_type == 'Pie Chart':
                fig = px.pie(data_for_plot, names=variable_x, title=f'Distribution de {variable_x}', hole=0.3, color='Personal Loan')
                st.plotly_chart(fig)


            # Si tout s'est bien déroulé, sortir de la boucle
            break

        except Exception as e:
            retry_count += 1
            st.warning(f"Une erreur s'est produite : {str(e)}")
            if retry_count < retry_limit:
                retry_button = st.button("Réessayer")
                if not retry_button:
                    break  # Si l'utilisateur n'a pas cliqué sur "Réessayer", sortir de la boucle
            else:
                st.error("Trop d'erreurs. Veuillez réessayer plus tard.")
                break
