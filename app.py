# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OrdinalEncoder

# Load models and encoders
ord_enc = joblib.load('ordinal_encoder.pkl')
tabpfn = joblib.load('tabpfn_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Function to add engineered features (copy from notebook)
def add_engineered_features(df):
    df_new = df.copy()
    
    # 1. Create age numeric feature
    age_map = {
        'Age 18 to 24': 21, 'Age 25 to 29': 27, 'Age 30 to 34': 32,
        'Age 35 to 39': 37, 'Age 40 to 44': 42, 'Age 45 to 49': 47,
        'Age 50 to 54': 52, 'Age 55 to 59': 57, 'Age 60 to 64': 62,
        'Age 65 to 69': 67, 'Age 70 to 74': 72, 'Age 75 to 79': 77,
        'Age 80 or older': 85
    }
    df_new['Age_Numeric'] = df_new['AgeCategory'].map(age_map)
    
    # 2. Create health conditions count
    health_conditions = ['HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD', 
                         'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis', 'HadDiabetes']
    df_new['HealthConditionsCount'] = df_new[health_conditions].apply(
        lambda x: (x == 'Yes').sum(), axis=1
    )
    
    # 3. Create disability count
    disabilities = ['DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 'DifficultyConcentrating',
                    'DifficultyWalking', 'DifficultyDressingBathing', 'DifficultyErrands']
    df_new['DisabilitiesCount'] = df_new[disabilities].apply(
        lambda x: (x == 'Yes').sum(), axis=1
    )
    
    # 4. Create BMI category
    def categorize_bmi(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif bmi < 25:
            return 'Normal'
        elif bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'
    
    df_new['BMI_Category'] = df_new['BMI'].apply(categorize_bmi)
    
    # 5. Create health days ratio
    df_new['HealthDaysRatio'] = df_new['PhysicalHealthDays'] / (df_new['MentalHealthDays'] + 1)
    
    # 6. Create vaccination count
    vaccinations = ['FluVaxLast12', 'PneumoVaxEver']
    df_new['VaccinationCount'] = df_new[vaccinations].apply(
        lambda x: (x == 'Yes').sum(), axis=1
    )
    
    # 7. Create binary features for smoker status
    df_new['IsSmoker'] = df_new['SmokerStatus'].apply(
        lambda x: 1 if x in ['Current smoker - now smokes every day', 'Current smoker - now smokes some days'] else 0
    )
    
    df_new['IsFormerSmoker'] = df_new['SmokerStatus'].apply(
        lambda x: 1 if x == 'Former smoker' else 0
    )
    
    # 8. Create binary feature for e-cigarette usage
    df_new['UsesECigarettes'] = df_new['ECigaretteUsage'].apply(
        lambda x: 1 if x in ['Use them every day', 'Use them some days'] else 0
    )
    
    return df_new

# Load preprocessor and model
preprocessor = joblib.load('preprocessor_engineered.pkl')
tabpfn = joblib.load('tabpfn_model.pkl')

st.title("Heart Attack Prediction Dashboard")
st.markdown("""
Ce dashboard permet de :
- Explorer les données et leur distribution
- Tester le moteur de prédiction sur des profils personnalisés
- Comparer la performance de plusieurs modèles récents
""")

# --- Analyse exploratoire interactive ---
st.header("Analyse exploratoire interactive")
df = pd.read_csv('data/heart_2022_no_nans.csv')

# Get all unique values for categorical columns (for dropdowns)
# You'll need this for creating complete user profiles

# Graphique interactif 1
age_dist = st.selectbox("Sélectionner une variable :", ['AgeCategory', 'Sex', 'BMI'])
st.bar_chart(df[age_dist].value_counts())

# Graphique interactif 2
if st.checkbox("Afficher la prévalence des Heart Attacks par sexe"):
    st.bar_chart(df.groupby('Sex')['HadHeartAttack'].value_counts(normalize=True).unstack()['Yes'])

# --- Saisie du profil utilisateur ---
st.header("Tester la prédiction sur un profil")

# Create input fields for all necessary features
col1, col2, col3 = st.columns(3)

with col1:
    age = st.selectbox('Âge', df['AgeCategory'].unique())
    sex = st.selectbox('Sexe', df['Sex'].unique())
    race = st.selectbox('Race/Ethnicité', df['RaceEthnicityCategory'].unique())
    bmi = st.slider('BMI', float(df['BMI'].min()), float(df['BMI'].max()), 25.0)
    height = st.slider('Taille (m)', 1.4, 2.2, 1.7)
    weight = st.slider('Poids (kg)', 40.0, 200.0, 70.0)

with col2:
    general_health = st.selectbox('Santé générale', df['GeneralHealth'].unique())
    smoker = st.selectbox('Statut tabagique', df['SmokerStatus'].unique())
    ecig = st.selectbox('E-cigarette', df['ECigaretteUsage'].unique())
    alcohol = st.selectbox('Consommation alcool', ['Yes', 'No'])
    physical_activities = st.selectbox('Activités physiques', ['Yes', 'No'])
    sleep_hours = st.slider('Heures de sommeil', 1, 24, 7)

with col3:
    # Health conditions
    st.subheader("Conditions médicales")
    had_angina = st.checkbox('Angine')
    had_stroke = st.checkbox('AVC')
    had_asthma = st.checkbox('Asthme')
    had_skin_cancer = st.checkbox('Cancer de la peau')
    had_copd = st.checkbox('BPCO')
    had_depression = st.checkbox('Dépression')
    had_kidney_disease = st.checkbox('Maladie rénale')
    had_arthritis = st.checkbox('Arthrite')
    had_diabetes = st.checkbox('Diabète')

# Additional inputs in expander
# In the expander section, add this input:
with st.expander("Paramètres additionnels"):
    col4, col5 = st.columns(2)
    
    with col4:
        physical_health_days = st.slider('Jours de mauvaise santé physique (30 derniers jours)', 0, 30, 0)
        mental_health_days = st.slider('Jours de mauvaise santé mentale (30 derniers jours)', 0, 30, 0)
        checkup_time = st.selectbox('Dernier checkup', df['LastCheckupTime'].unique())
        removed_teeth = st.selectbox('Dents extraites', df['RemovedTeeth'].unique())
        # Add this line:
        high_risk_last_year = st.selectbox('Comportement à risque élevé (12 derniers mois)', ['Yes', 'No'])

        
    with col5:
        # Disabilities
        deaf = st.checkbox('Sourd ou malentendant')
        blind = st.checkbox('Aveugle ou déficient visuel')
        diff_concentrating = st.checkbox('Difficulté à se concentrer')
        diff_walking = st.checkbox('Difficulté à marcher')
        diff_dressing = st.checkbox('Difficulté à s\'habiller/se laver')
        diff_errands = st.checkbox('Difficulté pour les courses')
        
        # Vaccinations
        flu_vax = st.checkbox('Vaccin grippe (12 derniers mois)')
        pneumo_vax = st.checkbox('Vaccin pneumonie')
        covid_pos = st.selectbox('COVID positif', df['CovidPos'].unique())
        
        # Other medical tests
        hiv_testing = st.selectbox('Test VIH', ['Yes', 'No'])
        chest_scan = st.selectbox('Radio thorax', ['Yes', 'No'])
        tetanus = st.selectbox('Vaccin tétanos', df['TetanusLast10Tdap'].unique())

# Create complete user DataFrame with all required columns
user_df = pd.DataFrame({
    'AgeCategory': [age],
    'Sex': [sex],
    'RaceEthnicityCategory': [race],
    'BMI': [bmi],
    'HeightInMeters': [height],
    'WeightInKilograms': [weight],
    'GeneralHealth': [general_health],
    'SmokerStatus': [smoker],
    'ECigaretteUsage': [ecig],
    'AlcoholDrinkers': [alcohol],
    'PhysicalActivities': [physical_activities],
    'SleepHours': [sleep_hours],
    'PhysicalHealthDays': [physical_health_days],
    'MentalHealthDays': [mental_health_days],
    'LastCheckupTime': [checkup_time],
    'RemovedTeeth': [removed_teeth],
    
    # Health conditions
    'HadAngina': ['Yes' if had_angina else 'No'],
    'HadStroke': ['Yes' if had_stroke else 'No'],
    'HadAsthma': ['Yes' if had_asthma else 'No'],
    'HadSkinCancer': ['Yes' if had_skin_cancer else 'No'],
    'HadCOPD': ['Yes' if had_copd else 'No'],
    'HadDepressiveDisorder': ['Yes' if had_depression else 'No'],
    'HadKidneyDisease': ['Yes' if had_kidney_disease else 'No'],
    'HadArthritis': ['Yes' if had_arthritis else 'No'],
    'HadDiabetes': ['Yes' if had_diabetes else 'No'],
    
    # Disabilities
    'DeafOrHardOfHearing': ['Yes' if deaf else 'No'],
    'BlindOrVisionDifficulty': ['Yes' if blind else 'No'],
    'DifficultyConcentrating': ['Yes' if diff_concentrating else 'No'],
    'DifficultyWalking': ['Yes' if diff_walking else 'No'],
    'DifficultyDressingBathing': ['Yes' if diff_dressing else 'No'],
    'DifficultyErrands': ['Yes' if diff_errands else 'No'],
    
    # Vaccinations and tests
    'FluVaxLast12': ['Yes' if flu_vax else 'No'],
    'PneumoVaxEver': ['Yes' if pneumo_vax else 'No'],
    'CovidPos': [covid_pos],
    'HIVTesting': [hiv_testing],
    'ChestScan': [chest_scan],
    'TetanusLast10Tdap': [tetanus],
    'HighRiskLastYear': [high_risk_last_year],
    
    # Additional columns that might be in your dataset
    # Add any other columns with default values as needed
})

# Apply feature engineering
user_df_engineered = add_engineered_features(user_df)

# Load feature columns order
feature_columns = joblib.load('feature_columns.pkl')

# Ensure user_df_engineered has the same columns in the same order
user_df_engineered = user_df_engineered[feature_columns]


# --- Prédiction ---
if st.button("Prédire le risque de Heart Attack"):
    try:
        # Apply feature engineering
        user_df_engineered = add_engineered_features(user_df)
        
        # Ensure correct column order
        user_df_engineered = user_df_engineered[feature_columns]
        
        # Transform with OrdinalEncoder
        user_encoded = ord_enc.transform(user_df_engineered)
        
        # Make prediction
        pred = tabpfn.predict(user_encoded)[0]
        proba = tabpfn.predict_proba(user_encoded)[0][1]
        
        # Display result
        if pred == 1:
            st.error(f"⚠️ Risque élevé de crise cardiaque détecté!")
        else:
            st.success(f"✅ Risque faible de crise cardiaque")
            
        st.metric("Probabilité de risque", f"{proba:.1%}")
        
        # Show confidence level
        confidence_level = abs(proba - 0.5) * 2
        st.progress(confidence_level)
        st.caption(f"Niveau de confiance: {confidence_level:.0%}")
        
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        st.info("Vérifiez que tous les champs sont correctement remplis.")
