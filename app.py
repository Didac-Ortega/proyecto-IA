import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier # Cambiamos a Random Forest
import numpy as np

st.set_page_config(page_title="IA Rendimiento", page_icon="🎓")
st.title("🎓 IA de Predicción de Rendimiento")

@st.cache_resource
def get_trained_model():
    d = pd.read_csv('student-por.csv', sep=';')
    
    # Umbral de aprobado: 30 puntos (la mitad)
    d['pass'] = d.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3']) >= 30 else 0, axis=1)
    
    y = d['pass']
    X = d.drop(['G1', 'G2', 'G3', 'pass'], axis=1)
    X = pd.get_dummies(X)
    
    # Usamos Random Forest: 100 árboles trabajando juntos
    # Esto es MUCHO más robusto que un solo árbol
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X, y)
    
    return rf, X.columns.tolist()

try:
    model, model_columns = get_trained_model()
    st.sidebar.success("✅ IA Robusta Lista")
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# Interfaz
col1, col2 = st.columns(2)
with col1:
    failures = st.number_input("Fracasos anteriores", 0, 4, 0)
    absences = st.slider("Ausencias", 0, 93, 5)
    studytime = st.slider("Tiempo estudio", 1, 4, 2)
with col2:
    Medu = st.selectbox("Edu. Madre", [0, 1, 2, 3, 4])
    Fedu = st.selectbox("Edu. Padre", [0, 1, 2, 3, 4])

if st.button("Analizar Estudiante"):
    entrada_dict = {col: [0] for col in model_columns}
    entrada_dict['failures'] = [failures]
    entrada_dict['absences'] = [absences]
    entrada_dict['studytime'] = [studytime]
    entrada_dict['Medu'] = [Medu]
    entrada_dict['Fedu'] = [Fedu]
    
    df_input = pd.DataFrame(entrada_dict)
    
    # Predicción y Probabilidad
    prediccion = model.predict(df_input)
    prob = model.predict_proba(df_input)
    
    st.markdown("---")
    # Si tiene muchísimas ausencias, forzamos un mensaje de alerta manual (lógica de negocio)
    if absences > 40:
        st.warning("⚠️ Alerta: El número de ausencias es crítico para la evaluación.")

    if prediccion[0] == 1:
        st.balloons()
        st.success(f"### 🎉 APROBADO (Confianza: {prob[0][1]*100:.1f}%)")
    else:
        st.error(f"### 📉 RIESGO DE SUSPENSO (Confianza: {prob[0][0]*100:.1f}%)")
