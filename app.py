import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.set_page_config(page_title="IA Rendimiento", page_icon="🎓")
st.title("🎓 IA de Predicción de Rendimiento")

@st.cache_resource
def get_trained_model():
    d = pd.read_csv('student-por.csv', sep=';')
    
    # 1. UMBRAL DE EXIGENCIA ALTO
    # Subimos a 35 puntos para que el "Aprobado" sea difícil de conseguir
    d['pass'] = d.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3']) >= 35 else 0, axis=1)
    
    y = d['pass']
    X = d.drop(['G1', 'G2', 'G3', 'pass'], axis=1)
    X = pd.get_dummies(X)
    
    # 2. RANDOM FOREST MUCHO MÁS PROFUNDO Y ESTRICTO
    # Aumentamos n_estimators y max_depth para que detecte las ausencias
    rf = RandomForestClassifier(
        n_estimators=200, 
        max_depth=15, 
        class_weight={0: 2.0, 1: 1.0}, # Penalizamos el doble fallar un suspenso
        random_state=42
    )
    rf.fit(X, y)
    
    return rf, X.columns.tolist()

try:
    model, model_columns = get_trained_model()
    st.sidebar.success("✅ IA Estricta Configurada")
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# Interfaz
col1, col2 = st.columns(2)
with col1:
    failures = st.number_input("Fracasos anteriores", 0, 4, 0)
    absences = st.slider("Ausencias", 0, 93, 5)
    studytime = st.slider("Tiempo estudio (1-4)", 1, 4, 2)
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
    
    # Predicción
    prediccion = model.predict(df_input)
    prob = model.predict_proba(df_input)
    
    st.markdown("---")
    
    # LÓGICA DE SEGURIDAD (Si la IA duda o hay demasiadas ausencias)
    # Si tiene más de 30 ausencias, el riesgo es crítico
    if absences > 30 or failures >= 2:
        es_riesgo_real = True
    else:
        es_riesgo_real = prediccion[0] == 0

    if not es_riesgo_real:
        st.balloons()
        st.success(f"### 🎉 PROBABLE APROBADO ({prob[0][1]*100:.1f}%)")
    else:
        st.error(f"### 📉 RIESGO DE SUSPENSO ({prob[0][0]*100:.1f}%)")
        st.warning("La IA detecta factores de riesgo críticos (ausencias o fracasos previos).")
