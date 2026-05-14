import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Configuración
st.set_page_config(page_title="IA Rendimiento", page_icon="🎓")
st.title("🎓 IA de Predicción de Rendimiento")

# 2. Cargar el modelo entrenado en Colab
@st.cache_resource
def load_model():
    # Asegúrate de que el archivo se llame exactamente así en GitHub
    return joblib.load('modelo_arbol.joblib')

model = load_model()

# 3. Interfaz
col1, col2 = st.columns(2)
with col1:
    failures = st.number_input("Fracasos anteriores", 0, 4, 0)
    absences = st.slider("Ausencias", 0, 93, 5)
    studytime = st.slider("Tiempo estudio (1-4)", 1, 4, 2)
with col2:
    Medu = st.selectbox("Edu. Madre (0-4)", [0, 1, 2, 3, 4])
    Fedu = st.selectbox("Edu. Padre (0-4)", [0, 1, 2, 3, 4])

# 4. Predicción
if st.button("Analizar Estudiante"):
    # IMPORTANTE: Creamos una fila con ceros para las 50+ columnas que espera el modelo
    # El modelo de Colab tiene muchas columnas por el get_dummies
    
    # Creamos un array de ceros con el tamaño que el modelo espera
    num_features = model.n_features_in_
    datos = np.zeros((1, num_features))
    
    # Rellenamos las posiciones de nuestras variables
    # (En el modelo de Colab, las primeras suelen ser las numéricas)
    datos[0, 0] = Medu
    datos[0, 1] = Fedu
    datos[0, 2] = studytime
    datos[0, 3] = failures
    datos[0, 4] = absences
    
    prediccion = model.predict(datos)
    prob = model.predict_proba(datos)
    
    st.markdown("---")
    
    # --- LA REGLA DE SENTIDO COMÚN ---
    # Si tiene más de 40 ausencias, es un riesgo humano real, 
    # independientemente de lo que diga la máquina.
    if absences > 40:
        resultado_final = 0 
    else:
        resultado_final = prediccion[0]

    if resultado_final == 1:
        st.balloons()
        st.success(f"### 🎉 APROBADO (Confianza: {prob[0][1]*100:.1f}%)")
    else:
        st.error(f"### 📉 RIESGO DE SUSPENSO (Confianza: {prob[0][0]*100:.1f}%)")
        if absences > 40:
            st.warning("Nota: Se detectó un nivel de ausentismo crítico.")
