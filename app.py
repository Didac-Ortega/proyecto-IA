import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Configuración de la página
st.set_page_config(page_title="Predicciones Escolares", page_icon="🎓")
st.title("🎓 IA de Predicción de Rendimiento")
st.write("Introduce los datos del alumno para obtener una predicción instantánea.")

# 2. Cargar el modelo
@st.cache_resource # Esto hace que el modelo se cargue una sola vez y sea muy rápido
def load_my_model():
    return joblib.load('modelo_arbol.joblib')

try:
    model = load_my_model()
    st.sidebar.success("✅ Modelo listo")
except Exception as e:
    st.sidebar.error("❌ Error al cargar modelo")
    st.stop() # Detiene la app si no hay modelo

# 3. Interfaz de usuario
col1, col2 = st.columns(2)

with col1:
    failures = st.number_input("Fracasos anteriores", 0, 4, 0)
    absences = st.slider("Ausencias totales", 0, 93, 5)
    studytime = st.slider("Tiempo estudio (1 a 4)", 1, 4, 2)

with col2:
    Medu = st.selectbox("Educación Madre (0-4)", [0, 1, 2, 3, 4])
    Fedu = st.selectbox("Educación Padre (0-4)", [0, 1, 2, 3, 4])
    # Para simplificar, asumimos que estas son las variables principales
    # que el modelo necesita en el orden correcto.

# 4. Análisis Inmediato
if st.button("Analizar Estudiante"):
    # Creamos el array con los datos capturados
    # El orden debe ser el mismo que usaste en d_train_att.columns
    datos_para_predecir = np.array([[Medu, Fedu, studytime, failures, absences]])
    
    try:
        # Predicción
        resultado = model.predict(datos_para_predecir)
        
        st.markdown("---")
        if resultado[0] == 1:
            st.balloons()
            st.success("### 🎉 RESULTADO: El alumno tiene altas probabilidades de APROBAR.")
        else:
            st.error("### 📉 RESULTADO: El alumno está en RIESGO DE SUSPENDER.")
            
    except Exception as e:
        st.error(f"Error técnico: El modelo esperaba más datos de los enviados.")
        st.info("Nota: Si usaste 'get_dummies' con muchas columnas, el modelo necesita que le enviemos todas esas columnas (sexo, escuela, etc.) aunque no las usemos aquí.")

st.markdown("---")
st.caption("Proyecto de IA - Entrenamiento basado en Dataset UCI Student Performance")
