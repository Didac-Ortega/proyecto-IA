
import streamlit as st\
import pandas as pd\
import joblib\
import numpy as np\
\
# 1. Configuraci\'f3n de la p\'e1gina\
st.set_page_config(page_title="Predicciones Escolares", page_icon="\uc0\u55356 \u57235 ")\
st.title("\uc0\u55356 \u57235  Predicci\'f3n de Rendimiento Estudiantil")\
st.write("Esta IA predice si un alumno aprobar\'e1 bas\'e1ndose en sus datos sociales y acad\'e9micos.")\
\
# 2. Cargar el modelo (Aseg\'farate de que 'modelo_arbol.joblib' est\'e9 en GitHub)\
try:\
    model = joblib.load('modelo_arbol.joblib')\
    # Necesitamos los nombres de las columnas para que coincidan con el entrenamiento\
    # Si guardaste las columnas en Colab, c\'e1rgalas aqu\'ed. Si no, usaremos una muestra.\
    st.sidebar.success("Modelo cargado correctamente")\
except:\
    st.sidebar.error("No se encontr\'f3 el archivo 'modelo_arbol.joblib'. S\'fabelo a GitHub.")\
\
# 3. Interfaz de usuario (Entradas de datos)\
st.subheader("Introduce los datos del estudiante:")\
\
col1, col2 = st.columns(2)\
\
with col1:\
    failures = st.number_input("Fracasos anteriores (clases suspendidas)", min_value=0, max_value=4, value=0)\
    absences = st.slider("N\'famero de ausencias", 0, 93, 10)\
    higher = st.radio("\'bfQuiere hacer estudios superiores?", ["S\'ed", "No"])\
\
with col2:\
    Medu = st.selectbox("Nivel educativo de la madre", [0, 1, 2, 3, 4], help="0: ninguno, 4: superior")\
    Fedu = st.selectbox("Nivel educativo del padre", [0, 1, 2, 3, 4])\
    studytime = st.slider("Tiempo de estudio semanal", 1, 4, 2, help="1: <2h, 4: >10h")\
\
# 4. Bot\'f3n de predicci\'f3n\
if st.button("Analizar Estudiante"):\
    # NOTA: Aqu\'ed deber\'edas transformar las entradas para que coincidan \
    # exactamente con las columnas que gener\'f3 el 'get_dummies' en Colab.\
    # Por ahora, mostramos un mensaje de \'e9xito:\
    st.info("Procesando datos...")\
    \
    # Aqu\'ed ir\'eda la l\'f3gica: prediccion = model.predict(datos_preparados)\
    # st.write(f"Resultado: \{prediccion\}")\
    st.warning("Recuerda que para que la predicci\'f3n funcione, los datos de entrada deben tener el mismo formato (columnas) que usaste en el entrenamiento.")}
