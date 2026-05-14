import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Configuración de la página
st.set_page_config(page_title="Predicciones Escolares", page_icon="🎓")
st.title("🎓 Predicción de Rendimiento Estudiantil")
st.write("Esta IA predice si un alumno aprobará basándose en sus datos sociales y académicos.")

# 2. Cargar el modelo
try:
    # Asegúrate de que el nombre del archivo coincida exactamente con el que subiste a GitHub
    model = joblib.load('modelo_arbol.joblib')
    st.sidebar.success("Modelo cargado correctamente")
except Exception as e:
    st.sidebar.error(f"Error al cargar el modelo: {e}")
    st.sidebar.info("Asegúrate de haber subido 'modelo_arbol.joblib' a tu repositorio.")

# 3. Interfaz de usuario (Entradas de datos)
st.subheader("Introduce los datos del estudiante:")

col1, col2 = st.columns(2)

with col1:
    failures = st.number_input("Fracasos anteriores (clases suspendidas)", min_value=0, max_value=4, value=0)
    absences = st.slider("Número de ausencias", 0, 93, 10)
    higher = st.radio("¿Quiere hacer estudios superiores?", ["Sí", "No"])

with col2:
    Medu = st.selectbox("Nivel educativo de la madre", [0, 1, 2, 3, 4], help="0: ninguno, 4: superior")
    Fedu = st.selectbox("Nivel educativo del padre", [0, 1, 2, 3, 4])
    studytime = st.slider("Tiempo de estudio semanal", 1, 4, 2, help="1: <2h, 4: >10h")

# 4. Botón de predicción
if st.button("Analizar Estudiante"):
    st.info("Procesando datos...")
    # Aquí la IA daría su veredicto. 
    # Para que funcione la predicción real, los datos de arriba deben convertirse 
    # al formato de columnas exacto que usó el modelo en Colab.
    st.warning("Interfaz conectada. Para que la IA prediga, asegúrate de que el formato de entrada coincida con el entrenamiento.")
