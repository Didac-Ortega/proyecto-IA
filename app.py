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
    try:
        # 1. Recuperamos las columnas exactas que el modelo espera
        # El modelo guardó el orden de las columnas cuando lo entrenaste
        model_features = model.feature_names_in_
        
        # 2. Creamos un diccionario con todas las columnas en 0
        datos_dict = {feature: [0] for feature in model_features}
        
        # 3. Rellenamos solo los datos que tenemos en la interfaz
        # NOTA: Asegúrate de que los nombres coincidan con los del CSV original
        datos_dict['failures'] = [failures]
        datos_dict['absences'] = [absences]
        datos_dict['studytime'] = [studytime]
        datos_dict['Medu'] = [Medu]
        datos_dict['Fedu'] = [Fedu]
        
        # Si tienes variables categóricas como "higher_yes", las activamos:
        # if higher == "Sí": datos_dict['higher_yes'] = [1]
        
        # 4. Convertimos a DataFrame para que mantenga el orden de columnas
        df_para_predecir = pd.DataFrame(datos_dict)
        
        # 5. Predicción
        resultado = model.predict(df_para_predecir)
        
        st.markdown("---")
        if resultado[0] == 1:
            st.balloons()
            st.success("### 🎉 RESULTADO: El alumno tiene altas probabilidades de APROBAR.")
        else:
            st.error("### 📉 RESULTADO: El alumno está en RIESGO DE SUSPENDER.")
            
    except Exception as e:
        st.error(f"Error técnico: {e}")
        st.info("Revisa que los nombres de las variables ('failures', 'absences', etc.) sean idénticos a los del CSV.")
