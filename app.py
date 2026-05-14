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
        # 1. Definimos MANUALMENTE la lista de columnas que el modelo vio en Colab
        # IMPORTANTE: Deben estar en el mismo orden exacto que después del get_dummies
        # Como son muchas, vamos a usar una lógica para asegurar que enviamos lo correcto
        
        # Primero cargamos el CSV para obtener las columnas reales
        df_original = pd.read_csv('student-por.csv', sep=';')
        df_original['pass'] = df_original.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3']) >= 35 else 0, axis=1)
        df_original = df_original.drop(['G1', 'G2', 'G3', 'pass'], axis=1)
        
        # Aplicamos el get_dummies igual que en el entrenamiento
        df_dummy = pd.get_dummies(df_original)
        model_features = df_dummy.columns.tolist()
        
        # 2. Creamos el DataFrame de una fila con ceros
        datos_dict = {feature: [0] for feature in model_features}
        
        # 3. Rellenamos los datos de la interfaz
        # Asegúrate de que estos nombres existan en tu CSV original
        if 'failures' in datos_dict: datos_dict['failures'] = [failures]
        if 'absences' in datos_dict: datos_dict['absences'] = [absences]
        if 'studytime' in datos_dict: datos_dict['studytime'] = [studytime]
        if 'Medu' in datos_dict: datos_dict['Medu'] = [Medu]
        if 'Fedu' in datos_dict: datos_dict['Fedu'] = [Fedu]
        
        # 4. Convertimos a DataFrame con el orden correcto
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
        st.info("Asegúrate de que el archivo 'student-por.csv' esté en tu GitHub.")
        st.info("Revisa que los nombres de las variables ('failures', 'absences', etc.) sean idénticos a los del CSV.")
