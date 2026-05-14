import streamlit as st
import pandas as pd
from sklearn import tree
import numpy as np

# 1. Configuración de la página
st.set_page_config(page_title="IA Rendimiento", page_icon="🎓")
st.title("🎓 IA de Predicción de Rendimiento")

# 2. Función para cargar datos y entrenar el modelo en vivo
@st.cache_resource
def get_trained_model():
    # Cargar los datos desde el GitHub
    d = pd.read_csv('student-por.csv', sep=';')
    
    # Crear la etiqueta 'pass' (la misma lógica que en Colab)
    d['pass'] = d.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3']) >= 35 else 0, axis=1)
    
    # Guardamos el objetivo y quitamos columnas innecesarias
    y = d['pass']
    X = d.drop(['G1', 'G2', 'G3', 'pass'], axis=1)
    
    # Convertir categorías a números (One-Hot Encoding)
    X = pd.get_dummies(X)
    
    # Entrenar el modelo aquí mismo
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t.fit(X, y)
    
    return t, X.columns.tolist()

try:
    model, model_columns = get_trained_model()
    st.sidebar.success("✅ IA lista para predecir")
except Exception as e:
    st.error(f"Error al preparar la IA: {e}")
    st.stop()

# 3. Interfaz de usuario
col1, col2 = st.columns(2)
with col1:
    failures = st.number_input("Fracasos anteriores", 0, 4, 0)
    absences = st.slider("Ausencias", 0, 93, 5)
    studytime = st.slider("Tiempo estudio", 1, 4, 2)
with col2:
    Medu = st.selectbox("Edu. Madre", [0, 1, 2, 3, 4])
    Fedu = st.selectbox("Edu. Padre", [0, 1, 2, 3, 4])

# 4. Botón de Predicción REAL
if st.button("Analizar Estudiante"):
    # Crear una fila vacía con todas las columnas necesarias
    entrada_dict = {col: [0] for col in model_columns}
    
    # Rellenar con los datos de los selectores
    entrada_dict['failures'] = [failures]
    entrada_dict['absences'] = [absences]
    entrada_dict['studytime'] = [studytime]
    entrada_dict['Medu'] = [Medu]
    entrada_dict['Fedu'] = [Fedu]
    
    # Convertir a DataFrame y predecir
    df_input = pd.DataFrame(entrada_dict)
    prediccion = model.predict(df_input)
    
    st.markdown("---")
    if prediccion[0] == 1:
        st.balloons()
        st.success("### 🎉 RESULTADO: El alumno probablemente APROBARÁ.")
    else:
        st.error("### 📉 RESULTADO: El alumno está en RIESGO DE SUSPENDER.")
