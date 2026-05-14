import streamlit as st
import pandas as pd
from sklearn import tree
import numpy as np

# 1. Configuración visual de la página
st.set_page_config(page_title="IA Rendimiento Escolar", page_icon="🎓")
st.title("🎓 IA de Predicción de Rendimiento")
st.write("Configura los datos del alumno para obtener una predicción coherente.")

# 2. Función de carga y entrenamiento equilibrado
@st.cache_resource
def get_trained_model():
    # Cargar datos desde GitHub
    d = pd.read_csv('student-por.csv', sep=';')
    
    # Ajustamos el umbral a 30 (más equilibrado)
    # Nota: G1+G2+G3 tiene un máximo de 60 puntos.
    d['pass'] = d.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3']) >= 30 else 0, axis=1)
    
    y = d['pass']
    X = d.drop(['G1', 'G2', 'G3', 'pass'], axis=1)
    
    # Transformar variables categóricas
    X = pd.get_dummies(X)
    
    # Entrenar con peso equilibrado para que no ignore a los que suspenden
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5, class_weight='balanced')
    t.fit(X, y)
    
    return t, X.columns.tolist()

# Intentar preparar el modelo
try:
    model, model_columns = get_trained_model()
    st.sidebar.success("✅ Sistema IA entrenado")
except Exception as e:
    st.error(f"Error al preparar la IA: {e}")
    st.stop()

# 3. Interfaz de usuario (Entrada de datos)
st.subheader("Introduce los datos del estudiante:")

col1, col2 = st.columns(2)

with col1:
    failures = st.number_input("Fracasos anteriores (clases suspendidas)", 0, 4, 0)
    absences = st.slider("Número de ausencias totales", 0, 93, 5)
    studytime = st.slider("Tiempo de estudio semanal (1 a 4)", 1, 4, 2)

with col2:
    Medu = st.selectbox("Nivel educativo de la madre (0-4)", [0, 1, 2, 3, 4])
    Fedu = st.selectbox("Nivel educativo del padre (0-4)", [0, 1, 2, 3, 4])
    st.info("Nota: 0 es educación nula y 4 es educación superior.")

# 4. Lógica de Predicción
if st.button("Analizar Estudiante"):
    # Creamos la estructura con todas las columnas en 0
    entrada_dict = {col: [0] for col in model_columns}
    
    # Mapeamos los valores de la interfaz a las columnas del modelo
    entrada_dict['failures'] = [failures]
    entrada_dict['absences'] = [absences]
    entrada_dict['studytime'] = [studytime]
    entrada_dict['Medu'] = [Medu]
    entrada_dict['Fedu'] = [Fedu]
    
    # Convertir a DataFrame y predecir
    df_input = pd.DataFrame(entrada_dict)
    prediccion = model.predict(df_input)
    probabilidades = model.predict_proba(df_input) # Para ver la seguridad de la IA
    
    st.markdown("---")
    
    if prediccion[0] == 1:
        st.balloons()
        st.success(f"### 🎉 RESULTADO: El alumno probablemente APROBARÁ.")
        st.write(f"Seguridad de la IA: {probabilidades[0][1]*100:.2f}%")
    else:
        st.error(f"### 📉 RESULTADO: El alumno está en ALTO RIESGO DE SUSPENDER.")
        st.write(f"Seguridad de la IA: {probabilidades[0][0]*100:.2f}%")

st.caption("Entrenamiento dinámico activo. Ajuste de pesos por desequilibrio aplicado.")
