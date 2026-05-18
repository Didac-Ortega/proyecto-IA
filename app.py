import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Configuración de la página
st.set_page_config(page_title="IA Rendimiento Pro", page_icon="🎓", layout="wide")
st.title("🎓 IA de Análisis de Rendimiento Estudiantil (Sistema Español)")
st.write("Introduce las notas del alumno (0 al 10) y sus datos para predecir si aprobará el curso.")

# 2. Cargar modelo y columnas
@st.cache_resource
def load_assets():
    model = joblib.load('modelo_arbol.joblib')
    columns = joblib.load('model_columns.joblib')
    return model, columns

try:
    model, model_columns = load_assets()
    st.sidebar.success("✅ Modelo cargado correctamente")
except Exception as e:
    st.sidebar.error("❌ Error al cargar archivos .joblib")
    st.stop()

# 3. Formulario de entrada de datos (ADAPTADO A ESPAÑA)
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 Notas Actuales (0 a 10)")
    # El usuario introduce las notas en el formato que conoce (0 a 10)
    g1_espanol = st.slider("Nota Primer Trimestre (G1)", 0.0, 10.0, 5.0, step=0.5)
    g2_espanol = st.slider("Nota Segundo Trimestre (G2)", 0.0, 10.0, 5.0, step=0.5)
    g3_espanol = st.slider("Nota Examen Final (G3)", 0.0, 10.0, 5.0, step=0.5)

with col2:
    st.subheader("🏫 Datos Académicos")
    failures = st.number_input("Fracasos anteriores (Asignaturas suspendidas)", 0, 4, 0)
    absences = st.slider("Ausencias totales en el año", 0, 93, 5)
    studytime = st.selectbox("Tiempo de estudio semanal", [1, 2, 3, 4], 
                            format_func=lambda x: f"{x} (Horas: {['<2 horas','2-5 horas','5-10 horas','>10 horas'][x-1]})")

with col3:
    st.subheader("👨‍👩‍👧 Entorno Familiar")
    medu = st.slider("Nivel educativo de la Madre (0-4)", 0, 4, 2)
    fedu = st.slider("Nivel educativo del Padre (0-4)", 0, 4, 2)
    higher = st.radio("¿Quiere hacer estudios superiores / Universidad?", ["Sí", "No"])

# 4. Lógica de Predicción
if st.button("🚀 Realizar Análisis"):
    # TRUCO MÁGICO: Multiplicamos por 2 internamente para adaptar el 0-10 al 0-20 que entiende la IA
    g1_portugal = g1_espanol * 2
    g2_portugal = g2_espanol * 2
    g3_portugal = g3_espanol * 2

    # Creamos la estructura que el modelo espera
    entrada = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
    
    # Rellenamos los datos mapeados
    entrada['G1'] = g1_portugal
    entrada['G2'] = g2_portugal
    entrada['G3'] = g3_portugal
    entrada['failures'] = failures
    entrada['absences'] = absences
    entrada['studytime'] = studytime
    entrada['Medu'] = medu
    entrada['Fedu'] = fedu
    
    if higher == "Sí":
        if 'higher_yes' in entrada.columns: entrada['higher_yes'] = 1
    else:
        if 'higher_no' in entrada.columns: entrada['higher_no'] = 1

    # Ejecutar la predicción
    pred = model.predict(entrada)
    prob = model.predict_proba(entrada)

    st.markdown("---")
    
    # Mostramos el resultado de forma muy visual
    if pred[0] == 1:
        st.balloons()
        st.success(f"### 🎉 RESULTADO: El alumno tiene altas probabilidades de APROBAR.")
        st.write(f"Seguridad del veredicto: {prob[0][1]*100:.1f}%")
    else:
        st.error(f"### 📉 RESULTADO: El alumno está en ALTO RIESGO DE SUSPENDER.")
        st.write(f"Seguridad del veredicto: {prob[0][0]*100:.1f}%")
    
    # Nota de aviso si la media es baja
    media_espanola = (g1_espanol + g2_espanol + g3_espanol) / 3
    if media_espanola < 5.0:
        st.warning(f"Aviso: La nota media del alumno es de un {media_espanola:.1f}/10. Está por debajo del aprobado.")

st.sidebar.markdown("---")
st.sidebar.info("Conversión de escala automática activa (0-10 Español ➔ 0-20 Portugal).")
