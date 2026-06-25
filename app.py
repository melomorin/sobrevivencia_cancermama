from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(title="API de Predicción con Random Survival Forest")

# Añade esto justo debajo de: app = FastAPI(...)
# Si la interfaz HTML se ejecutará en un sitio web (o localmente) y la API está en Render, 
# el navegador bloqueará la petición por seguridad si no habilitas CORS. 
# añadir este middleware antes de hacer el despliegue:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permite cualquier origen (puedes limitarlo a tu dominio web final)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Cargar el modelo al iniciar la aplicación
# Cargar el modelo y el escalador ajustado con tus datos
modelo = joblib.load("modelo_comprimido.pkl")
scaler = joblib.load("scaler_rsf.joblib")

# Orden de las columnas idéntico a tu lista 'features' de la imagen
FEATURES_ORDEN = [
    'Edad', 't_clinica', 'n_clinica', 'm_clinica', 'etapa_clinica',
    'receptores_estrogenos', 'receptores_progesterona', 'her2', 'grado_nuclear'
]

@app.post("/predict")
def predict(data: dict):
    # 1. Convertir JSON a DataFrame y asegurar el orden exacto de las columnas
    df_nuevo = pd.DataFrame([data])[FEATURES_ORDEN]
    
    # 2. Escalar los datos con el StandardScaler cargado
    # Extraemos .values para pasar una matriz limpia de NumPy, idéntica a tu X_scaled
    X_nuevo_scaled = scaler.transform(df_nuevo.values)
    
    # 3. Realizar las predicciones de supervivencia
    risk_score = modelo.predict(X_nuevo_scaled)
    chf_funcs = modelo.predict_cumulative_hazard_function(X_nuevo_scaled)
    surv_funcs = modelo.predict_survival_function(X_nuevo_scaled)
    
    # 4. Extraer ejes de tiempo y probabilidades para Chart.js
    # Como enviamos un solo registro, accedemos al índice [0] del arreglo resultante
    tiempos = chf_funcs[0].x.tolist()
    probabilidades_supervivencia = surv_funcs[0].y.tolist()
    
    return {
        "risk_score": float(risk_score[0]),
        "timeline_tiempos": tiempos,
        "probabilidades_supervivencia": probabilidades_supervivencia
    }

@app.get("/", response_class=HTMLResponse)
def read_index():
    # Sirve de forma automática tu interfaz gráfica al entrar a la URL principal
    with open("index.html", "r", encoding="utf-8") as file:
        return file.read()