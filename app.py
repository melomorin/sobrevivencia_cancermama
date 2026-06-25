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
modelo = joblib.load("modelo_rsf.joblib")

@app.post("/predict")
def predict(data: dict):
    # Convertir el diccionario JSON recibido en un DataFrame de Pandas
    df_nuevo = pd.DataFrame([data])
    
    # 1. Predecir la función de riesgo acumulado (Cumulative Hazard Function)
    chf_funcs = modelo.predict_cumulative_hazard_function(df_nuevo)
    
    # 2. Predecir la función de supervivencia (Survival Function)
    surv_funcs = modelo.predict_survival_function(df_nuevo)
    
    # Ejemplo: Obtener el riesgo o score de riesgo general (Riesgo Relativo)
    risk_score = modelo.predict(df_nuevo)[0]
    
    # Extraer los tiempos evaluados por el modelo
    tiempos = chf_funcs[0].x.tolist()
    probabilidades_supervivencia = surv_funcs[0].y.tolist()
    
    return {
        "risk_score": float(risk_score),
        "timeline_tiempos": tiempos,
        "probabilidades_supervivencia": probabilidades_supervivencia
    }
