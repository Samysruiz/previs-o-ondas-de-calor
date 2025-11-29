import pandas as pd
import numpy as np
from datetime import datetime

def preparar_features(lat, lon, data, gold):
    # Converter para datetime
    data = pd.to_datetime(data)
    
    # Buscar info do GOLD correspondente
    base = gold.copy()

    # Exemplo: nearest neighbor das coordenadas
    base["dist"] = np.sqrt((base["lat"] - lat)**2 + (base["lon"] - lon)**2)
    linha = base.sort_values("dist").iloc[0]

    # Montar vetor de features
    features = pd.DataFrame([{
        "lat": lat,
        "lon": lon,
        "doy": data.timetuple().tm_yday,
        "oni": linha["oni"],
        "tmax_anomaly": linha["tmax_anomaly"],
        "oni_lag_3m": linha["oni_lag_3m"],
    }])

    return features
