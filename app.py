# -*- coding: utf-8 -*-
"""
Sahel Taherian
"""

# 1. Librairies
import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI, HTTPException
import lightgbm as lgb
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt



# Creer l'objet app
app = FastAPI(title="Décision d'octroi de crédit",
    description=""""Cette API, va prédire si un client est en capacité de rembourser son crédit ou pas.""",
    version="0.1.0",
)

# Columns to read on CSVs
features = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'Montant_credit',
       'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION',
       'DAYS_ID_PUBLISH', 'CNT_FAM_MEMBERS', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
       'OBS_30_CNT_SOCIAL_CIRCLE', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'Age',
       'CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM',
       'DAYS_EMPLOYED_PERCENT', 'Statut_familial', 'Sexe', 'possède_voiture']

pickle_lgb = open("model_final.pkl", "rb")
final_model = pickle.load(pickle_lgb)

# Reading the csv
df_clients_to_predict = pd.read_csv("df_ex.csv")

@app.get('/')
def index():
    return{"text": "L'API est lancée"}

@app.get("/api/clients")
async def clients_id():
    """ 
    EndPoint pour obtenir l'identifiant des clients
    """
    
    clients_id = df_clients_to_predict["SK_ID_CURR"].tolist()

    return {"clientsId": clients_id}

 
@app.get("/api/predictions/clients/{id}")
async def predict(id: int):
    """ 
    EndPoint pour obtenir la prédiction
    """ 

    clients_id = df_clients_to_predict["SK_ID_CURR"].tolist()

    if id not in clients_id:
        raise HTTPException(status_code=404, detail="client's id not found")
    else:
        # Loading the model

        pickle_lgb = open("model_final.pkl", "rb")
        final_model = pickle.load(pickle_lgb)
        
        threshold = 0.150

        # Filtering by client's id
        df_prediction_by_id = df_clients_to_predict[df_clients_to_predict["SK_ID_CURR"] == id]
        df_prediction_by_id = df_prediction_by_id.drop(df_prediction_by_id.columns[[0, 1]], axis=1)

        # Predicting
        result_proba = final_model.predict_proba(df_prediction_by_id)
        y_prob = result_proba[:, 1]

        result = (y_prob >= threshold).astype(int)

        if (int(result[0]) == 0):
            result = "NON"
        else:
            result = "OUI"    

    return {
        "Client solvable" : result,
        "probability0" : result_proba[0][0],
        "probability1" : result_proba[0][1],
        "threshold" : threshold
    }
