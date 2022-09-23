

# Tabela 8188 - Índice e variação da receita nominal e do volume de vendas no comércio varejista ampliado, por atividades (2014 = 100) - Chamada de API
# https://apisidra.ibge.gov.br/values/t/8188/n1/all/v/11706/p/all/c11046/all/c85/2759,2762,90671,90672,90673,103155,103156,103157,103159/d/v11706%205

import pandas as pd
import numpy as np
import sidrapy
import janitor
import os
import matplotlib.pyplot as plt
from src.fct_mape import fct_mape
from src.fct_predhistoricalmean import fct_predhistoricalmean
from src.fct_naiveforecast import fct_naiveforecast
from src.fct_seasonalnaiveforecast import fct_seasonalnaiveforecast
os.curdir

# Options:
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 10)

# Parameters:
h = 12
data_address = "data/tabela8188.csv"
activity = "Veículos, motocicletas, partes e peças"
var_pred="receita"


# Load data:
df = pd.read_csv(data_address)
df.columns = ["region", "atividade", "period", "receita", "volume"]
df.head()
df.describe()
df.dtypes
df.isnull().sum()
df['atividade'].value_counts(normalize=True)
df[['mes', 'ano']] = df['period'].str.split(' ', 1, expand=True)
df = df.replace({
    "mes" : {
        "janeiro"  : "01",
        "fevereiro": "02",
        "março"    : "03",
        "abril"    : "04",
        "maio"     : "05",
        "junho"    : "06",
        "julho"    : "07",
        "agosto"   : "08",
        "setembro" : "09",
        "outubro"  : "10",
        "novembro" : "11",
        "dezembro" : "12"
    },
    "volume" : {
        "-" : 0
    },
    "receita" : {
        "-" : 0
    }
})
# df.dtypes

df['date'] = pd.to_datetime(df.ano + "-" + df.mes + "-01")
df['volume'] = df['volume'].astype(float)
df['receita'] = df['receita'].astype(float)

# filter:
df = df[df["atividade"]==activity]
df = df.loc[:, ["date", var_pred]]

# train/test:
train = df[:-h]
test = df[-h:]

# Prediction prep:
df_pred = pd.DataFrame()
df_pred['date'] = test.date
df_pred['pred_historicalmean'] = fct_predhistoricalmean(train[var_pred])
df_pred['pred_seasonalnaiveforecast'] = fct_seasonalnaiveforecast(train[[var_pred]], seasonal_n=12, n=12)
df_pred['pred_naiveforecast'] = fct_naiveforecast(train[var_pred])


# df_pred['pred_naivemodel'] = 
# train[[var_pred]][len(train[var_pred])-h:].values # Naive pred
# train[[var_pred]][-12:]
# train[[var_pred]][1:12]
# df_pred



fct_mape(test[var_pred], df_pred.pred_historicalmean)
fct_mape(test[var_pred], df_pred.pred_seasonalnaiveforecast)
fct_mape(test[var_pred], df_pred.pred_naiveforecast)


# def viz_scatterplot:
fig, ax = plt.subplots()
ax.plot(train.date, train[var_pred], 'k', label='Train')
ax.plot(test.date, test[var_pred], 'b', label='Test')
ax.plot(df_pred.date, df_pred.pred_historicalmean, 'r-', label='historical mean')
ax.plot(df_pred.date, df_pred.pred_seasonalnaiveforecast, 'r-', label='seasonal naive forecast')
ax.plot(df_pred.date, df_pred.pred_naiveforecast, 'r-', label='naive forecast')
ax.set_xlabel('date')
ax.set_ylabel('value')
ax.legend(loc=2) # Legend of colors and variables
plt.show()

viz_scatterplot()