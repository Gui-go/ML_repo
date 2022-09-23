

def fct_seasonalnaiveforecast(df_pred, seasonal_n=12, n=12):
    return df_pred[seasonal_n:][len(df_pred[seasonal_n:])-n:].values # Naive pred



