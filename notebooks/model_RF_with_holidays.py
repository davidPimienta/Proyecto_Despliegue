#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
import holidays
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import warnings
warnings.filterwarnings('ignore')
from IPython.display import display
import matplotlib.pyplot as plt
import holidays


# In[40]:


BASE_PATH = r"C:\Users\fdcontreras\OneDrive - Indra\Universidad\Despliegue de Soluciones Analíticas\Proyecto_Despliegue\Proyecto_Despliegue\data\01_raw"

def load_datasets():
    files = {
        'orders': 'olist_orders_dataset.csv',
        'order_items': 'olist_order_items_dataset.csv',
        'customers': 'olist_customers_dataset.csv',
        'products': 'olist_products_dataset.csv',
        'sellers': 'olist_sellers_dataset.csv',
        'order_payments': 'olist_order_payments_dataset.csv',
        'order_reviews': 'olist_order_reviews_dataset.csv',
        'geolocation': 'olist_geolocation_dataset.csv',
        'product_translation': 'product_category_name_translation.csv'
    }

    datasets = {}
    for name, filename in files.items():
        filepath = os.path.join(BASE_PATH, filename)
        datasets[name] = pd.read_csv(filepath)

    return datasets
data = load_datasets()


# In[41]:


for name, df in data.items():
    print(f"{name}: {df.shape}")


# In[42]:


merged_df = data['orders'].merge(
    data['order_items'], 
    on='order_id', 
    how='left'
)

print(f"Después de order_items: {merged_df.shape}")

# Agregar información de productos
merged_df = merged_df.merge(
    data['products'], 
    on='product_id', 
    how='left'
)

print(f"Después de products: {merged_df.shape}")

# Agregar traducción de categorías
merged_df = merged_df.merge(
    data['product_translation'], 
    on='product_category_name', 
    how='left'
)

print(f"Después de product_translation: {merged_df.shape}")
# Agregar información de vendedores
merged_df = merged_df.merge(
    data['sellers'], 
    on='seller_id', 
    how='left'
)

print(f"Después de sellers: {merged_df.shape}")

# Agregar información de clientes
merged_df = merged_df.merge(
    data['customers'], 
    on='customer_id', 
    how='left'
)

print(f"Después de customers: {merged_df.shape}")

# Agregar pagos
merged_df = merged_df.merge(
    data['order_payments'], 
    on='order_id', 
    how='left'
)

print(f"Después de order_payments: {merged_df.shape}")

# Agregar reviews
merged_df = merged_df.merge(
    data['order_reviews'], 
    on='order_id', 
    how='left'
)

print(f"Después de order_reviews: {merged_df.shape}")

# Agregar geolocalización de clientes
customer_geo = data['geolocation'].drop_duplicates(subset=['geolocation_zip_code_prefix'])
customer_geo = customer_geo.rename(columns={
    'geolocation_zip_code_prefix': 'customer_zip_code_prefix',
    'geolocation_lat': 'customer_lat',
    'geolocation_lng': 'customer_lng',
    'geolocation_city': 'customer_geo_city',
    'geolocation_state': 'customer_geo_state'
})

merged_df = merged_df.merge(
    customer_geo, 
    on='customer_zip_code_prefix', 
    how='left'
)

print(f"Después de customer geolocation: {merged_df.shape}")

# Agregar geolocalización de vendedores
seller_geo = data['geolocation'].drop_duplicates(subset=['geolocation_zip_code_prefix'])
seller_geo = seller_geo.rename(columns={
    'geolocation_zip_code_prefix': 'seller_zip_code_prefix',
    'geolocation_lat': 'seller_lat',
    'geolocation_lng': 'seller_lng',
    'geolocation_city': 'seller_geo_city',
    'geolocation_state': 'seller_geo_state'
})

merged_df = merged_df.merge(
    seller_geo, 
    on='seller_zip_code_prefix', 
    how='left'
)

print(f"Dataset final: {merged_df.shape}")

# Verificar información del dataset final
print("Información del dataset consolidado:")
print(f"Filas: {merged_df.shape[0]:,}")
print(f"Columnas: {merged_df.shape[1]}")
print(f"Memoria: {merged_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


# In[ ]:


merged_df.head()


# In[44]:


merged_df['order_purchase_timestamp'] = pd.to_datetime(merged_df['order_purchase_timestamp'])
merged_df['order_date'] = merged_df['order_purchase_timestamp'].dt.date


# In[45]:


category_days = (
    merged_df
      .groupby('product_category_name_english')['order_date']
      .nunique()
      .reset_index(name='unique_days')
      .sort_values('unique_days', ascending=False)
)

pd.set_option('display.max_rows',     None)
pd.set_option('display.max_columns',  None)
pd.set_option('display.width',        200)
pd.set_option('display.max_colwidth', None)

display(category_days)


# In[46]:


MIN_UNIQUE_DAYS = 45

valid_cats = (
    category_days[
        category_days['unique_days'] >= MIN_UNIQUE_DAYS
    ]['product_category_name_english']
    .tolist()
)

print(f"Categorías válidas (≥{MIN_UNIQUE_DAYS} días): {len(valid_cats)}\n", valid_cats)


# In[ ]:


daily = (
    merged_df
    .groupby(['product_category_name_english','order_date'])
    .size()
    .reset_index(name='units_sold')
)


daily.head()


# In[48]:


# Filtrar categorías con al menos 45 días de ventas
daily_filtered = daily[
    daily['product_category_name_english'].isin(valid_cats)
].copy()

print(f"daily_filtered shape: {daily_filtered.shape}")
daily_filtered['product_category_name_english'].nunique()


# In[49]:


pd.set_option('display.max_rows',   None)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',      200)

display(daily_filtered.head(20))


# In[50]:


categoria = 'furniture_decor'  

df_plot = merged_df[merged_df['product_category_name_english'] == categoria].copy()

df_plot['order_date'] = pd.to_datetime(df_plot['order_date']).dt.date

ts = (
    df_plot
      .groupby('order_date')
      .size()
      .reset_index(name='ventas')
)

plt.figure(figsize=(12, 4))
plt.plot(ts['order_date'], ts['ventas'])
plt.title(f'Ventas diarias de "{categoria}"')
plt.xlabel('Fecha')
plt.ylabel('Unidades vendidas')
plt.tight_layout()
plt.show()


# In[ ]:


categoria = 'furniture_decor' 


df_cat = merged_df[merged_df['product_category_name_english'] == categoria].copy()
df_cat['order_date'] = pd.to_datetime(df_cat['order_date'])


ventas_sem = df_cat.set_index('order_date').resample('W').size()


plt.figure(figsize=(12,4))
plt.plot(ventas_sem.index, ventas_sem.values)
plt.title(f'Ventas semanales de "{categoria}"')
plt.xlabel('Semana')
plt.ylabel('Unidades vendidas (semanal)')
plt.tight_layout()
plt.show()


# In[52]:


TRAIN_DAYS     = 15
TS_SPLITS      = 5
OOB_CANDIDATES = [100, 200, 500]
RANDOM_STATE   = 42


years = sorted(daily['order_date'].apply(lambda d: d.year).unique())
br_hols = holidays.Brazil(years=years)

results = []
tscv    = TimeSeriesSplit(n_splits=TS_SPLITS)


# In[53]:


for cat in valid_cats:
    df_cat = daily[
        daily['product_category_name_english'] == cat
    ].copy()
    df_cat['order_date'] = pd.to_datetime(df_cat['order_date'])
    df_cat.set_index('order_date', inplace=True)

    # 2.1) Feature Engineering diario
    df_cat['lag1']   = df_cat['units_sold'].shift(1)
    df_cat['roll7']  = df_cat['units_sold'].rolling(7).mean()
    idx = df_cat.index
    df_cat['dow_sin'] = np.sin(2*np.pi * idx.weekday / 7)
    df_cat['dow_cos'] = np.cos(2*np.pi * idx.weekday / 7)
    df_cat['mes_sin'] = np.sin(2*np.pi * idx.month   / 12)
    df_cat['mes_cos'] = np.cos(2*np.pi * idx.month   / 12)
    df_cat['festivo'] = idx.to_series().isin(br_hols).astype(int)

    bf_dates = []
    for y in idx.year.unique():
        nov = pd.date_range(f"{y}-11-01", f"{y}-11-30", freq="D")
        th  = nov[nov.weekday == 3]
        bf_dates.append((th[3] + pd.Timedelta(days=1)).date())
    df_cat['bf'] = df_cat.index.to_series().apply(lambda d: d.date() in bf_dates).astype(int)


    df_feat = df_cat.dropna()
    df_feat = df_feat.drop(columns=['product_category_name_english'])

    if len(df_feat) <= TRAIN_DAYS:
        continue



    X = df_feat.drop(columns=['units_sold'])
    y = df_feat['units_sold'].values
    X_train, X_test = X.iloc[:TRAIN_DAYS], X.iloc[TRAIN_DAYS:]
    y_train, y_test = y[:TRAIN_DAYS], y[TRAIN_DAYS:]
    if y_test.sum() == 0:
        continue


    best_oob, best_n = -np.inf, None
    for n in OOB_CANDIDATES:
        tmp = RandomForestRegressor(
            n_estimators = n,
            oob_score    = True,
            random_state = RANDOM_STATE,
            n_jobs       = -1
        )
        tmp.fit(X_train, y_train)
        if tmp.oob_score_ > best_oob:
            best_oob, best_n = tmp.oob_score_, n


    param_dist = {
        'n_estimators':    [max(5, best_n//2), best_n, best_n*2],
        'max_depth':       [None, 5, 10],
        'min_samples_leaf':[1, 2, 4],
        'max_features':    ['sqrt', 0.5]
    }
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, oob_score=True),
        param_distributions = param_dist,
        n_iter        = 20,
        cv            = tscv,
        scoring       = 'neg_mean_squared_error',
        n_jobs        = -1,
        random_state  = RANDOM_STATE
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_


    preds = best_model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mape  = np.mean(np.abs((y_test - preds) / y_test)[y_test > 0]) * 100

    results.append({
        'category':       cat,
        'best_n_oob':     best_n,
        'init_oob_score': best_oob,
        'MAE':            mae,
        'RMSE':           rmse,
        'MAPE (%)':       mape,
        **search.best_params_
    })

res_rf = pd.DataFrame(results) \
            .sort_values('MAPE (%)') \
            .reset_index(drop=True)
pd.set_option('display.max_rows', None)
display(res_rf)

