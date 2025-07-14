#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import holidays
import warnings
import logging

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

warnings.filterwarnings("ignore")
logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)

# ─── Tracking URI y experimento ───────────────────────────────────────────────
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8050")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Model_RF_with_holidays")

# ─── Rutas y carga de datos ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
RAW_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "01_raw"))

def load_datasets():
    files = {
        'orders'              : 'olist_orders_dataset.csv',
        'order_items'         : 'olist_order_items_dataset.csv',
        'customers'           : 'olist_customers_dataset.csv',
        'products'            : 'olist_products_dataset.csv',
        'sellers'             : 'olist_sellers_dataset.csv',
        'order_payments'      : 'olist_order_payments_dataset.csv',
        'order_reviews'       : 'olist_order_reviews_dataset.csv',
        'geolocation'         : 'olist_geolocation_dataset.csv',
        'product_translation' : 'product_category_name_translation.csv'
    }
    ds = {}
    for name, fname in files.items():
        path = os.path.join(RAW_DIR, fname)
        ds[name] = pd.read_csv(path)
    return ds

if __name__ == "__main__":
    data = load_datasets()
    # 1) Mostrar shapes
    for name, df in data.items():
        print(f"{name}: {df.shape}")

    # 2) Merge principal
    merged = (
        data['orders']
        .merge(data['order_items'],    on='order_id',               how='left')
        .merge(data['products'],       on='product_id',             how='left')
        .merge(data['product_translation'],
               on='product_category_name', how='left')
        .merge(data['sellers'],        on='seller_id',              how='left')
        .merge(data['customers'],      on='customer_id',            how='left')
        .merge(data['order_payments'], on='order_id',               how='left')
        .merge(data['order_reviews'],  on='order_id',               how='left')
    )

    # 3) Geolocalización clientes
    cust_geo = (
        data['geolocation']
        .drop_duplicates(subset=['geolocation_zip_code_prefix'])
        .rename(columns={
            'geolocation_zip_code_prefix': 'customer_zip_code_prefix',
            'geolocation_lat'            : 'customer_lat',
            'geolocation_lng'            : 'customer_lng',
            'geolocation_city'           : 'customer_geo_city',
            'geolocation_state'          : 'customer_geo_state'
        })
    )
    merged = merged.merge(cust_geo,
                          on='customer_zip_code_prefix',
                          how='left')

    # 4) Geolocalización vendedores
    sell_geo = (
        data['geolocation']
        .drop_duplicates(subset=['geolocation_zip_code_prefix'])
        .rename(columns={
            'geolocation_zip_code_prefix': 'seller_zip_code_prefix',
            'geolocation_lat'            : 'seller_lat',
            'geolocation_lng'            : 'seller_lng',
            'geolocation_city'           : 'seller_geo_city',
            'geolocation_state'          : 'seller_geo_state'
        })
    )
    merged = merged.merge(sell_geo,
                          on='seller_zip_code_prefix',
                          how='left')

    # 5) Timestamp y festivos
    merged['order_purchase_timestamp'] = pd.to_datetime(
        merged['order_purchase_timestamp']
    )
    merged['order_date'] = merged['order_purchase_timestamp'].dt.date
    years = merged['order_date'].apply(lambda d: d.year).unique().tolist()
    br_hols = holidays.Brazil(years=years)
    merged['is_holiday'] = merged['order_date'].apply(
        lambda d: int(d in br_hols)
    )

    # 6) Series diarias por categoría
    daily = (
        merged
        .groupby(['product_category_name_english', 'order_date'])
        .size()
        .reset_index(name='units_sold')
    )

    # 7) Filtrar categorías con suficientes días
    category_days = (
        daily
        .groupby('product_category_name_english')['order_date']
        .nunique()
        .reset_index(name='unique_days')
    )
    MIN_DAYS = 45
    valid_cats = category_days.query(
        "unique_days >= @MIN_DAYS"
    )['product_category_name_english'].tolist()
    print(f"Categorías válidas (≥{MIN_DAYS} días): {len(valid_cats)}")

    # 8) Parámetros del loop
    TRAIN_DAYS     = 30
    TEST_DAYS      = 15
    TS_SPLITS      = 5
    OOB_CANDIDATES = [100, 200, 500]
    RANDOM_STATE   = 42
    tscv = TimeSeriesSplit(n_splits=TS_SPLITS)

    results = []
    for cat in valid_cats:
        print(f"Entrenando categoría: {cat}")
        df_cat = daily[
            daily['product_category_name_english'] == cat
        ].copy()
        df_cat['order_date'] = pd.to_datetime(df_cat['order_date'])
        df_cat.set_index('order_date', inplace=True)

        # Feature engineering
        df_cat['lag1']  = df_cat['units_sold'].shift(1)
        df_cat['roll7'] = df_cat['units_sold'].rolling(7).mean()
        idx = df_cat.index
        df_cat['dow_sin'] = np.sin(2 * np.pi * idx.weekday / 7)
        df_cat['dow_cos'] = np.cos(2 * np.pi * idx.weekday / 7)
        df_cat['mes_sin'] = np.sin(2 * np.pi * idx.month / 12)
        df_cat['mes_cos'] = np.cos(2 * np.pi * idx.month / 12)
        df_cat['festivo'] = idx.to_series().isin(br_hols).astype(int)
        # Black Friday
        bf_dates = []
        for y in idx.year.unique():
            nov = pd.date_range(f"{y}-11-01", f"{y}-11-30", freq="D")
            th  = nov[nov.weekday == 3]
            bf_dates.append((th[3] + pd.Timedelta(days=1)).date())
        df_cat['bf'] = idx.to_series().apply(
            lambda d: int(d.date() in bf_dates)
        )

        # Preparamos train/test
        df_feat    = df_cat.dropna().drop(
            columns=['product_category_name_english']
        )
        total_days = len(df_feat)
        if total_days < TRAIN_DAYS + TEST_DAYS:
            continue

        X_train = df_feat.iloc[:TRAIN_DAYS].drop(
            columns=['units_sold']
        )
        y_train = df_feat.iloc[:TRAIN_DAYS]['units_sold'].values
        X_test  = df_feat.iloc[
                  TRAIN_DAYS:TRAIN_DAYS + TEST_DAYS
        ].drop(columns=['units_sold'])
        y_test  = df_feat.iloc[
                  TRAIN_DAYS:TRAIN_DAYS + TEST_DAYS
        ]['units_sold'].values
        if y_test.sum() == 0:
            continue

      
        best_oob, best_n = -np.inf, None
        for n in OOB_CANDIDATES:
            tmp = RandomForestRegressor(
                n_estimators=n,
                oob_score=True,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            tmp.fit(X_train, y_train)
            if tmp.oob_score_ > best_oob:
                best_oob, best_n = tmp.oob_score_, n

        
        param_dist = {
            'n_estimators'    : [max(5, best_n // 2), best_n, best_n * 2],
            'max_depth'       : [None, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features'    : ['sqrt', 0.5]
        }
        search = RandomizedSearchCV(
            RandomForestRegressor(
                random_state=RANDOM_STATE,
                oob_score=True
            ),
            param_distributions=param_dist,
            n_iter=20,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # 8.3) Predicción y métricas
        preds = best_model.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        mape  = np.mean(
            np.abs((y_test - preds) / y_test)[y_test > 0]
        ) * 100

        
        with mlflow.start_run(run_name=f"RF_{cat}"):
            mlflow.log_param("category",       cat)
            mlflow.log_param("train_days",     TRAIN_DAYS)
            mlflow.log_param("test_days",      TEST_DAYS)
            mlflow.log_param("ts_splits",      TS_SPLITS)
            mlflow.log_param("initial_best_n_oob", best_n)
            mlflow.log_metric("initial_oob_score", best_oob)
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("MAE",     mae)
            mlflow.log_metric("RMSE",    rmse)
            mlflow.log_metric("MAPE_pct", mape)

            input_example = X_train.head(1)
            signature     = infer_signature(
                X_train,
                best_model.predict(X_train)
            )
            mlflow.sklearn.log_model(
                sk_model      = best_model,
                name          = "rf_model",
                input_example = input_example,
                signature     = signature
            )

        results.append({
            'category' : cat,
            'best_n_oob': best_n,
            'init_oob' : best_oob,
            'MAE'      : mae,
            'RMSE'     : rmse,
            'MAPE_pct' : mape,
            **search.best_params_
        })

    
    res_df = pd.DataFrame(results).sort_values('MAPE_pct').reset_index(drop=True)
    pd.set_option('display.max_rows', None)
    print(res_df)
