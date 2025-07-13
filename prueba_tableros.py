import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pycaret.regression import setup, pull, plot_model, create_model, predict_model
import os
import shutil
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import base64
from datetime import datetime, timedelta

# ----------------------- CARGA Y PREPARACIÓN DE DATOS -----------------------

df = pd.read_csv("data/datos.csv")
df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
df["order_date"] = df["order_purchase_timestamp"].dt.date
df["order_year"] = df["order_purchase_timestamp"].dt.year
df["order_quarter"] = df["order_purchase_timestamp"].dt.quarter
df["order_weekday"] = df["order_purchase_timestamp"].dt.day_name()

fecha_limite = pd.to_datetime("2018-08-30")
df = df[df["order_purchase_timestamp"] <= fecha_limite]

# Crear columna delivery_time_days si es posible
if (
    "order_delivered_customer_date" in df.columns
    and "order_purchase_timestamp" in df.columns
):
    df["order_delivered_customer_date"] = pd.to_datetime(
        df["order_delivered_customer_date"]
    )
    df["delivery_time_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days
else:
    df["delivery_time_days"] = None

# Preparación para resumen diario
df_daily = df.groupby("order_date").agg({"payment_value": "sum"}).reset_index()
df_daily["day_of_week"] = pd.to_datetime(df_daily["order_date"]).dt.dayofweek
df_daily["month"] = pd.to_datetime(df_daily["order_date"]).dt.month

# Crear métricas adicionales
df['profit_margin'] = (df['price'] - df['freight_value']) / df['price'] * 100
df['high_value_order'] = df['payment_value'] > df['payment_value'].quantile(0.8)

default_predictors = [
    "price",
    "freight_value",
    "product_weight_g",
    "delivery_time_days",
    "review_score",
    "payment_type",
    "seller_state",
    "geolocation_state",
    "order_status",
]

# ----------------------- INICIALIZAR DASH APP -----------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Dashboard de Ventas Retail"

# ----------------------- COMPONENTES PERSONALIZADOS -----------------------

def create_metric_card(title, value, color="primary", icon=None):
    return dbc.Card(
        [
            dbc.CardBody([
                html.Div([
                    html.H6(title, className="card-subtitle text-muted"),
                    html.H3(value, className="card-title text-white mb-0"),
                ], className="d-flex flex-column")
            ])
        ],
        color=color,
        inverse=True,
        className="mb-3 shadow-sm"
    )

# ----------------------- LAYOUT MEJORADO -----------------------

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("📊 Dashboard de Ventas Retail", 
                   className="text-center text-primary mb-4"),
            html.Hr()
        ])
    ]),
    
    # Filtros globales
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🔍 Filtros Globales", className="card-title"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Productos:", className="fw-bold"),
                            dcc.Dropdown(
                                id="producto-dropdown",
                                options=[
                                    {"label": f"📦 {cat}", "value": cat}
                                    for cat in df["product_category_name_english"]
                                    .dropna()
                                    .unique()
                                ],
                                multi=True,
                                placeholder="Selecciona productos...",
                                className="mb-2"
                            ),
                        ], width=6),
                        dbc.Col([
                            html.Label("Rango de fechas:", className="fw-bold"),
                            dcc.DatePickerRange(
                                id="fecha-range",
                                start_date=df["order_purchase_timestamp"].min(),
                                end_date=df["order_purchase_timestamp"].max(),
                                display_format="DD/MM/YYYY",
                                className="mb-2"
                            ),
                        ], width=6),
                    ])
                ])
            ], className="mb-4")
        ])
    ]),
    
    # Tabs principales
    dcc.Tabs(id="main-tabs", value="tab-1", children=[
        dcc.Tab(label="📈 Análisis Descriptivo", value="tab-1", children=[
            html.Div(id="tab-descriptivo-content")
        ]),
        dcc.Tab(label="🤖 Modelos Predictivos", value="tab-2", children=[
            html.Div(id="tab-modelos-content")
        ]),
        dcc.Tab(label="🎯 Análisis Avanzado", value="tab-3", children=[
            html.Div(id="tab-avanzado-content")
        ])
    ])
], fluid=True)

# ----------------------- CALLBACKS PRINCIPALES -----------------------

@app.callback(
    Output("tab-descriptivo-content", "children"),
    Input("main-tabs", "value"),
    Input("producto-dropdown", "value"),
    Input("fecha-range", "start_date"),
    Input("fecha-range", "end_date"),
)
def render_tab_descriptivo(active_tab, productos, fecha_inicio, fecha_fin):
    if active_tab != "tab-1":
        return html.Div()
    
    # Filtrar datos
    df_filtrado = df[
        (df["order_purchase_timestamp"] >= fecha_inicio)
        & (df["order_purchase_timestamp"] <= fecha_fin)
    ]
    if productos:
        df_filtrado = df_filtrado[
            df_filtrado["product_category_name_english"].isin(productos)
        ]
    
    # Calcular KPIs
    total_pedidos = df_filtrado["order_id"].nunique()
    total_ventas = df_filtrado["price"].sum()
    
    # Entregas a tiempo
    if ("order_delivered_customer_date" in df_filtrado.columns and 
        "order_estimated_delivery_date" in df_filtrado.columns):
        entregas_a_tiempo = (
            pd.to_datetime(df_filtrado["order_delivered_customer_date"])
            <= pd.to_datetime(df_filtrado["order_estimated_delivery_date"])
        ).mean()
        entregas_text = f"{entregas_a_tiempo * 100:.1f}%"
    else:
        entregas_text = "N/A"
    
    # Review score
    puntaje = df_filtrado["review_score"].mean() if "review_score" in df_filtrado.columns else 0
    puntaje_text = f"{puntaje:.1f}/5" if puntaje > 0 else "N/A"
    
    # Margen promedio
    margen_promedio = df_filtrado["profit_margin"].mean()
    margen_text = f"{margen_promedio:.1f}%" if not pd.isna(margen_promedio) else "N/A"
    
    # Crear gráficos
    
    # 1. Gráfico de tendencia de ventas
    ventas_tiempo = df_filtrado.groupby("order_month").agg({
        "payment_value": "sum",
        "order_id": "nunique"
    }).reset_index()
    
    fig_tendencia = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Ventas Totales", "Número de Pedidos"),
        vertical_spacing=0.1
    )
    
    fig_tendencia.add_trace(
        go.Scatter(
            x=ventas_tiempo["order_month"],
            y=ventas_tiempo["payment_value"],
            mode="lines+markers",
            name="Ventas",
            line=dict(color="#1f77b4", width=3)
        ),
        row=1, col=1
    )
    
    fig_tendencia.add_trace(
        go.Scatter(
            x=ventas_tiempo["order_month"],
            y=ventas_tiempo["order_id"],
            mode="lines+markers",
            name="Pedidos",
            line=dict(color="#ff7f0e", width=3)
        ),
        row=2, col=1
    )
    
    fig_tendencia.update_layout(
        title_text="📈 Tendencia de Ventas en el Tiempo",
        height=600,
        showlegend=False
    )
    
    # 2. Análisis por día de la semana
    ventas_dia = df_filtrado.groupby("order_weekday").agg({
        "payment_value": "sum"
    }).reset_index()
    
    # Ordenar días de la semana
    dias_orden = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    ventas_dia["order_weekday"] = pd.Categorical(ventas_dia["order_weekday"], categories=dias_orden, ordered=True)
    ventas_dia = ventas_dia.sort_values("order_weekday")
    
    fig_dias = px.bar(
        ventas_dia,
        x="order_weekday",
        y="payment_value",
        title="💰 Ventas por Día de la Semana",
        color="payment_value",
        color_continuous_scale="Blues"
    )
    fig_dias.update_layout(xaxis_title="Día de la Semana", yaxis_title="Ventas ($)")
    
    # 3. Top productos
    top_productos = df_filtrado.groupby("product_category_name_english").agg({
        "payment_value": "sum",
        "order_id": "nunique"
    }).sort_values("payment_value", ascending=False).head(10).reset_index()
    
    fig_productos = px.bar(
        top_productos,
        x="payment_value",
        y="product_category_name_english",
        orientation="h",
        title="🏆 Top 10 Productos por Ventas",
        color="payment_value",
        color_continuous_scale="Viridis"
    )
    fig_productos.update_layout(yaxis_title="Categoría", xaxis_title="Ventas ($)")
    
    # 4. Distribución de métodos de pago
    if "payment_type" in df_filtrado.columns:
        pago_dist = df_filtrado["payment_type"].value_counts().reset_index()
        pago_dist.columns = ["Método", "Cantidad"]
        
        fig_pago = px.pie(
            pago_dist,
            values="Cantidad",
            names="Método",
            title="💳 Distribución de Métodos de Pago"
        )
    else:
        fig_pago = px.pie(title="💳 Sin datos de métodos de pago")
    
    return html.Div([
        # KPIs
        dbc.Row([
            dbc.Col([
                create_metric_card("Total Pedidos", f"{total_pedidos:,}", "primary")
            ], width=2),
            dbc.Col([
                create_metric_card("Ventas Totales", f"${total_ventas:,.0f}", "success")
            ], width=2),
            dbc.Col([
                create_metric_card("Entregas a Tiempo", entregas_text, "info")
            ], width=2),
            dbc.Col([
                create_metric_card("Rating Promedio", puntaje_text, "warning")
            ], width=2),
            dbc.Col([
                create_metric_card("Margen Promedio", margen_text, "danger")
            ], width=2),
            dbc.Col([
                create_metric_card("Productos Únicos", f"{df_filtrado['product_category_name_english'].nunique()}", "secondary")
            ], width=2),
        ], className="mb-4"),
        
        # Gráficos principales
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_tendencia)
            ], width=8),
            dbc.Col([
                dcc.Graph(figure=fig_dias)
            ], width=4),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_productos)
            ], width=8),
            dbc.Col([
                dcc.Graph(figure=fig_pago)
            ], width=4),
        ])
    ])

@app.callback(
    Output("tab-modelos-content", "children"),
    Input("main-tabs", "value"),
)
def render_tab_modelos(active_tab):
    if active_tab != "tab-2":
        return html.Div()
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("🤖 Configuración del Modelo", className="card-title"),
                        html.Label("Variables Predictoras:", className="fw-bold"),
                        dcc.Dropdown(
                            id="vars-predictoras",
                            options=[
                                {"label": f"📊 {col}", "value": col}
                                for col in default_predictors
                            ],
                            multi=True,
                            value=["price", "freight_value", "product_weight_g"],
                            placeholder="Selecciona variables predictoras...",
                            className="mb-3"
                        ),
                        html.Label("Algoritmo:", className="fw-bold"),
                        dcc.Dropdown(
                            id="modelo-dropdown",
                            options=[
                                {"label": "🌳 Random Forest", "value": "rf"},
                                {"label": "🌲 Extra Trees", "value": "et"},
                                {"label": "⚡ LightGBM", "value": "lightgbm"},
                                {"label": "📈 Linear Regression", "value": "lr"},
                                {"label": "🎯 XGBoost", "value": "xgboost"}
                            ],
                            value="rf",
                            className="mb-3"
                        ),
                        dbc.Button(
                            "🚀 Entrenar Modelo",
                            id="btn-entrenar",
                            color="primary",
                            size="lg",
                            className="w-100"
                        ),
                    ])
                ])
            ], width=4),
            dbc.Col([
                html.Div(id="resultados-modelo")
            ], width=8)
        ], className="mb-4"),
        
        # Visualizaciones del modelo
        html.Div(id="visualizaciones-modelo")
    ])

@app.callback(
    Output("tab-avanzado-content", "children"),
    Input("main-tabs", "value"),
    Input("producto-dropdown", "value"),
    Input("fecha-range", "start_date"),
    Input("fecha-range", "end_date"),
)
def render_tab_avanzado(active_tab, productos, fecha_inicio, fecha_fin):
    if active_tab != "tab-3":
        return html.Div()
    
    # Filtrar datos
    df_filtrado = df[
        (df["order_purchase_timestamp"] >= fecha_inicio)
        & (df["order_purchase_timestamp"] <= fecha_fin)
    ]
    if productos:
        df_filtrado = df_filtrado[
            df_filtrado["product_category_name_english"].isin(productos)
        ]
    
    # Análisis de correlaciones
    numeric_cols = df_filtrado.select_dtypes(include=[np.number]).columns
    corr_matrix = df_filtrado[numeric_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        title="🔗 Matriz de Correlaciones",
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    
    # Análisis de cohorts (si es posible)
    if "customer_unique_id" in df_filtrado.columns:
        # Análisis de cohortes simplificado
        cohort_data = df_filtrado.groupby([
            df_filtrado["order_purchase_timestamp"].dt.to_period("M"),
            "customer_unique_id"
        ]).size().reset_index()
        
        fig_cohort = px.line(
            title="👥 Análisis de Cohortes (Simplificado)"
        )
    else:
        fig_cohort = px.line(title="👥 Análisis de Cohortes - Datos no disponibles")
    
    # Análisis de estacionalidad
    estacional = df_filtrado.groupby([
        df_filtrado["order_purchase_timestamp"].dt.month,
        df_filtrado["order_purchase_timestamp"].dt.year
    ]).agg({
        "payment_value": "sum"
    }).reset_index()
    
    fig_estacional = px.box(
        df_filtrado,
        x=df_filtrado["order_purchase_timestamp"].dt.month,
        y="payment_value",
        title="📅 Análisis de Estacionalidad por Mes"
    )
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_corr)
            ], width=6),
            dbc.Col([
                dcc.Graph(figure=fig_estacional)
            ], width=6),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_cohort)
            ], width=12),
        ])
    ])

# ----------------------- CALLBACK PARA MODELOS -----------------------

@app.callback(
    Output("resultados-modelo", "children"),
    Output("visualizaciones-modelo", "children"),
    Input("btn-entrenar", "n_clicks"),
    State("vars-predictoras", "value"),
    State("modelo-dropdown", "value"),
    State("producto-dropdown", "value"),
    State("fecha-range", "start_date"),
    State("fecha-range", "end_date"),
)
def entrenar_modelos(n_clicks, vars_predictoras, modelo, productos, fecha_inicio, fecha_fin):
    if not n_clicks or not vars_predictoras:
        return (
            dbc.Alert(
                "⚠️ Selecciona variables predictoras y haz clic en 'Entrenar Modelo'",
                color="warning"
            ),
            html.Div()
        )
    
    # Filtrar datos
    df_model = df[
        (df["order_purchase_timestamp"] >= fecha_inicio)
        & (df["order_purchase_timestamp"] <= fecha_fin)
    ]
    if productos:
        df_model = df_model[df_model["product_category_name_english"].isin(productos)]
    
    cols_existentes = [col for col in vars_predictoras if col in df_model.columns]
    if not cols_existentes:
        return (
            dbc.Alert(
                "❌ Ninguna de las variables seleccionadas existe en los datos filtrados",
                color="danger"
            ),
            html.Div()
        )
    
    cols_req = cols_existentes + ["payment_value", "order_purchase_timestamp"]
    df_model = df_model[cols_req].dropna().copy()
    
    if df_model.empty:
        return (
            dbc.Alert("❌ No hay datos suficientes para entrenar", color="danger"),
            html.Div()
        )
    
    try:
        # Preparar datos
        df_model["order_year_month"] = df_model["order_purchase_timestamp"].dt.to_period("M")
        ultimo_mes = df_model["order_year_month"].max()
        
        train_df = df_model[df_model["order_year_month"] < ultimo_mes]
        test_df = df_model[df_model["order_year_month"] == ultimo_mes]
        
        if train_df.empty or test_df.empty:
            return (
                dbc.Alert(
                    "❌ No hay suficientes datos para entrenar/testear",
                    color="danger"
                ),
                html.Div()
            )
        
        # Entrenar modelo
        setup(
            data=train_df[cols_existentes + ["payment_value"]],
            target="payment_value",
            session_id=123,
            fold_strategy="timeseries",
            data_split_shuffle=False,
            fold_shuffle=False,
            verbose=False,
        )
        
        trained_model = create_model(modelo)
        
        # Predicciones
        pred = predict_model(trained_model, data=test_df[cols_existentes + ["payment_value"]])
        y_true = pred["payment_value"].values
        y_pred = pred["prediction_label"].values
        
        # Métricas
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Resultados
        resultados = dbc.Row([
            dbc.Col([
                create_metric_card("MAE", f"{mae:.2f}", "primary")
            ], width=4),
            dbc.Col([
                create_metric_card("RMSE", f"{rmse:.2f}", "warning")
            ], width=4),
            dbc.Col([
                create_metric_card("R²", f"{r2:.3f}", "success")
            ], width=4),
        ])
        
        # Gráfico de predicciones vs reales
        fig_pred = px.scatter(
            x=y_true,
            y=y_pred,
            title="🎯 Predicciones vs Valores Reales",
            labels={"x": "Valores Reales", "y": "Predicciones"}
        )
        
        # Línea de referencia perfecta
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig_pred.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Predicción Perfecta",
                line=dict(dash="dash", color="red")
            )
        )
        
        visualizaciones = html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=fig_pred)
                ], width=12)
            ])
        ])
        
        return resultados, visualizaciones
        
    except Exception as e:
        return (
            dbc.Alert(f"❌ Error al entrenar el modelo: {str(e)}", color="danger"),
            html.Div()
        )

if __name__ == "__main__":
    app.run(debug=True, port=8050)