import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from pycaret.regression import setup, pull, plot_model, create_model, predict_model
import os
import shutil
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import base64

# ----------------------- CARGA Y PREPARACIÓN DE DATOS -----------------------

df = pd.read_csv("data/datos.csv")
df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
df["order_date"] = df["order_purchase_timestamp"].dt.date
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

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.JOURNAL])

# ----------------------- LAYOUT -----------------------

app.layout = html.Div(
    [
        html.H1(
            "Ventas Retail - Dashboard", style={"textAlign": "center", "marginTop": 20}
        ),
        dcc.Tabs(
            [
                dcc.Tab(
                    label="Estadísticas Descriptivas",
                    children=[
                        html.Br(),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Selecciona productos:"),
                                        dcc.Dropdown(
                                            id="producto-dropdown",
                                            options=[
                                                {"label": cat, "value": cat}
                                                for cat in df[
                                                    "product_category_name_english"
                                                ]
                                                .dropna()
                                                .unique()
                                            ],
                                            multi=True,
                                        ),
                                    ],
                                    width=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Rango de fechas:"),
                                        dcc.DatePickerRange(
                                            id="fecha-range",
                                            start_date=df[
                                                "order_purchase_timestamp"
                                            ].min(),
                                            end_date=df[
                                                "order_purchase_timestamp"
                                            ].max(),
                                        ),
                                    ],
                                    width=4,
                                ),
                            ]
                        ),
                        html.Br(),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Total de Pedidos"),
                                            dbc.CardBody(
                                                html.H4(
                                                    id="kpi-pedidos",
                                                    className="card-title",
                                                )
                                            ),
                                        ],
                                        color="primary",
                                        inverse=True,
                                    ),
                                    width=3,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Ventas Totales"),
                                            dbc.CardBody(
                                                html.H4(
                                                    id="kpi-ventas",
                                                    className="card-title",
                                                )
                                            ),
                                        ],
                                        color="success",
                                        inverse=True,
                                    ),
                                    width=3,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Tasa de Entregas a Tiempo"),
                                            dbc.CardBody(
                                                html.H4(
                                                    id="kpi-entregas",
                                                    className="card-title",
                                                )
                                            ),
                                        ],
                                        color="info",
                                        inverse=True,
                                    ),
                                    width=3,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Punt. Promedio de Reseñas"),
                                            dbc.CardBody(
                                                html.H4(
                                                    id="kpi-reviews",
                                                    className="card-title",
                                                )
                                            ),
                                        ],
                                        color="warning",
                                        inverse=True,
                                    ),
                                    width=3,
                                ),
                            ]
                        ),
                        html.Br(),
                        dcc.Graph(id="line-chart"),
                        html.Br(),
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="mapa-ventas"), width=6),
                                dbc.Col(
                                    html.Div(
                                        id="tabla-por-estado",
                                        style={"overflowX": "auto"},
                                    ),
                                    width=6,
                                ),
                            ]
                        ),
                        html.Br(),
                        dcc.Graph(id="barras-pago"),
                    ],
                ),
                dcc.Tab(
                    label="Modelos Estadísticos",
                    children=[
                        html.Br(),
                        html.H5(
                            "Predicción de ventas con PyCaret",
                            style={"textAlign": "center"},
                        ),
                        html.Br(),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Selecciona variables predictoras:"),
                                        dcc.Dropdown(
                                            id="vars-predictoras",
                                            options=[
                                                {"label": col, "value": col}
                                                for col in default_predictors
                                            ],
                                            multi=True,
                                            value=[],
                                            placeholder="Selecciona variables predictoras",
                                        ),
                                        html.Br(),
                                        html.Label("Selecciona el modelo:"),
                                        dcc.Dropdown(
                                            id="modelo-dropdown",
                                            options=[
                                                {
                                                    "label": "Random Forest",
                                                    "value": "rf",
                                                },
                                                {"label": "Extra Trees", "value": "et"},
                                                {
                                                    "label": "LightGBM",
                                                    "value": "lightgbm",
                                                },
                                            ],
                                            value="rf",
                                        ),
                                        html.Br(),
                                        dbc.Button(
                                            "Entrenar Modelo",
                                            id="btn-entrenar",
                                            color="primary",
                                        ),
                                    ],
                                    width=4,
                                ),
                                dbc.Col(
                                    [
                                        html.H6("Resultados del modelo:"),
                                        html.Div(
                                            id="tabla-modelos",
                                            style={"overflowX": "scroll"},
                                        ),
                                    ],
                                    width=8,
                                ),
                            ]
                        ),
                        html.Br(),
                        html.H4(
                            "Importancia de Variables y Gráfico Residual",
                            style={"textAlign": "center"},
                        ),
                        html.Div(
                            id="importancia-vars",
                            style={
                                "textAlign": "center",
                                "display": "flex",
                                "justifyContent": "space-around",
                            },
                        ),
                        html.Br(),
                        html.H4("Predicción vs Real", style={"textAlign": "center"}),
                        html.Div(
                            id="grafico-predicciones", style={"textAlign": "center"}
                        ),
                        html.Br(),
                    ],
                ),
            ]
        ),
    ]
)

# ----------------------- CALLBACKS DESCRIPTIVOS -----------------------


@app.callback(
    Output("kpi-pedidos", "children"),
    Output("kpi-ventas", "children"),
    Output("kpi-entregas", "children"),
    Output("kpi-reviews", "children"),
    Input("producto-dropdown", "value"),
    Input("fecha-range", "start_date"),
    Input("fecha-range", "end_date"),
)
def actualizar_kpis(productos, fecha_inicio, fecha_fin):
    df_filtrado = df[
        (df["order_purchase_timestamp"] >= fecha_inicio)
        & (df["order_purchase_timestamp"] <= fecha_fin)
    ]
    if productos:
        df_filtrado = df_filtrado[
            df_filtrado["product_category_name_english"].isin(productos)
        ]

    total_pedidos = df_filtrado["order_id"].nunique()
    total_ventas = df_filtrado["price"].sum()

    if (
        "order_delivered_customer_date" in df_filtrado.columns
        and "order_estimated_delivery_date" in df_filtrado.columns
    ):
        entregas_a_tiempo = (
            pd.to_datetime(df_filtrado["order_delivered_customer_date"])
            <= pd.to_datetime(df_filtrado["order_estimated_delivery_date"])
        ).mean()
        entregas_a_tiempo = f"{entregas_a_tiempo * 100:.2f}%"
    else:
        entregas_a_tiempo = "N/A"

    puntaje = (
        df_filtrado["review_score"].mean()
        if "review_score" in df_filtrado.columns
        else "N/A"
    )
    return (
        total_pedidos,
        f"${total_ventas:,.2f}",
        entregas_a_tiempo,
        f"{puntaje:.2f}" if isinstance(puntaje, float) else puntaje,
    )


# ----------------------- CALLBACK PARA GRÁFICOS -----------------------


@app.callback(
    Output("line-chart", "figure"),
    Output("mapa-ventas", "figure"),
    Output("barras-pago", "figure"),
    Output("tabla-por-estado", "children"),
    Input("producto-dropdown", "value"),
    Input("fecha-range", "start_date"),
    Input("fecha-range", "end_date"),
)
def actualizar_graficos(productos, fecha_inicio, fecha_fin):
    df_filtrado = df[
        (df["order_purchase_timestamp"] >= fecha_inicio)
        & (df["order_purchase_timestamp"] <= fecha_fin)
    ]
    if productos:
        df_filtrado = df_filtrado[
            df_filtrado["product_category_name_english"].isin(productos)
        ]

    ventas_por_fecha = (
        df_filtrado.groupby(["order_month", "product_category_name_english"])
        .size()
        .reset_index(name="ventas")
    )
    line_fig = px.line(
        ventas_por_fecha,
        x="order_month",
        y="ventas",
        color="product_category_name_english",
        markers=True,
        title="Ventas por producto en el tiempo",
    )

    df_geo = df_filtrado.dropna(subset=["geolocation_lat", "geolocation_lng"])
    if not df_geo.empty:
        mapa_fig = px.scatter_mapbox(
            df_geo,
            lat="geolocation_lat",
            lon="geolocation_lng",
            hover_data=["order_id"],
            zoom=3,
            height=500,
        )
        mapa_fig.update_layout(mapbox_style="open-street-map", title="")
    else:
        mapa_fig = px.scatter_mapbox(title="")
        mapa_fig.update_layout(mapbox_style="open-street-map")

    if "payment_type" in df_filtrado.columns:
        barras = df_filtrado["payment_type"].value_counts().reset_index()
        barras.columns = ["Tipo de Pago", "Frecuencia"]
        barras_fig = px.bar(
            barras,
            x="Tipo de Pago",
            y="Frecuencia",
            title="Frecuencia por tipo de pago",
        )
    else:
        barras_fig = px.bar(title="No hay información de tipo de pago")

    if (
        "seller_state" in df_filtrado.columns
        and "product_category_name_english" in df_filtrado.columns
    ):
        tabla = (
            df_filtrado.groupby(["seller_state", "product_category_name_english"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        max_cols = 10
        dynamic_cols = tabla.columns[1 : max_cols + 1]
        tabla = tabla[["seller_state"] + list(dynamic_cols)].head(10)
        tabla_html = dbc.Table.from_dataframe(
            tabla, striped=True, bordered=True, hover=True
        )
    else:
        tabla_html = html.Div("No hay datos de estado y categoría disponibles.")

    return line_fig, mapa_fig, barras_fig, tabla_html


# ----------------------- CALLBACK PYCARET -----------------------


@app.callback(
    Output("tabla-modelos", "children"),
    Output("importancia-vars", "children"),
    Output("grafico-predicciones", "children"),
    Input("btn-entrenar", "n_clicks"),
    State("vars-predictoras", "value"),
    State("modelo-dropdown", "value"),
)
def entrenar_modelos(n_clicks, vars_predictoras, modelo):
    if not n_clicks or not vars_predictoras:
        return (
            "Selecciona variables y haz clic en Entrenar Modelo",
            "",
            "",
        )

    cols_existentes = [col for col in vars_predictoras if col in df.columns]
    if not cols_existentes:
        return (
            "Ninguna de las variables seleccionadas existe en el DataFrame.",
            "",
            "",
        )

    cols_req = cols_existentes + ["payment_value", "order_purchase_timestamp"]
    df_model = df[cols_req].dropna().copy()

    if df_model.empty:
        return "No hay datos para entrenar con esas variables.", "", ""

    df_model["order_year_month"] = df_model["order_purchase_timestamp"].dt.to_period(
        "M"
    )
    ultimo_mes = df_model["order_year_month"].max()

    train_df = df_model[df_model["order_year_month"] < ultimo_mes]
    test_df = df_model[df_model["order_year_month"] == ultimo_mes]

    if train_df.empty or test_df.empty:
        return (
            "No hay suficientes datos para entrenar o testear con esta selección.",
            "",
            "",
        )

    setup(
        data=train_df[cols_existentes + ["payment_value"]],
        target="payment_value",
        session_id=123,
        fold_strategy="timeseries",
        data_split_shuffle=False,
        fold_shuffle=False,
        verbose=False,
    )
    model = create_model(modelo)

    # Predicciones sobre test
    pred = predict_model(model, data=test_df[cols_existentes + ["payment_value"]])
    y_true = pred["payment_value"].values
    y_pred = pred["prediction_label"].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # KPIs como tarjetas
    kpi_cards = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("MAE", className="card-title"),
                            html.H2(f"{mae:.4f}", className="card-text"),
                        ]
                    ),
                    color="primary",
                    inverse=True,
                ),
                width=4,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("RMSE", className="card-title"),
                            html.H2(f"{rmse:.4f}", className="card-text"),
                        ]
                    ),
                    color="warning",
                    inverse=True,
                ),
                width=4,
            ),
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("R2", className="card-title"),
                            html.H2(f"{r2:.4f}", className="card-text"),
                        ]
                    ),
                    color="success",
                    inverse=True,
                ),
                width=4,
            ),
        ],
        className="mb-4",
        justify="center",
    )

    # Importancia de variables
    plot_model(model, plot="feature", save=True)
    feature_img_path = "Feature Importance.png"
    if os.path.exists(feature_img_path):
        with open(feature_img_path, "rb") as f:
            encoded_feature = base64.b64encode(f.read()).decode()
        importancia_img = html.Img(
            src="data:image/png;base64," + encoded_feature,
            ###           style={"width": "60%", "maxWidth": "500px", "margin": "10px"},
            alt="Importancia Variables",
        )
        os.remove(feature_img_path)
    else:
        importancia_img = html.Div("No se pudo generar gráfico de importancia.")

    # Gráfico residual
    plot_model(model, plot="residuals", save=True)
    residual_img_path = "Residuals.png"
    if os.path.exists(residual_img_path):
        with open(residual_img_path, "rb") as image_file:
            encoded_residual = base64.b64encode(image_file.read()).decode()
        residual_img = html.Img(
            src="data:image/png;base64," + encoded_residual,
            ##       style={"width": "100%", "maxWidth": "500px", "margin": "40px"},
            alt="Gráfico Residual",
        )
        os.remove(residual_img_path)
    else:
        residual_img = html.Div("No se pudo generar el gráfico residual.")

    combined_graphs = html.Div(
        [importancia_img, residual_img],
        style={"display": "flex", "justifyContent": "center"},
    )

    pred["order_date"] = test_df["order_purchase_timestamp"].dt.date.values
    pred_grouped = (
        pred.groupby("order_date")[["payment_value", "prediction_label"]]
        .sum()
        .reset_index()
    )
    fig_pred = px.line(
        pred_grouped,
        x="order_date",
        y=["payment_value", "prediction_label"],
        labels={"value": "Valor", "order_date": "Fecha"},
        title=f"Valores Reales vs Predichos - Mes {ultimo_mes.strftime('%Y-%m')}",
        markers=True,
    )

    return (
        kpi_cards,
        combined_graphs,
        dcc.Graph(figure=fig_pred),
    )


if __name__ == "__main__":
    app.run(debug=True)
# To run the app, save this script as `app.py` and execute it with Python.
# Ensure you have the required libraries installed:
