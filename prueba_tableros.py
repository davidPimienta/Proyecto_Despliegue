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

# Preparación para resumen diario - AGREGAMOS CANTIDAD DE PRODUCTOS
df_daily = df.groupby("order_date").agg({
    "payment_value": "sum",
    "order_id": "count"  # Cantidad de productos vendidos por día
}).reset_index()
df_daily.rename(columns={"order_id": "productos_vendidos"}, inplace=True)
df_daily["day_of_week"] = pd.to_datetime(df_daily["order_date"]).dt.dayofweek
df_daily["month"] = pd.to_datetime(df_daily["order_date"]).dt.month
df_daily["day_of_month"] = pd.to_datetime(df_daily["order_date"]).dt.day

# Variables predictoras - combinando variables originales y temporales
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
    "day_of_week",
    "month", 
    "day_of_month"
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
                        dcc.Graph(id="grafico-pedidos-tiempo"),
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
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="barras-pago"), width=6),
                                dbc.Col(dcc.Graph(id="grafico-dias-semana"), width=6),
                            ]
                        ),
                        html.Br(),
                    ],
                ),
                dcc.Tab(
                    label="Modelos Estadísticos",
                    children=[
                        html.Br(),
                        html.H5(
                            "Predicción de Cantidad de Productos Vendidos Diariamente",
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
                                                {"label": "Precio", "value": "price"},
                                                {"label": "Valor del flete", "value": "freight_value"},
                                                {"label": "Peso del producto (g)", "value": "product_weight_g"},
                                                {"label": "Tiempo de entrega (días)", "value": "delivery_time_days"},
                                                {"label": "Puntuación de reseña", "value": "review_score"},
                                                {"label": "Tipo de pago", "value": "payment_type"},
                                                {"label": "Estado del vendedor", "value": "seller_state"},
                                                {"label": "Estado de geolocalización", "value": "geolocation_state"},
                                                {"label": "Estado del pedido", "value": "order_status"},
                                                {"label": "Día de la semana", "value": "day_of_week"},
                                                {"label": "Mes", "value": "month"},
                                                {"label": "Día del mes", "value": "day_of_month"},
                                            ],
                                            multi=True,
                                            value=["price", "freight_value", "day_of_week", "month"],
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
                                                {
                                                    "label": "Linear Regression",
                                                    "value": "lr",
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
                        html.H4("Predicción vs Real - Productos Vendidos Diariamente", style={"textAlign": "center"}),
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
    Output("grafico-dias-semana", "figure"),
    Output("grafico-pedidos-tiempo", "figure"),
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

    # Gráfico de ventas por días de la semana
    if not df_filtrado.empty:
        df_filtrado_copy = df_filtrado.copy()
        df_filtrado_copy["day_name"] = pd.to_datetime(df_filtrado_copy["order_purchase_timestamp"]).dt.day_name()
        
        # Definir orden de días
        dias_orden = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dias_espanol = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        
        ventas_por_dia = df_filtrado_copy.groupby("day_name")["payment_value"].sum().reset_index()
        
        # Reordenar según días de la semana
        ventas_por_dia["day_name"] = pd.Categorical(ventas_por_dia["day_name"], categories=dias_orden, ordered=True)
        ventas_por_dia = ventas_por_dia.sort_values("day_name")
        
        # Mapear a español
        ventas_por_dia["dia_esp"] = ventas_por_dia["day_name"].map(dict(zip(dias_orden, dias_espanol)))
        
        dias_fig = px.bar(
            ventas_por_dia,
            x="dia_esp",
            y="payment_value",
            title="Ventas por Día de la Semana",
            labels={"dia_esp": "Día de la Semana", "payment_value": "Ventas ($)"},
            color="payment_value",
            color_continuous_scale="Blues"
        )
        dias_fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=False
        )
    else:
        dias_fig = px.bar(title="No hay datos para mostrar")

    # Gráfico de ventas en el tiempo (modificado)
    if not df_filtrado.empty:
        # Agrupar por mes y categoría de producto para obtener las ventas
        ventas_tiempo = df_filtrado.groupby(["order_month", "product_category_name_english"]).agg({
            "payment_value": "sum"
        }).reset_index()
        
        pedidos_fig = px.line(
            ventas_tiempo,
            x="order_month",
            y="payment_value",
            color="product_category_name_english",
            title="Ventas por Producto en el Tiempo",
            labels={"order_month": "Mes", "payment_value": "Ventas ($)", "product_category_name_english": "Categoría"},
            markers=True
        )
        pedidos_fig.update_layout(
            xaxis_title="Mes",
            yaxis_title="Ventas ($)",
            legend_title="Categoría de Producto"
        )
    else:
        pedidos_fig = px.line(title="No hay datos para mostrar")

    return line_fig, mapa_fig, barras_fig, tabla_html, dias_fig, pedidos_fig

# ----------------------- CALLBACK PYCARET MODIFICADO -----------------------

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

    # Crear dataset agregado por día para predicción
    df_prediccion = df.groupby("order_date").agg({
        "order_id": "count",  # Cantidad de productos vendidos por día
        "payment_value": "sum",
        "price": "mean",
        "freight_value": "mean",
        "product_weight_g": "mean",
        "delivery_time_days": "mean",
        "review_score": "mean"
    }).reset_index()
    
    df_prediccion.rename(columns={"order_id": "productos_vendidos"}, inplace=True)
    
    # Agregar variables temporales
    df_prediccion["order_date"] = pd.to_datetime(df_prediccion["order_date"])
    df_prediccion["day_of_week"] = df_prediccion["order_date"].dt.dayofweek
    df_prediccion["month"] = df_prediccion["order_date"].dt.month
    df_prediccion["day_of_month"] = df_prediccion["order_date"].dt.day
    
    # Para variables categóricas, usar la moda (valor más frecuente) por día
    categorical_vars = ["payment_type", "seller_state", "geolocation_state", "order_status"]
    for var in categorical_vars:
        if var in df.columns:
            df_prediccion[var] = df.groupby("order_date")[var].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown").values
    
    # Verificar que las variables seleccionadas existen
    cols_existentes = [col for col in vars_predictoras if col in df_prediccion.columns]
    if not cols_existentes:
        return (
            "Ninguna de las variables seleccionadas existe en el DataFrame.",
            "",
            "",
        )

    # Preparar datos para el modelo
    cols_req = cols_existentes + ["productos_vendidos"]
    df_model = df_prediccion[cols_req].dropna().copy()

    if df_model.empty:
        return "No hay datos para entrenar con esas variables.", "", ""

    if len(df_model) < 30:
        return "Se necesitan al menos 30 días de datos para entrenar el modelo.", "", ""

    # División temporal: usar últimos 7 días para test
    df_model = df_model.sort_values("order_date") if "order_date" in df_model.columns else df_model
    test_size = min(7, len(df_model) // 4)  # Usar 7 días o 25% de los datos para test
    
    train_df = df_model.iloc[:-test_size]
    test_df = df_model.iloc[-test_size:]

    if train_df.empty or test_df.empty:
        return (
            "No hay suficientes datos para entrenar o testear con esta selección.",
            "",
            "",
        )

    try:
        # Configurar PyCaret para predicción de cantidad de productos vendidos
        setup(
            data=train_df,
            target="productos_vendidos",
            session_id=123,
            fold_strategy="timeseries" if len(train_df) > 10 else "kfold",
            data_split_shuffle=False,
            fold_shuffle=False,
            verbose=False,
            train_size=0.8
        )
        
        # Crear y entrenar el modelo
        model_trained = create_model(modelo)

        # Predicciones sobre test
        pred = predict_model(model_trained, data=test_df)
        y_true = pred["productos_vendidos"].values
        y_pred = pred["prediction_label"].values

        # Calcular métricas
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
                                html.P("Error Absoluto Medio", className="card-text", style={"fontSize": "12px"}),
                                html.H3(f"{mae:.2f}", className="card-text"),
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
                                html.P("Raíz del Error Cuadrático Medio", className="card-text", style={"fontSize": "12px"}),
                                html.H3(f"{rmse:.2f}", className="card-text"),
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
                                html.H5("R²", className="card-title"),
                                html.P("Coeficiente de Determinación", className="card-text", style={"fontSize": "12px"}),
                                html.H3(f"{r2:.3f}", className="card-text"),
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
        try:
            plot_model(model_trained, plot="feature", save=True)
            feature_img_path = "Feature Importance.png"
            if os.path.exists(feature_img_path):
                with open(feature_img_path, "rb") as f:
                    encoded_feature = base64.b64encode(f.read()).decode()
                importancia_img = html.Img(
                    src="data:image/png;base64," + encoded_feature,
                    style={"width": "45%", "maxWidth": "400px", "margin": "10px"},
                    alt="Importancia Variables",
                )
                os.remove(feature_img_path)
            else:
                importancia_img = html.Div("No se pudo generar gráfico de importancia.")
        except:
            importancia_img = html.Div("No se pudo generar gráfico de importancia.")

        # Gráfico residual
        try:
            plot_model(model_trained, plot="residuals", save=True)
            residual_img_path = "Residuals.png"
            if os.path.exists(residual_img_path):
                with open(residual_img_path, "rb") as image_file:
                    encoded_residual = base64.b64encode(image_file.read()).decode()
                residual_img = html.Img(
                    src="data:image/png;base64," + encoded_residual,
                    style={"width": "45%", "maxWidth": "400px", "margin": "10px"},
                    alt="Gráfico Residual",
                )
                os.remove(residual_img_path)
            else:
                residual_img = html.Div("No se pudo generar el gráfico residual.")
        except:
            residual_img = html.Div("No se pudo generar el gráfico residual.")

        combined_graphs = html.Div(
            [importancia_img, residual_img],
            style={"display": "flex", "justifyContent": "center", "flexWrap": "wrap"},
        )

        # Preparar datos para gráfico de predicciones vs reales
        # Agregar fechas si están disponibles
        if "order_date" in test_df.columns:
            pred["order_date"] = test_df["order_date"].values
            x_axis = "order_date"
            x_label = "Fecha"
        else:
            pred["day_index"] = range(len(pred))
            x_axis = "day_index"
            x_label = "Día"
        
        # Crear gráfico de predicciones vs reales
        fig_pred = px.line(
            pred,
            x=x_axis,
            y=["productos_vendidos", "prediction_label"],
            labels={"value": "Productos Vendidos", x_axis: x_label},
            title="Productos Vendidos: Valores Reales vs Predichos",
            markers=True,
            color_discrete_map={
                "productos_vendidos": "blue",
                "prediction_label": "red"
            }
        )
        
        # Personalizar la leyenda
        fig_pred.for_each_trace(
            lambda trace: trace.update(
                name="Real" if trace.name == "productos_vendidos" else "Predicho"
            )
        )
        
        fig_pred.update_layout(
            xaxis_title=x_label,
            yaxis_title="Productos Vendidos",
            legend_title="Tipo"
        )

        return (
            kpi_cards,
            combined_graphs,
            dcc.Graph(figure=fig_pred),
        )
    
    except Exception as e:
        return (
            f"Error al entrenar el modelo: {str(e)}",
            "",
            "",
        )

if __name__ == "__main__":
    app.run(debug=True)