## FUNCIONES DE UTILIDAD PARA EL ETL Y EDA
# Importaciones
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def verifica_duplicados_por_columna(df, columna):
    """
    Verifica y muestra filas duplicadas en un DataFrame basado en una columna específica.

    Esta función toma como entrada un DataFrame y el nombre de una columna específica.
    Luego, identifica las filas duplicadas basadas en el contenido de la columna especificada,
    las filtra y las ordena para una comparación más sencilla.

    Parameters:
        df (pandas.DataFrame): El DataFrame en el que se buscarán filas duplicadas.
        columna (str): El nombre de la columna basada en la cual se verificarán las duplicaciones.

    Returns:
        pandas.DataFrame or str: Un DataFrame que contiene las filas duplicadas filtradas y ordenadas,
        listas para su inspección y comparación, o el mensaje "No hay duplicados" si no se encuentran duplicados.
    """
    # Se filtran las filas duplicadas
    duplicated_rows = df[df.duplicated(subset=columna, keep=False)]
    if duplicated_rows.empty:
        return "No hay duplicados"

    # se ordenan las filas duplicadas para comparar entre sí
    duplicated_rows_sorted = duplicated_rows.sort_values(by=columna)
    return duplicated_rows_sorted


def verificar_tipo_variable(df):
    """
    Realiza un análisis de los tipos de datos y la presencia de valores nulos en un DataFrame.

    Esta función toma un DataFrame como entrada y devuelve un resumen que incluye información sobre
    los tipos de datos en cada columna.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        pandas.DataFrame: Un DataFrame que contiene el resumen de cada columna, incluyendo:
        - 'nombre_campo': Nombre de cada columna.
        - 'tipo_datos': Tipos de datos únicos presentes en cada columna.
    """

    mi_dict = {"nombre_campo": [], "tipo_datos": []}

    for columna in df.columns:
        mi_dict["nombre_campo"].append(columna)
        mi_dict["tipo_datos"].append(df[columna].apply(type).unique())
    df_info = pd.DataFrame(mi_dict)

    return df_info


def convertir_a_time(x):
    """
    Convierte un valor a un objeto de tiempo (time) de Python si es posible.

    Esta función acepta diferentes tipos de entrada y trata de convertirlos en objetos de tiempo (time) de Python.
    Si la conversión no es posible, devuelve None.

    Parameters:
        x (str, datetime, or any): El valor que se desea convertir a un objeto de tiempo (time).

    Returns:
        datetime.time or None: Un objeto de tiempo (time) de Python si la conversión es exitosa,
        o None si no es posible realizar la conversión.
    """
    if isinstance(x, str):
        try:
            return datetime.strptime(x, "%H:%M:%S").time()
        except ValueError:
            return None
    elif isinstance(x, datetime):
        return x.time()
    return x


def imputa_valor_frecuente(df, columna):
    """
    Imputa los valores faltantes en una columna de un DataFrame con el valor más frecuente.

    Esta función reemplaza los valores "SD" con NaN en la columna especificada,
    luego calcula el valor más frecuente en esa columna y utiliza ese valor
    para imputar los valores faltantes (NaN).

    Parameters:
        df (pandas.DataFrame): El DataFrame que contiene la columna a ser imputada.
        columna (str): El nombre de la columna en la que se realizará la imputación.

    Returns:
        None
    """
    # Se reemplaza "SD" con NaN en la columna
    df[columna] = df[columna].replace("SD", pd.NA)

    # Se calcula el valor más frecuente en la columna
    valor_mas_frecuente = df[columna].mode().iloc[0]
    print(f"El valor mas frecuente es: {valor_mas_frecuente}")

    # Se imputan los valores NaN con el valor más frecuente
    df[columna].fillna(valor_mas_frecuente, inplace=True)


def verificar_tipo_datos_y_nulos(df):
    """
    Realiza un análisis de los tipos de datos y la presencia de valores nulos en un DataFrame.

    Esta función toma un DataFrame como entrada y devuelve un resumen que incluye información sobre
    los tipos de datos en cada columna, el porcentaje de valores no nulos y nulos, así como la
    cantidad de valores nulos por columna.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        pandas.DataFrame: Un DataFrame que contiene el resumen de cada columna, incluyendo:
        - 'nombre_campo': Nombre de cada columna.
        - 'tipo_datos': Tipos de datos únicos presentes en cada columna.
        - 'no_nulos_%': Porcentaje de valores no nulos en cada columna.
        - 'nulos_%': Porcentaje de valores nulos en cada columna.
        - 'nulos': Cantidad de valores nulos en cada columna.
    """

    mi_dict = {
        "nombre_campo": [],
        "tipo_datos": [],
        "no_nulos_%": [],
        "nulos_%": [],
        "nulos": [],
    }

    for columna in df.columns:
        porcentaje_no_nulos = (df[columna].count() / len(df)) * 100
        mi_dict["nombre_campo"].append(columna)
        mi_dict["tipo_datos"].append(df[columna].apply(type).unique())
        mi_dict["no_nulos_%"].append(round(porcentaje_no_nulos, 2))
        mi_dict["nulos_%"].append(round(100 - porcentaje_no_nulos, 2))
        mi_dict["nulos"].append(df[columna].isnull().sum())

    df_info = pd.DataFrame(mi_dict)

    return df_info.sort_values(ascending=False, by="nulos_%")


def graficos_eda_categoricos(cat):
    """
    Realiza gráficos de barras horizontales para explorar datos categóricos.

    Parámetros:
    - cat (DataFrame): DataFrame que contiene variables categóricas a visualizar.

    Retorna:
    - None: La función solo genera gráficos y no devuelve valores.

    La función toma un DataFrame con variables categóricas y genera gráficos de barras horizontales
    para visualizar la distribución de categorías en cada variable. Los gráficos se organizan en
    filas y columnas para facilitar la visualización.
    """
    # Calculamos el número de filas que necesitamos
    from math import ceil

    filas = ceil(cat.shape[1] / 2)

    # Definimos el gráfico
    f, ax = plt.subplots(nrows=filas, ncols=2, figsize=(16, filas * 6))

    # Aplanamos para iterar por el gráfico como si fuera de 1 dimensión en lugar de 2
    ax = ax.flat

    # Creamos el bucle que va añadiendo gráficos
    for cada, variable in enumerate(cat):
        cat[variable].value_counts().plot.barh(ax=ax[cada])
        ax[cada].set_title(variable, fontsize=12, fontweight="bold")
        ax[cada].tick_params(labelsize=12)


def estadisticos_cont(num):
    """
    Calcula estadísticas descriptivas para variables numéricas.

    Parámetros:
    - num (DataFrame o Series): Datos numéricos para los cuales se desean calcular estadísticas.

    Retorna:
    - DataFrame: Un DataFrame que contiene estadísticas descriptivas, incluyendo la media, la desviación estándar,
      los percentiles, el mínimo, el máximo y la mediana.

    La función toma datos numéricos y calcula estadísticas descriptivas, incluyendo la media, desviación estándar,
    percentiles (25%, 50%, 75%), mínimo, máximo y mediana. Los resultados se presentan en un DataFrame organizado
    para una fácil interpretación.

    Nota:
    - El DataFrame de entrada debe contener solo variables numéricas para obtener resultados significativos.
    """
    # Calculamos describe
    estadisticos = num.describe().T
    # Añadimos la mediana
    estadisticos["median"] = num.median()
    # Reordenamos para que la mediana esté al lado de la media
    estadisticos = estadisticos.iloc[:, [0, 1, 8, 2, 3, 4, 5, 6, 7]]
    # Lo devolvemos
    return estadisticos
