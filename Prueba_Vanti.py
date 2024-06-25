#!/usr/bin/env python
# coding: utf-8

# <div style="text-align: center;">
#     <img src="vanti-logo.png" alt="Logo" style="width:400px;">
#     <h1>Prueba técnica - Grupo Vanti</h1>
#     <h2>Elizabet Cano Mejía</h2>
# </div>

# El desafío que se trabajará en este *Notebook* consiste en segmentar una muestra de clientes del Grupo Vanti en función de su consumo, patrones de uso y características demográficas para desarrollar estrategias personalizadas que aumenten la satisfacción y la rentabilidad. 
# 
# El desarrollo del modelo se presentará a través de los siguientes capítulos:
# 
# 1. Carga y exploración de los datos
# 2. Preprocesamiento y limpieza de los datos
# 3. Análisis de datos atípicos
# 4. Agrupación de categorías minoritarias
# 4. Análisis Factorial de Datos Mixtos - FAMD 
# 5. Segmentación con *Kmeans* utilizando componentes de FAMD
# 6. Resultados finales 
# 
# Este enfoque metodológico proporcionará una guía clara para abordar el desafío de segmentación de clientes, facilitando la implementación de estrategias efectivas y personalizadas dentro del Grupo Vanti.

# # 1. Carga y exploración de los datos
# En este primer capítulo, abordaremos la carga de la base de datos en formato *csv*, el cual contiene la información necesaria para desarrollar el modelo de segmentación de clientes. A continuación, se describe el proceso y el código utilizado para lograr esta tarea.

# In[1]:


# Librerías a utilizar 
import pandas as pd
import numpy as np 
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Carga de datos
datos = pd.read_csv('data_challenge.csv')
datos


# ## 2. Preprocesamiento y limpieza de los datos
# En este capítulo, nos enfocaremos en preparar y limpiar los datos para asegurar que el conjunto esté óptimamente estructurado para el análisis y modelado subsiguiente. A través de una serie de pasos estructurados, eliminaremos registros duplicados, convertiremos variables numéricas en categóricas, y generaremos características agregadas. Estos procedimientos son cruciales para mejorar la calidad de los datos y asegurar una fase de modelado efectiva.

# ### 2.1 Eliminación de duplicados
# Durante la exploración inicial de los datos, se identificó la presencia de filas duplicadas que afectan todas las variables. Por lo tanto, se procede a eliminar estos registros duplicados para garantizar la integridad y la precisión de los datos.

# In[3]:


# Eliminar duplicados en todas las columnas
datos = datos.drop_duplicates()
datos


# ### 2.2. Agregación de características mensuales
# Dado que la información de cada cuenta se reporta mensualmente, procedemos a agregarla utilizando la mediana y la moda de cada variable. Este enfoque nos permite consolidar la información en un único registro por cliente o cuenta, facilitando así la segmentación y el análisis posterior.

# In[4]:


# Número de datos por cuentas o clientes
datos["cuenta"].value_counts()


# In[5]:


# Definir funciones de agregación
agg_functions = {
    'consumo': 'median',
    'categoria_cliente': lambda x: x.mode()[0] if not x.mode().empty else 'Desconocido',
    'porcion': lambda x: x.mode()[0] if not x.mode().empty else 'Desconocido',
    'dias_fact': 'median',
    'descrip_poblac_suministro': lambda x: x.mode()[0] if not x.mode().empty else 'Desconocido',
}

# Agrupar los datos
datos_agrupados = datos.groupby('cuenta').agg(agg_functions).reset_index()
datos_agrupados


# In[6]:


# Revisión de datos nulos y estructura de datos
datos_agrupados.info()


# ### 2.3 Cambio de columna numérica a categórica 
# La variable "categoría del cliente" está actualmente codificada como numérica. Por su naturaleza, esta variable no representa un valor numérico sino una categoría específica. Por lo tanto, procedemos a convertirla al tipo de dato objeto (o string) para reflejar su verdadera naturaleza categórica en el análisis.

# In[7]:


# Cambio de variable a categórica
datos_agrupados['categoria_cliente'] = datos_agrupados['categoria_cliente'].astype('object')


# In[8]:


# Se muestra las variables con el tipo correcto 
datos_agrupados.info()


# ### 2.4 Definición de columna como índice de la tabla
# La columna "cuenta" contiene un código único para cada cliente, por lo que se establece como el índice del DataFrame. Esto se realiza porque dicho código no representa una variable en sí misma para el análisis, sino una identificación única de cada registro en la tabla.

# In[9]:


# Se define la variable cuenta como índice de la tabla
datos_agrupados.set_index('cuenta', inplace=True)


# In[10]:


# Muestra de datos
datos_agrupados


# ### 2.5 Cambio de nombres de columnas
# Con el fin de mejorar la claridad y la interpretación de cada columna, procedemos a cambiar sus nombres por otros más descriptivos y comprensibles.

# In[11]:


# Definir un diccionario con los nuevos nombres
nuevos_nombres = {
    'consumo': 'Consumo_gas',
    'categoria_cliente': 'Categoria_cliente',
    'porcion': 'Porcion_cliente',
    'dias_fact': 'Dias_facturados',
    'descrip_poblac_suministro': 'Ciudad'
}

# Cambiar los nombres de las columnas
datos_agrupados.rename(columns=nuevos_nombres, inplace=True)
datos_agrupados


# ## 3. Análisis de datos atípicos
# Se observa la presencia de datos atípicos en las variables numéricas. Para evitar que estos valores influyan negativamente en la segmentación, se procede a separarlos del conjunto general. Estos clientes atípicos pueden tratarse como un grupo adicional en la segmentación, permitiendo estrategias de activación y tratamiento diferenciadas. La separación de datos atípicos es crucial para mantener la precisión y efectividad del análisis.

# In[12]:


# Se percibe la presencia de datos atípicos
datos_agrupados.describe()


# In[13]:


import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))

# Boxplot para Consumo de Gas
plt.subplot(1, 2, 1)
plt.boxplot(datos_agrupados['Consumo_gas'], patch_artist=True)
plt.title('Boxplot de Consumo de Gas')
plt.ylabel('Valor')

# Boxplot para Días Facturados
plt.subplot(1, 2, 2)
plt.boxplot(datos_agrupados['Dias_facturados'], patch_artist=True)
plt.title('Boxplot de Días Facturados')
plt.ylabel('Valor')

plt.tight_layout()
plt.show()


# Se procede a identificar y separar los clientes o cuentas atípicas mediante el uso del rango intercuartílico (IQR). Este enfoque estadístico nos permite detectar y aislar los valores extremos que podrían distorsionar los resultados del análisis. De esta manera, aseguramos que la segmentación sea más precisa y representativa de la mayoría de los clientes.

# In[14]:


variables_interes = ['Consumo_gas', 'Dias_facturados']

# Calcular Q1 y Q3 para cada variable
Q1 = datos_agrupados[variables_interes].quantile(0.25)
Q3 = datos_agrupados[variables_interes].quantile(0.75)

# Calcular IQR
IQR = Q3 - Q1

# Definir los límites para identificar outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identificar outliers para cada variable
outliers_gas = datos_agrupados[(datos_agrupados['Consumo_gas'] < lower_bound['Consumo_gas']) | (datos_agrupados['Consumo_gas'] > upper_bound['Consumo_gas'])]
outliers_dias = datos_agrupados[(datos_agrupados['Dias_facturados'] < lower_bound['Dias_facturados']) | (datos_agrupados['Dias_facturados'] > upper_bound['Dias_facturados'])]

# Unificar todos los outliers
outliers = pd.concat([outliers_gas, outliers_dias]).drop_duplicates()

# Mostrar los outliers identificados
print("Datos Atípicos:")
print(outliers)

# Eliminar outliers del DataFrame original
datos_limpios = datos_agrupados.drop(outliers.index)

# Mostrar el DataFrame sin outliers
print("\nDataFrame sin Outliers:")
print(datos_limpios)


# In[15]:


# Crear los boxplots en una sola fila y tres columnas
plt.figure(figsize=(15, 6))

# Boxplot para Consumo de Gas
plt.subplot(1, 2, 1)
plt.boxplot(datos_limpios['Consumo_gas'], patch_artist=True)
plt.title('Boxplot de Consumo de Gas')
plt.ylabel('Valor')

# Boxplot para Días Facturados
plt.subplot(1, 2, 2)
plt.boxplot(datos_limpios['Dias_facturados'], patch_artist=True)
plt.title('Boxplot de Días Facturados')
plt.ylabel('Valor')

# Ajustar el espaciado entre subplots
plt.tight_layout()

# Mostrar los gráficos
plt.show()


# A partir de los gráficos anteriores, se puede observar una notable mejora en la distribución de las variables al eliminar los datos atípicos. Esto nos permite continuar el análisis con mayor precisión y fiabilidad.

# ## 4. Agrupación de categorías minoritarias
# Para simplificar el análisis y reducir la cantidad de variables dummy necesarias, las categorías con menos datos se agruparon en una única categoría denominada "Otros". Este proceso ayuda a evitar la creación de múltiples variables dummy, lo que podría complicar el modelo y dificultar su interpretación. Al consolidar estas categorías menos representativas, mejoramos la eficiencia y la manejabilidad del análisis.

# In[16]:


# Se observa que hay categorías con pocos clientes
datos_limpios["Categoria_cliente"].value_counts()


# In[17]:


categorias_a_reemplazar = [50, 60, 80, 85]

# Reemplazar las categorías especificadas por 0
datos_limpios['Categoria_cliente'] = datos_limpios['Categoria_cliente'].replace(categorias_a_reemplazar, 0)

# Mostrar el DataFrame actualizado
datos_limpios['Categoria_cliente'].value_counts()  # Para verificar los valores después del reemplazo
datos_limpios['Categoria_cliente'] = datos_limpios['Categoria_cliente'].astype('object')


# In[18]:


# Se observa que hay ciudades con muy pocos clientes
datos_limpios["Ciudad"].value_counts()


# In[19]:


categorias_permitidas = {
    'BOGOTA': 4717,
    'SOACHA': 532,
    'BUCARAMANGA': 358,
    'TUNJA': 151
}

categorias_permitidas_lista = list(categorias_permitidas.keys())

mapeo_ciudad = {ciudad: ciudad if ciudad in categorias_permitidas_lista else 'Otras' for ciudad in datos_limpios['Ciudad'].unique()}

# Reemplazar las categorías en la columna 'Ciudad' usando el diccionario de mapeo
datos_limpios['Ciudad'] = datos_limpios['Ciudad'].replace(mapeo_ciudad)

# Mostrar el DataFrame actualizado
print(datos_limpios['Ciudad'].value_counts())


# ## 5. Análisis Factorial de Datos Mixtos (FAMD)
# En este capítulo, se lleva a cabo un Análisis Factorial de Datos Mixtos (FAMD) con el propósito de manejar simultáneamente variables categóricas y numéricas. La reducción de dimensionalidad previa a la segmentación ofrece múltiples ventajas, tales como:
# 
# - **Mejora de la eficiencia computacional:** Al reducir el número de variables, el tiempo de procesamiento y los recursos necesarios para el análisis disminuyen significativamente.
# - **Reducción del ruido:** La eliminación de variables redundantes o irrelevantes ayuda a enfocarse en la información más relevante, mejorando la calidad del modelo.
# - **Facilitación de la visualización:** Con menos dimensiones, es más sencillo visualizar y entender las relaciones entre las variables y los grupos de datos.
# - **Mitigación del problema de la multicolinealidad:** La reducción de dimensionalidad ayuda a prevenir la multicolinealidad, que puede afectar negativamente el rendimiento de los modelos de segmentación.
# - **Mejora de la interpretabilidad:** Un modelo con menos variables es más fácil de interpretar y explicar, lo cual es crucial para la toma de decisiones basada en datos.
# 
# Estos beneficios permiten una segmentación más efectiva y precisa, optimizando el análisis y la interpretación de los resultados.

# In[20]:


import prince

famd = prince.FAMD(
    n_components=28,
    n_iter=5,
    copy=True,
    check_input=True,
    random_state=42,
    engine="sklearn",
    handle_unknown="error"  # same parameter as sklearn.preprocessing.OneHotEncoder
)
famd = famd.fit(datos_limpios)


# In[21]:


import matplotlib.pyplot as plt

# Extraer los valores propios
varianza = famd.eigenvalues_summary["% of variance (cumulative)"]

# Crear un gráfico de barras para los valores propios
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(varianza) + 1), varianza, color='skyblue')
plt.xlabel('Componentes')
plt.ylabel('Varianza Explicada Acumulada')
plt.title('Gráfico de Varianza Explicada')
plt.xticks(range(1, len(varianza) + 1))
plt.grid(axis='y')

# Mostrar el gráfico
plt.show()


# In[22]:


# Valores propios y varianza explicada por cada componente de FAMD
famd.eigenvalues_summary


# In[23]:


# Tabla con coordenadas de cada cuenta o cliente en cada componente del FAMD
# Estas son las componentes que entrarían como insumo al kmeans
famd.row_coordinates(datos_limpios)


# In[24]:


# Extraer las contribuciones de las columnas a los componentes
column_contributions = famd.column_contributions_

# Crear un DataFrame con las contribuciones para los primeros 2 componentes
contributions_df = pd.DataFrame(column_contributions).iloc[:, :2]

# Graficar las contribuciones de las columnas para el primer componente
plt.figure(figsize=(10, 6))
contributions_df[0].plot(kind='bar', color='skyblue')
plt.xlabel('Variables')
plt.ylabel('Contribución')
plt.title('Contribuciones de las columnas a la primera componente')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')

plt.show()

# Graficar las contribuciones de las columnas para el segundo componente
plt.figure(figsize=(10, 6))
contributions_df[1].plot(kind='bar', color='skyblue')
plt.xlabel('Variables')
plt.ylabel('Contribución')
plt.title('Contribuciones de las columnas a la segunda componente')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')

plt.show()


# In[25]:


# Contribución de cada variable a la creación de cada componente
famd.column_contributions_.style.format('{:.0%}')


# ## 6. Segmentación con *Kmeans* utilizando componentes de FAMD
# En este capítulo, se exploran los resultados de la segmentación utilizando las dos primeras componentes del Análisis Factorial de Datos Mixtos (FAMD) como variables de entrada para el algoritmo *Kmeans*.
# 
# El objetivo principal es agrupar a los clientes en clusters homogéneos basados en patrones emergentes de comportamiento y características demográficas. Utilizando las primeras componentes del FAMD, se busca simplificar la representación de las características subyacentes de los clientes.

# In[26]:


# Obtener las coordenadas de las filas en el espacio factorial
coordenadas_filas = famd.row_coordinates(datos_limpios)

# Seleccionar solo las primeras 2 componentes principales
primeras_componentes = coordenadas_filas.iloc[:, :2]


# In[27]:


import matplotlib.pyplot as plt
from prince import FAMD
from sklearn.cluster import KMeans

# Calcular la suma de las distancias cuadradas intra-cluster para diferentes números de clusters
inertia = []
for k in range(1, 11):  
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(primeras_componentes)
    inertia.append(kmeans.inertia_)

# Graficar la curva del codo
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.xlabel('Número de Clústeres')
plt.ylabel('Inertia')
plt.title('Método del Codo para K-Means')
plt.grid(True)
plt.show()


# In[28]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42)  
kmeans.fit(primeras_componentes)

# Obtener las etiquetas de cluster asignadas
etiquetas_clusters = kmeans.labels_

# Añadir las etiquetas de cluster de vuelta al DataFrame original 
datos_limpios['Cluster'] = etiquetas_clusters

print(datos_limpios['Cluster'].value_counts())  # Para ver la distribución de los clusters


# In[29]:


from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(primeras_componentes, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg}")


# In[30]:


inertia = kmeans.inertia_
print(f"Inertia: {inertia}")


# In[31]:


import matplotlib.pyplot as plt
import seaborn as sns
df_plot = pd.DataFrame({
    'Componente 1': primeras_componentes[0],  # Primer componente del FAMD
    'Componente 2': primeras_componentes[1],  # Segundo componente del FAMD
    'Cluster': etiquetas_clusters  # Etiquetas de los clústeres asignadas por K-Means
})

# Graficar utilizando seaborn
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Componente 1', y='Componente 2', hue='Cluster', data=df_plot, palette='viridis', s=80, alpha=0.8)
plt.title('Gráfico de Componentes FAMD coloreados por Clúster')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.legend(title='Clúster')
plt.grid(True)
plt.show()


# # 7. Resultados finales
# En este capítulo final, se presenta un análisis detallado que muestra la base de datos original enriquecida con los resultados obtenidos, incluyendo las características promedio y moda para las variables originales. Este enfoque nos permite visualizar cómo se estructuran los datos después de aplicar el proceso de segmentación y análisis.

# In[32]:


# Tabla final
datos_limpios


# In[33]:


# Media de Consumo_gas y Dias_facturados por Cluster
medias = datos_limpios.groupby('Cluster')[['Consumo_gas', 'Dias_facturados']].mean().reset_index()

# Moda de Categoria_cliente, Porcion_cliente y Ciudad por Cluster
modas = datos_limpios.groupby('Cluster')[['Categoria_cliente', 'Porcion_cliente', 'Ciudad']].agg(lambda x: x.mode().iloc[0]).reset_index()

# Combinar resultados en un solo DataFrame por Cluster
resumen_por_cluster = pd.merge(medias, modas, on='Cluster')

# Mostrar resultados
print("Resumen por Cluster:")
print(resumen_por_cluster)


# # Conclusiones y recomendaciones

# Este análisis ha permitido realizar una segmentación efectiva de una muestra de cuentas o clientes del Grupo Vanti. La reducción de dimensionalidad mediante el Análisis Factorial de Datos Mixtos (FAMD) ha proporcionado un resultado inicial satisfactorio al simplificar la representación de las variables.
# 
# Principales conclusiones:
# - Segmentación efectiva: La aplicación del FAMD ha facilitado la agrupación de clientes en clusters homogéneos, permitiendo identificar patrones de comportamiento y características demográficas relevantes.
# 
# - Próximos pasos sugeridos:
#     - Explorar otros algoritmos: Considerar la implementación de métodos como *kmodes* debido a la presencia significativa de variables categóricas en los datos.
#     - Enriquecimiento del conjunto de variables: Incorporar más variables relevantes para enriquecer el análisis y capturar aspectos adicionales del comportamiento del cliente.
#     - Experimentar con métodos de agrupación adicionales: Evaluar la aplicación de otros métodos de agrupación que puedan ser más adecuados para variables mensuales y su comportamiento a lo largo del tiempo.
# 
# Recomendaciones finales:
# - Continuar explorando y refinando el modelo de segmentación para mejorar la precisión y la interpretación de los resultados.
# - Realizar pruebas adicionales con diferentes configuraciones de algoritmos y variables para validar y robustecer el análisis.
# - Implementar estrategias personalizadas basadas en los perfiles identificados para aumentar la satisfacción y la retención de clientes.
# 
# Estas recomendaciones buscan optimizar el análisis y la segmentación de clientes, promoviendo una estrategia más eficaz y adaptativa en el contexto del grupo Vanti.
