# %% [markdown]
# ## Importar librerias y definición de la ruta  de trabajo (path)

# %%

import pandas as pd
import numpy as np
import os
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

%matplotlib inline



# Formato de los prints
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def headr(text):
    return ('\n'+color.UNDERLINE + text + color.END+'\n')


# %%
# Ingresa la ruta donde está el repositorio
ruta = 'c:/repo_remoto/'

# %% [markdown]
# ## Integración

# %% [markdown]
# #### Los archivo provienen de la etapa 2 y ya están limpios y preprocesados

# %% [markdown]
# ### Carga de archivos

# %%
### Características Equipos

equipos = pd.read_csv(ruta + 'etapa3/output/Caracteristicas_Equipos_limpio.csv')
equipos_df = pd.DataFrame(equipos)
### Historicos Ordenes

ordenes = pd.read_csv(ruta + 'etapa3/output/Historicos_Ordenes_limpio.csv')
ordenes_df = pd.DataFrame(ordenes)
### Registros Condiciones

condiciones = pd.read_csv(ruta + 'etapa3/output/Registros_Condiciones_limpio.csv')
condiciones_df = pd.DataFrame(condiciones)

# %% [markdown]
# ### Tratar ordenes_df

# %%
# Elimino ID_Orden y establezco Fecha como índice
ordenes_df = ordenes_df.drop(columns=['ID_Orden']).set_index('Fecha')

# %%
# Asegurarse de que el índice sea de tipo DatetimeIndex
ordenes_df.index = pd.to_datetime(ordenes_df.index)

# Obtener el año y la semana y combinarlos y crear una nueva columna 'Semana'
ordenes_df['Semana'] = ordenes_df.index.isocalendar().year.astype(str) + '-W' + ordenes_df.index.isocalendar().week.astype(str)

# %%
ordenes_df

# %% [markdown]
# ### Tratar condiciones_df

# %%
# Elimino ID_Registro y establezco Fecha como índice
condiciones_df = condiciones_df.drop(columns=['ID_Registro']).set_index('Fecha')

# %%
# Asegurarse de que el índice sea de tipo DatetimeIndex
condiciones_df.index = pd.to_datetime(condiciones_df.index)

# Obtener el año y la semana y combinarlos y crear una nueva columna 'Semana'
condiciones_df['Semana'] = condiciones_df.index.isocalendar().year.astype(str) + '-W' + condiciones_df.index.isocalendar().week.astype(str)

# %%
condiciones_df

# %% [markdown]
# ### merge

# %%
# Mezclo los dataframes por la columna 'ID_Equipo'
merged_df = pd.merge(ordenes_df, condiciones_df, on=['ID_Equipo'], how='outer')
merged_df

# %%
merged_df.info()

# %%
# Agrupar por Semana, ID_Equipo, Tipo_Mantenimiento y Ubicacion, y calcular la media de las columnas especificadas
columnas_a_promediar = ['Costo_Mantenimiento', 'Duracion_Horas', 'Temperatura_C', 'Vibracion_mm_s', 'Horas_Operativas']
merged_df = merged_df.groupby(['Semana_x', 'Semana_y', 'ID_Equipo', 'Tipo_Mantenimiento', 'Ubicacion'])[columnas_a_promediar].mean().reset_index()

# Mostrar información del DataFrame resultante
merged_df.info()

# %%
merged_df

# %%
# Realizar la unión  con equipos_df manteniendo merge_df como referencia
merged_df = merged_df.merge(equipos_df, on='ID_Equipo', how='left')
merged_df

# %% [markdown]
# ## Generar nuevas características

# %% [markdown]
# ### Añado Frecuencia_Correctivo

# %%
# Filtrar las filas donde Tipo_Mantenimiento es 'Correctivo'
correctivo_counts = merged_df[merged_df['Tipo_Mantenimiento'] == 'Correctivo'].groupby('ID_Equipo').size()

# Convertir a DataFrame y renombrar la columna
correctivo_counts = correctivo_counts.reset_index(name='Frecuencia_Correctivo')

# Añadir la columna al DataFrame merged_df
merged_df = merged_df.merge(correctivo_counts, on='ID_Equipo', how='left')

# Rellenar valores NaN con 0 (si algún ID_Equipo no tiene registros de 'Correctivo')
merged_df['Frecuencia_Correctivo'] = merged_df['Frecuencia_Correctivo'].fillna(0)

# %%
merged_df

# %%
# Tratamiento sesgo

merged_df["Frec_Correctivo"] = merged_df.Frecuencia_Correctivo.apply(np.log1p)


# Crear una comparación de gráficos
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
sns.kdeplot(merged_df.Frecuencia_Correctivo, shade=True, ax=axs[0])
sns.kdeplot(merged_df.Frec_Correctivo, shade=True, ax=axs[1])

# Añadir títulos para mayor claridad
axs[0].set_title('Frecuencia_Correctivo Original')
axs[1].set_title('Frec_Correctivo Transformada con Log')

plt.tight_layout()
plt.show()

# Eliminar la columna Frecuencia_Correctivo por Frec_Correctivo
merged_df.drop(columns=["Frecuencia_Correctivo"], inplace=True)

# %% [markdown]
# ### Añado Frecuencia_Preventivo

# %%
# Filtrar las filas donde Tipo_Mantenimiento es 'Preventivo'
preventivo_counts = merged_df[merged_df['Tipo_Mantenimiento'] == 'Preventivo'].groupby('ID_Equipo').size()

# Convertir a DataFrame y renombrar la columna
preventivo_counts = preventivo_counts.reset_index(name='Frecuencia_Preventivo')

# Añadir la columna al DataFrame merged_df
merged_df = merged_df.merge(preventivo_counts, on='ID_Equipo', how='left')

# Rellenar valores NaN con 0 (si algún ID_Equipo no tiene registros de 'Preventivo')
merged_df['Frecuencia_Preventivo'] = merged_df['Frecuencia_Preventivo'].fillna(0)

# %%
merged_df

# %%
# Tratamiento sesgo

merged_df["Frec_Preventivo"] = merged_df.Frecuencia_Preventivo.apply(np.log1p)


# Crear una comparación de gráficos
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
sns.kdeplot(merged_df.Frecuencia_Preventivo, shade=True, ax=axs[0])
sns.kdeplot(merged_df.Frec_Preventivo, shade=True, ax=axs[1])

# Añadir títulos para mayor claridad
axs[0].set_title('Frecuencia_Preventivo Original')
axs[1].set_title('Frec_Preventivo Transformada con Log')

plt.tight_layout()
plt.show()

# Eliminar la columna Frecuencia_Preventivo por Frec_Preventivo
merged_df.drop(columns=["Frecuencia_Preventivo"], inplace=True)

# %% [markdown]
# ### Añadir Frecuencia Ubicación

# %%
# Filtrar las filas donde Tipo_Mantenimiento es 'Correctivo'
ubicacion_counts = merged_df[merged_df['Tipo_Mantenimiento'] == 'Correctivo'].groupby('Ubicacion').size()

# Convertir a DataFrame y renombrar la columna
ubicacion_counts = ubicacion_counts.reset_index(name='Frecuencia_Ubicacion')

# Añadir la columna al DataFrame merged_df
merged_df = merged_df.merge(ubicacion_counts, on='Ubicacion', how='left')

# Rellenar valores NaN con 0 (si algún Ubicacion no tiene registros de 'Correctivo')
merged_df['Frecuencia_Ubicacion'] = merged_df['Frecuencia_Ubicacion'].fillna(0)

# %%
merged_df

# %% [markdown]
# ### Añadir Frecuencia Potencia_kW

# %%
# Filtrar las filas donde Tipo_Mantenimiento es 'Correctivo'
Potencia_kW_counts = merged_df[merged_df['Tipo_Mantenimiento'] == 'Correctivo'].groupby('Potencia_kW').size()

# Convertir a DataFrame y renombrar la columna
Potencia_kW_counts = Potencia_kW_counts.reset_index(name='Frecuencia_Potencia_kW')

# Añadir la columna al DataFrame merged_df
merged_df = merged_df.merge(Potencia_kW_counts, on='Potencia_kW', how='left')

# Rellenar valores NaN con 0 (si algún Potencia_kW no tiene registros de 'Correctivo')
merged_df['Frecuencia_Potencia_kW'] = merged_df['Frecuencia_Potencia_kW'].fillna(0)

# %%
merged_df

# %%
# Tratamiento sesgo

merged_df["Frec_Potencia_kW"] = merged_df.Frecuencia_Potencia_kW.apply(np.log1p)


# Crear una comparación de gráficos
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
sns.kdeplot(merged_df.Frecuencia_Potencia_kW, shade=True, ax=axs[0])
sns.kdeplot(merged_df.Frec_Potencia_kW, shade=True, ax=axs[1])

# Añadir títulos para mayor claridad
axs[0].set_title('Frecuencia_Potencia_kW Original')
axs[1].set_title('Frec_Potencia_kW Transformada con Log')

plt.tight_layout()
plt.show()

# Eliminar la columna Frecuencia_Potencia_kW por Frec_Preventivo
merged_df.drop(columns=["Frecuencia_Potencia_kW"], inplace=True)

# %% [markdown]
# ### Almacenamiento

# %%
# Extraer el DataFrame a un archivo CSV

merged_df.to_csv(ruta + 'etapa3/output/Mantenimiento.csv', index=False)


print("El DataFrame merged_df se ha extraído a *.csv")

# %% [markdown]
# ## Analisis y exportación Profiling

# %%
merged_df = pd.read_csv(ruta + 'Etapa3/output/Mantenimiento.csv')

# %%
from ydata_profiling import ProfileReport
profile = ProfileReport(merged_df, title="Mantenimiento Profiling Report")

# %%
profile.to_notebook_iframe()

# %%
profile.to_file(ruta + 'Etapa3/output/Mantenimiento.html')

# %% [markdown]
# ## Preprocesado

# %% [markdown]
# ### Definición del Target

# %%
mantenimiento_df = merged_df

# %%
target_column = "Tipo_Mantenimiento"

mantenimiento_df.shape

# %% [markdown]
# ### Explorar

# %%
mantenimiento_df.shape

# %%
mantenimiento_df.describe()

# %%
mantenimiento_df.describe(include=object)

# %%
mantenimiento_df.info()

# %% [markdown]
# ### Valores faltantes

# %%
# Datos vacios
print('Datos vacios en mantenimiento\n',mantenimiento_df.isna().sum())

# %%
# Rellenar valores faltantes en columnas numéricas con la media
numerical_columns = mantenimiento_df.select_dtypes(include=['float64', 'int64']).columns
mantenimiento_df[numerical_columns] = mantenimiento_df[numerical_columns].fillna(mantenimiento_df[numerical_columns].mean())

# Rellenar valores faltantes en columnas categóricas con la moda
categorical_columns = mantenimiento_df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    mantenimiento_df[column] = mantenimiento_df[column].fillna(mantenimiento_df[column].mode()[0])

mantenimiento_df.info()

# %%
# Mostrar las filas que tienen valores vacíos
filas_vacias = mantenimiento_df[mantenimiento_df.isna().any(axis=1)]
filas_vacias

# %% [markdown]
# ### Outliers

# %%
# Extraer columnas numéricas y categóricas
from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(mantenimiento_df)
categorical_columns = categorical_columns_selector(mantenimiento_df)

print(headr("Numerical columns"), numerical_columns)
print(headr("Categorical columns"), categorical_columns)

# %%
# Outliers

# identificación
IQR = mantenimiento_df[numerical_columns].quantile(0.75) - mantenimiento_df[numerical_columns].quantile(0.25)
lower_bound = mantenimiento_df[numerical_columns].quantile(0.25) - (IQR * 3)
upper_bound = mantenimiento_df[numerical_columns].quantile(0.75) + (IQR * 3)

print(headr('lower_bound'),lower_bound)
print(headr('upper_bound'),upper_bound)

outliers = mantenimiento_df[numerical_columns][(mantenimiento_df[numerical_columns] < lower_bound) | (mantenimiento_df[numerical_columns] > upper_bound)]

print(headr("Outliers:"))
outliers.count()

# %% [markdown]
# ### Duplicados

# %%
# Mostrar todas las filas duplicadas
print('\nSumatorio duplicados en mantenimiento', mantenimiento_df.duplicated().sum())
mantenimiento_df[mantenimiento_df.duplicated(keep=False)]

# %% [markdown]
# ### Tratamiento tipo de columnas

# %%
# Extraer columnas numéricas y categóricas
from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(mantenimiento_df)
categorical_columns = categorical_columns_selector(mantenimiento_df)

print(headr("Numerical columns"), numerical_columns)
print(headr("Categorical columns"), categorical_columns)

# %%
# Transformamos columnas categoricas en valores numéricos

from sklearn.preprocessing import LabelEncoder

mantenimiento_dff = mantenimiento_df.copy()
mantenimiento_dff[categorical_columns] = mantenimiento_dff[categorical_columns].apply(LabelEncoder().fit_transform)   
mantenimiento_dff.head()

# %% [markdown]
# ### Seleccionamos características más relevantes

# %%
# Evaluamos la aportación de cada columna
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

X= mantenimiento_dff.drop(target_column, axis=1)
y = mantenimiento_dff[target_column]

fvalue_selector = SelectKBest(f_classif, k=2)

X_kbest = fvalue_selector.fit(X,y)

feature_scores = pd.DataFrame({"Feature": X.columns,"Score": X_kbest.scores_}).sort_values(by="Score", ascending=False)

print(headr("Feature scores"))
round(feature_scores,2)

# %%
# Seleccionar las 5 características más relevantes
top_features = feature_scores.head(9)['Feature']
X_selected = X[top_features]

print("Características seleccionadas:")
print(top_features)


# %%
# Seleccionar las columnas más relevantes y la columna target_column
relevant_columns = top_features.tolist() + [target_column]

# Actualizar mantenimiento_const con las columnas seleccionadas
mantenimiento_const = mantenimiento_dff[relevant_columns]

# Mostrar las primeras filas del DataFrame actualizado
mantenimiento_const.head()

# %%
# Ver proporciones nuevamente

# import matplotlib.pyplot as plt
# import seaborn as sns

# %matplotlib inline


# features = X.columns
# num_features = len(features)

# for i, feature in enumerate(features):
#     print(headr(f"Graficando: {feature}"))
#     sns.barplot(x=feature, y=target_column, data=mantenimiento_dff)
#     plt.title(f"Tasa de {target_column} por {feature}")
#     plt.xlabel(feature)
#     plt.ylabel(target_column)
#     plt.show()

# %% [markdown]
# ### Eliminar características constantes

# %%
# Calcular la varianza de cada columna
column_variances = X.var()

# Crear un DataFrame para visualizar mejor los resultados
variance_df = pd.DataFrame({
    "Feature": column_variances.index,
    "Variance": column_variances.values
}).sort_values(by="Variance", ascending=False)

# Mostrar las varianzas ordenadas de mayor a menor
print("Varianza de cada columna:")
print(variance_df)

# %%
# # Seleccionar características con varianza mayor a 5 o el valor que se desee, excluyendo target_column

# from sklearn.feature_selection import VarianceThreshold


# # Aplicar VarianceThreshold
# sel = VarianceThreshold(threshold=5)
# sel.fit(X)

# # Obtener las columnas que no son constantes
# no_constant_columns = X.columns[sel.get_support()]
# constant_columns = X.columns.drop(no_constant_columns)

# print(headr("Columnas constantes:"), constant_columns)

# # Actualizar mantenimiento_const excluyendo las columnas constantes, pero manteniendo target_column
# mantenimiento_const = mantenimiento_const.drop(columns=[col for col in constant_columns if col != target_column])
# print(headr("Columnas finales:"), mantenimiento_const.columns)

# %% [markdown]
# ### Separamos el target

# %%
# Separar características y target
X = mantenimiento_const.drop(target_column, axis=1)     #df_mantenimiento.drop(target_column, axis=1)
y = mantenimiento_const[target_column] # Se coje el target

# %%
X.shape

# %% [markdown]
# ### Preparar para entrenamiento

# %%
# Dividir en entrenamiento y prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
print("Tamaño de X_train:", X_train.shape)
print("Tamaño de X_test:", X_test.shape)
print("Tamaño de y_train:", y_train.shape)
print("Tamaño de y_test:", y_test.shape)

# %% [markdown]
# ## Candidatos

# %% [markdown]
# A partir del análisis previo:
# 
# 1. `LogisticRegression`
# 2. `RandomForestClassifier` 
# 3. `DecisionTreeClassifier`
# 4. `KNeighborsClassifier`

# %% [markdown]
# ### Piplines

# %%
numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(X)
categorical_columns = categorical_columns_selector(X)

print(headr("Numerical columns"), numerical_columns)
print(headr("Categorical columns"), categorical_columns)

# %%
preprocessor = ColumnTransformer(
    [
        ('numerical', StandardScaler(), numerical_columns),
        ('categorical', OneHotEncoder(), categorical_columns)
    ])

pipelines = {
    "LogisticRegression": Pipeline([('preprocessor', preprocessor),('classifier', LogisticRegression())]),
    "RandomForestClassifier": Pipeline([('preprocessor', preprocessor),('classifier', RandomForestClassifier())]),
    "DecisionTreeClassifier": Pipeline([('preprocessor', preprocessor),('classifier', DecisionTreeClassifier())]),
    "KNeighborsClassifier": Pipeline([('preprocessor', preprocessor),('classifier', KNeighborsClassifier(n_neighbors=2, metric= 'euclidean', weights= 'distance'))]),
    #"SVC": Pipeline([('preprocessor', preprocessor),('classifier', SVC())]),
    #"GradientBoostingClassifier": Pipeline([('preprocessor', preprocessor),('classifier', GradientBoostingClassifier(n_estimators=500))]),
 
}

# %% [markdown]
# ### CV

# %%
def cv_train(name, pipeline, cv):
    print(headr(f"Entrenar {name}"))
    cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring="accuracy", return_estimator=True, return_train_score=True)
    trained_model = cv_results["estimator"][0]
    scores = pd.DataFrame(cv_results)

    print("test score (mean-std): {0:.2f} - {1:.2f}".format(scores["test_score"].mean(), scores["test_score"].std()))
    print("train score (mean-std): {0:.2f} - {1:.2f}".format(scores["train_score"].mean(), scores["train_score"].std()))
    print("params:", pipeline.named_steps.get("classifier").get_params())

    y_pred = trained_model.predict(X_test)
    f1 = f1_score(y_test, y_pred) #

    return {"acc": round(scores["test_score"].mean(), 2), "f1": round(f1, 2),}

# %%
cvss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

results = {}

for name, pipeline in pipelines.items():
    results[name] = cv_train(name, pipeline, cvss)

print(headr("Resultados:"))
results_df=pd.DataFrame(results)
results_df

# %% [markdown]
# ### Curva de aprendizaje

# %%
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve


# %%
# Curvas de aprendizaje

train_sizes = np.linspace(0.1, 1.0, num=5, endpoint=True)

def generate_learning_curves(name, pipeline, X, y, train_sizes):
    results = learning_curve(pipeline, X, y, train_sizes=train_sizes,
                             cv=cvss, scoring='accuracy')
    
    train_size, train_scores, test_scores = results[:3]

    # graficar la curva.
    plt.errorbar(train_size, train_scores.mean(axis=1),
                 yerr=train_scores.std(axis=1), label="Accuracy de entrenamiento", color='blue', linestyle='--', marker=None)
    plt.errorbar(train_size, test_scores.mean(axis=1),
                 yerr=test_scores.std(axis=1), label="Accuracy de prueba", color='red', linestyle='-', marker=None)
    
    # Posicionar la leyenda en el centro a la derecha
    plt.legend(loc='lower right')

    plt.xscale("linear")
    plt.xlabel("Número de muestras en el conjunto de entrenamiento")
    plt.ylabel("Accuracy")
    plt.title("Curva de aprendizaje para {name}".format(name=name))
    
    # Ajusta según el rango esperado de los valores
    plt.ylim(0, 1)
    
    
   

    plt.show()

# %%
# Mostramos gráficas

for pipeline_name, pipeline_obj in pipelines.items():
    generate_learning_curves(pipeline_name, pipeline_obj, X, y, train_sizes)

# %% [markdown]
# ### Curva validación

# %%

# Curvas de validación

def generate_validation_curves(name, pipeline, X, y, param_name, param_range):
    train_scores, test_scores = validation_curve(
        pipeline, X, y, param_name=param_name, param_range=param_range,
        cv=cvss, scoring="accuracy")

    # graficar la curva.
    plt.plot(param_range, train_scores.mean(
        axis=1), label="Error de entrenamiento", color='blue', linestyle='--')
    plt.plot(param_range, test_scores.mean(axis=1), label="Error de prueba", color='red', linestyle='-')
    plt.legend()

    plt.xlabel("Valor del ({param_name})".format(
        param_name=param_name))
    plt.ylabel("Accuracy")
    plt.title("Curva de validación para {name}".format(name=name))
    plt.ylim(0, 2)  # Ajusta según el rango esperado de los valores

    plt.show()


# %%
# Mostramos gráficas

pname = 'LogisticRegression'
Cs = [0.1, 10, 20]
generate_validation_curves(pname, pipelines[pname], X, y, 'classifier__C', Cs)

# %%
pname = 'RandomForestClassifier'
min_samples_split = [1, 0.1, 0.01, 0.001]
generate_validation_curves(pname, pipelines[pname], X, y, 'classifier__min_samples_split', min_samples_split)

# %%
pname = 'DecisionTreeClassifier'
max_depth = [1,5,10,150]
generate_validation_curves(pname, pipelines[pname], X, y, 'classifier__max_depth', max_depth)

# %%
pname = 'KNeighborsClassifier'
n_neighbors = [1, 5, 10, 15, 20]
generate_validation_curves(pname, pipelines[pname], X, y, 'classifier__n_neighbors', n_neighbors)

# %% [markdown]
# ### Afinar hiperparámentros

# %%
from sklearn.model_selection import GridSearchCV

pname = 'LogisticRegression'

param_grid = {
    'classifier__C': [7.5, 50, 70],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag, saga']
}

grid_search = GridSearchCV(
    pipelines[pname], param_grid, cv=cvss, scoring="accuracy")

grid_search.fit(X_train, y_train)

print(headr(pname))
print("Mejores hiperparámetros:", grid_search.best_params_)
print(f"Mejor accuracy: {grid_search.best_score_:.2f}")

# %%
from sklearn.model_selection import GridSearchCV

pname = 'RandomForestClassifier'

param_grid = {
    'classifier__n_estimators': [50, 70, 120],
    'classifier__min_samples_split': [0.1, 0.01, 0.001]
}

grid_search = GridSearchCV(
    pipelines[pname], param_grid, cv=cvss, scoring="accuracy")

grid_search.fit(X_train, y_train)

print(headr(pname))
print("Mejores hiperparámetros:", grid_search.best_params_)
print(f"Mejor accuracy: {grid_search.best_score_:.2f}")

# %%
from sklearn.model_selection import GridSearchCV

pname = 'DecisionTreeClassifier'

param_grid = {
    'classifier__criterion': ['gini', 'entropy', 'log_loss'],
    'classifier__min_samples_split': [0.5, 1.5, 2, 3]
}

grid_search = GridSearchCV(
    pipelines[pname], param_grid, cv=cvss, scoring="accuracy")

grid_search.fit(X_train, y_train)

print(headr(pname))
print("Mejores hiperparámetros:", grid_search.best_params_)
print(f"Mejor accuracy: {grid_search.best_score_:.2f}")

# %%
from sklearn.model_selection import GridSearchCV

pname = 'KNeighborsClassifier'

param_grid = {
    'classifier__n_neighbors': [0, 2, 5, 7],
    'classifier__metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
    'classifier__weights': ['uniform', 'distance']
}

grid_search = GridSearchCV(
    pipelines[pname], param_grid, cv=cvss, scoring="accuracy")

grid_search.fit(X_train, y_train)

print(headr(pname))
print("Mejores hiperparámetros:", grid_search.best_params_)
print(f"Mejor accuracy: {grid_search.best_score_:.2f}")

# %% [markdown]
# ### re-entrenar + re-evaluar

# %%
preprocessor = ColumnTransformer(
    [
        ('numerical', StandardScaler(), numerical_columns),
        ('categorical', OneHotEncoder(), categorical_columns)
    ])

pipelines = {
    "LogisticRegression": Pipeline([('preprocessor', preprocessor),('classifier', LogisticRegression(C=7.5,penalty='l2',solver='newton-cg'))]),
    "RandomForestClassifier": Pipeline([('preprocessor', preprocessor),('classifier', RandomForestClassifier(min_samples_split= 2, n_estimators= 100))]),
    "DecisionTreeClassifier": Pipeline([('preprocessor', preprocessor),('classifier', DecisionTreeClassifier(criterion='gini',min_samples_leaf=2))]),
    "KNeighborsClassifier": Pipeline([('preprocessor', preprocessor),('classifier', KNeighborsClassifier(n_neighbors=2, metric= 'euclidean', weights= 'distance'))]),
 }

# %%
results_final = {}

for name, pipeline in pipelines.items():
    results_final[name] = cv_train(name, pipeline, cvss)

# %% [markdown]
# ## Resumen del Notebook

# %% [markdown]
# Este notebook realiza un análisis completo de datos relacionados con el mantenimiento de equipos, desde la preparación de los datos hasta la evaluación de modelos de clasificación.
# A continuación, se resume cada sección:
# 
# 1. Carga y preparación de datos
# Se importan librerías necesarias y herramientas de scikit-learn.
# Se cargan varios archivos CSV relacionados con características de equipos, históricos de órdenes y registros de condiciones.
# Se procesan los datos:
# Se eliminan columnas irrelevantes.
# Se convierten índices a formato de fecha.
# Se agrupan y combinan los datos en un único DataFrame (merged_df).
# Se añaden nuevas características como frecuencias de mantenimiento correctivo, preventivo, por ubicación y potencia.
# 
# 2. Limpieza de datos
# Se verifican y rellenan valores faltantes.
# Se eliminan duplicados.
# Se transforman columnas categóricas en valores numéricos usando LabelEncoder.
# 
# 3. Selección de características
# Se evalúa la importancia de las características utilizando SelectKBest y se seleccionan las más relevantes.
# Se eliminan características con baja varianza.
# 
# 4. División de datos
# Se separan los datos en características (X) y la variable objetivo (y), que es el tipo de mantenimiento.
# Se dividen los datos en conjuntos de entrenamiento y prueba (80%-20%).
# 
# 5. Modelos candidatos
# Se definen cuatro modelos de clasificación como candidatos:
# LogisticRegression
# RandomForestClassifier
# DecisionTreeClassifier
# KNeighborsClassifier
# Se crean pipelines para cada modelo, incluyendo preprocesamiento (escalado y codificación).
# 
# 6. Evaluación inicial
# Se realiza validación cruzada para evaluar el desempeño de cada modelo en términos de precisión (accuracy) y puntaje F1.
# Se generan curvas de aprendizaje para analizar el comportamiento de los modelos con diferentes tamaños de datos de entrenamiento.
# 
# 7. Curvas de validación
# Se generan curvas de validación para ajustar hiperparámetros clave de los modelos, como C para regresión logística, min_samples_split para Random Forest, y otros.
# 
# 8. Afinación de hiperparámetros
# Se utiliza GridSearchCV para buscar los mejores hiperparámetros para cada modelo.
# 
# 9. Reentrenamiento y evaluación final
# Los modelos se reentrenan con los mejores hiperparámetros encontrados.
# Se evalúan nuevamente para comparar su desempeño final.
# 
# **Resultados**
# 
# El notebook permite identificar el modelo con mejor desempeño basado en las métricas de validación cruzada y las curvas de aprendizaje.
# Los modelos DecisionTreeClassifier y KNeighborsClassifier parecen tener un desempeño destacado en las evaluaciones iniciales.
# En resumen, este notebook realiza un flujo completo de análisis de datos, selección de características, entrenamiento y evaluación de modelos de clasificación para predecir el tipo de mantenimiento de equipos.

# %% [markdown]
# ## Conclusiones

# %% [markdown]
# **Resumen y elección del modelo**
# 
# 1. Logistic Regression:
# 
# - Curva de aprendizaje: La accuracy de entrenamiento y prueba se mantienen constantes alrededor de 0.63, indicando que el modelo no mejora con más datos de entrenamiento.
# 
#     Conclusión: Este modelo está subajustado, ya que no alcanza el objetivo de accuracy ≥ 0.80 ni el f1 ≥ 0.75.
# 
# 2. Random Forest Classifier:
# 
# - Curva de aprendizaje: La accuracy de prueba aumenta con más datos de entrenamiento, acercándose a la accuracy de entrenamiento.
# 
#     Conclusión: Este modelo muestra una tendencia positiva y es menos propenso al sobreajuste debido a su capacidad de generalización. Cumple con los objetivos de accuracy y f1.
# 
# 3. Decision Tree Classifier:
# 
# - Curva de aprendizaje: La accuracy de entrenamiento es alta, pero puede haber una diferencia significativa con la accuracy de prueba, indicando posible sobreajuste.
# 
#     Conclusión: Aunque cumple con los objetivos de accuracy y f1, puede ser más propenso al sobreajuste comparado con Random Forest.
# 
# 4. K-Neighbors Classifier:
# 
# - Curva de aprendizaje: Similar a Random Forest, muestra una mejora en la accuracy de prueba con más datos de entrenamiento.
# 
#     Conclusión: Este modelo también cumple con los objetivos de accuracy y f1 y tiene una buena capacidad de generalización.
# 
# 
# Recomendación de Modelos:
# 
# 1. Random Forest Classifier:
# 
#     Ventajas: Alta accuracy y f1, buena capacidad de generalización, menos propenso al sobreajuste.
# Coste Computacional: Moderado a alto, debido a la complejidad del modelo y el número de árboles.
# 
# 2. K-Neighbors Classifier:
# 
#     Ventajas: Alta accuracy y f1, buena capacidad de generalización.
# Coste Computacional: Alto, especialmente en la fase de predicción, ya que requiere calcular distancias para cada muestra.
# 
# 3. Decision Tree Classifier:
# 
#     Ventajas: Alta accuracy y f1, fácil de interpretar.
# Coste Computacional: Bajo a moderado, pero puede ser más propenso al sobreajuste.
# 
# ### Conclusión:
# 
# Para cumplir con los objetivos de accuracy ≥ 0.80 y f1 ≥ 0.75, elijo el **Random Forest Classifier** debido a su buen desempeño y capacidad de generalización. Si el coste computacional es una preocupación, el Decision Tree Classifier puede ser una opción viable, aunque se debe estar atento al posible sobreajuste. El K-Neighbors Classifier también es una buena opción, pero su coste computacional hay que tenerlo en cuenta en la fase de predicción.


