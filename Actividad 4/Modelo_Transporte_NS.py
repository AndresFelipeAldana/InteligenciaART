# ---------------------------------------
# 1. Librerías necesarias
# ---------------------------------------
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------
# 2. Generación de datos sintéticos
# ---------------------------------------
origenes = ['A', 'B', 'C', 'D']
destinos = ['A', 'B', 'C', 'D']
horas_dia = ['mañana', 'tarde', 'noche']
climas = ['soleado', 'lluvioso', 'nublado']
dias_semana = ['lunes', 'martes', 'miércoles', 'jueves', 'viernes', 'sábado', 'domingo']

def generar_duracion(origen, destino, hora, clima):
    if origen == destino:
        return 0
    base = abs(ord(origen) - ord(destino)) * 5
    modificador = 0
    if hora == 'noche':
        modificador += 2
    if clima == 'lluvioso':
        modificador += 3
    return base + random.randint(-2, 3) + modificador

muestras = []
for _ in range(100):
    o = random.choice(origenes)
    d = random.choice(destinos)
    h = random.choice(horas_dia)
    c = random.choice(climas)
    dia = random.choice(dias_semana)
    duracion = generar_duracion(o, d, h, c)
    muestras.append([o, d, h, c, dia, duracion])

df = pd.DataFrame(muestras, columns=['origen', 'destino', 'hora_dia', 'clima', 'dia_semana', 'duracion'])

# Reutilizamos el mismo dataset `df`
df_cluster = df.copy().drop('duracion', axis=1)

# Codificación one-hot
df_cluster_encoded = pd.get_dummies(df_cluster)

# Escalado
scaler = StandardScaler()
X_cluster = scaler.fit_transform(df_cluster_encoded)

# ---------------------------------------
# 3. Entrenamiento del modelo
# ---------------------------------------
X = df.drop('duracion', axis=1)
y = df['duracion']

columnas_categoricas = ['origen', 'destino', 'hora_dia', 'clima', 'dia_semana']
preprocesamiento = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(), columnas_categoricas)
])

modelo = Pipeline(steps=[
    ('preprocesamiento', preprocesamiento),
    ('regresor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo.fit(X_train, y_train)

score = modelo.score(X_test, y_test)
print(f"\nPrecisión del modelo (R²): {score:.2f}")

kmeans = KMeans(n_clusters=3, random_state=42)
df['grupo_viaje'] = kmeans.fit_predict(X_cluster)

# ---------------------------------------
# 4. Predicción de ejemplo en tiempo real
# ---------------------------------------
nueva_ruta = pd.DataFrame([{
    'origen': 'A',
    'destino': 'D',
    'hora_dia': 'tarde',
    'clima': 'lluvioso',
    'dia_semana': 'viernes'
}])

prediccion = modelo.predict(nueva_ruta)
print(f"Duración estimada del viaje (A -> D): {prediccion[0]:.2f} minutos")

print("\nAgrupaciones de viajes (no supervisado):")
print(df[['origen', 'destino', 'hora_dia', 'clima', 'dia_semana', 'grupo_viaje']].head(10))


# 5. (Opcional) Visualización con PCA
# ---------------------------------------

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster)

plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['grupo_viaje'], cmap='viridis')
plt.title('Agrupamiento de viajes con K-Means (2D PCA)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.grid(True)
plt.show()