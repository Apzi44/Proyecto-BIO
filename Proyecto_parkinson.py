import pandas as pd
import numpy as np
import os
import pygad
import warnings
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title = "Deteccion de Parkinson", layout = "centered")

st.title("Deteccion de Parkinson con Algoritmo Genetico")

@st.cache_data
def cargar_datos():
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_archivo = os.path.join(directorio_actual, 'parkinsons.data')
    try:
        df = pd.read_csv(ruta_archivo)
        return df
    except FileNotFoundError:
        return None
    
df = cargar_datos()

if df is None:
    st.error("No se encontro el archivo 'parkinsons.data'")
    st.stop()
else:
    st.success("Datos cargados correctamente")
    if st.checkbox("Ver primeros datos"):
        st.dataframe(df.head())

X = df.drop(['name', 'status'], axis=1)
y = df['status']
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

st.sidebar.header("Configuracion del Algoritmo Genetico")
num_generaciones = st.sidebar.slider("Numero de Generaciones", 10, 200, 50)
poblacion = st.sidebar.slider("Tamaño de Poblacion", 10, 100, 20)

def fitness_func(ga_instance, solution, solution_idx):
    selected_features = [bool(bit) for bit in solution[:-1]]
    k_value = int(solution[-1])

    if sum(selected_features) == 0 or k_value < 1:
        return 0

    X_subset = X_scaled.iloc[:, selected_features]
    knn = KNeighborsClassifier(n_neighbors=k_value)
    scores = cross_val_score(knn, X_subset, y, cv=5)

    penalizacion = 0.005 * (sum(selected_features)/len(selected_features))
    return scores.mean() - penalizacion

if st.button("Iniciar Proceso"):
    barra_progreso = st.progress(0)
    texto_estado = st.empty()

    def on_generation(ga_instance):
        progreso = (ga_instance.generations_completed/num_generaciones)
        barra_progreso.progress(progreso)
        texto_estado.text(f"Generacion {ga_instance.generations_completed}: Fitness actual = {ga_instance.best_solution()[1]:.4f}")

    espacio_genes = [[0,1]] * len(X.columns) + [list(range(1, 16))]

    ga_instance = pygad.GA(
        num_generations=num_generaciones,
        num_parents_mating=int(poblacion/2),
        fitness_func=fitness_func,
        sol_per_pop=poblacion,
        num_genes=len(X.columns) + 1,
        gene_space=espacio_genes,
        parent_selection_type="sss",
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=10,
        on_generation=on_generation
    )

    ga_instance.run()

    solution, fitness, idx = ga_instance.best_solution()
    features_bits = solution[:-1]
    best_k = int(solution[-1])
    real_accuracy = fitness + (0.005 * (sum(features_bits)/len(features_bits)))

    st.divider()
    st.subheader("Resultados del Proceso")

    col1, col2, col3 = st.columns(3)
    col1.metric("Mejor Accuracy obtenido", f"{real_accuracy * 100:.2f}%")
    col2.metric("Valor Optimo de K", best_k)
    col3.metric("Columnas seleccionadas", f"{int(sum(features_bits))} de {len(X.columns)}")

    st.write("Caracteristicas Elegidas:")
    cols_names = X.columns[np.array(features_bits, dtype=bool)].tolist()
    st.info(", ".join(cols_names))

    st.write("Grafica de Evolucion")
    fig = ga_instance.plot_fitness(title="Mejora del Modelo por Generacion", save_dir = None)
    st.pyplot(fig)

#para encenderlo, usa este comando para instalar las librerias:pip install -r requerimientos.txt
#luego desde la terminal usa este comando: python -m streamlit run Proyecto_parkinson.py
#asegurate de estar dentro de la carpeta del proyecto, puedes usar: cd Proyecto-BIO