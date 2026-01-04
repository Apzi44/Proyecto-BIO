import streamlit as st
import pandas as pd
import numpy as np
import os
import pygad
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Ignorar advertencias no críticas para mantener limpia la consola
warnings.filterwarnings("ignore", category=UserWarning)

# Seccion 1: CONFIGURACIÓN DE LA INTERFAZ
st.set_page_config(page_title="Detección de Parkinson", layout="centered")

st.title("Detección de Parkinson con Algoritmos Genéticos")
st.markdown("""
**Descripción del Proyecto:**
Este programa utiliza Inteligencia Artificial Evolutiva para optimizar el diagnóstico de Parkinson.
El algoritmo selecciona automáticamente las características de voz más relevantes y el mejor 
hiperparámetro 'K' para un clasificador KNN, maximizando la precisión y reduciendo el ruido
""")

# Seccion 2: CARGA Y PREPROCESAMIENTO DE DATOS
@st.cache_data
def cargar_datos():
    """Carga la base de datos y maneja errores de ruta"""
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_archivo = os.path.join(directorio_actual, 'parkinsons.data')
    try:
        df = pd.read_csv(ruta_archivo)
        return df
    except FileNotFoundError:
        return None
    
df = cargar_datos()

if df is None:
    st.error("No se encontró el archivo 'parkinsons.data'")
    st.stop()
else:
    st.success("Base de datos cargada correctamente")
    if st.checkbox("Ver vista previa de los datos"):
        st.dataframe(df.head())

# Preparacion
# Separamos características 'x' y etiqueta 'y'
X = df.drop(['name', 'status'], axis=1)
y = df['status']

# Escalado
# Estandarizamos los datos para que el KNN funcione correctamente
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Seccion 3: CONFIGURACIÓN DEL ALGORITMO GENÉTICO
st.sidebar.header("Panel de Control")
st.sidebar.info("Ajusta los parámetros de la evolución:")

num_generaciones = st.sidebar.slider("Número de Generaciones", 10, 200, 50)
poblacion = st.sidebar.slider("Tamaño de Población", 10, 100, 20)

# Seccion 4: FUNCIÓN DE FITNESS
def fitness_func(ga_instance, solution, solution_idx):
    """
    Evalúa la calidad de una solución
    - Genes (0-21): Selección de columnas
    - Gen 22: Valor de K para KNN (Entero)
    """
    selected_features = [bool(bit) for bit in solution[:-1]]
    k_value = int(solution[-1])

    # Si no elige columnas o K es inválido, el individuo muere (fitness 0)
    if sum(selected_features) == 0 or k_value < 1:
        return 0

    # Crear subconjunto de datos
    X_subset = X_scaled.iloc[:, selected_features]
    
    # Crear modelo KNN con el K propuesto por el genético
    knn = KNeighborsClassifier(n_neighbors=k_value)
    
    # Validación Cruzada para robustez
    scores = cross_val_score(knn, X_subset, y, cv=5)
    
    # Penalización por complejidad (preferimos menos columnas)
    penalizacion = 0.005 * (sum(selected_features)/len(selected_features))
    
    return scores.mean() - penalizacion

# Seccion 5: EJECUCIÓN DEL ALGORITMO
if st.button("Iniciar Optimización Evolutiva"):
    
    # Barra de progreso visual
    barra_progreso = st.progress(0)
    texto_estado = st.empty()

    # Función para actualizar la barra mientras piensa
    def on_generation(ga_instance):
        progreso = (ga_instance.generations_completed / num_generaciones)
        barra_progreso.progress(progreso)
        texto_estado.text(f"Generación {ga_instance.generations_completed}: Fitness actual = {ga_instance.best_solution()[1]:.4f}")

    # Definir espacio de búsqueda: 22 genes binarios + 1 gen numérico (K)
    espacio_genes = [[0,1]] * len(X.columns) + [list(range(1, 16))]

    # Configuración de PyGAD
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

    # Correr la evolución
    ga_instance.run()

    # Resultados
    solution, fitness, idx = ga_instance.best_solution()
    features_bits = solution[:-1]
    best_k = int(solution[-1])
    
    # Recalcular accuracy real (sin la penalización) para mostrar al usuario
    real_accuracy = fitness + (0.005 * (sum(features_bits)/len(features_bits)))

    st.divider()
    st.subheader("Resultados de la Optimización")

    # Mostrar métricas
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy (Validación Cruzada)", f"{real_accuracy * 100:.2f}%", delta="Precisión")
    col2.metric("Mejor Valor de K", best_k, delta="Vecinos")
    col3.metric("Columnas Seleccionadas", f"{int(sum(features_bits))} de {len(X.columns)}", delta="Reducción de Ruido")

    # Mostrar nombres de columnas
    st.write("Biomarcadores Seleccionados:")
    cols_names = X.columns[np.array(features_bits, dtype=bool)].tolist()
    st.info(", ".join(cols_names))

    # Gráfica
    st.write("Evolución del Aprendizaje")
    fig = ga_instance.plot_fitness(title="Mejora del Fitness por Generación", save_dir=None)
    st.pyplot(fig)

#para encenderlo, usa este comando para instalar las librerias:pip install -r requerimientos.txt
#luego desde la terminal usa este comando: python -m streamlit run Proyecto_parkinson.py
#asegurate de estar dentro de la carpeta del proyecto, puedes usar: cd Proyecto-BIO