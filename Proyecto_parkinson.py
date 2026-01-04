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

# Ignora advertencias de versiones o deprecaciones para limpiar la salida de la consola.
warnings.filterwarnings("ignore")

# --- CONFIGURACIÓN DE LA INTERFAZ (STREAMLIT) ---
# Configura el título de la pestaña del navegador y el diseño centrado.
st.set_page_config(page_title="Detección de Parkinson", layout="centered")

# Encabezado principal y descripción de la aplicación.
st.title("🧬 Detección de Parkinson con algoritmo genetico")
st.markdown("""
Esta aplicación utiliza **Algoritmos Genéticos** para encontrar la combinación óptima de biomarcadores de voz
que permitan diagnosticar la enfermedad de Parkinson con la mayor precisión posible.
""")

# --- GESTIÓN DE DATOS ---

# Decorador para cachear los datos: evita recargar el CSV en cada interacción de la UI.
@st.cache_data
def cargar_datos():
    # Obtiene la ruta absoluta del script actual para localizar el archivo de datos.
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_archivo = os.path.join(directorio_actual, 'parkinsons.data')
    try:
        df = pd.read_csv(ruta_archivo)
        return df
    except FileNotFoundError:
        return None

# Carga el DataFrame en memoria.
df = cargar_datos()

# Validación de seguridad: detiene la app si no hay datos.
if df is None:
    st.error("No se encontró el archivo 'parkinsons.data'. Asegúrate de que esté en la misma carpeta.")
    st.stop()

# --- ANÁLISIS EXPLORATORIO DE DATOS (EDA) ---
st.divider()
st.subheader("-- Análisis de la Población")

# Conteo de clases (1: Parkinson, 0: Sano).
conteo = df['status'].value_counts()
total = len(df)
parkinson_count = conteo.get(1, 0)
healthy_count = conteo.get(0, 0)

# Visualización de métricas clave en columnas.
col1, col2, col3 = st.columns(3)
col1.metric("Total de Muestras", total)
col2.metric("Pacientes con Parkinson", parkinson_count, delta=f"{(parkinson_count/total)*100:.1f}%")
col3.metric("Pacientes Sanos", healthy_count, delta_color="inverse", delta=f"{(healthy_count/total)*100:.1f}%")

# Opción para inspeccionar la estructura de los datos.
if st.checkbox("Ver datos crudos"):
    st.dataframe(df.head())

# --- PREPROCESAMIENTO DE DATOS ---
# Separación de características (X) y etiqueta objetivo (y).
# Se elimina 'name' (irrelevante) y 'status' (target).
X = df.drop(['name', 'status'], axis=1)
y = df['status']

# Estandarización (Scaling): Crucial para KNN ya que se basa en distancias Euclidianas.
# Transforma los datos para que tengan media 0 y desviación estándar 1.
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# --- CONFIGURACIÓN DEL ALGORITMO GENÉTICO (SIDEBAR) ---
st.sidebar.header("-- Configuración Genética")
# Sliders para ajustar la intensidad de la búsqueda evolutiva.
num_generaciones = st.sidebar.slider("Generaciones", 10, 200, 50)
poblacion = st.sidebar.slider("Tamaño de Población", 10, 100, 20)

# --- LÓGICA DE EVALUACIÓN (FITNESS FUNCTION) ---
def fitness_func(ga_instance, solution, solution_idx):
    """
    Evalúa qué tan buena es una solución (individuo).
    El cromosoma tiene estructura mixta: [Bits de Features] + [Valor K]
    """
    # 1. Decodificación del Cromosoma:
    # Los primeros N genes son binarios (Selección de características).
    selected_features = [bool(bit) for bit in solution[:-1]]
    # El último gen es un entero (Hiperparámetro K para KNN).
    k_value = int(solution[-1])

    # Validación: Si no se selecciona ninguna característica, el fitness es 0 (inválido).
    if sum(selected_features) == 0:
        return 0
    
    # 2. Construcción del modelo con el subconjunto de datos seleccionado.
    X_subset = X_scaled.iloc[:, selected_features]
    
    # Inicializa KNN con el K sugerido por el genoma.
    knn = KNeighborsClassifier(n_neighbors=k_value) 
    
    # 3. Validación Cruzada (Cross-Validation).
    # Evalúa el modelo 5 veces con diferentes particiones para evitar overfitting.
    scores = cross_val_score(knn, X_subset, y, cv=5)
    
    # 4. Cálculo del Fitness con Penalización (Regularización).
    # Se penaliza el uso excesivo de características para buscar el modelo más simple posible (Parsimonia).
    # Penalización = 0.005 * (% de características usadas).
    penalizacion = 0.005 * (sum(selected_features)/len(selected_features))
    
    # El fitness final es la precisión media menos la penalización.
    return scores.mean() - penalizacion

# --- MOTOR DE OPTIMIZACIÓN ---
if st.button("---> Iniciar Optimización"):
    
    # Elementos de UI para feedback en tiempo real.
    barra = st.progress(0)
    status_text = st.empty()

    # Callback: Se ejecuta al finalizar cada generación para actualizar la UI.
    def on_generation(ga_instance):
        progreso = (ga_instance.generations_completed / num_generaciones)
        barra.progress(progreso)
        best_sol = ga_instance.best_solution()[1]
        status_text.caption(f"Generación {ga_instance.generations_completed} | Mejor Fitness: {best_sol:.4f}")

    # Definición del Espacio de Búsqueda (Gene Space):
    # - Genes de características: Binarios [0, 1].
    # - Gen de K: Enteros impares entre 1 y 15 (para evitar empates en votación KNN).
    espacio_genes = [[0,1]] * len(X.columns) + [list(range(1, 16, 2))]

    # Configuración de la instancia PyGAD.
    ga_instance = pygad.GA(
        num_generations=num_generaciones,
        num_parents_mating=int(poblacion/2),    # El 50% de la población se reproduce.
        fitness_func=fitness_func,
        sol_per_pop=poblacion,
        num_genes=len(X.columns) + 1,           # N features + 1 hiperparámetro.
        gene_space=espacio_genes,
        parent_selection_type="sss",            # Steady State Selection.
        crossover_type="single_point",          # Cruce de un punto.
        mutation_type="random",                 # Mutación aleatoria simple.
        mutation_percent_genes=10,              # 10% de probabilidad de mutación.
        on_generation=on_generation
    )

    # Ejecución del algoritmo.
    ga_instance.run()

    # --- PRESENTACIÓN DE RESULTADOS ---
    
    # Obtención del mejor individuo.
    solution, fitness, idx = ga_instance.best_solution()
    
    # Decodificación final.
    features_bits = solution[:-1]
    best_k = int(solution[-1])
    
    # Recálculo de la precisión real (restando la penalización aplicada en el fitness).
    accuracy_final = fitness + (0.005 * (sum(features_bits)/len(features_bits)))
    
    st.success("¡Optimización Completada!")
    
    # Métricas finales.
    c1, c2, c3 = st.columns(3)
    c1.metric("Precisión (Accuracy)", f"{accuracy_final*100:.2f}%")
    c2.metric("Vecinos (K)", best_k)
    c3.metric("Características", f"{int(sum(features_bits))}/{len(X.columns)}")

    # Gráfico de convergencia.
    st.subheader("Curva de Aprendizaje")
    fig = ga_instance.plot_fitness(title="Mejora del Modelo por Generación", save_dir=None)
    st.pyplot(fig)

    # Interpretación en lenguaje natural.
    st.divider()
    st.subheader("----> Interpretación de los Resultados")
    
    st.info(f"""
    **Conclusión del Analisis:**
    El sistema ha encontrado que para diagnosticar Parkinson en esta población específica,
    no hace falta medir todo. Usando solo **{int(sum(features_bits))}** variables clave y comparando
    con **{best_k}** pacientes similares, se logra una efectividad del **{accuracy_final*100:.1f}%**.
    """)
    
    # Listado de nombres de las características seleccionadas.
    cols_names = X.columns[np.array(features_bits, dtype=bool)].tolist()
    st.write("**Biomarcadores seleccionados:** " + ", ".join(cols_names))