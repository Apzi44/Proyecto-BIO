import pandas as pd
import numpy as np
import os
import pygad
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore", category=UserWarning)

directorio_actual = os.path.dirname(os.path.abspath(__file__))
ruta_archivo = os.path.join(directorio_actual, 'parkinsons.data')
try:
    df = pd.read_csv(ruta_archivo)
except FileNotFoundError:
    print(f"No se encuentra el archivo en: {ruta_archivo}")
    exit()

X = df.drop(['name', 'status'], axis=1)
y = df['status']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

def fitness_func(ga_instance, solution, solution_idx):
    selected_features = [bool(bit) for bit in solution[:-1]]
    k_value = int(solution[-1])

    if sum(selected_features) == 0 or k_value < 1:
        return 0

    X_subset = X_scaled.iloc[:, selected_features]

    knn = KNeighborsClassifier(n_neighbors=k_value)
    scores = cross_val_score(knn, X_subset, y, cv=5)
    accuracy_promedio = scores.mean()

    penalizacion = 0.005 * (sum(selected_features)/len(selected_features))

    return accuracy_promedio - penalizacion

espacio_genes = [[0,1]] * len(X.columns) + [list(range(1, 16))]

ga_instance = pygad.GA(
    num_generations=100,
    num_parents_mating=10,
    fitness_func=fitness_func,
    sol_per_pop=20,
    num_genes=len(X.columns) + 1,
    gene_space=espacio_genes,
    parent_selection_type="sss",
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=10
)

print("Optimizando selección de características...")
ga_instance.run()

solution, fitness, idx = ga_instance.best_solution()
features_bits = solution[:-1]
best_k = int(solution[-1])

print("\n" + "=" * 40)
print(f"Mejor Accuracy obtenido: {fitness + (0.005 * (sum(features_bits)/22)):.4f}")
print(f"Valor Optimo de K: {best_k}")
print(f"Columnas seleccionadas: {int(sum(features_bits))}")
print(f"Caracteristicas: {X.columns[np.array(features_bits, dtype=bool)].tolist()}")
print("="*40)

print("Generacion de grafica de evolucion")
ga_instance.plot_fitness(
    title = "Evolucion del Fitness",
    xlabel = "Generacion",
    ylabel = "Fitness Score"
)