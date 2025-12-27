import pandas as pd
import numpy as np
import pygad
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


df = pd.read_csv('parkinsons.data')

X = df.drop(['name', 'status'], axis=1)
y = df['status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_df = pd.DataFrame(X_scaled, columns=X.columns)

def fitness_func(ga_instance, solution, solution_idx):
    selected_features = [bool(bit) for bit in solution]

    if sum(selected_features) == 0:
        return 0

    X_subset = X_df.iloc[:, selected_features]

    X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.3, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    predictions = knn.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return accuracy

num_genes = X.shape[1]

ga_instance = pygad.GA(
    num_generations=50,
    num_parents_mating=5,
    fitness_func=fitness_func,
    sol_per_pop=20,
    num_genes=num_genes,
    gene_space=[0, 1],
    parent_selection_type="sss",
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=10
)

print("Optimizando selección de características...")
ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()

print("-" * 30)
print(f"Mejor Accuracy obtenido: {solution_fitness:.4f}")
print(f"Número de columnas seleccionadas: {int(sum(solution))}")

features_seleccionadas = X.columns[np.array(solution, dtype=bool)].tolist()
print(f"Columnas óptimas: {features_seleccionadas}")

ga_instance.plot_fitness()