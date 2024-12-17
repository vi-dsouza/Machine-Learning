# Importação de bibliotecas
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

# 1. Carregar a base de dados Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 2. Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Treinar um modelo (árvore de decisão como exemplo)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Fazer previsões
y_pred = model.predict(X_test)

# 5. Calcular a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.2f}")

# 5. Calcular a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(conf_matrix)

# 6. Calcular métricas de avaliação
# Precisão
precision = precision_score(y_test, y_pred, average='weighted')  # Média ponderada para múltiplas classes
print(f"Precisão: {precision}")

# Sensibilidade (Recall) - Foco nas classes positivas (nesse caso cada classe)
sensitivity = recall_score(y_test, y_pred, average='weighted')
print(f"Sensibilidade: {sensitivity}")

# Especificidade: 1 - Falso Positivo / (Verdadeiro Negativo + Falso Positivo)
specificity = []
for i in range(len(conf_matrix)):
    tn = sum([conf_matrix[j][j] for j in range(len(conf_matrix))]) - conf_matrix[i][i]
    fp = sum(conf_matrix[i]) - conf_matrix[i][i]
    specificity.append(tn / (tn + fp))
specificity = np.mean(specificity)
print(f"Especificidade: {specificity}")

# F1-Score (Média harmônica entre precisão e sensibilidade)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1-Score: {f1}")

# 7. Curva ROC e AUC (Área sob a curva)
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)
y_pred_bin = lb.transform(y_pred)

# Para calcular a Curva ROC e AUC para múltiplas classes
fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_pred_bin.ravel())
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc}")

# Plotar a Curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()
