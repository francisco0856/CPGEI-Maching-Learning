# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 20:02:22 2025
@author: Francisco
"""

import os
import pandas as pd
import numpy as np
from scipy.io import loadmat

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

from sklearn.ensemble import RandomForestClassifier



# Caminho da pasta com os arquivos .mat
data_path = r'D:\UTFPR\Maching Learning Curitiba\Códigos Python\assigment 3\sameWindowData'

# Lista e ordena os arquivos .mat
mat_files = sorted([f for f in os.listdir(data_path) if f.endswith('.mat')])

# Nomes das colunas
colunas = ['tempo',
           'acc_X', 'acc_Y', 'acc_Z', # Aceleração nos eixos X, Y e Z
           'gyro_X', 'gyro_Y', 'gyro_Z', # Giroscópio nos eixos X, Y e Z
           'mag_X', 'mag_Y', 'mag_Z'] # Magnetômetro nos eixos X, Y e Z

# Frequência de amostragem (Hz) e período de amostragem (s)
Fs = 100
Ts = 1 / Fs

# Inicializa um dicionário para armazenar os dados de todos os arquivos
dados = {}


# Itera sobre cada arquivo .mat encontrado
for file in mat_files:
    # Monta o caminho completo do arquivo
    filepath = os.path.join(data_path, file) 
    # Carrega o conteúdo do arquivo .mat
    conteudo = loadmat(filepath) 
    # Extrai a matriz chamada 'newData'
    newData = conteudo['newData'] 
    # Constrói um DataFrame pandas com os dados e os nomes de coluna definidos
    df = pd.DataFrame(newData[:, :10], columns=colunas)
    # Converte a coluna de tempo de amostras para segundos
    df['tempo'] = df['tempo'] * Ts
    # Extrai o nome do arquivo sem a extensão e divide pelas partes usando ponto
    nome_base = os.path.splitext(file)[0]
    partes = nome_base.split('.')
    
    # Extrai identificadores numéricos M, R e VO a partir do nome do arquivo
    M = int(partes[0])  # Queda
    R = int(partes[2])  # Repetição
    VO = int(partes[3]) # Voluntário

    dados[nome_base] = {
        'M': M,
        'R': R,
        'VO': VO,
        'df': df # DataFrame com os dados sensoriais
    }

# Divide os dados em conjunto de treino e teste com base no identificador do voluntário (VO)
# Voluntários com VO de 1 a 18 são usados para treino
dados_train = {k: v for k, v in dados.items() if v['VO'] <= 18}

# Voluntários com VO 19 ou maior são usados para teste
dados_test = {k: v for k, v in dados.items() if v['VO'] >= 19}


print(f"Total de exemplos de treino: {len(dados_train)}")
print(f"Total de exemplos de teste: {len(dados_test)}")


# Função para extrair estatísticas de um exemplo (ignorando a coluna de tempo)
def extrair_features(exemplo):
    df = exemplo['df']

    # Inicializa dicionário com rótulo (queda ou não) e identificadores
    features = {
        'Fall': 'yes' if exemplo['M'] == 2 else 'no',  # Define rótulo de queda: M=2 indica queda
    }

    # Para cada sensor (aceleração, giroscópio, magnetômetro nos eixos X, Y, Z)
    for col in colunas[1:]:  # Ignora a coluna 'tempo'
        x = df[col].astype(np.float64)  # Garante precisão numérica
        # Extrai estatísticas básicas da série temporal
        features[f'{col}_mean'] = x.mean()                 # Média
        features[f'{col}_std'] = x.std()                   # Desvio padrão
        features[f'{col}_var'] = x.var()                   # Variância
        features[f'{col}_min'] = x.min()                   # Mínimo
        features[f'{col}_max'] = x.max()                   # Máximo
        features[f'{col}_range'] = x.max() - x.min()       # Amplitude
        features[f'{col}_rms'] = np.sqrt(np.mean(x**2))    # RMS (root mean square)

    return features


# Cria o DataFrame de treino aplicando a função de extração de features em cada exemplo
# Isso transforma os dados brutos do sensor em atributos estatísticos fixos prontos para classificação
# Cada linha do DataFrame representa um exemplo com as features extraídas
df_train = pd.DataFrame([extrair_features(ex) for ex in dados_train.values()])

df_test = pd.DataFrame([extrair_features(ex) for ex in dados_test.values()])

# Codificação dos rótulos de classe para uso em algoritmos de classificação
# 'yes' (caiu) será codificado como 1
# 'no'  (não caiu) será codificado como 0
le = LabelEncoder()

y_train = le.fit_transform(df_train['Fall'].values)
y_test = le.transform(df_test['Fall'].values)
X_train = df_train.drop(columns=['Fall']).values  
X_test = df_test.drop(columns=['Fall']).values
feature_names = df_train.drop(columns=['Fall']).columns

# Visualiza as classes aprendidas pelo codificador
print('\nClasse (Fall):')
print(le.classes_)

# Mostra o resultado da transformação das classes para números
print("\nRótulo codificado (LabelEncoder):")
print(le.transform(['no', 'yes']))  # Esperado: [0 1]


# ======================================================
# 1. Pipeline: Random Forest 
# ======================================================

pipe_rf = make_pipeline(
    RandomForestClassifier(random_state=1, 
                           n_jobs=-1,
                           class_weight={0: 1, 1: 1})
)

param_grid_rf = {
    'randomforestclassifier__n_estimators': [25, 50, 75, 100],  # Número de árvores na floresta
    'randomforestclassifier__max_depth': [2, 3, 4], # Profundidade máxima das árvores
    'randomforestclassifier__min_samples_leaf': [2, 4, 6],
    'randomforestclassifier__ccp_alpha': [0.001, 0.002] #parametro de poda
}

# ======================================================
# 3. Métrica personalizada: F2-score (importância maior para recall)
# ======================================================
scorer = make_scorer(fbeta_score, beta=2, pos_label=1)


# ======================================================
# 4. splits fixos
# ======================================================

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


# ======================================================
# 5. Busca em grade com validação cruzada
# ======================================================
gs_rf = GridSearchCV(
    estimator=pipe_rf,
    param_grid=param_grid_rf,
    scoring=scorer,
    refit=True,
    return_train_score=True,
    cv=cv, 
    verbose=1
)


gs_rf.fit(X_train, y_train)

clf_rf = gs_rf.best_estimator_

print("\nMelhor F2 score médio :", gs_rf.best_score_)

# Avalia o desempenho do modelo final no conjunto de teste final (VO 19 a 22)

acc_test = clf_rf.score(X_test, y_test)
y_pred = clf_rf.predict(X_test)
f2_test = fbeta_score(y_test, y_pred, beta=2, pos_label=1)
print("\nMelhores hyperparâmetros (RF):", gs_rf.best_params_)

print(f"\n>>> Acurácia no conjunto de teste final (VO 19 a 22): {acc_test:.3f}")
print(f">>> F2-score no conjunto de teste final:              {f2_test:.3f}")


y_pred = clf_rf.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

# ============================================================
# Métricas no teste
# ============================================================

# Matriz de confusão

labels = ['Não queda', 'Queda']

fig, ax = plt.subplots(figsize=(4, 4))
sns.heatmap(confmat, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
ax.set_xlabel('Classe prevista')
ax.set_ylabel('Classe verdadeira')
ax.set_title('Matriz de Confusão no Teste')
plt.savefig("matriz_confusao_teste_random_forest.png", dpi=300)
plt.show()

precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)
mcc = matthews_corrcoef(y_test, y_pred)
f2 = fbeta_score(y_test, y_pred, beta=4, pos_label=1)

print("\nMétricas no conjunto de teste:")
print(f"Precision {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:           {f1:.3f}")
print(f"Matthews CorrCoef:  {mcc:.3f}")
print(f"F2-score:            {f2:.3f}")

# ============================================================
# Curvas ROC
# ============================================================

plt.figure(figsize=(6, 5))
y_score_test = clf_rf.predict_proba(X_test)[:, 1]
fpr_test, tpr_test, _ = roc_curve(y_test, y_score_test)
roc_auc_test = auc(fpr_test, tpr_test)

plt.plot(fpr_test, tpr_test, color='black', linestyle='-', lw=2.5,
         label=f'Teste Final (AUC = {roc_auc_test:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle=':')
plt.xlabel('Taxa de Falso Positivo')
plt.ylabel('Taxa de Verdadeiro Positivo')
plt.title('Curva ROC - Conjunto de Teste')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_test_random_forest.png', dpi=300)
plt.close()

# ===============================
# Importância das features
# ===============================
rf_model = clf_rf.named_steps['randomforestclassifier']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Ranking textual
print("\nImportância das features (ordem decrescente):")
for f in range(X_train.shape[1]):
    print("%2d) %-30s %f" % (
        f + 1,
        feature_names[indices[f]],
        importances[indices[f]]
    ))

# Gráfico
plt.figure(figsize=(12, 6))
plt.title('Importância das features (Random Forest)')
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), feature_names[indices], rotation=90)
plt.tight_layout()

# Salvar imagem com 300 dpi
plt.savefig('importancia_features_RF_pipeline.png', dpi=300, bbox_inches='tight')
plt.show()

