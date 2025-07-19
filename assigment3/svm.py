# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 20:02:22 2025
@author: Francisco
"""

import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
import collections
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

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
        'VO': exemplo['VO'],  # Voluntário
        'R': exemplo['R'],    # Repetição
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

# Converte 'yes' (queda) em 1 e 'no' (não queda) em 0
y_train = le.fit_transform(df_train['Fall'].values)

# Visualiza as classes aprendidas pelo codificador
print('\nClasse (Fall):')
print(le.classes_)

# Mostra o resultado da transformação das classes para números
print("\nRótulo codificado (LabelEncoder):")
print(le.transform(['no', 'yes']))  # Esperado: [0 1]

# Prepara o conjunto de entrada X contendo apenas as features numéricas
# Remove as colunas: 
# - 'Fall': rótulo de classe (será usado separadamente como y)
# - 'VO': identificador do voluntário (não é uma feature do movimento)
# - 'R' : número da repetição (também não é uma feature do movimento)
X_train = df_train.drop(columns=['Fall', 'VO', 'R']).values  # .values retorna como array NumPy


# Prepara o conjunto de teste final com dados de voluntários nunca vistos (VO ≥ 19)
# Isso simula a generalização para novos usuários

X_test = df_test.drop(columns=['Fall', 'VO', 'R']).values
y_test = le.transform(df_test['Fall'].values)

# Salva os nomes das features para futura visualização ou interpretação dos modelos
feature_names = df_train.drop(columns=['Fall', 'VO', 'R']).columns

print("\nDistribuição original das classes no treino:")
print(collections.Counter(y_train))


# Pipeline com normalização + SVM

pipe_svc = make_pipeline(
    StandardScaler(),
    SVC(class_weight={0: 1, 1: 10},  # 1: classe positiva tem mais peso
        probability=True,           
        random_state=1)
)


scorer = make_scorer(fbeta_score, beta=4, pos_label=1)


c_range=[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009, 0.01]
c_range_rbf = [0.1, 0.25, 0.5, 0.9]
gamma_range= [0.001, 0.0025, 0.003]


param_grid = [{'svc__C': c_range,
               'svc__kernel': ['linear']},
              {'svc__C': c_range_rbf,
               'svc__gamma': gamma_range,
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring=scorer,
                  cv=10,
                  verbose=1,
                  n_jobs=-1)

gs = gs.fit(X_train, y_train)

print(gs.best_score_)
print(gs.best_params_)

clf_svc = gs.best_estimator_

# Exibe a melhor combinação de hiperparâmetros encontrada
print("\nMelhores parâmetros (RF):", gs.best_params_)

# Avalia o desempenho do modelo final no conjunto de teste final (VO 19 a 22)
acc_test = clf_svc.score(X_test, y_test)
print(f">>> Acurácia no conjunto de teste final (VO 19 a 22): {acc_test:.3f}")

y_pred = clf_svc.predict(X_test)

confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

# Rótulos legíveis
labels = ['Não queda', 'Queda']


# Matriz de confusão
confmat = confusion_matrix(y_test, y_pred)

# Plot com seaborn
fig, ax = plt.subplots(figsize=(4, 3.5))
sns.heatmap(confmat,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
            ax=ax)

ax.set_xlabel('Classe prevista')
ax.set_ylabel('Classe verdadeira')
ax.set_title('Matriz de Confusão no Teste')
plt.tight_layout()

precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)
mcc = matthews_corrcoef(y_test, y_pred)

print("\nMétricas no conjunto de teste:")
print(f"Precision (queda): {precision:.3f}")
print(f"Recall (queda):    {recall:.3f}")
print(f"F1-score:           {f1:.3f}")
print(f"Matthews CorrCoef:  {mcc:.3f}")

