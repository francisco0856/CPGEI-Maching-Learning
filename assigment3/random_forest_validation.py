# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 20:02:22 2025
@author: Francisco
"""

import os
import pandas as pd
import numpy as np
from scipy.io import loadmat

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import validation_curve


from sklearn.pipeline import make_pipeline

from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score

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

y_train = le.fit_transform(df_train['Fall'].values)
X_train = df_train.drop(columns=['Fall']).values 
X_test = df_test.drop(columns=['Fall']).values
y_test = le.transform(df_test['Fall'].values)
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
    RandomForestClassifier(n_estimators =20, random_state=1, n_jobs=-1)
)

# ======================================================
# 3. Métrica personalizada: F2-score (importância maior para recall)
# ======================================================
scorer = make_scorer(fbeta_score, beta=2, pos_label=1)

# ======================================================
# 4. splits fixos
# ======================================================

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Intervalo de profundidades máximas a testar
max_depth_range = [1, 2, 4, 6, 8, 10,12,14,16,18,20]

# Curva de validação variando max_depth
train_scores_rf, test_scores_rf = validation_curve(
    estimator=pipe_rf,
    X=X_train,
    y=y_train,
    param_name='randomforestclassifier__max_depth',
    param_range=max_depth_range,
    cv=cv,
    scoring=scorer,
    n_jobs=-1
)
# Estatísticas por ponto
train_mean = np.mean(train_scores_rf, axis=1)
train_std = np.std(train_scores_rf, axis=1)
test_mean = np.mean(test_scores_rf, axis=1)
test_std = np.std(test_scores_rf, axis=1)

# Plot
plt.plot(max_depth_range, train_mean, color='blue', marker='o',
         markersize=5, label='Treino')
plt.fill_between(max_depth_range, train_mean + train_std, train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(max_depth_range, test_mean, color='green', linestyle='--',
         marker='s', markersize=5, label='Validação')
plt.fill_between(max_depth_range, test_mean + test_std, test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.legend(loc='lower right')
plt.xlabel('Hiperparâmetro max_depth')
plt.ylabel('F2-score')
plt.ylim([0.8, 1.02])
plt.title('Curva de Validação - Random Forest (n_estimators = 20) ', fontsize=12)
plt.tight_layout()
plt.savefig("validation_curve_rf_max_depth.png", format='png', dpi=300)
plt.savefig("validation_curve_rf_max_depth.pdf", format='pdf')
plt.show()


pipe_rf = make_pipeline(
    RandomForestClassifier(max_depth = 4    , 
                           random_state=1, 
                           n_jobs=-1,
                           class_weight={0: 1, 1: 1})
)


# Intervalo de número de estimadores (árvores)
n_estimators_range = [10, 50, 100, 150, 200, 300, 400]

# Curva de validação variando n_estimators
train_scores_rf, test_scores_rf = validation_curve(
    estimator=pipe_rf,
    X=X_train,
    y=y_train,
    param_name='randomforestclassifier__n_estimators',
    param_range=n_estimators_range,
    cv=cv,
    scoring=scorer,
    n_jobs=-1
)

# Estatísticas por ponto
train_mean = np.mean(train_scores_rf, axis=1)
train_std = np.std(train_scores_rf, axis=1)
test_mean = np.mean(test_scores_rf, axis=1)
test_std = np.std(test_scores_rf, axis=1)

# Plot
plt.plot(n_estimators_range, train_mean, color='blue', marker='o',
         markersize=5, label='Treino')
plt.fill_between(n_estimators_range, train_mean + train_std, train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(n_estimators_range, test_mean, color='green', linestyle='--',
         marker='s', markersize=5, label='Validação')
plt.fill_between(n_estimators_range, test_mean + test_std, test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.legend(loc='lower right')
plt.xlabel('Hiperparâmetro n_estimators')
plt.ylabel('F2-score')
plt.ylim([0.8, 1.02])
plt.title('Curva de Validação - Random Forest (max_depth=4)', fontsize=12)
plt.tight_layout()
plt.savefig("validation_curve_rf_n_estimators.png", format='png', dpi=300)
plt.savefig("validation_curve_rf_n_estimators.pdf", format='pdf')
plt.show()
