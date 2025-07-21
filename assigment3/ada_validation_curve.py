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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import validation_curve


from sklearn.pipeline import make_pipeline

from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score


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

# Converte 'yes' (queda) em 1 e 'no' (não queda) em 0
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
# 1. Pipeline: Adaline com descida do gradiente
# ======================================================

pipe_adaline_l1 = make_pipeline(
    StandardScaler(),
    SGDClassifier(
        loss='squared_error',         # Erro quadrático → Adaline clássico
        learning_rate='constant',     # Taxa de aprendizado fixa
        eta0=0.0001,                  
        max_iter=100,                 # Número de épocas
        tol=1e-3,                     
        random_state=1,
        penalty='l1',               # <- Regularização L1
        class_weight={0: 1, 1: 10}    # Penaliza mais as quedas (classe 1)
    )
)

pipe_adaline_l2 = make_pipeline(
    StandardScaler(),
    SGDClassifier(
        loss='squared_error',         # Erro quadrático → Adaline clássico
        learning_rate='constant',     # Taxa de aprendizado fixa
        eta0=0.0001,                  
        max_iter=100,                 # Número de épocas
        tol=1e-3,                     
        random_state=1,
        penalty='l2',               # <- Regularização L2
        class_weight={0: 1, 1: 10}    # Penaliza mais as quedas (classe 1)
    )
)


# ======================================================
# 3. Métrica personalizada: F2-score (importância maior para recall)
# ======================================================
scorer = make_scorer(fbeta_score, beta=2, pos_label=1)

# ======================================================
# 4. Objeto de validação cruzada com splits fixos
# ======================================================

# Define a validação cruzada com 5 folds

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

alpha_range = np.logspace(-2, 0, 20)  # De 10^-3 até 10^0 (1.0), 100 valores
alpha_rangeL2 = np.logspace(-2, 2, 20)  # De 10^-3 até 10^0 (1.0), 100 valores

# Executa validation_curve corretamente
train_scores_ada, test_scores_ada = validation_curve(
    estimator=pipe_adaline_l1,
    X=X_train,
    y=y_train,
    param_name='sgdclassifier__alpha',
    param_range=alpha_range,
    cv=cv,
    scoring=scorer,  # make_scorer(fbeta_score, beta=2 ou 4)
    n_jobs=-1
)

train_mean = np.mean(train_scores_ada, axis=1)
train_std = np.std(train_scores_ada, axis=1)
test_mean = np.mean(test_scores_ada, axis=1)
test_std = np.std(test_scores_ada, axis=1)

plt.plot(alpha_range, train_mean,
         color='blue', marker='o',
         markersize=5, label='Treino')

plt.fill_between(alpha_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')

plt.plot(alpha_range, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validação')

plt.fill_between(alpha_range,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parâmetro de reguralização alpha para  (L1)')
plt.ylabel('F2-score')
plt.ylim([0.8, 1.0])
plt.title('Curva de Validação (eta fixo = 0.0001)', fontsize=12)
plt.tight_layout()
plt.savefig("validation_curve_adaline_l1.png", format='png', dpi=300)
plt.savefig("validation_curve_adaline_l1.pdf", format='pdf')  # <- esta linha salva o gráfico
plt.show()

# Executa validation_curve corretamente
train_scores_ada, test_scores_ada = validation_curve(
    estimator=pipe_adaline_l2,
    X=X_train,
    y=y_train,
    param_name='sgdclassifier__alpha',
    param_range=alpha_rangeL2,
    cv=cv,
    scoring=scorer,  # make_scorer(fbeta_score, beta=2 ou 4)
    n_jobs=-1
)

train_mean = np.mean(train_scores_ada, axis=1)
train_std = np.std(train_scores_ada, axis=1)
test_mean = np.mean(test_scores_ada, axis=1)
test_std = np.std(test_scores_ada, axis=1)

plt.plot(alpha_rangeL2, train_mean,
         color='blue', marker='o',
         markersize=5, label='Treino')

plt.fill_between(alpha_rangeL2, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')

plt.plot(alpha_rangeL2, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validação')

plt.fill_between(alpha_rangeL2,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parâmetro de reguralização alpha para  (L2)')
plt.ylabel('F2-score')
plt.ylim([0.8, 1.0])
plt.title('Curva de Validação (eta fixo = 0.0001)', fontsize=12)
plt.tight_layout()
plt.savefig("validation_curve_adaline_l2.png", format='png', dpi=300)
plt.savefig("validation_curve_adaline_l2.pdf", format='pdf')  # <- esta linha salva o gráfico
plt.show()
