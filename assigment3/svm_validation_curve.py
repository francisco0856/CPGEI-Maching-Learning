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


from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import validation_curve

from sklearn.model_selection import learning_curve

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

pipe_svc_rbf_gamma = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf',  
        C=0.1,# valor fixo de C            
        class_weight={0: 1, 1: 3},
        probability=True,
        random_state=1)
)

pipe_svc_rbf_c = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf',  
        gamma=0.005,# valor fixo de gamma           
        class_weight={0: 1, 1: 3},
        probability=True,
        random_state=1)
)

pipe_svc_leinar = make_pipeline(
    StandardScaler(),
    SVC(kernel='linear',               
        class_weight={0: 1, 1: 3},
        probability=True,
        random_state=1)
)


X_train, y_train = shuffle(X_train, y_train, random_state=1)

tkfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# F2-score como métrica
scorer = make_scorer(fbeta_score, beta=2, pos_label=1)

# Faixas de hiperparâmetros
gamma_range = np.logspace(-4, -2, num=100)  # 10 valores logaritmicamente espaçados
c_range = [0.001, 0.01, 0.1, 1, 10, 100]


# Curva 1 – Variando gamma (com C fixo)
train_scores_rbf_gamma, test_scores_rbf_gamma = validation_curve(
    estimator=pipe_svc_rbf_gamma,
    X=X_train,
    y=y_train,
    param_name='svc__gamma',
    param_range=gamma_range,
    cv=tkfold,
    scoring=scorer,
    n_jobs=-1
)

# Curva 2 – Variando C (com gamma fixo)
train_scores_rbf_c, test_scores_rbf_c = validation_curve(
    estimator=pipe_svc_rbf_c,
    X=X_train,
    y=y_train,
    param_name='svc__C',
    param_range=c_range,
    cv=tkfold,
    scoring=scorer,
    n_jobs=-1
)

# Curva 3 – SVC Linear variando C
train_scores_linear, test_scores_linear = validation_curve(
    estimator=pipe_svc_leinar,
    X=X_train,
    y=y_train,
    param_name='svc__C',
    param_range=c_range,
    cv=tkfold,
    scoring=scorer,
    n_jobs=-1
)

train_mean = np.mean(train_scores_rbf_gamma, axis=1)
train_std = np.std(train_scores_rbf_gamma, axis=1)
test_mean = np.mean(test_scores_rbf_gamma, axis=1)
test_std = np.std(test_scores_rbf_gamma, axis=1)

plt.figure(figsize=(6, 4))
plt.plot(gamma_range, train_mean, color='blue', marker='o', label='Treino')
plt.fill_between(gamma_range, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
plt.plot(gamma_range, test_mean, color='green', linestyle='--', marker='s',markersize=3,  label='Validação')
plt.fill_between(gamma_range, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')
plt.xscale('linear')
plt.xlabel('gamma (RBF)')
plt.ylabel('F2-score')
plt.title('SVC RBF - F2 x gamma (C fixo em 0.1)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('svc_rbf_f2_vs_gamma.png', dpi=300)
plt.show()

train_mean = np.mean(train_scores_rbf_c, axis=1)
train_std = np.std(train_scores_rbf_c, axis=1)
test_mean = np.mean(test_scores_rbf_c, axis=1)
test_std = np.std(test_scores_rbf_c, axis=1)

plt.figure(figsize=(6, 4))
plt.plot(c_range, train_mean, color='blue', marker='o', label='Treino')
plt.fill_between(c_range, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
plt.plot(c_range, test_mean, color='green', linestyle='--', marker='s', label='Validação')
plt.fill_between(c_range, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')
plt.xscale('log')
plt.xlabel('C (RBF)')
plt.ylabel('F2-score')
plt.title('SVC RBF - F2 x C (gamma fixo em 0.005)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('svc_rbf_f2_vs_C.png', dpi=300)
plt.show()

train_mean = np.mean(train_scores_linear, axis=1)
train_std = np.std(train_scores_linear, axis=1)
test_mean = np.mean(test_scores_linear, axis=1)
test_std = np.std(test_scores_linear, axis=1)

plt.figure(figsize=(6, 4))
plt.plot(c_range, train_mean, color='blue', marker='o', label='Treino')
plt.fill_between(c_range, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
plt.plot(c_range, test_mean, color='green', linestyle='--', marker='s', label='Validação')
plt.fill_between(c_range, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')
plt.xscale('log')
plt.xlabel('C (Linear)')
plt.ylabel('F2-score')
plt.title('SVC Linear - F2 x C')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('svc_linear_f2_vs_C.png', dpi=300)
plt.show()