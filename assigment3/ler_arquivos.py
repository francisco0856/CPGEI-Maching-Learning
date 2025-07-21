import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

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


# Caminho da pasta com os arquivos .mat
data_path = r'D:\UTFPR\Maching Learning Curitiba\Códigos Python\assigment 3\sameWindowData'

# Lista e ordena os arquivos .mat
mat_files = sorted([f for f in os.listdir(data_path) if f.endswith('.mat')])

# Nomes das colunas
colunas = ['tempo',
           'acc_X', 'acc_Y', 'acc_Z',
           'gyro_X', 'gyro_Y', 'gyro_Z',
           'mag_X', 'mag_Y', 'mag_Z']

# Frequência de amostragem
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


# Divide o dicionário principal em treino e teste
dados_train = {k: v for k, v in dados.items() if v['VO'] <= 18}
dados_test = {k: v for k, v in dados.items() if v['VO'] >= 19}

print(f"Total de exemplos de treino: {len(dados_train)}")
print(f"Total de exemplos de teste: {len(dados_test)}")

# Extrai features para treino
features_train = [extrair_features(ex) for ex in dados_train.values()]
df_train = pd.DataFrame(features_train)

# Extrai features para teste
features_test = [extrair_features(ex) for ex in dados_test.values()]
df_test = pd.DataFrame(features_test)

# Visualiza
print("\nFeatures de treino:")
print(df_train.head())
print("\nFeatures de teste:")
print(df_test.head())

le = LabelEncoder()

# Codifica os rótulos de classe:
# - yes (caiu) →  '1' 
# - no (nao caiu) → '0' 

# Prepara o conjunto de entrada X contendo apenas as features numéricas
# Remove as colunas: 
# - 'Fall': rótulo de classe (será usado separadamente como y)
# - 'VO': identificador do voluntário (não é uma feature do movimento)
# - 'R' : número da repetição (também não é uma feature do movimento)

X_train = df_train.drop(columns=['Fall']).values
y_train = le.fit_transform(df_train['Fall'].values)

X_test = df_test.drop(columns=['Fall']).values
y_test = le.transform(df_test['Fall'].values)

feature_names = df_train.drop(columns=['Fall']).columns



# Visualiza as classes aprendidas pelo codificador
print('\nClasse (Fall):')
print(le.classes_)

# Mostra o resultado da transformação das classes para números
print("\nRótulo codificado (LabelEncoder):")
print(le.transform(['no', 'yes']))  # Esperado: [0 1]



X_train2, X_val, y_train2, y_val = \
    train_test_split(X_train, y_train, test_size=0.3,
                     stratify=y_train,
                     random_state=0)
    
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train2)
X_test_std = sc.transform(X_val)

pca = PCA(n_components=None)

X_train_pca = pca.fit_transform(X_train_std)

print(pca.explained_variance_ratio_)

eigen_vals=pca.explained_variance_;
eigen_vecs=(-1)*pca.components_.T ;
tot = sum(eigen_vals)

var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1, 64), var_exp, align='center',
        label='Individual explained variance')
plt.step(range(1, 64), cum_var_exp, where='mid',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('variancia_explicada.png', dpi=300)
plt.show()


def agrupar_sinais_interpolados(dados, n_amostras=500):
    sensores = {
        'Aceleração': ['acc_X', 'acc_Y', 'acc_Z'],
        'Giroscópio': ['gyro_X', 'gyro_Y', 'gyro_Z'],
        'Magnetômetro': ['mag_X', 'mag_Y', 'mag_Z']
    }

    sinais = {
        'queda': {grupo: {eixo: [] for eixo in eixos} for grupo, eixos in sensores.items()},
        'nao_queda': {grupo: {eixo: [] for eixo in eixos} for grupo, eixos in sensores.items()}
    }

    for ex in dados.values():
        classe = 'queda' if ex['M'] == 2 else 'nao_queda'
        df = ex['df']
        for grupo, eixos in sensores.items():
            for eixo in eixos:
                sinal = df[eixo].astype(float).values
                x = np.linspace(0, 1, len(sinal))
                f = interp1d(x, sinal, kind='linear')
                sinal_interp = f(np.linspace(0, 1, n_amostras))
                sinais[classe][grupo][eixo].append(sinal_interp)

    return sinais, sensores

def plotar_media_por_classe(sinais, sensores, n_amostras=648):
    tempo = np.linspace(0, 1, n_amostras)
    for classe in ['queda', 'nao_queda']:
        for grupo, eixos in sensores.items():
            fig, axs = plt.subplots(len(eixos), 1, figsize=(10, 8), sharex=True)
            fig.suptitle(f"{grupo} - Classe: {classe.replace('_', ' ').title()}", fontsize=16)

            for i, eixo in enumerate(eixos):
                dados = np.array(sinais[classe][grupo][eixo])
                media = dados.mean(axis=0)
                desvio = dados.std(axis=0)

                axs[i].plot(tempo, media, label=f'{eixo} média')
                axs[i].fill_between(tempo, media - desvio, media + desvio, alpha=0.3)
                axs[i].set_ylabel(eixo)
                axs[i].legend()

            axs[-1].set_xlabel('Tempo Normalizado')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            nome_arquivo = f"media_{classe}_{grupo.lower()}.png".replace(" ", "_")
            plt.savefig(nome_arquivo, dpi=300)
            plt.show()
            
sinais, sensores = agrupar_sinais_interpolados(dados_train, n_amostras=648)
plotar_media_por_classe(sinais, sensores)
