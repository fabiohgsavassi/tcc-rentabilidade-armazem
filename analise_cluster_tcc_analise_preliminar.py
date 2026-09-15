# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:09:16 2026

Trabalho de TCC do MBA em Data Science e Analytics USP ESALQ
Modelagem para Análise de Padrões de Rentabilidade de 
Clientes em Armazéns Logísticos segundo Técnica de Clustering

Aluno(a): Fábio Henrique Gonçales Savassi
Orientador: Elias Soares de Figueiredo

"""
#%% Instalando os pacotes

## Executar na linha de comando do console (sem o #)

# pip install pandas
# pip install numpy
# pip install matplotlib
# pip install seaborn
# pip install plotly
# pip install scipy
# pip install scikit-learn
# pip install pingouin

#%% Importando os pacotes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pingouin as pg
import plotly.express as px 
import plotly.io as pio
import seaborn as sns
pio.renderers.default='browser'

#%% Importando o banco de dados

# Objetivo: agrupar os clientes de um Operador Logístico
# Analisar os grupos de clientes baseados na similaridade do perfil operacional, 
# volumétrico e de rentabilidade, dentro de um armazém
modelo_armazem = pd.read_csv('modelo_analise.csv', sep=';')

#%% Visualizando informações sobre os dados e variáveis

# Estrutura do banco de dados

print(modelo_armazem.info())

#%% Convertendo as colunas de Object em Float64 ( por causa das virgulas )


# Lista com as colunas que estão marcadas incorretamente como 'object'
colunas_para_corrigir = [
    'Índice de Intensidade Operacional',
    'Lead Time Médio em Dias',
    'Média de Movimentações por Mês',
    'Movimentações Médias Internas em Pallets por Mês',
    'Ocupação Média de Pallets por Mês',
    'Receita Mensal Bruta de Armazenagem',
    'Receita Mensal Bruta de Serviços',
    'Valor Mensal Bruta de Mercadorias',
    'Volume Total Movimentados em Pallets',
    'Margem de Contribuição Estimada'
]

# Forçar a limpeza e a conversão de tipo para cada coluna
for col in colunas_para_corrigir:
    if col in modelo_armazem.columns:
        # 1. Garante que o dado seja tratado como texto para podermos limpar
        modelo_armazem[col] = modelo_armazem[col].astype(str)
        
        # 2. Converte vírgula em ponto
        modelo_armazem[col] = modelo_armazem[col].str.replace(',', '.', regex=False)
        
        # 3. Transforma o texto limpo em número decimal puro (float64)
        modelo_armazem[col] = pd.to_numeric(modelo_armazem[col], errors='coerce')

# Confere o resultado no console
print(modelo_armazem.info())

#%% Estatísticas descritivas das variáveis

# Primeiramente, vamos excluir as variáveis que não serão utilizadas

armazem = modelo_armazem.drop(columns=['Cliente', 'Segmento'])

# Obtendo as estatísticas descritivas das variáveis

tab_descritivas = armazem.describe().T

# Matriz de correlações das variáveis

# Gerando a matriz de correlações de Pearson
matriz_corr = pg.rcorr(armazem, method = 'pearson', upper = 'pval', 
                       decimals = 4, 
                       pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})

#%% Mapa de calor indicando a correlação entre os atributos

# Matriz de correlações básica
corr = armazem.corr()

f, ax = plt.subplots(figsize=(11, 9), dpi=600)

sns.heatmap(corr, 
            cmap = plt.cm.coolwarm,
            vmax=1, 
            vmin = -1,
            center=0,
            square=True, 
            linewidths=.5,
            annot = True,
            fmt='.2f', 
            annot_kws={'size': 12},
            cbar_kws={"shrink": 0.50})

plt.title('Matriz de Correlações')
plt.tight_layout()
ax.tick_params(axis = 'x', labelsize = 11)
ax.tick_params(axis = 'y', labelsize = 11)
ax.set_ylim(len(corr))

plt.show()

#%% Padronização por meio do Z-Score

# Aplicando o procedimento de ZScore
armazem_pad = modelo_armazem_cluster.apply(zscore, ddof=1)

# Visualizando o resultado do procedimento na média e desvio padrão
print(np.round(armazem_pad.mean(), 3))
print(np.round(armazem_pad.std(), 3))

#%% Gráfico 3D das observações

fig = px.scatter_3d(armazem_pad, 
                    x='Índice de Intensidade Operacional', 
                    y='Lead Time Médio em Dias', 
                    z='Média de Movimentações por Mês')

fig.write_html('armazem_inicial.html')

#%% Identificação da quantidade de clusters (Método Elbow)

elbow = []
K = range(1,11) # ponto de parada pode ser parametrizado manualmente
for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(cartao_pad)
    elbow.append(kmeanElbow.inertia_)
    
plt.figure(figsize=(16,8), dpi=600)
plt.plot(K, elbow, marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.xticks(range(1,11)) # ajustar range de acordo com K acima
plt.ylabel('WCSS', fontsize=16)
plt.title('Método de Elbow', fontsize=16)
plt.show()

#%% Identificação da quantidade de clusters (Método da Silhueta)

silhueta = []
I = range(2,11) # ponto de parada pode ser parametrizado manualmente
for i in I: 
    kmeansSil = KMeans(n_clusters=i, init='random', random_state=100).fit(cartao_pad)
    silhueta.append(silhouette_score(cartao_pad, kmeansSil.labels_))

plt.figure(figsize=(16,8), dpi=600)
plt.plot(range(2, 11), silhueta, color = 'purple', marker='o') # ajustar range
plt.xlabel('Nº Clusters', fontsize=16)
plt.ylabel('Silhueta Média', fontsize=16)
plt.title('Método da Silhueta', fontsize=16)
plt.axvline(x = silhueta.index(max(silhueta))+2, linestyle = 'dotted', color = 'red') 
plt.show()

#%% Cluster Não Hierárquico K-means

# Vamos considerar 3 clusters, considerando as evidências anteriores!

kmeans_final = KMeans(n_clusters = 3, init = 'random', random_state=100).fit(cartao_pad)

# Gerando a variável para identificarmos os clusters gerados

kmeans_clusters = kmeans_final.labels_
cartao_cluster['cluster_kmeans'] = kmeans_clusters
cartao_pad['cluster_kmeans'] = kmeans_clusters
cartao_cluster['cluster_kmeans'] = cartao_cluster['cluster_kmeans'].astype('category')
cartao_pad['cluster_kmeans'] = cartao_pad['cluster_kmeans'].astype('category')

#%% Análise de variância de um fator (ANOVA)

# Interpretação do output:

## cluster_kmeans MS: indica a variabilidade entre grupos
## Within MS: indica a variabilidade dentro dos grupos
## F: estatística de teste (cluster_kmeans MS / Within MS)
## p-unc: p-valor da estatística F
## se p-valor < 0.05: pelo menos um cluster apresenta média estatisticamente diferente dos demais

# Avg_Credit_Limit
pg.anova(dv='Avg_Credit_Limit', 
         between='cluster_kmeans', 
         data=cartao_pad,
         detailed=True).T

# Total_Credit_Cards
pg.anova(dv='Total_Credit_Cards', 
         between='cluster_kmeans', 
         data=cartao_pad,
         detailed=True).T

# Total_visits_bank
pg.anova(dv='Total_visits_bank', 
         between='cluster_kmeans', 
         data=cartao_pad,
         detailed=True).T

# Total_visits_online
pg.anova(dv='Total_visits_online', 
         between='cluster_kmeans', 
         data=cartao_pad,
         detailed=True).T

# Total_calls_made
pg.anova(dv='Total_calls_made', 
         between='cluster_kmeans', 
         data=cartao_pad,
         detailed=True).T

#%% Gráfico 3D dos clusters

# Perspectiva 1

fig1 = px.scatter_3d(cartao_cluster, 
                     x='Avg_Credit_Limit', 
                     y='Total_Credit_Cards', 
                     z='Total_visits_online',
                     color='cluster_kmeans')

fig1.write_html('cartao_1.html')

# Perspectiva 2

fig2 = px.scatter_3d(cartao_cluster, 
                     x='Avg_Credit_Limit', 
                     y='Total_Credit_Cards', 
                     z='Total_visits_bank',
                     color='cluster_kmeans')

fig2.write_html('cartao_2.html')

# Perspectiva 3

fig3 = px.scatter_3d(cartao_cluster, 
                     x='Avg_Credit_Limit', 
                     y='Total_Credit_Cards', 
                     z='Total_calls_made',
                     color='cluster_kmeans')

fig3.write_html('cartao_3.html')

#%% Identificação das características dos clusters

# Agrupando o banco de dados

cartao_grupo = cartao_cluster.groupby(by=['cluster_kmeans'], observed=True)

# Estatísticas descritivas por grupo

tab_media_grupo = cartao_grupo.mean().T

#%% FIM