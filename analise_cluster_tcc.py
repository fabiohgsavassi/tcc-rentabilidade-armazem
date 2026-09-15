# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:09:16 2026

Trabalho de TCC do MBA em Data Science e Analytics USP ESALQ
Modelagem para Análise de Padrões de Rentabilidade de
Clientes em Armazéns Logísticos segundo Técnica de Clustering

Aluno(a): Fábio Henrique Gonçales Savassi
Orientador: Prof. Elias Soares de Figueiredo

"""
# %% Instalando os pacotes

## Executar na linha de comando do console (sem o #)

# pip install pandas
# pip install numpy
# pip install matplotlib
# pip install seaborn
# pip install plotly
# pip install scipy
# pip install scikit-learn
# pip install pingouin

# %% Importando os pacotes

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

pio.renderers.default = 'browser'

# %% Importando o banco de dados

# Objetivo: agrupar os clientes de um Operador Logístico
# Analisar os grupos de clientes baseados na similaridade do perfil operacional,
# volumétrico e de rentabilidade, dentro de um armazém
modelo_armazem = pd.read_csv('dados_tcc_armazem.csv', sep=';', encoding='utf-8-sig')

# %% Visualizando informações sobre os dados e variáveis

# Estrutura do banco de dados

print(modelo_armazem.info())

# %% Convertendo as colunas de Object em Float64 ( por causa das virgulas )

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
    'Margem de Contribuição Estimada',
    'Giro Mensal Médio de Estoque',
    'Volatilidade de Ocupação',
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

# %% Definição das variáveis da análise

# Conjunto inicial extraído do WMS
variaveis_iniciais = [
    'Índice de Intensidade Operacional',
    'Lead Time Médio em Dias',
    'Média de Movimentações por Mês',
    'Movimentações Médias Internas em Pallets por Mês',
    'Ocupação Média de Pallets por Mês',
    'Receita Mensal Bruta de Armazenagem',
    'Receita Mensal Bruta de Serviços',
    'Valor Mensal Bruta de Mercadorias',
    'Volume Total Movimentados em Pallets',
    'Margem de Contribuição Estimada',
]

# Conjunto restante após a eliminação das redundâncias e das receitas,
# acrescido das duas variáveis derivadas
variaveis = [
    'Índice de Intensidade Operacional',
    'Lead Time Médio em Dias',
    'Ocupação Média de Pallets por Mês',
    'Margem de Contribuição Estimada',
    'Giro Mensal Médio de Estoque',
    'Volatilidade de Ocupação',
]

# Variáveis que formam a matriz de distância do K-means
variaveis_cluster = [
    'Ocupação Média de Pallets por Mês',
    'Giro Mensal Médio de Estoque',
    'Volatilidade de Ocupação',
]

# Variáveis que não participam da clusterização e servem para validar o modelo
variaveis_externas = [
    'Margem de Contribuição Estimada',
    'Índice de Intensidade Operacional',
    'Lead Time Médio em Dias',
]

# %% Mapa de calor das correlações do conjunto inicial

corr_inicial = modelo_armazem[variaveis_iniciais].corr()

f, ax = plt.subplots(figsize=(11, 9), dpi=200)

sns.heatmap(corr_inicial,
            cmap=plt.cm.coolwarm,
            vmax=1,
            vmin=-1,
            center=0,
            square=True,
            linewidths=.5,
            annot=True,
            fmt='.2f',
            annot_kws={'size': 10},
            cbar_kws={"shrink": 0.50})

plt.tight_layout()
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
plt.savefig('figura1_correlacoes_iniciais.png', dpi=200, bbox_inches='tight')
plt.show()

# %% Verificação de Outliers por Boxplots

plt.figure(figsize=(12, 6))

modelo_armazem[variaveis].boxplot()

plt.ylabel('Valores')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('figura3_boxplot.png', dpi=200, bbox_inches='tight')
plt.show()

# %% Estatísticas descritivas das variáveis

armazem_cluster = modelo_armazem[variaveis].copy()

# Obtendo as estatísticas descritivas das variáveis

tab_descritivas = armazem_cluster.describe().T

# Matriz de correlações das variáveis

# Gerando a matriz de correlações de Pearson
matriz_corr = pg.rcorr(armazem_cluster, method='pearson', upper='pval',
                       decimals=4,
                       pval_stars={0.01: '***', 0.05: '**', 0.10: '*'})

# %% Mapa de calor indicando a correlação entre os atributos

# Matriz de correlações básica
corr = armazem_cluster.corr()

f, ax = plt.subplots(figsize=(10, 8), dpi=200)

sns.heatmap(corr,
            cmap=plt.cm.coolwarm,
            vmax=1,
            vmin=-1,
            center=0,
            square=True,
            linewidths=.5,
            annot=True,
            fmt='.2f',
            annot_kws={'size': 11},
            cbar_kws={"shrink": 0.75})

plt.tight_layout()
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
plt.savefig('figura2_correlacoes.png', dpi=200, bbox_inches='tight')
plt.show()

# %% Padronização por meio do Z-Score

# Aplicando o procedimento de ZScore nas variáveis que formam os clusters
armazem_pad = modelo_armazem[variaveis_cluster].apply(zscore, ddof=1)

# Visualizando o resultado do procedimento na média e desvio padrão
print(np.round(armazem_pad.mean(), 3))
print(np.round(armazem_pad.std(), 3))

# %% Identificação da quantidade de clusters (Método Elbow)

elbow = []
K = range(1, 11)  # ponto de parada pode ser parametrizado manualmente
for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', n_init=10, random_state=100).fit(armazem_pad)
    elbow.append(kmeanElbow.inertia_)

plt.figure(figsize=(10, 5), dpi=200)
plt.plot(K, elbow, marker='o')
plt.axvline(x=3, linestyle='dotted', color='red')
plt.xlabel('Nº Clusters', fontsize=12)
plt.xticks(range(1, 11))  # ajustar range de acordo com K acima
plt.ylabel('WCSS', fontsize=12)
plt.tight_layout()
plt.savefig('figura4_elbow.png', dpi=200, bbox_inches='tight')
plt.show()

print(np.round(elbow, 1))

# %% Identificação da quantidade de clusters (Método da Silhueta)

silhueta = []
I = range(2, 11)  # ponto de parada pode ser parametrizado manualmente
for i in I:
    kmeansSil = KMeans(n_clusters=i, init='random', n_init=10, random_state=100).fit(armazem_pad)
    silhueta.append(silhouette_score(armazem_pad, kmeansSil.labels_))

plt.figure(figsize=(10, 5), dpi=200)
plt.plot(range(2, 11), silhueta, color='purple', marker='o')  # ajustar range
plt.xlabel('Nº Clusters', fontsize=12)
plt.ylabel('Silhueta Média', fontsize=12)
plt.xticks(range(2, 11))
plt.axvline(x=silhueta.index(max(silhueta)) + 2, linestyle='dotted', color='red')
plt.tight_layout()
plt.savefig('figura5_silhueta.png', dpi=200, bbox_inches='tight')
plt.show()

print(np.round(silhueta, 3))

# %% Cluster Não Hierárquico K-means

# Vamos considerar 3 clusters, considerando as evidências anteriores!

kmeans_final = KMeans(n_clusters=3, init='random', n_init=10, random_state=100).fit(armazem_pad)

# Gerando a variável para identificarmos os clusters gerados

kmeans_clusters = kmeans_final.labels_
armazem_cluster['cluster_kmeans'] = kmeans_clusters
armazem_pad['cluster_kmeans'] = kmeans_clusters
armazem_cluster['cluster_kmeans'] = armazem_cluster['cluster_kmeans'].astype('category')
armazem_pad['cluster_kmeans'] = armazem_pad['cluster_kmeans'].astype('category')

print(armazem_cluster['cluster_kmeans'].value_counts())

# %% Análise de variância de um fator (ANOVA)

# Interpretação do output:

## cluster_kmeans MS: indica a variabilidade entre grupos
## Within MS: indica a variabilidade dentro dos grupos
## F: estatística de teste (cluster_kmeans MS / Within MS)
## p-unc: p-valor da estatística F
## se p-valor < 0.05: pelo menos um cluster apresenta média estatisticamente diferente dos demais

# Ocupação Média de Pallets por Mês
pg.anova(dv='Ocupação Média de Pallets por Mês',
         between='cluster_kmeans',
         data=armazem_pad,
         detailed=True).T

# Giro Mensal Médio de Estoque
pg.anova(dv='Giro Mensal Médio de Estoque',
         between='cluster_kmeans',
         data=armazem_pad,
         detailed=True).T

# Volatilidade de Ocupação
pg.anova(dv='Volatilidade de Ocupação',
         between='cluster_kmeans',
         data=armazem_pad,
         detailed=True).T

## A variável mais discriminante contém a maior estatística F
## O valor da estatística F é sensível ao tamanho da amostra

# %% ANOVA das variáveis que não participaram da clusterização

# Objetivo: verificar se as variáveis mantidas fora da matriz de distância
# ainda assim diferem entre os grupos formados. A Margem de Contribuição é a
# principal delas: por não ter participado da formação dos clusters, uma
# diferença significativa constitui validação externa do modelo.

armazem_externas = modelo_armazem[variaveis_externas].apply(zscore, ddof=1)
armazem_externas['cluster_kmeans'] = armazem_pad['cluster_kmeans'].values

# Margem de Contribuição Estimada
pg.anova(dv='Margem de Contribuição Estimada',
         between='cluster_kmeans',
         data=armazem_externas,
         detailed=True).T

# Índice de Intensidade Operacional
pg.anova(dv='Índice de Intensidade Operacional',
         between='cluster_kmeans',
         data=armazem_externas,
         detailed=True).T

# Lead Time Médio em Dias
pg.anova(dv='Lead Time Médio em Dias',
         between='cluster_kmeans',
         data=armazem_externas,
         detailed=True).T

# %% Teste de Robustez: reclusterização incluindo a margem de contribuição

# Objetivo: confirmar que a exclusão da Margem de Contribuição na etapa de
# clusterização não compromete a estrutura dos agrupamentos formados. Para
# isso, reincluímos a variável (padronizada) e reclusterizamos com o mesmo
# K=3, comparando o resultado com a classificação original.

armazem_robustez = modelo_armazem[variaveis_cluster +
                                  ['Margem de Contribuição Estimada']].apply(zscore, ddof=1)

kmeans_robustez = KMeans(n_clusters=3, init='random', n_init=10, random_state=100).fit(armazem_robustez)
armazem_cluster['cluster_robustez'] = kmeans_robustez.labels_

# Tabela cruzada: classificação original (3 variáveis) x classificação de
# robustez (4 variáveis)
comparacao_robustez = pd.crosstab(
    armazem_cluster['cluster_kmeans'].astype(int),
    armazem_cluster['cluster_robustez'],
    rownames=['Cluster Original (3 var.)'],
    colnames=['Cluster Robustez (4 var.)']
)
print(comparacao_robustez)

# Os rótulos de cluster são arbitrários entre rodadas, então a contagem usa a
# correspondência dominante de cada linha da tabela cruzada
mudaram = len(armazem_cluster) - comparacao_robustez.max(axis=1).sum()
print(f'Clientes que mudaram de cluster: {mudaram} de {len(armazem_cluster)}')

# %% Matriz de comparação do teste de robustez

fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
valor_max = comparacao_robustez.values.max()
ax.imshow(comparacao_robustez.values, cmap='Greens', vmin=0, vmax=valor_max)

for i in range(comparacao_robustez.shape[0]):
    for j in range(comparacao_robustez.shape[1]):
        valor = comparacao_robustez.values[i, j]
        cor_texto = 'white' if valor > valor_max / 2 else '#333333'
        ax.text(j, i, str(valor), ha='center', va='center',
                fontsize=16, fontweight='bold', color=cor_texto)

ax.set_xticks(range(comparacao_robustez.shape[1]))
ax.set_yticks(range(comparacao_robustez.shape[0]))
ax.set_xticklabels([f'Cluster {c}' for c in comparacao_robustez.columns], fontsize=10)
ax.set_yticklabels([f'Cluster {c}' for c in comparacao_robustez.index], fontsize=10)
ax.set_xlabel('Teste de robustez (4 variáveis, com a margem)', fontsize=10.5)
ax.set_ylabel('Modelo original (3 variáveis)', fontsize=10.5)
plt.tight_layout()
plt.savefig('figura8_robustez.png', dpi=200, bbox_inches='tight')
plt.show()

# %% Identificação das características dos clusters

# Agrupando o banco de dados

armazem_grupo = armazem_cluster.groupby(by=['cluster_kmeans'], observed=True)

# Estatísticas descritivas por grupo

tab_media_grupo = armazem_grupo[variaveis].mean().T

# Centroides em escores padronizados
armazem_z = modelo_armazem[variaveis].apply(zscore, ddof=1)
armazem_z['cluster_kmeans'] = kmeans_clusters
tab_media_grupo_z = armazem_z.groupby('cluster_kmeans')[variaveis].mean().T

print(tab_media_grupo)
print(tab_media_grupo_z.round(2))

# %% Gráfico do perfil padronizado dos clusters

cores_cluster = {0: '#C0504D', 1: '#4F81BD', 2: '#9BBB59'}
rotulos_cluster = {0: 'Cluster 0 – baixa escala e alta volatilidade (n = 8)',
                   1: 'Cluster 1 – alto giro e alta rentabilidade (n = 2)',
                   2: 'Cluster 2 – alta ocupação e estabilidade (n = 2)'}

rotulos_eixo = ['Índice de\nIntensidade\nOperacional', 'Lead Time\nMédio\n(dias)',
                'Ocupação\nMédia de\nPallets/mês', 'Margem de\nContribuição\nEstimada',
                'Giro Mensal\nMédio de\nEstoque', 'Volatilidade\nde Ocupação']

x = np.arange(len(variaveis))
largura = 0.26

fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)
for i, c in enumerate([0, 1, 2]):
    ax.bar(x + (i - 1) * largura, tab_media_grupo_z[c].values, largura,
           label=rotulos_cluster[c], color=cores_cluster[c], edgecolor='white')

ax.axhline(0, color='#444', lw=0.9)
ax.set_xticks(x)
ax.set_xticklabels(rotulos_eixo, fontsize=9.5)
ax.set_ylabel('Média padronizada (z-score)')
ax.grid(axis='y', linestyle=':', color='#bbb')
ax.set_axisbelow(True)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=9.5)
plt.tight_layout()
plt.savefig('figura6_perfil_clusters.png', dpi=200, bbox_inches='tight')
plt.show()

# %% Gráfico 3D dos clusters

fig2 = px.scatter_3d(armazem_cluster,
                     x='Ocupação Média de Pallets por Mês',
                     y='Margem de Contribuição Estimada',
                     z='Volatilidade de Ocupação',
                     color='cluster_kmeans')

fig2.write_html('figura7_clusters_3d.html')

# %% Reintrodução do Faturamento Bruto para análise financeira dos clusters

# Objetivo: trazer de volta a dimensão financeira (propositalmente excluída
# da clusterização) para avaliar os clusters sob a ótica de rentabilidade.
# Faturamento Bruto = Receita de Armazenagem + Receita de Serviços.

colunas_financeiras = [
    'Receita Mensal Bruta de Armazenagem',
    'Receita Mensal Bruta de Serviços',
]

armazem_cluster['Cliente'] = modelo_armazem['Cliente'].values
armazem_cluster['Faturamento Bruto'] = modelo_armazem[colunas_financeiras].sum(axis=1).values

tab_faturamento_cluster = armazem_cluster[[
    'Cliente',
    'Ocupação Média de Pallets por Mês',
    'Margem de Contribuição Estimada',
    'Giro Mensal Médio de Estoque',
    'Volatilidade de Ocupação',
    'cluster_kmeans',
    'Faturamento Bruto'
]]

print(armazem_cluster.groupby('cluster_kmeans', observed=True)['Faturamento Bruto'].sum())

# %% Gráfico de Bolhas: Faturamento Bruto x Perfil Operacional

fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

faturamento_max = armazem_cluster['Faturamento Bruto'].max()


def tamanho_bolha(faturamento):
    # escala em raiz quadrada para que a área da bolha seja proporcional
    # ao faturamento
    return 200 + 3200 * (faturamento / faturamento_max) ** 0.5


for c in [0, 1, 2]:
    grupo = armazem_cluster[armazem_cluster['cluster_kmeans'] == c]
    ax.scatter(grupo['Volatilidade de Ocupação'],
               grupo['Margem de Contribuição Estimada'] * 100,
               s=tamanho_bolha(grupo['Faturamento Bruto']),
               color=cores_cluster[c],
               alpha=0.75,
               edgecolors='white',
               linewidth=1.2,
               zorder=3)

for _, linha in armazem_cluster.iterrows():
    ax.annotate(linha['Cliente'],
                (linha['Volatilidade de Ocupação'], linha['Margem de Contribuição Estimada'] * 100),
                textcoords='offset points', xytext=(0, 16), ha='center', fontsize=9.5, zorder=4)

ax.set_xlabel('Volatilidade de Ocupação (CV, %)', fontsize=12)
ax.set_ylabel('Margem de Contribuição Estimada (%)', fontsize=12)
ax.grid(linestyle=':', color='#bbb')
ax.set_axisbelow(True)
for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

marcadores = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cores_cluster[c],
                         markersize=11, label=rotulos_cluster[c]) for c in [0, 1, 2]]
marcadores.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#999999',
                             markersize=14, label='Área da bolha ∝ faturamento bruto anual'))
ax.legend(handles=marcadores, loc='upper center', bbox_to_anchor=(0.5, -0.14),
          frameon=False, fontsize=9.5)
plt.tight_layout()
plt.savefig('figura9_grafico_bolhas.png', dpi=200, bbox_inches='tight')
plt.show()

# %% FIM
