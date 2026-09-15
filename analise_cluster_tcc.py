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
modelo_armazem = pd.read_csv('modelo_analise_revisado.csv', sep=';')

# %% Visualizando informações sobre os dados e variáveis

# Estrutura do banco de dados

print(modelo_armazem.info())

# %% Convertendo as colunas de Object em Float64 ( por causa das virgulas )


# Lista com as colunas que estão marcadas incorretamente como 'object'
colunas_para_corrigir = [
    'Índice de Intensidade Operacional',
    'Lead Time Médio em Dias',
    'Ocupação Média de Pallets por Mês',
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

# %% Verificação de Outliers por Boxplots

variaveis = [
    'Índice de Intensidade Operacional',
    'Lead Time Médio em Dias',
    'Ocupação Média de Pallets por Mês',
    'Margem de Contribuição Estimada',
    'Giro Mensal Médio de Estoque',
    'Volatilidade de Ocupação',
]

plt.figure(figsize=(12, 6))

modelo_armazem[variaveis].boxplot()

plt.title('Boxplots das Variáveis Utilizadas na Clusterização')
plt.ylabel('Valores')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

# %% Estatísticas descritivas das variáveis

# Primeiramente, vamos excluir as variáveis que não serão utilizadas

armazem_cluster = modelo_armazem.drop(
    columns=['Cliente', 'Segmento', 'Índice de Intensidade Operacional', 'Lead Time Médio em Dias'])

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

f, ax = plt.subplots(figsize=(10, 8), dpi=600)

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

plt.title('Matriz de Correlações')
plt.tight_layout()
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
ax.set_ylim(len(corr))

plt.show()

# %% Padronização por meio do Z-Score

# Aplicando o procedimento de ZScore
armazem_pad = armazem_cluster.apply(zscore, ddof=1)

# Visualizando o resultado do procedimento na média e desvio padrão
print(np.round(armazem_pad.mean(), 3))
print(np.round(armazem_pad.std(), 3))

# %% Gráfico 3D das observações

fig = px.scatter_3d(armazem_pad,
                    x='Giro Mensal Médio de Estoque',
                    y='Volatilidade de Ocupação',
                    z='Margem de Contribuição Estimada')

fig.write_html('armazem_inicial.html')

# %% Identificação da quantidade de clusters (Método Elbow)

elbow = []
K = range(1, 11)  # ponto de parada pode ser parametrizado manualmente
for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(armazem_pad)
    elbow.append(kmeanElbow.inertia_)

plt.figure(figsize=(16, 8), dpi=600)
plt.plot(K, elbow, marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.xticks(range(1, 11))  # ajustar range de acordo com K acima
plt.ylabel('WCSS', fontsize=16)
plt.title('Método de Elbow', fontsize=16)
plt.show()

# %% Identificação da quantidade de clusters (Método da Silhueta)

silhueta = []
I = range(2, 11)  # ponto de parada pode ser parametrizado manualmente
for i in I:
    kmeansSil = KMeans(n_clusters=i, init='random', random_state=100).fit(armazem_pad)
    silhueta.append(silhouette_score(armazem_pad, kmeansSil.labels_))

plt.figure(figsize=(16, 8), dpi=600)
plt.plot(range(2, 11), silhueta, color='purple', marker='o')  # ajustar range
plt.xlabel('Nº Clusters', fontsize=16)
plt.ylabel('Silhueta Média', fontsize=16)
plt.title('Método da Silhueta', fontsize=16)
plt.axvline(x=silhueta.index(max(silhueta)) + 2, linestyle='dotted', color='red')
plt.show()

# %% Cluster Não Hierárquico K-means

# Vamos considerar 3 clusters, considerando as evidências anteriores!

kmeans_final = KMeans(n_clusters=3, init='random', random_state=100).fit(armazem_pad)

# Gerando a variável para identificarmos os clusters gerados

kmeans_clusters = kmeans_final.labels_
armazem_cluster['cluster_kmeans'] = kmeans_clusters
armazem_pad['cluster_kmeans'] = kmeans_clusters
armazem_cluster['cluster_kmeans'] = armazem_cluster['cluster_kmeans'].astype('category')
armazem_pad['cluster_kmeans'] = armazem_pad['cluster_kmeans'].astype('category')

# %% Análise de variância de um fator (ANOVA)

# Interpretação do output:

## cluster_kmeans MS: indica a variabilidade entre grupos
## Within MS: indica a variabilidade dentro dos grupos
## F: estatística de teste (cluster_kmeans MS / Within MS)
## p-unc: p-valor da estatística F
## se p-valor < 0.05: pelo menos um cluster apresenta média estatisticamente diferente dos ## demais


# Margem de Contribuição Estimada
pg.anova(dv='Margem de Contribuição Estimada',
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

# Ocupação Média de Pallets por Mês
pg.anova(dv='Ocupação Média de Pallets por Mês',
         between='cluster_kmeans',
         data=armazem_pad,
         detailed=True).T

## Todas as variáveis foram estatisticamente relevantes para a formação de pelo menos um dos clusters
## A variável mais discriminante contém a maior estatística F
## O valor da estatística F é sensível ao tamanho da amostra

# %% ANOVA complementar: variáveis excluídas da clusterização

# Objetivo: verificar se Índice de Intensidade Operacional e Lead Time Médio
# em Dias -- que não participaram da formação dos clusters -- ainda assim
# diferem entre os grupos formados. Se não diferirem, isso reforça que a
# exclusão dessas variáveis na etapa de clusterização foi uma decisão segura.

armazem_excluidas = modelo_armazem[['Índice de Intensidade Operacional',
                                    'Lead Time Médio em Dias']].copy()
armazem_excluidas['cluster_kmeans'] = armazem_pad['cluster_kmeans'].values

# Índice de Intensidade Operacional
pg.anova(dv='Índice de Intensidade Operacional',
         between='cluster_kmeans',
         data=armazem_excluidas,
         detailed=True).T

# Lead Time Médio em Dias
pg.anova(dv='Lead Time Médio em Dias',
         between='cluster_kmeans',
         data=armazem_excluidas,
         detailed=True).T

## Ambas as variáveis apresentaram p-valor > 0.05, ou seja, não diferenciam
## estatisticamente os clusters formados pelas 4 variáveis operacionais

# %% Teste de Robustez: reclusterização incluindo as variáveis excluídas

# Objetivo: confirmar que a exclusão de "Índice de Intensidade Operacional"
# e "Lead Time Médio em Dias" na etapa de clusterização não compromete a
# estrutura dos agrupamentos formados. Para isso, reincluímos as duas
# variáveis (padronizadas) e reclusterizamos com o mesmo K=3, comparando
# o resultado com a classificação original (cluster_kmeans).

armazem_robustez = armazem_pad.drop(columns='cluster_kmeans').copy()
armazem_robustez[['Índice de Intensidade Operacional', 'Lead Time Médio em Dias']] = (
    modelo_armazem[['Índice de Intensidade Operacional', 'Lead Time Médio em Dias']]
    .apply(zscore, ddof=1)
)

kmeans_robustez = KMeans(n_clusters=3, init='random', random_state=100).fit(armazem_robustez)
armazem_cluster['cluster_robustez'] = kmeans_robustez.labels_

# Tabela cruzada: classificação original (4 variáveis) x classificação de
# robustez (6 variáveis)
comparacao_robustez = pd.crosstab(
    armazem_cluster['cluster_kmeans'].astype(int),
    armazem_cluster['cluster_robustez'],
    rownames=['Cluster Original (4 var.)'],
    colnames=['Cluster Robustez (6 var.)']
)
print(comparacao_robustez)

mudaram = (armazem_cluster['cluster_kmeans'].astype(int) != armazem_cluster['cluster_robustez']).sum()
print(f'Clientes que mudaram de cluster: {mudaram} de {len(armazem_cluster)}')

# %% Figura 21 - Matriz de comparação do teste de robustez

fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=200)
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
ax.set_xticklabels(comparacao_robustez.columns, fontsize=10, rotation=20, ha='right')
ax.set_yticklabels(comparacao_robustez.index, fontsize=10)
ax.set_title(
    f'Estabilidade dos Agrupamentos ao Reincluir\n'
    f'Índice de Intensidade Operacional e Lead Time\n'
    f'({mudaram} de {len(armazem_cluster)} clientes mudaram de cluster)',
    fontsize=11
)
plt.tight_layout()
plt.savefig('figura21_teste_robustez.png', dpi=200, bbox_inches='tight')
plt.show()

# %% Gráfico 3D dos clusters

# Perspectiva 1

fig1 = px.scatter_3d(armazem_cluster,
                     x='Margem de Contribuição Estimada',
                     y='Giro Mensal Médio de Estoque',
                     z='Volatilidade de Ocupação',
                     color='cluster_kmeans')

fig1.write_html('armazem_1.html')

# Perspectiva 2

fig2 = px.scatter_3d(armazem_cluster,
                     x='Ocupação Média de Pallets por Mês',
                     y='Margem de Contribuição Estimada',
                     z='Volatilidade de Ocupação',
                     color='cluster_kmeans')

fig2.write_html('armazem_2.html')

# Perspectiva 3

fig3 = px.scatter_3d(armazem_cluster,
                     x='Ocupação Média de Pallets por Mês',
                     y='Margem de Contribuição Estimada',
                     z='Giro Mensal Médio de Estoque',
                     color='cluster_kmeans')

fig3.write_html('armazem_3.html')

# %% Identificação das características dos clusters

# Agrupando o banco de dados

armazem_grupo = armazem_cluster.groupby(by=['cluster_kmeans'], observed=True)

# Estatísticas descritivas por grupo

tab_media_grupo = armazem_grupo.mean().T

# %% Reintrodução do Faturamento Bruto para análise financeira dos clusters

# Objetivo: trazer de volta a dimensão financeira (propositalmente excluída
# da clusterização, por ser uma variável dependente da ocupação e do giro)
# para avaliar os clusters sob a ótica de rentabilidade.
#
# OBS: o Faturamento Bruto não está no CSV revisado usado na clusterização
# ('modelo_analise_revisado.csv'), pois esse arquivo já vem com as colunas
# financeiras removidas. Os valores de receita estão no CSV original,
# anterior à revisão ('modelo_analise.csv'). Carregamos esse arquivo à
# parte e mesclamos por 'Cliente'.

modelo_financeiro = pd.read_csv('modelo_analise.csv', sep=';', encoding='utf-8-sig')

colunas_financeiras = [
    'Receita Mensal Bruta de Armazenagem',
    'Receita Mensal Bruta de Serviços',
]

for col in colunas_financeiras:
    modelo_financeiro[col] = modelo_financeiro[col].astype(str).str.replace(',', '.', regex=False)
    modelo_financeiro[col] = pd.to_numeric(modelo_financeiro[col], errors='coerce')

# Faturamento Bruto = Receita de Armazenagem + Receita de Serviços
modelo_financeiro['Faturamento Bruto'] = modelo_financeiro[colunas_financeiras].sum(axis=1)

armazem_cluster['Cliente'] = modelo_armazem['Cliente']
armazem_cluster = armazem_cluster.merge(
    modelo_financeiro[['Cliente', 'Faturamento Bruto']],
    on='Cliente',
    how='left'
)

tab_faturamento_cluster = armazem_cluster[[
    'Cliente',
    'Ocupação Média de Pallets por Mês',
    'Margem de Contribuição Estimada',
    'Giro Mensal Médio de Estoque',
    'Volatilidade de Ocupação',
    'cluster_kmeans',
    'Faturamento Bruto'
]]

# %% Figura 20 - Gráfico de Bolhas: Faturamento Bruto x Perfil Operacional

cores_cluster = {0: '#e74c3c', 1: '#27ae60', 2: '#8e44ad'}
rotulos_cluster = {0: 'Cluster 0 (Risco)', 1: 'Cluster 1 (Alto Giro)', 2: 'Cluster 2 (Alta Eficiência)'}

fig, ax = plt.subplots(figsize=(9, 6), dpi=200)

faturamento_max = armazem_cluster['Faturamento Bruto'].max()


def tamanho_bolha(faturamento):
    # escala em raiz quadrada para que a área da bolha seja proporcional
    # ao faturamento
    return 200 + 3200 * (faturamento / faturamento_max) ** 0.5


clusters_plotados = set()
for _, linha in armazem_cluster.iterrows():
    cl = int(linha['cluster_kmeans'])
    rotulo = rotulos_cluster[cl] if cl not in clusters_plotados else None
    clusters_plotados.add(cl)
    ax.scatter(
        linha['Volatilidade de Ocupação'],
        linha['Margem de Contribuição Estimada'] * 100,
        s=tamanho_bolha(linha['Faturamento Bruto']),
        color=cores_cluster[cl],
        alpha=0.75,
        edgecolors='white',
        linewidth=1.2,
        label=rotulo,
        zorder=3
    )
    ax.annotate(
        linha['Cliente'],
        (linha['Volatilidade de Ocupação'], linha['Margem de Contribuição Estimada'] * 100),
        textcoords='offset points', xytext=(0, 14), ha='center', fontsize=9, color='#222'
    )

ax.set_xlabel('Volatilidade de Ocupação', fontsize=12)
ax.set_ylabel('Margem de Contribuição Estimada (%)', fontsize=12)
ax.set_title(
    'Faturamento Bruto x Perfil Operacional por Cluster\n(tamanho da bolha = Faturamento Bruto)',
    fontsize=13
)
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.savefig('figura20_grafico_bolhas.png', dpi=200, bbox_inches='tight')
plt.show()

# %% FIM
