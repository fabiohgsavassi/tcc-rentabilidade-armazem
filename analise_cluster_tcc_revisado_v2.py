# -*- coding: utf-8 -*-
"""
Trabalho de TCC do MBA em Data Science e Analytics USP ESALQ
Modelagem para Análise de Padrões de Rentabilidade de
Clientes em Armazéns Logísticos segundo Técnica de Clustering

Aluno(a): Fábio Henrique Gonçales Savassi
Orientador: Prof. Elias Soares de Figueiredo

===============================================================================
VERSÃO 2 - correção metodológica sobre 'analise_cluster_tcc_revisado.py'
===============================================================================
O QUE MUDOU E POR QUÊ

1) A Margem de Contribuição Estimada SAIU da matriz de distância do K-means.
   Na versão anterior a clusterização usava 4 variáveis (ocupação, margem, giro
   e volatilidade) e, em seguida, a ANOVA da margem era apresentada como
   validação. Isso é circular: a margem diferia entre os grupos porque havia
   sido usada para criá-los. Agora a matriz contém apenas as três variáveis de
   consumo de recursos, e a margem passa para o bloco de validação externa.
   Resultado verificado: a partição dos 12 clientes é IDÊNTICA e todos os
   valores de F, p e eta² permanecem os mesmos. A silhueta em k=3 sobe de
   0,530 para 0,604.

2) O teste de robustez foi invertido. Antes: base de 4 variáveis, teste
   reincluindo intensidade e lead time. Agora: base de 3 variáveis, teste
   reincluindo a margem — que é o teste que interessa e que se mantém estável
   em todas as sementes.

3) A comparação do teste de robustez passou a alinhar os rótulos antes de
   contar. Rótulos do K-means são arbitrários: comparar 'cluster_kmeans' com
   'cluster_robustez' diretamente só funcionava por coincidência da semente.

4) Elbow, silhueta e mapa de calor passaram a ser salvos em arquivo, para que
   as figuras do texto venham comprovadamente deste script.

5) n_init explicitado. A semente fica declarada no texto para permitir réplica.
===============================================================================
"""
# %% Importando os pacotes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pingouin as pg
import plotly.express as px
import plotly.io as pio

pio.renderers.default = 'browser'

SEED = 100     # semente declarada no texto
N_INIT = 10    # nº de inicializações do K-means

# %% Importando o banco de dados

modelo_armazem = pd.read_csv('modelo_analise_revisado.csv', sep=';')
print(modelo_armazem.info())

# %% Convertendo as colunas de Object em Float64 (por causa das vírgulas)

colunas_para_corrigir = [
    'Índice de Intensidade Operacional',
    'Lead Time Médio em Dias',
    'Ocupação Média de Pallets por Mês',
    'Margem de Contribuição Estimada',
    'Giro Mensal Médio de Estoque',
    'Volatilidade de Ocupação',
]

for col in colunas_para_corrigir:
    if col in modelo_armazem.columns:
        modelo_armazem[col] = modelo_armazem[col].astype(str)
        modelo_armazem[col] = modelo_armazem[col].str.replace(',', '.', regex=False)
        modelo_armazem[col] = pd.to_numeric(modelo_armazem[col], errors='coerce')

print(modelo_armazem.info())
print('Registros nulos:', modelo_armazem[colunas_para_corrigir].isna().sum().sum())

# %% Papel de cada variável   <<< MUDANÇA CENTRAL

# Candidatas: todas as que passaram pelo diagnóstico de correlação.
VARIAVEIS_CANDIDATAS = colunas_para_corrigir

# Formadoras: compõem a MATRIZ DE DISTÂNCIA do K-means (consumo de recursos).
VARIAVEIS_CLUSTER = [
    'Ocupação Média de Pallets por Mês',
    'Giro Mensal Médio de Estoque',
    'Volatilidade de Ocupação',
]

# Externas: NÃO entram no K-means. Servem para validar o modelo.
VARIAVEIS_EXTERNAS = [
    'Margem de Contribuição Estimada',      # <<< saiu da clusterização
    'Índice de Intensidade Operacional',
    'Lead Time Médio em Dias',
]

armazem = modelo_armazem[VARIAVEIS_CANDIDATAS].copy()

# %% Figura 3 - Boxplots das seis variáveis candidatas

plt.figure(figsize=(12, 6))
armazem.boxplot()
plt.ylabel('Valores (unidades originais)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('figura3_boxplot.png', dpi=200, bbox_inches='tight')
plt.show()

# %% Figura 2 - Mapa de calor das seis candidatas

tab_descritivas = armazem.describe().T

matriz_corr = pg.rcorr(armazem, method='pearson', upper='pval', decimals=4,
                       pval_stars={0.01: '***', 0.05: '**', 0.10: '*'})

corr = armazem.corr()
f, ax = plt.subplots(figsize=(10, 8), dpi=200)
sns.heatmap(corr, cmap=plt.cm.coolwarm, vmax=1, vmin=-1, center=0, square=True,
            linewidths=.5, annot=True, fmt='.2f', annot_kws={'size': 11},
            cbar_kws={"shrink": 0.75})
plt.tight_layout()
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
plt.savefig('figura2_correlacoes.png', dpi=200, bbox_inches='tight')
plt.show()
# Obs.: a linha ax.set_ylim(len(corr)) da versão anterior foi removida — era
# um contorno para um bug antigo do seaborn e hoje corta a figura.

# %% Padronização por meio do Z-Score - APENAS as variáveis formadoras

armazem_pad = armazem[VARIAVEIS_CLUSTER].apply(zscore, ddof=1)
print(np.round(armazem_pad.mean(), 3))   # ~0
print(np.round(armazem_pad.std(), 3))    # ~1

# %% Figura 4 - Método de Elbow

elbow = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, init='random', n_init=N_INIT, random_state=SEED).fit(armazem_pad)
    elbow.append(km.inertia_)

plt.figure(figsize=(10, 5), dpi=200)
plt.plot(list(K), elbow, marker='o')
plt.axvline(x=3, linestyle='dotted', color='red')
plt.xlabel('Nº de clusters (k)', fontsize=12)
plt.ylabel('WCSS (soma dos quadrados intragrupos)', fontsize=12)
plt.xticks(list(K))
plt.tight_layout()
plt.savefig('figura4_elbow.png', dpi=200, bbox_inches='tight')
plt.show()
print('WCSS:', [round(v, 1) for v in elbow])      # esperado 33,0 / 17,4 / 7,0 / 3,6 ...

# %% Figura 5 - Método da Silhueta

silhueta = []
I = range(2, 11)
for i in I:
    km = KMeans(n_clusters=i, init='random', n_init=N_INIT, random_state=SEED).fit(armazem_pad)
    silhueta.append(silhouette_score(armazem_pad, km.labels_))

plt.figure(figsize=(10, 5), dpi=200)
plt.plot(list(I), silhueta, color='purple', marker='o')
plt.axvline(x=silhueta.index(max(silhueta)) + 2, linestyle='dotted', color='red')
plt.xlabel('Nº de clusters (k)', fontsize=12)
plt.ylabel('Coeficiente médio de silhueta', fontsize=12)
plt.xticks(list(I))
plt.tight_layout()
plt.savefig('figura5_silhueta.png', dpi=200, bbox_inches='tight')
plt.show()
print('Silhueta:', [round(v, 3) for v in silhueta])   # esperado máx. 0,604 em k=3

# %% Cluster Não Hierárquico K-means (k = 3, três variáveis)

kmeans_final = KMeans(n_clusters=3, init='random', n_init=N_INIT,
                      random_state=SEED).fit(armazem_pad)
kmeans_clusters = kmeans_final.labels_

armazem_cluster = armazem.copy()
armazem_cluster['cluster_kmeans'] = kmeans_clusters
armazem_pad['cluster_kmeans'] = pd.Categorical(kmeans_clusters)
print(armazem_cluster['cluster_kmeans'].value_counts())   # esperado 8 / 2 / 2

# %% Tabela 3 e Figura 6 - centroides dos clusters

centroides_orig = armazem_cluster.groupby('cluster_kmeans')[VARIAVEIS_CANDIDATAS].mean()

z_todas = armazem[VARIAVEIS_CANDIDATAS].apply(zscore, ddof=1)
z_todas['cluster_kmeans'] = kmeans_clusters
centroides_z = z_todas.groupby('cluster_kmeans')[VARIAVEIS_CANDIDATAS].mean()

print(centroides_orig.round(3))
print(centroides_z.round(2))

rotulos = {0: 'Cluster 0 – baixa escala e alta volatilidade (n = 8)',
           1: 'Cluster 1 – alto giro e alta rentabilidade (n = 2)',
           2: 'Cluster 2 – alta ocupação e estabilidade (n = 2)'}
cores = {0: '#C0504D', 1: '#4F81BD', 2: '#9BBB59'}

rot_eixo = ['Índice de\nIntensidade\nOperacional', 'Lead Time\nMédio\n(dias)',
            'Ocupação\nMédia de\nPallets/mês', 'Margem de\nContribuição\nEstimada',
            'Giro Mensal\nMédio de\nEstoque', 'Volatilidade\nde Ocupação']

x = np.arange(len(VARIAVEIS_CANDIDATAS)); largura = 0.26
fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)
for i, c in enumerate([0, 1, 2]):
    ax.bar(x + (i - 1) * largura, centroides_z.loc[c, VARIAVEIS_CANDIDATAS].values,
           largura, label=rotulos[c], color=cores[c], edgecolor='white')
ax.axhline(0, color='#444', lw=0.9)
ax.set_xticks(x); ax.set_xticklabels(rot_eixo, fontsize=9.5)
ax.set_ylabel('Média padronizada (z-score)')
ax.grid(axis='y', ls=':', color='#bbb'); ax.set_axisbelow(True)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=9.5)
plt.tight_layout()
plt.savefig('figura6_perfil_clusters.png', dpi=200, bbox_inches='tight')
plt.show()

# %% Tabela 4 (parte 1) - ANOVA das variáveis FORMADORAS
# Mede o poder discriminante relativo de cada variável na definição dos grupos.

def resumo_anova(dv, dados):
    """ANOVA de um fator, tolerante à versão do pingouin.
    O nome da coluna do p-valor mudou de 'p-unc' para 'p_unc' a partir da 0.6."""
    r = pg.anova(dv=dv, between='cluster_kmeans', data=dados, detailed=True)
    col_p = 'p-unc' if 'p-unc' in r.columns else 'p_unc'
    return r.loc[0, 'F'], r.loc[0, col_p], r.loc[0, 'np2']


print('\n--- ANOVA: variáveis que formaram os clusters ---')
for v in VARIAVEIS_CLUSTER:
    F, p, eta2 = resumo_anova(v, armazem_pad)
    print(f'{v}: F={F:.2f}  p={p:.4f}  eta2={eta2:.3f}')

# %% Tabela 4 (parte 2) - ANOVA das variáveis EXTERNAS  <<< validação do modelo
# A margem entra AQUI. Como não participou da formação dos grupos, uma
# diferença significativa é evidência externa e não consequência do método.

armazem_externas = modelo_armazem[VARIAVEIS_EXTERNAS].apply(zscore, ddof=1)
armazem_externas['cluster_kmeans'] = pd.Categorical(kmeans_clusters)

print('\n--- ANOVA: variáveis externas (não participaram da clusterização) ---')
for v in VARIAVEIS_EXTERNAS:
    F, p, eta2 = resumo_anova(v, armazem_externas)
    print(f'{v}: F={F:.2f}  p={p:.4f}  eta2={eta2:.3f}')

# Esperado:
#   Margem de Contribuição     F=7,64  p=0,0115  eta2=0,629  -> validação externa
#   Índice de Intensidade Op.  F=2,39  p=0,1468  eta2=0,347  -> não discrimina
#   Lead Time Médio em Dias    F=0,86  p=0,4571  eta2=0,160  -> não discrimina

# %% Figura 8 - Teste de robustez: reincluir a margem

armazem_robustez = modelo_armazem[VARIAVEIS_CLUSTER +
                                  ['Margem de Contribuição Estimada']].apply(zscore, ddof=1)
kmeans_rob = KMeans(n_clusters=3, init='random', n_init=N_INIT,
                    random_state=SEED).fit(armazem_robustez)

# Alinhamento de rótulos: os números de cluster do K-means são arbitrários,
# então a comparação direta entre duas rodadas não é válida sem casá-los antes.
from itertools import permutations

def alinhar(base, outro, k=3):
    melhor, menor = None, len(base) + 1
    for p in permutations(range(k)):
        cand = np.array([p[v] for v in outro])
        n = int((np.asarray(base) != cand).sum())
        if n < menor:
            menor, melhor = n, cand
    return melhor, menor

rob_alinhado, mudaram = alinhar(kmeans_clusters, kmeans_rob.labels_)
armazem_cluster['cluster_robustez'] = rob_alinhado

comparacao_robustez = pd.crosstab(
    armazem_cluster['cluster_kmeans'],
    armazem_cluster['cluster_robustez'],
    rownames=['Modelo original (3 variáveis)'],
    colnames=['Robustez (4 variáveis, com a margem)'])
print(comparacao_robustez)
print(f'Clientes que mudaram de cluster: {mudaram} de {len(armazem_cluster)}')

fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
valor_max = comparacao_robustez.values.max()
ax.imshow(comparacao_robustez.values, cmap='Greens', vmin=0, vmax=valor_max)
for i in range(comparacao_robustez.shape[0]):
    for j in range(comparacao_robustez.shape[1]):
        v = comparacao_robustez.values[i, j]
        ax.text(j, i, str(v), ha='center', va='center', fontsize=16, fontweight='bold',
                color='white' if v > valor_max / 2 else '#333333')
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels([f'Cluster {c}' for c in comparacao_robustez.columns], fontsize=10)
ax.set_yticklabels([f'Cluster {c}' for c in comparacao_robustez.index], fontsize=10)
ax.set_xlabel('Teste de robustez (4 variáveis, com a margem)', fontsize=10.5)
ax.set_ylabel('Modelo original (3 variáveis)', fontsize=10.5)
plt.tight_layout()
plt.savefig('figura8_robustez.png', dpi=200, bbox_inches='tight')
plt.show()

# Estabilidade em várias sementes (número reportado no texto)
iguais = 0
for s in (100, 0, 1, 2, 7, 42, 123, 2024):
    lab = KMeans(n_clusters=3, init='random', n_init=N_INIT, random_state=s).fit(armazem_robustez).labels_
    iguais += (alinhar(kmeans_clusters, lab)[1] == 0)
print(f'Partição preservada em {iguais} de 8 sementes')

# %% Figura 7 - Gráfico 3D dos clusters

armazem_plot = armazem_cluster.copy()
armazem_plot['cluster'] = armazem_plot['cluster_kmeans'].astype(str)

fig7 = px.scatter_3d(armazem_plot,
                     x='Ocupação Média de Pallets por Mês',
                     y='Margem de Contribuição Estimada',
                     z='Volatilidade de Ocupação',
                     color='cluster')
fig7.write_html('figura7_clusters_3d.html')

# %% Reintrodução do Faturamento Bruto - análise financeira dos clusters
# O faturamento entra SÓ AQUI, depois de formados E validados os grupos.
# Faturamento Bruto = Receita de Armazenagem + Receita de Serviços.
# O "Valor Mensal Bruto de Mercadorias" é o valor da carga custodiada, não é
# receita do operador, e por isso não entra na soma.

modelo_financeiro = pd.read_csv('modelo_analise.csv', sep=';', encoding='utf-8-sig')

colunas_financeiras = ['Receita Mensal Bruta de Armazenagem',
                       'Receita Mensal Bruta de Serviços']
for col in colunas_financeiras:
    modelo_financeiro[col] = modelo_financeiro[col].astype(str).str.replace(',', '.', regex=False)
    modelo_financeiro[col] = pd.to_numeric(modelo_financeiro[col], errors='coerce')

modelo_financeiro['Faturamento Bruto'] = modelo_financeiro[colunas_financeiras].sum(axis=1)

armazem_cluster['Cliente'] = modelo_armazem['Cliente'].values
armazem_cluster = armazem_cluster.merge(
    modelo_financeiro[['Cliente', 'Faturamento Bruto']], on='Cliente', how='left')

tab_faturamento_cluster = armazem_cluster[[
    'Cliente', 'Ocupação Média de Pallets por Mês', 'Margem de Contribuição Estimada',
    'Giro Mensal Médio de Estoque', 'Volatilidade de Ocupação',
    'cluster_kmeans', 'Faturamento Bruto']]
print(armazem_cluster.groupby('cluster_kmeans')['Faturamento Bruto'].agg(['sum', 'mean']))

# %% Figura 9 - Faturamento bruto x perfil operacional
# Sem título interno: a legenda da figura cumpre esse papel no texto.

from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
fat_max = armazem_cluster['Faturamento Bruto'].max()

for c in [0, 1, 2]:
    sub = armazem_cluster[armazem_cluster['cluster_kmeans'] == c]
    ax.scatter(sub['Volatilidade de Ocupação'], sub['Margem de Contribuição Estimada'] * 100,
               s=200 + 3200 * (sub['Faturamento Bruto'] / fat_max) ** 0.5,
               color=cores[c], alpha=0.72, edgecolors='white', linewidth=1.2, zorder=3)

for _, linha in armazem_cluster.iterrows():
    ax.annotate(linha['Cliente'],
                (linha['Volatilidade de Ocupação'], linha['Margem de Contribuição Estimada'] * 100),
                textcoords='offset points', xytext=(0, 16), ha='center', fontsize=9.5, zorder=4)

ax.set_xlabel('Volatilidade de Ocupação (CV, %)', fontsize=12)
ax.set_ylabel('Margem de Contribuição Estimada (%)', fontsize=12)
ax.grid(ls=':', color='#bbb'); ax.set_axisbelow(True)
for s in ['top', 'right']: ax.spines[s].set_visible(False)

handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=cores[c],
                  markersize=11, label=rotulos[c]) for c in [0, 1, 2]]
handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#999',
                      markersize=14, label='Área da bolha ∝ faturamento bruto anual'))
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.14),
          frameon=False, fontsize=9.5)
plt.tight_layout()
plt.savefig('figura9_grafico_bolhas.png', dpi=200, bbox_inches='tight')
plt.show()

# %% FIM
