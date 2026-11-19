── CC_Partie_A (2).ipynb (28800 tokens)


/CC_Partie_A (2).ipynb:
--------------------------------------------------------------------------------
   1 | {
   2 |   "cells": [
   3 |     {
   4 |       "cell_type": "markdown",
   5 |       "source": [
   6 |         "\n",
   7 |         "[🚀 Lancer le TP des 29 algorithmes dans Google Colab](https://colab.research.google.com/github/larhlimi-ia/CC-Partie-A-FIN-3/blob/main/CC%20Partie%20A%20Exemples_29_Algorithmes_Colab_consigne_etudiants.ipynb)\n"
   8 |       ],
   9 |       "metadata": {
  10 |         "id": "axrmLljFFVhX"
  11 |       },
  12 |       "id": "axrmLljFFVhX"
  13 |     },
  14 |     {
  15 |       "cell_type": "markdown",
  16 |       "id": "490c1b79",
  17 |       "metadata": {
  18 |         "id": "490c1b79"
  19 |       },
  20 |       "source": [
  21 |         "# 29 exemples Python — principaux algorithmes de Machine Learning\n",
  22 |         "\n",
  23 |         "**Objectif :** expérimenter chaque méthode sur des données reproductibles. Les données de démonstration sont synthétiques, sauf `load_wine` ; elles ne constituent pas des résultats économiques réels. Les exercices appliqués utilisent des sources publiques dans les TP séparés.\n",
  24 |         "\n",
  25 |         "Google Colab : exécuter les cellules dans l’ordre."
  26 |       ]
  27 |     },
  28 |     {
  29 |       "cell_type": "code",
  30 |       "execution_count": 1,
  31 |       "id": "9135a155",
  32 |       "metadata": {
  33 |         "id": "9135a155"
  34 |       },
  35 |       "outputs": [],
  36 |       "source": [
  37 |         "!pip -q install scikit-learn xgboost umap-learn gymnasium"
  38 |       ]
  39 |     },
  40 |     {
  41 |       "cell_type": "code",
  42 |       "execution_count": 2,
  43 |       "id": "cf24f0ea",
  44 |       "metadata": {
  45 |         "colab": {
  46 |           "base_uri": "https://localhost:8080/"
  47 |         },
  48 |         "id": "cf24f0ea",
  49 |         "outputId": "7437fd44-1956-4962-97cb-f7b4fded9bf8"
  50 |       },
  51 |       "outputs": [
  52 |         {
  53 |           "output_type": "stream",
  54 |           "name": "stdout",
  55 |           "text": [
  56 |             "Régression: (280, 5) Classification: (400, 6)\n"
  57 |           ]
  58 |         }
  59 |       ],
  60 |       "source": [
  61 |         "# Données pédagogiques reproductibles (les étiquettes sont connues pour l'évaluation)\n",
  62 |         "import numpy as np\n",
  63 |         "import pandas as pd\n",
  64 |         "import matplotlib.pyplot as plt\n",
  65 |         "from sklearn.datasets import make_regression, make_classification, make_blobs, load_wine\n",
  66 |         "from sklearn.model_selection import train_test_split\n",
  67 |         "from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score, silhouette_score\n",
  68 |         "from sklearn.preprocessing import StandardScaler\n",
  69 |         "from sklearn.pipeline import make_pipeline\n",
  70 |         "GRAINE = 42\n",
  71 |         "X_reg, y_reg = make_regression(n_samples=280, n_features=5, noise=13, random_state=GRAINE)\n",
  72 |         "X_cls, y_cls = make_classification(n_samples=400, n_features=6, n_informative=4, n_redundant=0, random_state=GRAINE)\n",
  73 |         "X_clu, etiquettes_reference = make_blobs(n_samples=240, centers=3, cluster_std=0.85, random_state=GRAINE)\n",
  74 |         "Xr_ent, Xr_test, yr_ent, yr_test = train_test_split(X_reg, y_reg, test_size=.25, random_state=GRAINE)\n",
  75 |         "Xc_ent, Xc_test, yc_ent, yc_test = train_test_split(X_cls, y_cls, test_size=.25, stratify=y_cls, random_state=GRAINE)\n",
  76 |         "print('Régression:',X_reg.shape,'Classification:',X_cls.shape)"
  77 |       ]
  78 |     },
  79 |     {
  80 |       "cell_type": "markdown",
  81 |       "id": "e8502bdc",
  82 |       "metadata": {
  83 |         "id": "e8502bdc"
  84 |       },
  85 |       "source": [
  86 |         "## 1. Régression linéaire\n",
  87 |         "\n",
  88 |         "Construire le modèle, afficher une métrique ou une sortie et interpréter ses limites. Les exemples n’utilisent pas de test final pour le réglage d’hyperparamètres."
  89 |       ]
  90 |     },
  91 |     {
  92 |       "cell_type": "code",
  93 |       "execution_count": null,
  94 |       "id": "8250aa51",
  95 |       "metadata": {
  96 |         "colab": {
  97 |           "base_uri": "https://localhost:8080/"
  98 |         },
  99 |         "id": "8250aa51",
 100 |         "outputId": "61aec64d-089a-42a6-acd6-9b85aafa8787"
 101 |       },
 102 |       "outputs": [
 103 |         {
 104 |           "output_type": "stream",
 105 |           "name": "stdout",
 106 |           "text": [
 107 |             "MAE: 9.353512045154824\n"
 108 |           ]
 109 |         }
 110 |       ],
 111 |       "source": [
 112 |         "from sklearn.linear_model import LinearRegression\n",
 113 |         "modele=LinearRegression().fit(Xr_ent,yr_ent)\n",
 114 |         "print('MAE:',mean_absolute_error(yr_test,modele.predict(Xr_test)))"
 115 |       ]
 116 |     },
 117 |     {
 118 |       "cell_type": "markdown",
 119 |       "id": "23b9dfa0",
 120 |       "metadata": {
 121 |         "id": "23b9dfa0"
 122 |       },
 123 |       "source": [
 124 |         "## 2. Ridge\n",
 125 |         "\n",
 126 |         "Construire le modèle, afficher une métrique ou une sortie et interpréter ses limites. Les exemples n’utilisent pas de test final pour le réglage d’hyperparamètres."
 127 |       ]
 128 |     },
 129 |     {
 130 |       "cell_type": "code",
 131 |       "execution_count": null,
 132 |       "id": "db49b9c1",
 133 |       "metadata": {
 134 |         "colab": {
 135 |           "base_uri": "https://localhost:8080/"
 136 |         },
 137 |         "id": "db49b9c1",
 138 |         "outputId": "544194dd-1964-477a-967e-25e4af763231"
 139 |       },
 140 |       "outputs": [
 141 |         {
 142 |           "output_type": "stream",
 143 |           "name": "stdout",
 144 |           "text": [
 145 |             "MAE: 9.367832312622433\n"
 146 |           ]
 147 |         }
 148 |       ],
 149 |       "source": [
 150 |         "from sklearn.linear_model import Ridge\n",
 151 |         "modele=make_pipeline(StandardScaler(),Ridge(alpha=1.0)).fit(Xr_ent,yr_ent)\n",
 152 |         "print('MAE:',mean_absolute_error(yr_test,modele.predict(Xr_test)))"
 153 |       ]
 154 |     },
 155 |     {
 156 |       "cell_type": "markdown",
 157 |       "id": "c43cd181",
 158 |       "metadata": {
 159 |         "id": "c43cd181"
 160 |       },
 161 |       "source": [
 162 |         "## 3. Lasso\n",
 163 |         "\n",
 164 |         "Construire le modèle, afficher une métrique ou une sortie et interpréter ses limites. Les exemples n’utilisent pas de test final pour le réglage d’hyperparamètres."
 165 |       ]
 166 |     },
 167 |     {
 168 |       "cell_type": "code",
 169 |       "execution_count": null,
 170 |       "id": "3294e772",
 171 |       "metadata": {
 172 |         "colab": {
 173 |           "base_uri": "https://localhost:8080/"
 174 |         },
 175 |         "id": "3294e772",
 176 |         "outputId": "8dafc704-cdb0-484a-c6cb-97309aed9a98"
 177 |       },
 178 |       "outputs": [
 179 |         {
 180 |           "output_type": "stream",
 181 |           "name": "stdout",
 182 |           "text": [
 183 |             "R²: 0.986045476236913\n"
 184 |           ]
 185 |         }
 186 |       ],
 187 |       "source": [
 188 |         "from sklearn.linear_model import Lasso\n",
 189 |         "modele=make_pipeline(StandardScaler(),Lasso(alpha=.1,max_iter=10000)).fit(Xr_ent,yr_ent)\n",
 190 |         "print('R²:',r2_score(yr_test,modele.predict(Xr_test)))"
 191 |       ]
 192 |     },
 193 |     {
 194 |       "cell_type": "markdown",
 195 |       "id": "d951d32a",
 196 |       "metadata": {
 197 |         "id": "d951d32a"
 198 |       },
 199 |       "source": [
 200 |         "## 4. Régression polynomiale\n",
 201 |         "\n",
 202 |         "Construire le modèle, afficher une métrique ou une sortie et interpréter ses limites. Les exemples n’utilisent pas de test final pour le réglage d’hyperparamètres."
 203 |       ]
 204 |     },
 205 |     {
 206 |       "cell_type": "code",
 207 |       "execution_count": null,
 208 |       "id": "26fc61ac",
 209 |       "metadata": {
 210 |         "colab": {
 211 |           "base_uri": "https://localhost:8080/"
 212 |         },
 213 |         "id": "26fc61ac",
 214 |         "outputId": "257462d3-d560-40d7-f758-5f9417189f1f"
 215 |       },
 216 |       "outputs": [
 217 |         {
 218 |           "output_type": "stream",
 219 |           "name": "stdout",
 220 |           "text": [
 221 |             "MAE: 10.896908068570918\n"
 222 |           ]
 223 |         }
 224 |       ],
 225 |       "source": [
 226 |         "from sklearn.preprocessing import PolynomialFeatures\n",
 227 |         "from sklearn.linear_model import Ridge\n",
 228 |         "modele=make_pipeline(StandardScaler(),PolynomialFeatures(degree=2,include_bias=False),Ridge(alpha=10)).fit(Xr_ent,yr_ent)\n",
 229 |         "print('MAE:',mean_absolute_error(yr_test,modele.predict(Xr_test)))"
 230 |       ]
 231 |     },
 232 |     {
 233 |       "cell_type": "markdown",
 234 |       "id": "eb26714d",
 235 |       "metadata": {
 236 |         "id": "eb26714d"
 237 |       },
 238 |       "source": [
 239 |         "## 5. Arbre de régression\n",
 240 |         "\n",
 241 |         "Construire le modèle, afficher une métrique ou une sortie et interpréter ses limites. Les exemples n’utilisent pas de test final pour le réglage d’hyperparamètres."
 242 |       ]
 243 |     },
 244 |     {
 245 |       "cell_type": "code",
 246 |       "execution_count": null,
 247 |       "id": "e0518bf4",
 248 |       "metadata": {
 249 |         "colab": {
 250 |           "base_uri": "https://localhost:8080/"
 251 |         },
