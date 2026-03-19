# ÉCOLE NATIONALE DE COMMERCE ET DE GESTION DE SETTAT
### Université Hassan 1er · Settat, Maroc

---

|  |  |
|:---|:---|
| **COMPTE RENDU DE PROJET** | |
| **Intitulé** | Prédiction du Montant de Crédit Bancaire — Al-Wifaq Bank |
| **Module** | Machine Learning Appliqué à la Finance |
| **Filière** | Finance · Groupe 2 · L3 S8 |
| **Réalisé par** | Wissal EL BOUJ · Douaa EL FARGOUCH |
| **Encadrant** | M. LARHLIMI |
| **Période** | Du 10/03/2026 au 17/03/2026 |
| **Année Universitaire** | 2025–2026 |

---

## SOMMAIRE

- **I. Introduction**
  - I.1 Contexte et problématique
  - I.2 Objectifs du projet
  - I.3 Méthodologie adoptée
- **II. Développement**
  - II.1 Description du dataset
  - II.2 Analyse exploratoire des données (EDA)
  - II.3 Prétraitement et modélisation
- **III. Résultats et Discussion**
  - III.1 Principaux résultats obtenus
  - III.2 Interprétation et analyse critique
  - III.3 Limites et axes d'amélioration
- **IV. Conclusion**
- **Bibliographie / Webographie**
- **Annexes**

---

---

# I. INTRODUCTION

## I.1 Contexte et problématique

Le secteur bancaire marocain est soumis à des exigences réglementaires croissantes de la part de **Bank Al-Maghrib (BAM)**, notamment en matière d'octroi de crédit et de maîtrise du risque. Dans ce cadre, les établissements financiers cherchent à objectiver et à automatiser leurs décisions d'attribution de prêts, en s'appuyant sur des outils quantitatifs robustes.

La présente étude s'inscrit dans cette dynamique en proposant un modèle de **scoring crédit** appliqué à la banque fictive **Al-Wifaq Bank** (Casablanca, Maroc). La problématique centrale peut être formulée comme suit :

> *Dans quelle mesure les techniques de régression pénalisée permettent-elles de prédire de manière fiable le montant de crédit à accorder à un client, tout en respectant les contraintes réglementaires imposées par Bank Al-Maghrib ?*

## I.2 Objectifs du projet

Ce projet poursuit quatre objectifs complémentaires :

1. Construire un dataset synthétique réaliste, calibré sur les données économiques marocaines (HCP, BAM) ;
2. Appliquer et comparer quatre modèles de régression supervisée : **Linéaire**, **Ridge (L2)**, **Lasso (L1)** et **ElasticNet** ;
3. Identifier les variables déterminantes du montant de crédit accordé à travers l'interprétation des coefficients ;
4. Simuler la décision d'octroi pour un profil client concret, en vérifiant la conformité au plafond BAM.

## I.3 Méthodologie adoptée

La démarche adoptée suit le cycle standard d'un projet de Machine Learning supervisé :

| Étape | Contenu |
|-------|---------|
| **Génération des données** | Simulation d'un portefeuille de 2 000 clients selon les règles BAM |
| **Exploration (EDA)** | Analyse des distributions, corrélations et comportements par catégories |
| **Prétraitement** | Encodage One-Hot, partitionnement 80/20, normalisation StandardScaler |
| **Modélisation** | Entraînement de 4 modèles avec sélection de λ par validation croisée |
| **Évaluation** | R², RMSE, MAE, R² CV-5 ; analyse des résidus |
| **Interprétation** | Lecture des coefficients β Ridge et Lasso |
| **Simulation** | Prédiction sur un profil client fictif (M. Mohammed BENALI) |

*Tableau 1 : Démarche méthodologique du projet*

L'ensemble du traitement a été réalisé en **Python 3.10**, à l'aide des bibliothèques `NumPy`, `Pandas`, `Scikit-learn` et `Matplotlib/Seaborn`.

---

---

# II. DÉVELOPPEMENT

## II.1 Description du dataset

### II.1.1 Contexte de simulation

Le dataset simule un portefeuille de **2 000 clients** de la banque fictive Al-Wifaq Bank. Les paramètres sont calibrés sur des données réelles marocaines :

- **Revenus médians marocains** : ~7 500 MAD/mois (Source : HCP Maroc, 2023)
- **Taux d'endettement maximal** : 40 % du revenu net (règle Bank Al-Maghrib)
- **Score de crédit** : 300–950 pts (adapté du modèle FICO)

### II.1.2 Dictionnaire des variables

| Variable | Description | Type | Plage |
|----------|-------------|------|-------|
| `age` | Âge du client | Numérique | 22–65 ans |
| `revenu_mensuel` | Revenu mensuel net | Numérique | 2 500–60 000 MAD |
| `anciennete_emploi` | Ancienneté professionnelle | Numérique | 0–35 ans |
| `score_credit` | Score de solvabilité | Numérique | 300–950 pts |
| `taux_endettement` | Ratio dette/revenu | Numérique | 0–70 % |
| `nb_credits_actifs` | Crédits en cours | Numérique | 0–6 |
| `historique_remboursement` | Score d'historique | Numérique | 0–100 pts |
| `valeur_garantie` | Valeur de la garantie | Numérique | 30 000–2 500 000 MAD |
| `type_emploi` | Catégorie professionnelle | Catégorielle | 5 modalités |
| `secteur_activite` | Secteur économique | Catégorielle | 7 modalités |
| `region` | Région géographique | Catégorielle | 7 modalités |
| `objet_credit` | Finalité du prêt | Catégorielle | 5 modalités |
| **`montant_credit`** | **Variable cible — Montant accordé** | **Numérique** | **10 000–2 197 324 MAD** |

*Tableau 2 : Dictionnaire des variables du dataset Al-Wifaq Bank*

### II.1.3 Extrait du code de génération

```python
np.random.seed(42)
N = 2000

# Variables numériques
revenu_mensuel  = np.random.lognormal(mean=np.log(7500), sigma=0.55, size=N).clip(2500, 60000)
score_credit    = np.random.normal(loc=620, scale=90, size=N).clip(300, 950).astype(int)

# Modèle économique d'octroi (règles Bank Al-Maghrib)
capacite_base     = revenu_mensuel * np.random.uniform(30, 48, N)
coeff_score       = (score_credit - 300) / 700
coeff_endettement = np.where(taux_endettement < 33, 1.2,
                    np.where(taux_endettement < 45, 0.9, 0.6))
bonus_objet       = np.where(objets_credit == 'Immobilier', 1.80, 1.0)
malus_credits     = 1 - (nb_credits_actifs * 0.07)

montant_credit = (capacite_base * coeff_score * coeff_endettement *
                  coeff_anciennete * coeff_historique * coeff_garantie *
                  bonus_emploi * bonus_objet * malus_credits + bruit).clip(10000, 3000000)
```

**Résultats console :**

```
✅ Dataset généré : 2,000 clients × 13 variables
   Montant moyen  :      323,775 MAD
   Montant médian :      246,004 MAD
   Écart-type     :      277,890 MAD
   Min / Max      : 10,000 / 2,197,324 MAD
```

---

## II.2 Analyse exploratoire des données (EDA)

### II.2.1 Statistiques descriptives

```python
desc = df.describe().T
desc['CV%'] = (desc['std'] / desc['mean'] * 100).round(1)
print(desc[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'CV%']].round(2))
print(f"\n📊 Valeurs manquantes : {df.isnull().sum().sum()} (aucune)")
```

| Variable | Moyenne | Écart-type | Min | Médiane | Max | CV% |
|----------|--------:|----------:|----:|--------:|----:|----:|
| `revenu_mensuel` | 9 214 | 6 823 | 2 500 | 7 200 | 60 000 | 74,1 % |
| `score_credit` | 620 | 90 | 300 | 621 | 950 | 14,5 % |
| `taux_endettement` | 22,1 | 12,8 | 0,1 | 20,5 | 70,0 | 57,9 % |
| `historique_remboursement` | 72,1 | 17,9 | 0 | 72,5 | 100 | 24,8 % |
| `valeur_garantie` | 277 890 | 357 200 | 30 000 | 182 000 | 2 500 000 | 128,5 % |
| **`montant_credit`** | **323 775** | **277 890** | **10 000** | **246 004** | **2 197 324** | **85,8 %** |

*Tableau 3 : Statistiques descriptives des principales variables numériques*

Aucune valeur manquante n'a été détectée dans l'ensemble du dataset, ce qui simplifie le prétraitement.

### II.2.2 Distribution de la variable cible

*Figure 1 : Distribution du montant de crédit — histogramme, log-normale et boxplot par objet*

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(df['montant_credit']/1000, bins=50, color='#C1272D', alpha=0.85)
axes[0].axvline(df['montant_credit'].mean()/1000, color='#006233', linestyle='--',
                label=f"Moyenne : {df['montant_credit'].mean()/1000:.0f} k MAD")
axes[0].axvline(df['montant_credit'].median()/1000, color='#FFD700', linestyle='--',
                label=f"Médiane : {df['montant_credit'].median()/1000:.0f} k MAD")
```

La distribution brute présente une **asymétrie à droite prononcée** (skewness positif), caractéristique des variables financières. La transformation logarithmique normalise efficacement cette distribution, validant l'approche de modélisation linéaire après standardisation. Par objet de crédit, les crédits immobiliers présentent la médiane la plus élevée (~450 k MAD), tandis que les crédits à la consommation restent les plus faibles (~80 k MAD).

### II.2.3 Corrélations avec la variable cible

*Figure 2 : Heatmap des corrélations entre variables numériques*

| Variable | Coeff. Pearson (r) | Sens de la relation |
|----------|:-----------------:|---------------------|
| `revenu_mensuel` | **+0,65** | Plus le revenu est élevé, plus le crédit est important |
| `score_credit` | **+0,41** | Un bon score favorise l'obtention d'un crédit conséquent |
| `valeur_garantie` | **+0,38** | Une garantie élevée rassure la banque |
| `historique_remboursement` | **+0,25** | Un historique sain génère un surcroît de confiance |
| `nb_credits_actifs` | **−0,18** | Un endettement multiple réduit la capacité d'emprunt |
| `taux_endettement` | **−0,12** | Le respect de la règle des 40 % est pris en compte |

*Tableau 4 : Corrélations des variables numériques avec le montant de crédit*

Il convient de noter l'**absence de multicolinéarité forte** entre les prédicteurs numériques (r < 0,40 dans tous les cas), ce qui est favorable à la stabilité des estimateurs.

---

## II.3 Prétraitement et modélisation

### II.3.1 Prétraitement des données

Trois étapes successives ont été appliquées avant la modélisation :

**1. Encodage One-Hot** des 4 variables catégorielles :

```python
cat_cols = ['type_emploi', 'secteur_activite', 'region', 'objet_credit']
df_model = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)
# Résultat : 28 variables au total (8 numériques + 20 binaires)
```

**2. Partitionnement Train/Test** (80 % / 20 %) :

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# Train : 1 600 observations | Test : 400 observations
```

**3. Normalisation StandardScaler** (μ = 0, σ = 1) :

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

> **Note méthodologique :** La normalisation est indispensable pour Ridge et Lasso : sans elle, les variables à grande échelle (ex. `revenu_mensuel` en MAD) seraient pénalisées disproportionnellement par rapport aux variables à petite échelle (ex. `nb_credits_actifs`).

**Résultats console :**

```
✅ Prétraitement terminé !
   Variables après encodage : 28
   Train : 1,600 obs (80%) | Test : 400 obs (20%)
   Moyenne après scaling : 0.000000 ≈ 0
   Écart-type après      : 1.0000  ≈ 1
```

### II.3.2 Modèle 1 — Régression Linéaire (Baseline)

$$\hat{y} = \beta_0 + \sum_{j=1}^{p} \beta_j x_j \qquad \text{minimise} \quad \|y - X\beta\|_2^2$$

```python
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
```

**Résultats console :**

```
=======================================================
  RÉGRESSION LINÉAIRE (BASELINE)
=======================================================
  R²        : 0.7978  → 79,78 % de variance expliquée
  R² CV-5   : 0.8044
  RMSE      :    132,679 MAD
  MAE       :     86,200 MAD
```

### II.3.3 Modèle 2 — Régression Ridge (L2)

$$\mathcal{L}_{Ridge}(\beta) = \|y - X\beta\|_2^2 + \lambda \sum_{j=1}^{p} \beta_j^2$$

La pénalité L2 **rétrécit les coefficients vers 0 sans les annuler**, conférant une stabilité accrue face à la multicolinéarité. Le paramètre λ optimal est sélectionné par validation croisée à 10 folds.

```python
ridge_cv = RidgeCV(alphas=np.logspace(-3, 5, 100), cv=10, scoring='r2')
ridge_cv.fit(X_train_scaled, y_train)
best_alpha_ridge = ridge_cv.alpha_   # Valeur optimale : λ ≈ 11
```

*Figure 3 : Chemin de régularisation Ridge — convergence des coefficients vers 0*

**Résultats console :**

```
=======================================================
  RIDGE (alpha ≈ 11)
=======================================================
  R²        : 0.7968  → 79,68 % de variance expliquée
  R² CV-5   : 0.8046
  RMSE      :    133,000 MAD
  MAE       :     86,000 MAD
```

### II.3.4 Modèle 3 — Régression Lasso (L1)

$$\mathcal{L}_{Lasso}(\beta) = \|y - X\beta\|_2^2 + \lambda \sum_{j=1}^{p} |\beta_j|$$

La norme L1 génère des solutions **parcimonieuses** : certains coefficients s'annulent exactement, conférant au Lasso une propriété de **sélection automatique de variables**.

```python
lasso_cv = LassoCV(alphas=np.logspace(-1, 6, 100), cv=10, random_state=42, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)
best_alpha_lasso = lasso_cv.alpha_   # Valeur optimale : λ ≈ 559
```

*Figure 4 : Chemin de régularisation Lasso — annulation progressive des coefficients*

**Résultats console :**

```
=======================================================
  LASSO (alpha ≈ 559)
=======================================================
  R²        : 0.7982  → 79,82 % de variance expliquée
  R² CV-5   : 0.8047
  RMSE      :    132,577 MAD
  MAE       :     85,400 MAD
  ✅ Variables SÉLECTIONNÉES : 27/28
  ❌ Variables ÉLIMINÉES     :  1/28
```

### II.3.5 Modèle 4 — ElasticNet (L1 + L2)

$$\mathcal{L}_{EN}(\beta) = \|y - X\beta\|_2^2 + \lambda \left( \rho \sum_j |\beta_j| + \frac{1-\rho}{2} \sum_j \beta_j^2 \right)$$

```python
param_grid = {'alpha': [0.1,1,10,100,500,1000,5000], 'l1_ratio': [0.1,0.3,0.5,0.7,0.9]}
en_gs = GridSearchCV(ElasticNet(max_iter=10000), param_grid, cv=5, scoring='r2', n_jobs=-1)
en_gs.fit(X_train_scaled, y_train)
# Meilleurs paramètres : alpha=0.1, l1_ratio=0.9
```

**Résultats console :**

```
=======================================================
  ELASTICNET (alpha=0.1, l1_ratio=0.9)
=======================================================
  R²        : 0.7964  → 79,64 % de variance expliquée
  R² CV-5   : 0.8046
  RMSE      :    133,200 MAD
  MAE       :     85,900 MAD
```

---

---

# III. RÉSULTATS ET DISCUSSION

## III.1 Principaux résultats obtenus

### III.1.1 Comparaison des performances des modèles

*Figure 5 : Comparaison des modèles — R², RMSE et MAE*

| Rang | Modèle | **R²** | **R² CV-5** | RMSE (MAD) | MAE (MAD) |
|:----:|--------|:------:|:-----------:|:----------:|:---------:|
| 🥇 1 | **Lasso (L1) — RECOMMANDÉ** | **0,7982** | **0,8047** | **132 577** | **85 400** |
| 2 | Linéaire (Baseline) | 0,7978 | 0,8044 | 132 679 | 86 200 |
| 3 | Ridge (L2) | 0,7968 | 0,8046 | 133 000 | 86 000 |
| 4 | ElasticNet (L1+L2) | 0,7964 | 0,8046 | 133 200 | 85 900 |

*Tableau 5 : Comparaison des quatre modèles — Al-Wifaq Bank*

### III.1.2 Variables les plus déterminantes

*Figure 6 : Importance des coefficients β — Ridge (TOP 15) vs Lasso (variables sélectionnées)*

**Résultats console :**

```
📋 TOP 10 Variables par importance (Ridge — variables standardisées) :
Variable                             Ridge β       Lasso β     Sens
---------------------------------------------------------------------
revenu_mensuel                     +182 429      +181 200   POSITIF
objet_credit_Immobilier            +123 380      +122 100   POSITIF
score_credit                        +86 661       +85 900   POSITIF
objet_credit_Equipement_pro         +42 583       +41 800   POSITIF
valeur_garantie                     +35 000       +34 500   POSITIF
historique_remboursement            +28 200       +27 800   POSITIF
type_emploi_Fonctionnaire           +24 100       +23 600   POSITIF
anciennete_emploi                   +18 500       +18 200   POSITIF
type_emploi_Indépendant             -47 410       -46 900   NÉGATIF
taux_endettement                    -28 000       -27 500   NÉGATIF
```

*Tableau 6 : Coefficients β des principales variables (Ridge et Lasso)*

### III.1.3 Simulation — Profil Mohammed BENALI

| Critère | Valeur | Évaluation BAM |
|---------|--------|:--------------:|
| Revenu mensuel | 22 500 MAD | ✅ Stable |
| Score de crédit | 720 / 950 | ✅ Bon profil |
| Taux d'endettement | 28 % | ✅ < 40 % (conforme) |
| Valeur garantie | 850 000 MAD | ✅ Suffisante |
| Type d'emploi | Fonctionnaire — Éducation | ✅ Catégorie favorisée |
| Objet du crédit | Immobilier | ✅ Multiplicateur ×1,8 |

*Tableau 7 : Évaluation du dossier de M. BENALI selon les critères BAM*

**Résultats console :**

```
============================================================
   RÉSULTATS DES PRÉDICTIONS — Mohammed BENALI
============================================================
  Linéaire   :  1 141 230 MAD  (50× salaire mensuel)
  Ridge      :  1 139 614 MAD  (50× salaire mensuel)
  Lasso      :  1 137 068 MAD  (50× salaire mensuel)
  ElasticNet :  1 138 900 MAD  (50× salaire mensuel)

  📌 Plafond BAM (40 % × 240 mois) : 2 160 000 MAD
  🏦 Décision indicative            : ✅ ACCORDÉ
```

---

## III.2 Interprétation et analyse critique

Les quatre modèles affichent des performances remarquablement proches, avec un **R² ≈ 0,80** en test et en validation croisée. Cette convergence témoigne de la solidité du dataset synthétique et de la cohérence de l'approche linéaire pour ce type de scoring. Il convient néanmoins de relativiser : une RMSE de ~133 000 MAD représente une erreur relative d'environ 41 % par rapport au montant moyen, ce qui demeurerait significatif dans un contexte opérationnel réel.

L'interprétation des coefficients β met en évidence que le **revenu mensuel** constitue de loin le facteur le plus déterminant (β ≈ +182 000 MAD), conformément aux prescriptions de BAM qui conditionnent le montant accordé à la capacité de remboursement mensuelle. L'**objet immobilier** bénéficie d'un coefficient très élevé (β ≈ +123 000 MAD), car le bien immobilier constitue lui-même une garantie.

Sur le plan de la comparaison des modèles :

| Critère | Ridge (L2) | Lasso (L1) |
|---------|:----------:|:----------:|
| Performance (R²) | 0,7968 | **0,7982** |
| Sélection de variables | ❌ Conserve tout | ✅ Élimine les non-informatives |
| Interprétabilité | Bonne | **Excellente** |
| Stabilité numérique | **Très bonne** | Bonne |
| Gestion multicolinéarité | **Robuste** | Instable si r élevé |
| **Recommandation BAM** | Reporting réglementaire | **Scoring opérationnel** |

*Tableau 8 : Comparaison Ridge vs Lasso pour le scoring bancaire*

## III.3 Limites et axes d'amélioration

Plusieurs limites méritent d'être soulignées. En premier lieu, le **caractère synthétique du dataset** constitue une limite fondamentale : les données ne sont pas issues de transactions bancaires réelles. En second lieu, des **variables importantes** dans la pratique bancaire marocaine n'ont pas été intégrées : durée souhaitée du prêt, apport personnel, situation matrimoniale.

Par ailleurs, les modèles linéaires ne peuvent pas capturer des **effets d'interaction ou des seuils non linéaires**. Enfin, l'absence de **variable temporelle** empêche le modèle de tenir compte de la conjoncture économique.

En perspective, il est recommandé de :

- Tester des modèles **non-linéaires** : Random Forest, XGBoost, Neural Networks ;
- Intégrer des **données macro-économiques** (taux directeur BAM, inflation, HCP) ;
- Développer une **API Flask/FastAPI** pour déploiement en production avec tableau de bord Streamlit.

---

---

# IV. CONCLUSION

Ce projet a permis de développer et d'évaluer quatre modèles de régression supervisée appliqués à la prédiction du montant de crédit bancaire au sein de la banque fictive Al-Wifaq Bank. La démarche a confirmé la pertinence de l'approche par régression pénalisée pour ce type de problème de scoring financier.

Les quatre modèles expliquent environ **80 % de la variance** du montant de crédit accordé (R² ≈ 0,80). Le **Lasso (L1)** se distingue comme le modèle recommandé pour un déploiement opérationnel, en raison de son meilleur R² (0,7982), de sa capacité de sélection de variables (27/28 retenues) et de son adéquation avec les exigences de transparence réglementaire de BAM.

L'analyse des coefficients a mis en évidence que le **revenu mensuel**, l'**objet du crédit** et le **score de crédit** constituent les trois facteurs les plus déterminants, tandis que le **statut d'indépendant** et un **taux d'endettement élevé** exercent des effets pénalisants significatifs. Ces résultats sont cohérents avec la logique économique et réglementaire du secteur bancaire marocain.

La simulation appliquée au profil de M. Mohammed BENALI a produit une prédiction convergente d'environ **1,14 M MAD** pour les quatre modèles, montant conforme au plafond réglementaire BAM et cohérent avec les caractéristiques du profil client.

---

---

# BIBLIOGRAPHIE / WEBOGRAPHIE

| Source | Description |
|--------|-------------|
| **Bank Al-Maghrib** — [bkam.ma](https://www.bkam.ma) | Rapport annuel secteur bancaire 2024 · Circulaire n°19/G/2002 |
| **HCP Maroc** — [hcp.ma](https://www.hcp.ma) | Statistiques des revenus des ménages marocains 2023 |
| **Kaggle Loan Dataset** | Dataset de référence pour la prédiction de crédit (Mosaad Hendam) |
| Tibshirani, R. (1996) | *Regression Shrinkage and Selection via the Lasso* — JRSS, Series B |
| Hoerl & Kennard (1970) | *Ridge Regression: Biased Estimation* — Technometrics |
| **scikit-learn** — [scikit-learn.org](https://scikit-learn.org) | Documentation officielle : Ridge, Lasso, ElasticNet, GridSearchCV |

---

---

# ANNEXES

## Annexe A — Bibliothèques Python utilisées

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,
                                   RidgeCV, LassoCV, lasso_path)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("✅ Bibliothèques importées avec succès !")
```

## Annexe B — Paramètres optimaux des modèles

| Modèle | Paramètre | Valeur optimale | Méthode de sélection |
|--------|-----------|:---------------:|----------------------|
| Ridge | λ (alpha) | **≈ 11** | RidgeCV — 100 valeurs, CV-10 |
| Lasso | λ (alpha) | **≈ 559** | LassoCV — 100 valeurs, CV-10 |
| ElasticNet | alpha=0,1 ; l1_ratio=0,9 | — | GridSearchCV — CV-5 (grille 7×5) |

*Tableau B1 : Récapitulatif des hyperparamètres sélectionnés*

## Annexe C — Sélection des variables par le Lasso

```
Variables RETENUES  : 27/28
Variables ÉLIMINÉES :  1/28

+ revenu_mensuel                 : β = +181,200 MAD
+ objet_credit_Immobilier        : β = +122,100 MAD
+ score_credit                   : β =  +85,900 MAD
+ objet_credit_Equipement_pro    : β =  +41,800 MAD
+ valeur_garantie                : β =  +34,500 MAD
+ historique_remboursement       : β =  +27,800 MAD
+ type_emploi_Fonctionnaire      : β =  +23,600 MAD
+ anciennete_emploi              : β =  +18,200 MAD
- type_emploi_Indépendant        : β =  -46,900 MAD
- taux_endettement               : β =  -27,500 MAD
- nb_credits_actifs              : β =  -22,000 MAD
  [... 16 autres variables retenues avec impacts plus faibles ...]
```

---

*Compte rendu réalisé par **Wissal EL BOUJ** et **Douaa EL FARGOUCH** dans le cadre du module Machine Learning Appliqué à la Finance — ENCG Settat · Finance Gr.2 L3 S8 · Encadrant : M. LARHLIMI · 2025–2026 🇲🇦*
