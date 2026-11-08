# Détection d’intrusions réseau — UNSW-NB15 et modèles hybrides

Projet académique de classification binaire du trafic réseau : **normal (0)** ou **attaque (1)**. Cette version conserve les deux hybridations LSTM d’origine et ajoute trois modèles, avec enregistrement automatique des figures et des résultats.

Le code d’origine est fourni sous licence MIT, copyright 2026 Soufian Haskouk. La notice [LICENSE](LICENSE) est conservée. Les anciens scripts et modèles restent dans la copie locale archives-originales, hors du dépôt à publier.

## Aperçu des résultats

Graphiques UNSW-NB15 : H1/H2 présentent les résultats historiques fournis ; H3/H4/H5 proviennent des nouveaux entraînements. Les protocoles diffèrent : ne pas comparer directement les scores historiques aux scores recalculés.

### H1 — LSTM + XGBoost (résultats historiques)

Ces figures proviennent de l’exécution d’origine et n’ont pas été recalculées avec le pipeline corrigé.

![H1 — Courbes d’entraînement — historique](imageH1/03_loss_accuracy.png)

![H1 — Matrice de confusion — historique](imageH1/04_confusion_matrix.png)

![H1 — Courbe précision-rappel — historique](imageH1/06_precision_recall_curve.png)

### H2 — LSTM + Random Forest (résultats historiques)

Ces figures proviennent de l’exécution d’origine et n’ont pas été recalculées avec le pipeline corrigé.

![H2 — Courbes d’entraînement — historique](imageH2/03_lstm_loss_accuracy.png)

![H2 — Matrice de confusion — historique](imageH2/04_confusion_matrix.png)

![H2 — Courbe précision-rappel — historique](imageH2/06_precision_recall_curve.png)

### Comparaison des trois nouveaux modèles

![Comparaison des modèles H3, H4 et H5](comparison/comparison.png)

### Entraînement de l’extracteur MLP — H3

![Perte et exactitude de l’extracteur MLP pour H3](imageH3/03_loss_accuracy.png)

### Matrice de confusion — H4

![Matrice de confusion du modèle H4 sur le test officiel](imageH4/04_confusion_matrix.png)

### Courbe précision-rappel — H4

![Courbe précision-rappel du modèle H4](imageH4/06_precision_recall_curve.png)

Les autres figures sont accessibles dans les sections dépliables « Graphiques par hybridation ».

## Modèles disponibles

| Modèle | Hybridation | Dépendances |
|---|---|---|
| H1 | LSTM + XGBoost | TensorFlow et XGBoost |
| H2 | LSTM + Random Forest | TensorFlow |
| H3 | MLP + Extra Trees | scikit-learn |
| H4 | MLP + Histogram Gradient Boosting | scikit-learn |
| H5 | PCA + Random Forest | scikit-learn |

Les modèles H3/H4 conviennent à des données tabulaires sans supposer un ordre temporel. H5 fournit une comparaison avec réduction de dimension. Voir [la méthodologie et ses limites](docs/METHODOLOGIE.md).

## Résultats recalculés sur les données complètes

Entraînement : 140 145 lignes ; validation : 35 196 ; test officiel : 82 332. Seuil fixé à 0,5.

| Modèle | Exactitude | Rappel attaque | F1 attaque | AP | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| H3 | 88.62% | 94.45% | 90.14% | 0.9769 | 0.9700 |
| H4 | 88.86% | 93.72% | 90.26% | 0.9762 | 0.9688 |
| H5 | 87.39% | 96.50% | 89.39% | 0.9703 | 0.9640 |

Ces scores décrivent une exécution du benchmark : 8 541 lignes de test ont un vecteur de caractéristiques également présent dans le training. Cette limite est conservée et documentée ; il ne s’agit pas d’une preuve de performance sur un réseau réel.

## Graphiques par hybridation

<details>
<summary>H1 — résultats historiques, non recalculés</summary>

![H1 — 01 class distribution](imageH1/01_class_distribution.png)

![H1 — 02 attack types](imageH1/02_attack_types.png)

![H1 — 03 loss accuracy](imageH1/03_loss_accuracy.png)

![H1 — 04 confusion matrix](imageH1/04_confusion_matrix.png)

![H1 — 05 feature importance](imageH1/05_feature_importance.png)

![H1 — 06 precision recall curve](imageH1/06_precision_recall_curve.png)

[Résultats détaillés H1](imageH1/results_H1.txt)

</details>

<details>
<summary>H2 — résultats historiques, non recalculés</summary>

![H2 — 01 class distribution](imageH2/01_class_distribution.png)

![H2 — 02 attack types](imageH2/02_attack_types.png)

![H2 — 03 lstm loss accuracy](imageH2/03_lstm_loss_accuracy.png)

![H2 — 04 confusion matrix](imageH2/04_confusion_matrix.png)

![H2 — 05 feature importance](imageH2/05_feature_importance.png)

![H2 — 06 precision recall curve](imageH2/06_precision_recall_curve.png)

[Résultats détaillés H2](imageH2/results_H2.txt)

</details>

<details>
<summary>H3 — nouvel entraînement complet</summary>

![H3 — 01 class distribution](imageH3/01_class_distribution.png)

![H3 — 02 attack types](imageH3/02_attack_types.png)

![H3 — 03 loss accuracy](imageH3/03_loss_accuracy.png)

![H3 — 04 confusion matrix](imageH3/04_confusion_matrix.png)

![H3 — 05 feature importance](imageH3/05_feature_importance.png)

![H3 — 06 precision recall curve](imageH3/06_precision_recall_curve.png)

![H3 — 07 roc curve](imageH3/07_roc_curve.png)

[Résultats détaillés H3](imageH3/results_H3.txt)

</details>

<details>
<summary>H4 — nouvel entraînement complet</summary>

![H4 — 01 class distribution](imageH4/01_class_distribution.png)

![H4 — 02 attack types](imageH4/02_attack_types.png)

![H4 — 03 loss accuracy](imageH4/03_loss_accuracy.png)

![H4 — 04 confusion matrix](imageH4/04_confusion_matrix.png)

![H4 — 05 feature importance](imageH4/05_feature_importance.png)

![H4 — 06 precision recall curve](imageH4/06_precision_recall_curve.png)

![H4 — 07 roc curve](imageH4/07_roc_curve.png)

[Résultats détaillés H4](imageH4/results_H4.txt)

</details>

<details>
<summary>H5 — nouvel entraînement complet</summary>

![H5 — 01 class distribution](imageH5/01_class_distribution.png)

![H5 — 02 attack types](imageH5/02_attack_types.png)

![H5 — 03 loss accuracy](imageH5/03_loss_accuracy.png)

![H5 — 04 confusion matrix](imageH5/04_confusion_matrix.png)

![H5 — 05 feature importance](imageH5/05_feature_importance.png)

![H5 — 06 precision recall curve](imageH5/06_precision_recall_curve.png)

![H5 — 07 roc curve](imageH5/07_roc_curve.png)

[Résultats détaillés H5](imageH5/results_H5.txt)

</details>

## Installation

Python 3.12 a été utilisé pour les vérifications. Depuis ce dossier :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Pour H1 et H2, ajouter les bibliothèques de deep learning :

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-deep.txt
```

Les versions réellement utilisées pour H3/H4/H5 sont consignées dans requirements-tested.txt. Pour reproduire cet environnement Python 3.12, installer ce fichier à la place de requirements.txt.

## Données

Déposer les CSV training et testing dans DATA/. Ils sont présents dans la copie locale préparée, mais exclus de Git et du ZIP de publication. [Instructions, source et conditions](DATA/README.md).

## Entraîner et enregistrer les graphiques

Pour entraîner les trois nouveaux modèles sur les données complètes :

```powershell
.\.venv\Scripts\python.exe train.py --models H3 H4 H5
```

Pour lancer une seule hybridation :

```powershell
.\.venv\Scripts\python.exe model_H1_LSTM_XGBoost.py
.\.venv\Scripts\python.exe model_H2_LSTM_RF.py
.\.venv\Scripts\python.exe model_H3_MLP_ExtraTrees.py
.\.venv\Scripts\python.exe model_H4_MLP_HistGradientBoosting.py
.\.venv\Scripts\python.exe model_H5_PCA_RF.py
```

Pour un essai court, séparé des résultats complets :

```powershell
.\.venv\Scripts\python.exe train.py --models H3 H4 H5 --smoke --epochs 2 --trees 12
```

Les essais courts utilisent 4 000 lignes du training et 2 000 du test, puis écrivent dans smoke/. Ils ne constituent pas les résultats finaux du benchmark.

Paramètres disponibles : --epochs, --trees, --batch-size, --features, --patience, --seed, --jobs, --data-dir et --output-dir. Par défaut : 15 époques maximum, patience 4, 100 arbres/itérations et deux threads. Une relance au même emplacement remplace les résultats du modèle ; utiliser --output-dir pour conserver plusieurs expériences.

## Organisation des résultats

Chaque hybridation crée automatiquement un dossier imageH1, imageH2, etc. :

```text
imageH3/
├── 01_class_distribution.png
├── 02_attack_types.png
├── 03_loss_accuracy.png
├── 04_confusion_matrix.png
├── 05_feature_importance.png
├── 06_precision_recall_curve.png
├── 07_roc_curve.png
├── results_H3.txt
├── metrics_H3.json
└── ORIGINE.md
```

Les modèles et le prétraitement sont enregistrés dans models/H3/, etc. Ces fichiers volumineux sont ignorés par Git.

## Comparer les résultats recalculés

```powershell
.\.venv\Scripts\python.exe compare_results.py
```

Cela produit comparison/comparison.csv et comparison/comparison.png. Les anciens résultats H1/H2, sans fichier metrics_H*.json, sont exclus de cette comparaison.

## Prédire avec un modèle sauvegardé

```powershell
.\.venv\Scripts\python.exe predict.py --model-dir models/H3 --input DATA/UNSW_NB15_testing-set.csv --output predictions/H3.csv
```

Charger uniquement des fichiers joblib provenant d’une source de confiance. Le CSV d’inférence doit contenir les mêmes colonnes d’entrée que le training.

## Résultats et graphiques

Les dossiers imageH1 et imageH2 contiennent les **résultats historiques fournis**, pas une réexécution du protocole corrigé. Dans le README d’origine, le rappel d’attaque H2 était mal recopié : son fichier results_H2.txt indique 0,9373, soit 93,73 %. La valeur AP ne doit pas être présentée comme ROC-AUC.

Les résultats nouveaux et leur état de validation sont détaillés dans [VALIDATION.md](docs/VALIDATION.md).

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Publication GitHub

Publier le contenu de ce dossier avec son .gitignore. Les CSV DATA/, les modèles sérialisés, les essais smoke/ et les environnements Python restent locaux. Les PNG, TXT, JSON, scripts et documents sont publiables. Les résultats historiques doivent garder leur étiquette d’origine.
