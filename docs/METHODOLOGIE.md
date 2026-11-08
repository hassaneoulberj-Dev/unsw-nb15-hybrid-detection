# Protocole d’entraînement et d’évaluation

1. Charger les partitions officielles training et testing sans les fusionner.
2. Retirer id, label et attack_cat des entrées ; label reste la cible binaire.
3. Réserver environ 20 % du training pour la validation, en regroupant les vecteurs de caractéristiques identiques. Une copie du même vecteur ne traverse donc pas la frontière entraînement/validation.
4. Ajuster imputation, normalisation, encodage catégoriel et sélection de variables uniquement sur la partie fit. Les catégories inconnues sont ignorées par OneHotEncoder.
5. Entraîner l’extracteur sur fit ; l’arrêt anticipé du réseau utilise uniquement validation. Aucun SMOTE n’est appliqué. Les arbres utilisent des poids de classe lorsque prévu ; les MLP/LSTM apprennent la distribution originale.
6. Entraîner le classifieur final sur les caractéristiques extraites de fit, puis évaluer une fois sur le fichier testing avec un seuil fixé à 0,5.
7. Sauvegarder le prétraitement, l’extracteur, le classifieur, les métriques et les paramètres. L’inférence recharge ces objets sans réajuster le prétraitement.

## Interprétation des hybridations

- H1/H2 : une observation tabulaire est représentée par une séquence de longueur 1. Cela ne démontre pas l’apprentissage de dépendances temporelles entre flux. Le nouvel extracteur comporte deux LSTM 64/32 et un vecteur de 32 composantes.
- H3/H4 : un MLP supervisé (64 puis 32 neurones) fournit les activations de sa dernière couche cachée au classifieur. La meilleure époque est choisie sur la perte de validation.
- H5 : la PCA réduit les caractéristiques à 16 composantes maximum ; une forêt classe ces composantes. C’est une combinaison réduction de dimension/classification, pas du deep learning.

## Graphiques

01 : distribution binaire du CSV d’entraînement avant séparation.
02 : catégories d’attaque, uniquement pour la description des données.
03 : perte et exactitude du réseau extracteur pour H1–H4 ; pour H5, performances de la forêt à 25 %, 50 %, 75 % et 100 % du nombre d’arbres. La PCA n’a pas de loss d’entraînement neuronale.
04 : matrice de confusion sur le test.
05 : importance des composantes transformées, pas des variables réseau originales. H4 utilise une importance par permutation calculée sur un échantillon de validation.
06 : précision-rappel et Average Precision (AP).
07 : ROC et ROC-AUC. AP et ROC-AUC sont deux mesures distinctes.

## Limites du benchmark

Les fichiers contiennent des doublons : 67 601 lignes du training et 26 387 du testing sont des doublons après retrait de id (cible et catégorie conservées dans ce comptage). Le chevauchement de vecteurs entre fichiers est mesuré pour chaque exécution et indiqué dans metrics_H*.json. Les partitions officielles sont conservées ; les résultats ne prouvent donc pas une généralisation sur des flux entièrement nouveaux.

Les métriques historiques et nouvelles ne partagent pas le même protocole. Ne pas les classer dans un tableau unique sans les réentraîner. Aucun modèle n’est garanti supérieur avant les mesures. Les durées dépendent du matériel ; training_seconds inclut chargement/prétraitement/extraction/classification, mais exclut graphiques et sauvegarde.

Les courbes enregistrées servent à l’analyse du benchmark ; elles ne constituent pas une validation de sécurité opérationnelle.

## Références

- Dataset : https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Séparation et prévention des fuites : https://scikit-learn.org/stable/common_pitfalls.html
- MLP : https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
- Arrêt anticipé : https://keras.io/api/callbacks/early_stopping/
