# Vérifications réalisées

## Données et protocole

- Lecture des deux CSV : 175 341 lignes training et 82 332 testing.
- Données originales copiées sans modification, empreintes SHA-256 conservées.
- Séparation par groupes de vecteurs identiques : 140 145 lignes fit et 35 196 validation, graine 42.
- 8 541 lignes test ont leurs caractéristiques présentes dans le training. Ce chevauchement préexistant est déclaré dans chaque métrique ; aucune ligne test n’est utilisée pour ajuster le prétraitement ou sélectionner l’époque du réseau.

## Tests réussis

- Quatre tests automatisés : exclusion des cibles, catégorie inconnue sans réajustement, séparation des doublons, cohérence des activations cachées du MLP avec sa prédiction.
- Essai court H3/H4/H5, deux époques et 12 arbres/itérations, sur sous-échantillons réels.
- Entraînement complet H3/H4/H5, 15 époques maximum pour le MLP, 100 arbres/itérations.
- Génération de sept graphiques PNG, rapport TXT, métriques JSON et sauvegarde du pipeline pour chaque nouveau modèle.
- Rechargement des trois pipelines : exactitude et ROC-AUC identiques sur les 82 332 observations de test.
- Vérification visuelle d’un échantillon des graphiques et des liens du README.

## Limites de validation

TensorFlow et XGBoost ne sont pas installés dans l’environnement utilisé. H1/H2 ont été réorganisés et corrigés dans le pipeline commun, mais leur nouvel entraînement n’a pas été exécuté. Les images H1/H2 conservées sont celles fournies par l’auteur, avec leur provenance explicite.

Une installation propre avec requirements-deep.txt et un entraînement H1/H2 restent à effectuer. Les nouvelles architectures et paramètres diffèrent des scripts historiques ; les scores historiques ne doivent pas servir de comparaison directe.

Versions testées : requirements-tested.txt. Python 3.12.4. Les modèles ne constituent pas un dispositif de cybersécurité validé en production.
