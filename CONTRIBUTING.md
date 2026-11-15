# Contribuer — UNSW-NB15

Ce guide aide à proposer des corrections reproductibles pour ce projet académique.

## Signaler un problème

Décrire le comportement observé, le résultat attendu et les étapes pour reproduire le problème. Indiquer les versions des outils utilisés et joindre un exemple minimal lorsque cela est possible.

## Vérifications adaptées au projet

- Indiquer le modèle, la commande, la graine, les versions des bibliothèques et les fichiers de données utilisés pour reproduire une expérience.
- Ajuster les transformations sur le jeu d’apprentissage uniquement ; conserver la séparation apprentissage, validation et test.
- Exécuter `python -m unittest discover -s tests -v` après un changement du pipeline.
- Joindre les paramètres et les métriques calculées à toute nouvelle figure ; ne pas remplacer une figure par une illustration de résultats non mesurés.
- Présenter séparément les résultats historiques H1/H2 et les nouvelles expériences lorsque leurs protocoles diffèrent.
- Conserver la notice de licence et éviter de publier les jeux de données ou modèles exclus par `.gitignore` sans examiner leur provenance.

## Proposer une modification

Créer une branche dédiée, limiter chaque commit à un changement cohérent et décrire sa motivation. Dans la demande de fusion, préciser les vérifications réalisées, leur résultat et les limites restantes. Actualiser la documentation lorsque le comportement ou les prérequis changent.
