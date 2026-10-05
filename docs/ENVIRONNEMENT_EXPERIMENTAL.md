# Environnement expérimental et évaluation

## Environnement conseillé

- Python avec NumPy et pytest pour le banc minimal; PyTorch ou JAX peuvent servir
  aux expériences apprises plus larges. PyTorch est optionnel et n'est pas installé
  par la CI de base.
- Pour des signatures de chemin, `iisignature` et `signatory` sont des noms de
  bibliothèques à examiner, pas des dépendances recommandées ici. Vérifier avant
  toute adoption leur maintenance, compatibilité Python/PyTorch, licence, API et
  résultats numériques contre l'implémentation NumPy et les tests de Chen.
- C++ pour étudier le point d'intégration éventuel avec `llama.cpp`; ne pas
  présumer qu'une interface ou un plugin suffit sans tester la version visée.
- GitHub Actions exécute pytest et le smoke test du banc; les métriques JSON du
  smoke test sont ajoutées au résumé de l'exécution CI. Les sorties de CI ne sont
  pas des résultats linguistiques généralisables.

Installation et validation locales :

```bash
pip install -r requirements.txt
python -m pytest -q
python -m experiments.continuous_vs_vision.run_experiment \
  --lang fr --seed 17 --permutations 1000 --output out/continuous-vs-vision.json
```

Le script choisit un dictionnaire local `data/dict/<lang>.tsv` s'il existe.
Sinon, `corpus_source` et `toy_corpus` indiquent sans ambiguïté le mini-corpus jouet.
Les données téléchargées et les résultats générés ne sont pas ajoutés au dépôt par
défaut. Publier avec chaque comparaison le corpus réellement utilisé, sa provenance,
les seeds, les paramètres, le budget de calcul et les métriques.

## Protocole d'évaluation des modèles de code

Donner à chaque modèle exactement la même tâche d'implémentation : écrire ou modifier
`Advance()` pour composer une signature de chemin tronquée aux niveaux 1 et 2, avec
les tests pytest de la relation de Chen et des cas limites. Réinitialiser le dépôt
entre essais et fixer les mêmes versions d'environnement et consignes.

Évaluer chaque soumission sur :

1. le nombre et la nature des tests passés, incluant composition, chemin vide,
   segment unique et entrées invalides ;
2. la reproductibilité à seeds fixées (mêmes sorties et métriques) ;
3. le respect des contrôles du benchmark : one-hot, permutation circulaire,
   rasterisation contre coordonnées, paramètres/FLOPs appariés, motif aléatoire,
   tests de permutation et correction des comparaisons multiples ;
4. la déclaration explicite des limites, sources et résultats non vérifiés.

Aucun classement de modèles de code n'est affirmé sans mesure reproduite selon ce
protocole. Un meilleur taux de tests sur cette tâche ne prouve pas une supériorité
générale.
