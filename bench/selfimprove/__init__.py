"""Boucle d'auto-amélioration de DSH local : enregistrements, score externe, auto-réglage.

Trois principes, tirés de l'audit du harnais H1 — où le verdict de tête était calculé à partir de la
`correlation_map.solid` **que le modèle avait lui-même écrite** :

1. **Aucun modèle ne se note.** Le score vient d'un **résultat externe** (tâche réussie, tests verts,
   tokens consommés, temps mural, reprises) fourni par un fichier d'issues. `bench/selfimprove/score.py`
   refuse explicitement tout champ d'auto-évaluation comme source de vérité.
2. **Acceptation pré-déclarée.** Un candidat ne remplace l'incumbent que s'il le bat sur des
   répétitions tenues à l'écart, au-delà d'une marge fixée **avant** la mesure. Sinon il est rejeté,
   avec le chiffre qui l'a rejeté.
3. **Tout est versionné par hash.** Un profil promu, un jeu de données, un candidat : chacun porte son
   empreinte, pour qu'une amélioration soit reproductible et pas une impression.

Sous-modules :

- `records` : lecture du journal JSONL du plugin DSH et du fichier d'issues, jointure, empreinte ;
- `score`   : métriques **externes** et agrégation, avec garde-fou anti-auto-évaluation ;
- `tune`    : auto-réglage continu piloté par les ressources locales (VRAM, RAM, température) ;
- `run`     : point d'entrée en ligne de commande.
"""
