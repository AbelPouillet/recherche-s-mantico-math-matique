# EMBEDBABEL-BENCH V2 — Protocole compétition (LLM vs rigueur corrélative)

Tu es un LLM participant à un benchmark comparatif. Produis une analyse **scientifique, falsifiable, anti-apophénie** du projet EmbedBabel (cf. PROMPT.md à la racine du repo pour la description complète du projet).

## Contrat épistémique
Tague chaque affirmation importante : `[DEFINI]`, `[TESTABLE]`, `[PLAUSIBLE]`, `[SPECULATIF]`, `[PROBABLEMENT_FAUX]`, `[NON_FALSIFIABLE]`.
Ne confonds jamais corrélation/causalité, compression/gain de coût, gématrie/sémantique. Toute claim centrale a une justification ou un test de réfutation.

## Travail demandé
Formalisation (G, nœuds, arêtes, relations, S_t, Advance, Merge, Compress, ProjectToLLM) ; corrélations math/géométriques (robustesse OOD, permutations, bijections) ; audit anti-mystique (SDM 0–100) ; 3–7 intuitions fructueuses (SIF 0–100) ; plan expérimental (9 baselines : caractères, byte-level, tokenizer natif, sous-chaînes sans gématrie, trie, features aléatoires, embeddings appris, petit pré-encodeur, sans EmbedBabel ; + contrôles : permutation, dictionnaire bruité, langue OOD, scramble numérique) ; intégration llama.cpp (pré-tokenizer, tokenizer, prefill, draft, KV-cache, couche externe) avec coûts cachés ; GO/NO-GO.

## Sortie : JSON strict uniquement (aucun texte hors JSON)
Clés : `meta`, `executive_verdict`, `formalization`, `correlation_map` (solid/fragile/illusory), `anti_mystical_audit`, `fruitful_intuitions`, `experimental_plan`, `system_integration`, `smallest_decisive_experiment`, `claim_tagging_summary`.

La sortie est invalide si : texte hors JSON, champ manquant, tag hors vocabulaire, aucun test de réfutation, aucune baseline, SDM/SIF sans justification.

## Directive finale
Ne défends pas EmbedBabel : tente de le casser, puis isole ce qui survit.

> Version condensée du V2 discuté ; remplace ce fichier par le texte complet si besoin.
