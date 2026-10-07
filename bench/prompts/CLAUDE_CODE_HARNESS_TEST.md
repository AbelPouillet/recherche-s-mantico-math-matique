# Prompt Claude Code — test du harnais de bench EmbedBabel

Colle le bloc ci-dessous dans Claude Code, lancé à la racine du dépôt `recherche-s-mantico-math-matique`.

```text
Contexte : dépôt EmbedBabel (benchmark de recherche, en français). Installe d'abord les dépendances
(`pip install -e ".[dev]"` — sans elles, quatre modules de test ne collectent même pas), puis lis
README.md, PROMPT.md, bench/prompts/EMBEDBABEL_BENCH_V2.md, docs/LED7_SIGNATURE_BUS.md,
docs/EMBEDBABEL_REPORT.md, bench/guard.py (garde anti-pollution canonique) et le vrai plugin DeepSeek
Harness dsh/embedbabel-dsh/ (package.json, cordis.patch.yml, lib/guard.js, lib/index.js, README.md).
Le merge est résolu et l'arbre est propre (vérifie-le avec `git status`) : README.md ne contient plus
de marqueurs de conflit.

Objectif : tester le harnais de bench. Un « harnais » = le programme qui donne le même prompt à chaque modèle, borne et mesure son contexte, valide sa sortie JSON stricte, puis fait évaluer les sorties par les autres modèles. Je veux pouvoir le lancer hors ligne avec des modèles factices avant de payer le moindre appel.

Règles de travail (impératives) :
1. Plan avant code. Commence en LECTURE SEULE : `git status`, `git log --oneline -15 --all`, `python -m pytest -q`. Donne-moi l'état réel (tests qui passent/échouent, ce que fait déjà adapter.py, ce que couvre test_context_guard.py) puis un plan numéroté avec, pour chaque étape, les fichiers touchés et les effets de bord possibles. ATTENDS mon accord avant d'écrire quoi que ce soit.
2. Modifications minimales et ciblées, en diff. Ne touche pas README.md, PROMPT.md, .env, ni aucun fichier hors du périmètre du plan. Ne modifie aucun fichier couvert par le hash d'un harnais inscrit sans incrémenter sa version (`bench/registry/` est en lecture seule pour le lanceur : seul `--register` l'écrit). Pas de refactor non demandé, pas de renommage, pas de formatage global.
3. Ne fais ni commit, ni push, ni merge, ni `git add`. Travaille sur une branche locale `harness-test` créée depuis l'état actuel ; si la création de branche est impossible, arrête-toi et dis-le-moi.
4. Aucun accès réseau, aucun appel d'API, aucun téléchargement. Ne lis pas le contenu de .env et ne l'affiche jamais.
5. N'invente ni API, ni option de CLI, ni nom de champ : si tu n'es pas sûr, lis le code ou demande.
6. Chaque affirmation sur le projet est taguée [DEFINI], [TESTABLE], [PLAUSIBLE], [SPECULATIF] ou [PROBABLEMENT_FAUX] ; la gématrie n'est pas supposée porteuse de sens (H0).

À construire (après mon accord sur le plan) :
A. Un lanceur déterministe en ligne de commande (machine à états en code, pas de serveur MCP) : `python -m bench.run --models <fichier> --task <id> --budget <tokens> --out runs/<date>/`. Les états : DECLARER (le modèle annonce sa limite de contexte) → PLANIFIER (compte rendu budgété par étapes, avec paquets de contexte modulaires et points de reprise) → EXECUTER → VALIDER (JSON strict selon EMBEDBABEL_BENCH_V2.md) → EVALUER (un autre modèle audite) → RAPPORT. Les modèles ne sont appelés qu'à l'intérieur des états.
B. Trois adaptateurs de modèle factices et reproductibles (graine fixe) : « honnête » (limite et budget corrects), « optimiste » (sous-estime son budget de 3×), « bavard » (dépasse sa limite de contexte). Ils servent à vérifier que le harnais distingue ces comportements.
C. Une garde de contexte : refuse ou tronque proprement tout paquet dépassant la limite déclarée, et consigne l'événement dans le journal (réutilise `bench/guard.py` plutôt que de dupliquer la règle ; cette règle est aussi implémentée en JavaScript dans `dsh/embedbabel-dsh/lib/guard.js`, les deux étant vérifiées par le même jeu de cas `tests/fixtures/guard_cases.json`).
D. Le compte rendu budgété : par étape, tokens estimés vs consommés, ce qui est chargé, ce qui est laissé hors contexte et comment le retrouver, point de reprise. Test de reprise : un second adaptateur reprend à l'étape n depuis le seul compte rendu et doit produire le même résultat.
E. Une évaluation croisée : une grille déterministe calculée par le harnais pour chaque compte rendu — honnêteté de la limite, exactitude du budget annoncé (adaptateurs qui en annoncent un) ou respect du budget accordé (modèles réels, qui n'annoncent rien), taille du plus gros paquet, reprise réussie. Aucun modèle n'audite le compte rendu d'un autre.
F. Des tests pytest pour A à E dans tests/ ; les tests existants doivent rester verts. Aucun test ne doit dépendre du réseau ou d'une horloge, et aucun run ne doit modifier un fichier suivi par git.

Critères d'acceptation (tu les vérifies en exécutant, pas en affirmant) :
- `python -m pytest -q` passe, y compris les anciens tests ; donne la sortie réelle.
- Lancé deux fois avec la même graine, le harnais produit des sorties identiques octet à octet.
- L'adaptateur « bavard » est détecté et journalisé ; l'adaptateur « optimiste » reçoit une erreur de budget mesurée (écart en %) ; l'adaptateur « honnête » passe.
- Une sortie JSON invalide (champ manquant, tag hors vocabulaire) est rejetée avec la raison.
- La reprise à l'étape n donne un résultat identique.
- `git status --porcelain` est identique avant et après un run.

Arrêt et signalement : si un point du plan exige de modifier un fichier hors périmètre, de résoudre le merge, ou si deux lectures du code se contredisent, arrête-toi et pose la question. À la fin, donne : la liste des fichiers créés/modifiés, la commande pour tout relancer, les tests exécutés avec leur sortie, ce qui n'a PAS été testé, et les effets de bord. Rien d'autre.
```
