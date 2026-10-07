# Prompt Claude Code — test du harnais de bench EmbedBabel

Colle le bloc ci-dessous dans Claude Code, lancé à la racine du dépôt `recherche-s-mantico-math-matique`.

```text
Contexte : dépôt EmbedBabel (benchmark de recherche, en français). Lis d'abord README.md (il contient des marqueurs de conflit de merge : ne les résous pas), PROMPT.md, bench/prompts/EMBEDBABEL_BENCH_V2.md, docs/LED7_SIGNATURE_BUS.md, docs/EMBEDBABEL_REPORT.md, le plugin plugins/deepseek-r1-gématriphonéticospatiale-v0.1.0/ (adapter.py, manifest.json, strategy.md) et integrations/deepseek-harness-labsia/tests/test_context_guard.py.

Objectif : tester le harnais de bench. Un « harnais » = le programme qui donne le même prompt à chaque modèle, borne et mesure son contexte, valide sa sortie JSON stricte, puis fait évaluer les sorties par les autres modèles. Je veux pouvoir le lancer hors ligne avec des modèles factices avant de payer le moindre appel.

Règles de travail (impératives) :
1. Plan avant code. Commence en LECTURE SEULE : `git status`, `git log --oneline -15 --all`, `python -m pytest -q`. Donne-moi l'état réel (tests qui passent/échouent, ce que fait déjà adapter.py, ce que couvre test_context_guard.py) puis un plan numéroté avec, pour chaque étape, les fichiers touchés et les effets de bord possibles. ATTENDS mon accord avant d'écrire quoi que ce soit.
2. Modifications minimales et ciblées, en diff. Ne touche pas README.md, PROMPT.md, .env, ni aucun fichier hors du périmètre du plan. Pas de refactor non demandé, pas de renommage, pas de formatage global.
3. Ne fais ni commit, ni push, ni merge, ni `git add`. Travaille sur une branche locale `harness-test` créée depuis l'état actuel ; si la création de branche est impossible à cause du merge en cours, arrête-toi et dis-le-moi.
4. Aucun accès réseau, aucun appel d'API, aucun téléchargement. Ne lis pas le contenu de .env et ne l'affiche jamais.
5. N'invente ni API, ni option de CLI, ni nom de champ : si tu n'es pas sûr, lis le code ou demande.
6. Chaque affirmation sur le projet est taguée [DEFINI], [TESTABLE], [PLAUSIBLE], [SPECULATIF] ou [PROBABLEMENT_FAUX] ; la gématrie n'est pas supposée porteuse de sens (H0).

À construire (après mon accord sur le plan) :
A. Un lanceur déterministe en ligne de commande (machine à états en code, pas de serveur MCP) : `python -m bench.run --models <fichier> --task <id> --budget <tokens> --out runs/<date>/`. Les états : DECLARER (le modèle annonce sa limite de contexte) → PLANIFIER (compte rendu budgété par étapes, avec paquets de contexte modulaires et points de reprise) → EXECUTER → VALIDER (JSON strict selon EMBEDBABEL_BENCH_V2.md) → EVALUER (un autre modèle audite) → RAPPORT. Les modèles ne sont appelés qu'à l'intérieur des états.
B. Trois adaptateurs de modèle factices et reproductibles (graine fixe) : « honnête » (limite et budget corrects), « optimiste » (sous-estime son budget de 3×), « bavard » (dépasse sa limite de contexte). Ils servent à vérifier que le harnais distingue ces comportements.
C. Une garde de contexte : refuse ou tronque proprement tout paquet dépassant la limite déclarée, et consigne l'événement dans le journal (réutilise ce que fait déjà test_context_guard.py plutôt que de le dupliquer).
D. Le compte rendu budgété : par étape, tokens estimés vs consommés, ce qui est chargé, ce qui est laissé hors contexte et comment le retrouver, point de reprise. Test de reprise : un second adaptateur reprend à l'étape n depuis le seul compte rendu et doit produire le même résultat.
E. Une évaluation croisée : chaque modèle audite le plan d'un autre sans l'exécuter ; grille = honnêteté de la limite, exactitude du budget, taille du plus gros paquet, reprise réussie.
F. Des tests pytest pour A à E dans tests/ ; les tests existants doivent rester verts. Aucun test ne doit dépendre du réseau ou d'une horloge.

Critères d'acceptation (tu les vérifies en exécutant, pas en affirmant) :
- `python -m pytest -q` passe, y compris les anciens tests ; donne la sortie réelle.
- Lancé deux fois avec la même graine, le harnais produit des sorties identiques octet à octet.
- L'adaptateur « bavard » est détecté et journalisé ; l'adaptateur « optimiste » reçoit une erreur de budget mesurée (écart en %) ; l'adaptateur « honnête » passe.
- Une sortie JSON invalide (champ manquant, tag hors vocabulaire) est rejetée avec la raison.
- La reprise à l'étape n donne un résultat identique.

Arrêt et signalement : si un point du plan exige de modifier un fichier hors périmètre, de résoudre le merge, ou si deux lectures du code se contredisent, arrête-toi et pose la question. À la fin, donne : la liste des fichiers créés/modifiés, la commande pour tout relancer, les tests exécutés avec leur sortie, ce qui n'a PAS été testé, et les effets de bord. Rien d'autre.
```
