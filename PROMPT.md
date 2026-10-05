# EMBEDBABEL — Research challenge

Tu es une IA participant à un benchmark de recherche indépendant. Tu dois analyser, critiquer et tenter de formaliser le projet **EmbedBabel** décrit ci-dessous.

## 0. Règle fondamentale

Ne pars pas du principe que l'idée est correcte.

Tu dois distinguer explicitement :
1. ce qui est mathématiquement définissable ;
2. ce qui est empiriquement testable ;
3. ce qui est plausible mais non démontré ;
4. ce qui est probablement faux ou incompatible avec les contraintes connues des LLM ;
5. ce qui pourrait néanmoins conduire à une optimisation réelle de l'inférence.

Ne confonds jamais « représentation plus compacte » avec « moins de FLOPs », « moins de mémoire », « moins de bande passante » ou « meilleure perplexité ».

Lorsque tu avances une affirmation technique importante, donne soit une justification, soit une expérience qui permettrait de la falsifier.

---

# 1. Idée générale

**EmbedBabel** est une couche expérimentale destinée à un harness autour de `llama.cpp`.

L'objectif est d'explorer si un dictionnaire linguistique structuré sous forme d'arbres de sous-chaînes, enrichi de relations numériques/gématriques, ordinales, sémantiques et éventuellement sensorielles, peut produire **incrémentalement**, à mesure que l'utilisateur saisit du texte caractère par caractère, un **état vectoriel sémantique extrêmement compact**.

Cet état pourrait ensuite être utilisé pour compléter/préparer un **prefill/draft** destiné au modèle principal.

L'hypothèse radicale est la suivante :

> Au lieu de faire porter au modèle principal toute la découverte progressive du contexte à partir des tokens, une structure externe pourrait accumuler les contraintes sémantiques induites par chaque nouveau caractère et maintenir un état latent compact représentant l'espace des interprétations encore compatibles avec le préfixe.

Le projet ne prétend pas que cette hypothèse est vraie. C'est précisément ce que le benchmark doit déterminer.

---

# 2. Arbre gématrique

Dans EmbedBabel, une entrée textuelle n'est pas seulement :

    string -> embedding

Elle peut être représentée comme un arbre de sous-chaînes.

Exemple conceptuel :

    "maison"
       |
       +-- "m"
       |
       +-- "ma"
       |     +-- "m"
       |     +-- "a"
       |
       +-- "mai"
       |
       +-- "mais"
       |
       +-- "maison"
             +-- "mai"
             +-- "son"

Cette structure doit cependant être généralisée.

Chaque nœud pourrait contenir par exemple :

    Node {
        string / grapheme sequence
        parent(s)
        children[]
        order
        length
        language
        gematria value(s)
        normalized numerical representation
        semantic relations
        sensory relations
        phonetic relations
        morphological relations
        embedding prototype / residual
        confidence
        frequency
    }

Les relations ne doivent pas être supposées valides simplement parce qu'elles existent dans un dictionnaire.

---

# 2bis. Dimension phonétique et diachronique

La v1 ne distingue pas la similarité **graphique** (lettres) de la similarité **sonore** (phonétique), et n'a aucun axe diachronique (évolution, emprunts, contractions). Chaque nœud/feuille de l'arbre doit donc pouvoir porter une dimension phonétique, associée à son territoire d'origine **et** aux zones côtières/de contact.

Extension du nœud :

    Node.phon {
        ipa[]            // transcriptions IPA, par variété/territoire
        syllables[]      // découpage syllabique (uniquement si la source le fournit)
        region/dialect   // territoire d'origine ET zones côtières/de contact
        period           // datation de la forme
        phon_features    // traits articulatoires (voisement, lieu, mode, voyelle)
    }
    Edge types ajoutés (avec provenance `source` et `confidence`) :
        - cognate(l1,l2)         // origine commune
        - borrowing(src→dst)     // emprunt
        - sound_change(rule)     // règle de changement phonétique
        - contraction(rule)      // ex. es- → é-
        - contact_blend(l1,l2)   // mélange culturel

**Exemple *écureuil* / *squirrel*** (à formuler prudemment) : les deux mots sont rattachés au latin tardif *sciurus* (du grec *skíouros*) ; le français passe par l'ancien français *escurel* puis *écureuil*, et l'anglais a emprunté à l'ancien français. La contraction « es- → é- » (chute du *s* préconsonantique) est une évolution documentée du français. Le système doit **retrouver ces règles à partir de données documentées** (étymologie Wiktionnaire/EtymWordNet, listes de cognats, `scripts/fetch_etymology.py`) et les valider sur des cas connus. Il ne doit **jamais** « déduire » librement une histoire plausible : une règle sans source est refusée.

Ressources du dépôt : `data/` (dictionnaires IPA téléchargés, jamais commités), `data/SOURCES.md` (sources, licences, couverture), `graph/` (schéma Neo4j), `scripts/build_graph.py`.

## Hypothèses à tester

- **H0** : phonétique + morphologie suffisent ; la gématrie n'apporte rien au-delà.
- **H1** : la gématrie apporte un gain mesurable **au-dessus** des baselines H0.

Avertissement : la gématrie n'a aucune raison de survivre aux changements phonétiques (les valeurs changent avec l'orthographe) ; la phonétique est le canal diachronique le plus plausible. Le benchmark doit le tester explicitement, pas le supposer.

## Contrôles statistiques

- Toute corrélation gématrico-sensorielle entre langues est facile à trouver par hasard (beaucoup de fonctions candidates, peu de données). Exiger des **tests de permutation** (étiquettes de mots/sens permutées, au moins 10 000 tirages) et une **correction des comparaisons multiples** (Bonferroni/Holm ou FDR de Benjamini–Hochberg, en déclarant le nombre total de tests essayés).
- Le « sensoriel » doit être **opérationnalisé** par des normes psycholinguistiques existantes (par ex. concrétude, imageabilité, valeurs sensorielles, symbolisme sonore), citées avec leur source vérifiée, et non par intuition.
- Les baselines minimales : lettres seules, IPA seul, motif aléatoire de même longueur.

---

# 2ter. Mots « à inventer » (espéranto IA-friendly)

Objectif à rendre mesurable : composer, pour chaque mélange culturel, des formes candidates où chaque lettre tapée et/ou chaque syllabe prononcée est exploitable par un prefill/draft `llama.cpp` composé avec EmbedBabel pour cadrer la compréhension de la requête et la recherche de contexte (RAG).

- **Tâche** : étant donné deux langues A et B en contact, générer des formes néologiques candidates.
- **Évaluation** : (a) rétro-test sur des mélanges réels et documentés (pidgins, créoles, mots-valises attestés) en masquant la forme et en la reconstruisant ; (b) panel de locuteurs ; (c) efficacité en tokens du lexique obtenu face à un tokenizer standard.
- **Contrôles** : un espéranto/interlangue existant (Esperanto, Interlingua) et un générateur aléatoire contraint par la phonotactique.
- **Piège hors distribution** : un lexique « IA-friendly » doit l'être pour un LLM *pré-entraîné*. Un vocabulaire nouveau est hors distribution et probablement **pire** sans fine-tuning ; le benchmark doit mesurer ce coût, pas l'ignorer.

---

# 2quater. Désambiguïsation d'homophones par contexte

Deux mots peuvent sonner pareil mais avoir une proximité contextuelle différente (ex. *vers / verre / ver / vert*, *sain / saint / sein*).

- **Tâche** : `sons(w1) = sons(w2)` mais sens différent ; choisir le bon mot selon l'historique et la requête.
- **Sketch de contexte** : les hash des mots/syllabes de l'historique et de la requête forment un sketch compact. Il doit être comparé explicitement à des baselines **MinHash, SimHash et count-min** sur un simple bag-of-n-grams hashé, pour savoir si la structure gématrico-phonétique apporte quoi que ce soit en plus.
- **Mesures** : accuracy en streaming après k syllabes de requête, latence par désambiguïsation, comparaison avec un n-gramme de contexte et un petit modèle de langage.

---

# 2quinquies. Couche RAG : Neo4j + Qdrant contre solution maison

Architecture à évaluer :

    requête (flux caractère/syllabe)
        → EmbedBabel (état incrémental)
        → graphe lexical phonético-gématrique (arbres de sous-chaînes dans Neo4j)
        → ancrage du contexte : graphe de connaissances (entités/chunks liés)
        → vecteurs de chunks (Qdrant, indexés sur les arbres Neo4j)
        → prefill/draft llama.cpp

Les IA sont **notées** sur leur comparaison technique entre un Neo4j/Qdrant et une solution maison pour EmbedBabel, afin de juger si le choix relève de la facilité de développement ou de raisons techniques valables.

| Critère | Poids | Question posée |
|---|---|---|
| Adéquation du modèle de données | 15 | Préfixes, sous-chaînes partagées et relations typées sont-ils naturels dans un graphe de propriétés ? |
| Latence en streaming | 20 | Un lookup par caractère tient-il dans le budget (µs contre ms) ? Un aller-retour réseau/IPC par frappe est un risque majeur. |
| Structure en mémoire | 10 | Un trie/DAWG/FST embarqué bat-il une base serveur pour la partie lexicale ? |
| Recherche vectorielle | 10 | Qdrant (HNSW, filtres payload, quantification) est-il réellement plus efficace qu'une implémentation maison ? Sinon, pourquoi la refaire ? |
| Filtrage hybride | 10 | Graphe + vecteurs combinés sans aller-retour coûteux ? |
| Coût de maintenance | 10 | Le coût dev/ops d'une solution maison est-il justifié par un gain mesuré ? |
| Reproductibilité | 10 | Un autre chercheur peut-il reproduire le banc (`docker-compose.yml`) ? |
| Évolutivité | 5 | Mise à jour du dictionnaire, versioning, multilingue. |
| Honnêteté de la justification | 10 | A-t-on mesuré, ou affirmé ? |

**Test de découplage** : séparer (1) le **lexique embarqué** (trie/FST, très chaud, contraintes de latence), probablement mieux servi par une structure maison ou une bibliothèque embarquée, et (2) le **graphe de contexte et les chunks** (volumineux, évolutif, requêtes ad hoc), qui justifie plus souvent Neo4j/Qdrant. Exiger des mesures sur le même jeu de données : **latence p50 et p99** par opération, **mémoire**, **débit**. Pénaliser tout choix d'outil sans chiffre ni analyse du goulot d'étranglement.

Hypothèse de travail à falsifier : « une architecture hybride (lexique embarqué maison + Neo4j/Qdrant pour le contexte) domine les deux extrêmes ». Plausible, non démontrée.

## Rappels

- **Petit vecteur ≠ moins de FLOPs** (ni moins de mémoire ni moins de bande passante).
- **Graphe riche ≠ meilleur contexte** : une base de graphe volumineuse n'améliore pas la désambiguïsation tant qu'on n'a pas mesuré le gain contre les baselines.
- Ne cite aucune source que tu n'as pas réellement vérifiée.

---

# 3. Relations inter-langues

Le projet cherche à étudier des relations entre arbres issus de différentes langues.

Une relation peut être :

- numérique ;
- ordinale ;
- morphologique ;
- phonétique ;
- lexicale ;
- sémantique ;
- compositionnelle ;
- éventuellement sensorielle/perceptuelle ;
- ou une combinaison de plusieurs de ces dimensions.

La question difficile est :

> Existe-t-il des invariants ou des transformations entre ces arbres qui permettent de prédire une partie de la représentation sémantique d'un préfixe sans exécuter le coût complet du modèle principal ?

Il faut être particulièrement vigilant contre les corrélations artificielles, l'overfitting linguistique, les coïncidences numériques et les biais du dictionnaire.

---

# 4. Mise à jour caractère par caractère

L'état EmbedBabel doit idéalement être incrémental.

Pour un préfixe :

    c1 c2 c3 ... cn

on cherche quelque chose de la forme :

    S0 = initial_state

    S(n+1) = F(Sn, c(n+1), Graph)

plutôt que :

    S(n+1) = F(c1...c(n+1))

L'état pourrait être :

    S = compressed semantic state

et l'opération d'ajout :

    S' = Advance(S, character)

devrait exploiter les branches compatibles de l'arbre plutôt que recalculer toute l'histoire.

Une formulation possible est :

    S' = Compress(
            Merge(
                S,
                Delta(character, active_graph_branches)
            )
         )

Il faut proposer plusieurs formulations concurrentes, pas en imposer une.

---

# 5. « Pompable jusqu'au désechement »

Expression utilisée pour décrire l'objectif expérimental :

> pousser la réduction aussi loin que possible tout en conservant l'information utile à la prédiction.

La compression recherchée n'est donc pas seulement textuelle.

On cherche à savoir si l'on peut passer de :

    caractères
       ↓
    sous-chaînes
       ↓
    relations
       ↓
    hypothèses sémantiques
       ↓
    représentation latente
       ↓
    fusion / factorisation
       ↓
    état compact

avec une perte contrôlée.

Mais il faut mesurer la perte.

Définir notamment :

    semantic_loss
    predictive_loss
    reconstruction_loss
    perplexity_delta
    next-token accuracy delta
    latency
    FLOPs
    memory traffic
    KV-cache footprint
    energy, si mesurable

---

# 6. Intégration llama.cpp

Le prototype doit être pensé pour un harness `llama.cpp`.

Une architecture conceptuelle possible :

    User input
        |
        v
    EmbedBabel
        |
        +-- Graph lookup
        +-- incremental traversal
        +-- relation propagation
        +-- state compression
        |
        v
    VectorState
        |
        v
    Draft / prefill adapter
        |
        v
    llama.cpp
        |
        v
    main model

Une interface C++ possible, à critiquer :

    class EmbedBabel {
    public:
        VectorState advance(
            const VectorState& state,
            char32_t c
        );

        VectorState finalize(
            const VectorState& state
        );
    };

    class VectorState {
    public:
        ...
    };

Le candidat doit déterminer si cette interface est pertinente ou proposer mieux.

Il faut notamment étudier où intervenir réellement dans `llama.cpp` :

- tokenizer ;
- input embedding ;
- batch/prefill ;
- KV cache ;
- draft model ;
- speculative decoding ;
- logits ;
- GGML graph ;
- ou une couche extérieure.

Ne prétends pas qu'un « plugin » peut modifier arbitrairement `llama.cpp` sans examiner les contraintes réelles de son architecture.

---

# 7. Question centrale du benchmark

Réponds à cette question :

> **Peut-on construire un système EmbedBabel qui exploite des arbres linguistiques/gématriques et des relations structurées entre sous-chaînes pour maintenir, caractère par caractère, un état latent très compact capable de remplacer ou réduire une partie du travail de contextualisation d'un LLM, tout en obtenant un gain mesurable de latence, mémoire ou coût d'inférence dans un système réel basé sur llama.cpp ?**

Il faut répondre à la fois :

### Théoriquement
Quelle information peut être conservée, perdue ou factorisée ?

### Mathématiquement
Comment définir les graphes, les relations et l'opérateur de composition ?

### Linguistiquement
Quelles relations sont réellement invariantes entre langues ?

### ML
Comment apprendre ou calibrer les transformations ?

### Systèmes
Où le gain de calcul peut-il réellement apparaître ?

### llama.cpp
Quel point d'intégration est réaliste ?

### Expérimentalement
Quelle expérience falsifierait l'hypothèse ?

---

# 8. Exigence particulière : ne pas confondre gematria et sémantique

La gématrie fournit des fonctions numériques sur des chaînes ou caractères.

Cela ne démontre pas qu'une valeur numérique encode leur sens.

Tu dois donc séparer :

    valeur gématrique
    ≠
    représentation sémantique

et proposer un mécanisme expérimental permettant de vérifier si les deux présentent une information prédictive exploitable.

Si la gématrie n'apporte aucune information supplémentaire par rapport à des baselines simples, il faut le dire clairement.

---

# 9. Baselines obligatoires

Comparer au minimum contre :

1. caractères seuls ;
2. byte-level ;
3. tokenizer normal du modèle ;
4. embeddings de sous-chaînes sans gématrie ;
5. trie/prefix tree classique ;
6. hash/features numériques aléatoires ;
7. embeddings appris ;
8. petit modèle spécialisé de pré-encodage ;
9. éventuellement un modèle n'utilisant aucune structure EmbedBabel.

Il faut contrôler le nombre de paramètres et le coût de calcul.

---

# 10. Compression

Étudier au minimum :

- low-rank ;
- quantification ;
- hashing ;
- product quantization ;
- sparse representation ;
- factorisation d'arbre ;
- shared subtrees ;
- residual vectors ;
- delta encoding ;
- codebooks ;
- clustering ;
- éventuellement une représentation hyperbolique ou autre espace adapté aux arbres.

Ne pas supposer qu'une compression est bénéfique : mesurer son coût de décodage.

---

# 11. Critère de succès

Une démonstration convaincante doit montrer au moins un des phénomènes suivants :

### A — même qualité, moins de coût

    quality ≈ baseline
    latency < baseline

### B — même coût, meilleur contexte

    latency ≈ baseline
    quality > baseline

### C — même qualité, mémoire réduite

    quality ≈ baseline
    memory << baseline

### D — préfill réellement évité

Une partie mesurable du travail du modèle principal devient inutile.

Le simple fait de produire un petit vecteur n'est PAS un succès si le modèle doit ensuite refaire exactement le même travail pour l'exploiter.

---

# 12. Expérience minimale demandée

Proposer une expérience reproductible :

    corpus multilingue
        ↓
    construction des arbres
        ↓
    génération des relations
        ↓
    entraînement/calibration
        ↓
    streaming caractère par caractère
        ↓
    VectorState
        ↓
    llama.cpp
        ↓
    benchmark

Mesurer :

    tokens/s
    prefill latency
    decode latency
    peak RAM
    VRAM
    KV-cache
    FLOPs estimés
    perplexity
    next-token accuracy
    semantic retrieval
    qualité multilingue

Comparer plusieurs niveaux de compression.

## Expérience additionnelle : visualisation circulaire (`viz/circle_trace.py`)

Les n lettres de l'alphabet sont placées à égale distance sur un cercle (angle 2πk/n, n rayons). Une entrée (texte, phonèmes IPA, ou transcription ASR optionnelle) trace un parcours lettre → lettre ; chaque préfixe produit une image, d'où un GIF de la construction incrémentale, éventuellement coloré selon la valeur gématrique, avec un panneau « motif phonétique » (même construction sur les phonèmes) côte à côte.

À tester, **sans présupposer qu'il y ait quoi que ce soit à apprendre** : le motif visuel a-t-il une information prédictive au-delà de baselines — lettres seules, IPA seul, motif aléatoire de même longueur ? Avec permutations et correction des comparaisons multiples (section 2bis). Voir la section « Hypothèses à tester » du `README.md`.

---

# 13. Langues

Étudier au minimum plusieurs familles linguistiques, par exemple :

- français ;
- anglais ;
- allemand ;
- espagnol ;
- arabe ;
- hébreu ;
- grec ;
- éventuellement chinois.

Ne pas supposer que le même arbre ou la même relation fonctionne partout.

Étudier notamment :

- alphabets différents ;
- scripts ;
- morphologie ;
- ordre des mots ;
- segmentation ;
- langues flexionnelles ;
- langues agglutinantes ;
- systèmes non alphabétiques.

---

# 14. Attention aux faux gains

Analyser explicitement les pièges :

- coût de construction du dictionnaire ;
- coût de recherche dans l'arbre ;
- coût de calcul des relations ;
- coût du mapping vers l'espace latent du LLM ;
- overhead CPU/GPU ;
- transfert mémoire ;
- synchronisations ;
- cache misses ;
- coût de décompression ;
- coût de conversion entre espaces vectoriels.

Un système n'est pas plus rapide parce que son vecteur intermédiaire est plus petit.

---

# 15. Ce que je veux dans ton rapport

Structure impérative :

## 1. Verdict exécutif

En quelques paragraphes :

- idée viable ?
- idée partiellement viable ?
- idée probablement fausse ?
- partie la plus prometteuse ?

## 2. Reformulation mathématique

Définir proprement :

    G
    Node
    Edge
    Relation
    State
    Advance()
    Compress()
    ProjectToLLM()

## 3. Architecture

Diagramme clair.

## 4. Algorithmes

Pseudo-code précis.

## 5. Intégration llama.cpp

Indiquer précisément où et comment intervenir.

## 6. Ce qui peut réellement accélérer l'inférence

Distinguer :

- gain théorique ;
- gain mesurable ;
- gain probable ;
- gain illusoire.

## 7. Expérience falsifiable

Donner le protocole exact.

## 8. Baselines

## 9. Risques et objections

Chercher activement à réfuter EmbedBabel.

## 10. Prototype

Proposer une architecture C++ concrète et, si possible, du pseudo-code ou du code compilable.

## 11. Recherche bibliographique

Identifier les travaux pertinents sur :

- speculative decoding ;
- prefix caching ;
- prompt caching ;
- token-free / byte-level models ;
- learned compression ;
- semantic hashing ;
- graph embeddings ;
- hierarchical embeddings ;
- incremental contextual representations ;
- multilingual representation alignment ;
- tree-structured representations ;
- llama.cpp / GGML ;
- KV-cache compression ;
- early exit ;
- latent-space inference.

Ne cite aucune source que tu n'as pas réellement vérifiée.

## 12. Conclusion

Répondre explicitement :

> Quelle est la plus petite expérience capable de prouver que cette idée mérite d'être développée ?

---

# 16. Deuxième passe

Ton rapport sera donné à d'autres IA qui devront le critiquer.

Tu dois donc rendre tes hypothèses, équations, interfaces et affirmations suffisamment explicites pour qu'un autre chercheur puisse :

- reproduire ;
- critiquer ;
- réfuter ;
- améliorer.

Ne cherche pas à être flatteur envers le projet.

**Le but est de trouver si EmbedBabel contient une idée réellement nouvelle et exploitable, et si oui, où se trouve précisément cette idée.**
