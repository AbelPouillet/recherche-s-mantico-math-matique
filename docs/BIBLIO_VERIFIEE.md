# EmbedBabel — Bibliographie vérifiée (deep research, 2026-10-06)

Règle : une source n'est listée en « Vérifiée » que si sa page (arXiv, éditeur, dépôt) a été ouverte pendant cette session et que titre/auteurs/date/venue ont été relus. Les chiffres cités sont ceux **annoncés par les auteurs**, pas reproduits par nous. Rien ici ne prouve qu'EmbedBabel fonctionne ; la bibliographie sert à positionner le projet et à fixer les baselines.

## 1. Décodage spéculatif et drafts (axe le plus prometteur selon `docs/EMBEDBABEL_REPORT.md`)

| Réf. | Source | Ce qu'elle apporte à EmbedBabel |
|---|---|---|
| Leviathan, Kalman, Matias, *Fast Inference from Transformers via Speculative Decoding*, ICML 2023 — https://arxiv.org/abs/2211.17192 | Vérifiée | Cadre formel : un proposeur bon marché + vérification parallèle par le modèle principal, sortie identique ; annonce 2–3× (T5X). Le gain dépend du taux d'acceptation × coût du draft : c'est exactement la métrique à mesurer pour `propose()`. |
| Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, 2023 — https://arxiv.org/abs/2302.01318 | Vérifiée | Résultat indépendant, 2–2,5× annoncés. |
| Cai et al., *Medusa*, 2024 — https://arxiv.org/abs/2401.10774 | Vérifiée | Têtes de décodage multiples (nécessite entraînement) : baseline « petit module appris » (baseline 8). |
| Li et al., *EAGLE*, 2024 — https://arxiv.org/abs/2401.15077 | Vérifiée | Draft au niveau des features ; baseline apprise forte à battre. |
| Saxena, *Prompt Lookup Decoding* — https://github.com/apoorvumang/prompt-lookup-decoding | Vérifiée (dépôt, non évaluée par des pairs) | Draft par n-gram du prompt, sans modèle : 2,4× annoncé en moyenne sur résumé/QA, gain faible sur génération « originale ». **C'est la baseline directe d'un trie EmbedBabel-proposeur.** |
| llama.cpp `examples/lookup` — https://github.com/ggml-org/llama.cpp/blob/master/examples/lookup/README.md | Vérifiée (README minimal : paramètres ngram_min/ngram_max/n_draft) | Implémentation déjà existante dans la cible d'intégration. |

## 2. Prefix / prompt caching (le « gain réel sans EmbedBabel »)

| Réf. | Source | Apport |
|---|---|---|
| Kwon et al., *PagedAttention / vLLM*, SOSP 2023 — https://arxiv.org/abs/2309.06180 | Vérifiée | Gestion paginée du KV cache ; 2–4× de débit annoncés. |
| Zheng et al., *SGLang* (RadixAttention), 2023 — https://arxiv.org/abs/2312.07104 | Vérifiée | Cache de préfixes en arbre radix : **un trie de préfixes existe déjà pour le KV**, c'est la référence à comparer à l'arbre de sous-chaînes d'EmbedBabel. Jusqu'à 6,4× annoncés. |
| Gim et al., *Prompt Cache*, MLSys 2024 — https://arxiv.org/abs/2311.04934 | Vérifiée | Réutilisation modulaire d'états d'attention ; 8× (GPU) à 60× (CPU) annoncés sur des segments répétés. |

## 3. Représentations sans token / sous-mots / hachage

| Réf. | Source | Apport |
|---|---|---|
| Xue et al., *ByT5*, TACL 2022 — https://arxiv.org/abs/2105.13626 | Vérifiée | Baseline « byte-level » (baseline 2) ; robustesse au bruit de saisie. |
| Pagnoni et al., *Byte Latent Transformer*, 2024 — https://arxiv.org/abs/2412.09871 | Vérifiée | Patches dynamiques d'octets ≈ tokenisation à l'échelle 8B : la voie « état latent sur octets » est déjà explorée, en entraînant le modèle de bout en bout. |
| Svenstrup et al., *Hash Embeddings*, 2017 — https://arxiv.org/abs/1709.03933 | Vérifiée | Embeddings hachés : paramètres réduits, performance comparable annoncée. Baseline 6 (features hachées) et piste « réduction mémoire de la table d'embeddings ». |

## 4. Compression, KV cache, calcul adaptatif

| Réf. | Source | Apport |
|---|---|---|
| Zhang et al., *H2O*, 2023 — https://arxiv.org/abs/2306.14048 | Vérifiée | Éviction de KV (20 % conservés) : compression *du cache réel*, pas d'un substitut externe. |
| Jiang et al., *LLMLingua*, EMNLP 2023 — https://arxiv.org/abs/2310.05736 | Vérifiée | Compression de prompt jusqu'à 20× annoncée : concurrent direct de « pomper jusqu'au dessèchement ». |
| Schuster et al., *CALM*, NeurIPS 2022 — https://arxiv.org/abs/2207.07061 | Vérifiée | Sortie anticipée adaptative (jusqu'à ×3 annoncé) : réduit le calcul *dans* le modèle. |
| Kusupati et al., *Matryoshka Representation Learning*, 2022 — https://arxiv.org/abs/2205.13147 | Vérifiée | Embeddings emboîtés à dimension réglable : baseline pour la courbe m ∈ {16…256}. |
| Johnson, Douze, Jégou, *Billion-scale similarity search with GPUs* (FAISS), 2017 — https://arxiv.org/abs/1702.08734 | Vérifiée | Recherche vectorielle comprimée (PQ) : baseline « solution existante » face à une recherche vectorielle maison. |
| Jégou, Douze, Schmid, *Product quantization for nearest neighbor search*, IEEE TPAMI 2011 — notice HAL https://hal.univ-grenoble-alpes.fr/inria-00514462v1 | **Partielle** (existence/titre/venue vus dans les résultats de recherche ; page bloquée) | À relire avant citation formelle. |

## 5. Alignement multilingue et espaces hiérarchiques

| Réf. | Source | Apport |
|---|---|---|
| Conneau et al., *Word Translation Without Parallel Data*, ICLR 2018 — https://arxiv.org/abs/1710.04087 | Vérifiée | Alignement non supervisé d'espaces monolingues, sans info de caractères : base pour tester si des « relations inter-langues » battent un alignement appris. |
| Nickel & Kiela, *Poincaré Embeddings*, 2017 — https://arxiv.org/abs/1705.08039 | Vérifiée | Espace hyperbolique pour hiérarchies : candidat pour représenter l'arbre de sous-chaînes. |

## 6. Volet gématrie / sensoriel / statistiques

| Réf. | Source | Apport |
|---|---|---|
| Blasi et al., *Sound–meaning association biases evidenced across thousands of languages*, PNAS 113(39):10818–10823, 2016 — https://hraf.yale.edu/documents/1105 | Vérifiée (notice : référence et conclusion ; méthode de contrôle non relue, page PNAS bloquée en 403) | Seul canal « son↔sens » avec évidence interlangue sérieuse : justifie la phonétique, **pas la gématrie** (valeurs numériques d'orthographe). À relire pour la méthode de contrôle avant de s'en inspirer. |
| Brysbaert, Warriner, Kuperman, *Concreteness ratings for 40 thousand generally known English word lemmas*, Behav. Res. Methods 46(3):904–911, 2014 — https://norare.clld.org/sources/Brysbaert2014 | Vérifiée (notice) | Norme de concrétude pour opérationnaliser « sensoriel ». |
| Lynott et al., *The Lancaster Sensorimotor Norms*, Behav. Res. Methods 52:1271–1291, 2020 — https://norare.clld.org/sources/Lynott2020 | Vérifiée (notice) | Normes perception/action (40 000 mots anglais). |
| Benjamini & Hochberg, *Controlling the False Discovery Rate*, JRSS-B 57(1):289–300, 1995 — https://www.dcscience.net/Benjamini-Hochberg-1995-FDR.pdf | Vérifiée | Correction FDR exigée par `PROMPT.md` §2bis. |

## 7. Point d'intégration llama.cpp (vérifié dans `llama.h`, branche master)

- `llama_batch.embd` : « token embeddings (float vector of size n_embd), used when token is NULL » → on **peut** injecter des embeddings d'entrée ; le modèle les traite ensuite par toutes ses couches (le prefill n'est pas évité).
- `llama_memory_seq_rm / cp / keep / add / div` : manipulation du KV cache **par séquence**, donc réutilisation de préfixe et backtracking (backspace) possibles côté harness, sans fork de ggml.
- Aucune API ne permet d'injecter des K/V arbitraires par couche : confirme « K/V par couche non réaliste » du rapport.
- Non vérifié : exigences exactes du draft model de `examples/speculative` (README vide d'information dans la lecture faite).

## 8. Ce que la littérature change dans le verdict

1. **Confirmé** : la voie réaliste est « proposeur de drafts + prefix cache » (§1–2). Les baselines sont déjà fortes : lookup n-gram (llama.cpp), RadixAttention, Prompt Cache. EmbedBabel doit les battre à coût égal, sinon il n'apporte rien.
2. **Fragilisé** : « petit vecteur pour compresser le contexte » est déjà couvert par LLMLingua (prompt), H2O (KV), BLT/ByT5 (octets), hash embeddings (table). La seule part non couverte est l'usage de relations structurées inter-langues comme prior, non démontrée.
3. **Gématrie** : aucune source trouvée ne soutient un contenu sémantique de la gématrie ; l'évidence interlangue disponible (Blasi et al.) concerne les sons. H0_g reste l'hypothèse nulle par défaut.
4. **Expérience minimale proposée** (révisée) : sur un corpus de code/formulaires multilingue, comparer le taux d'acceptation et les tokens/s de `propose()` d'un trie EmbedBabel contre (a) lookup n-gram llama.cpp, (b) prefix cache seul ; puis ablation avec/sans gématrie via permutations ≥10 000 et FDR.

## 9. Limites de cette recherche

Pas de lecture intégrale des articles (résumés/pages de notice seulement). Non cherchés : hachage sémantique (Salakhutdinov & Hinton), modèles à état récurrent (Mamba/RWKV), travaux de linguistique historique computationnelle (cognats, EtymWordNet), CacheBlend. Les chiffres de speedup sont des annonces d'auteurs, sur leurs bancs.
