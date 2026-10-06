# EmbedBabel — Rapport de recherche critique

Statut des affirmations : **[D]** définissable, **[T]** testable, **[P]** plausible non démontré, **[F]** probablement faux, **[O]** piste d'optimisation réelle.
Aucune source externe n'a été consultée pour ce rapport : la section 11 ne liste que des *axes de recherche*, pas des citations. Rien n'a été benchmarké ; seul le prototype minimal `prototype/embedbabel_min.cpp` a été compilé et exécuté (il ne mesure aucun gain).

## 1. Verdict exécutif

- **Version radicale (état latent externe remplaçant du prefill du LLM) : probablement fausse [F].** Le prefill d'un LLM calcule des K/V par couche, conditionnés par toutes les couches précédentes. Un vecteur unique par préfixe ne peut pas reproduire ces K/V ; sans eux, le modèle doit refaire son travail (critère D du sujet non satisfait). Le coût du prefill est de ~2·N_params FLOPs/token, indépendant de la façon dont le vecteur d'entrée est produit.
- **Gématrie comme signal sémantique : très probablement nulle [F/T].** C'est une somme/fonction d'index de caractères ; elle est non injective et sans lien causal avec le sens. Hypothèse nulle à tester : elle n'apporte rien par rapport à un hash aléatoire (baseline 6).
- **Partiellement viable [P/O]** : (a) un trie/arbre de sous-chaînes incrémental comme *cache de préfixes et accélérateur de drafting* (type lookup/n-gram draft) pour le décodage spéculatif ; (b) pré-encodage de l'entrée en cours de frappe (le prefill du préfixe tapé peut démarrer avant la fin de la saisie : c'est du *prefix caching* classique, gain réel, sans EmbedBabel) ; (c) embeddings de sous-chaînes/n-grammes (idée de type hash-embeddings) pour réduire la matrice d'embedding — gain mémoire possible, gain latence marginal.
- **Partie la plus prometteuse** : EmbedBabel comme **proposeur de drafts** (draft sans réseau, ou petit modèle auxiliaire), dont le gain se mesure par taux d'acceptation × coût du draft. Le reste (gématrie, inter-langues) n'est justifié que si une ablation le montre.
- **Nouveauté** : l'idée d'une mise à jour incrémentale caractère par caractère existe déjà (RNN, tries, prefix caching). La seule partie potentiellement nouvelle est l'usage de *relations structurées inter-langues* comme prior ; elle n'est pas démontrée.

## 2. Reformulation mathématique

- Alphabet Σ (graphèmes Unicode, normalisation NFC), langues L.
- **G = (V, E, R)** : V = ensemble de nœuds, un par couple (langue, chaîne) ; E ⊆ V×V arêtes typées (préfixe, sous-chaîne, composition `mai+son`) ; R = ensemble de relations pondérées r : V×V → [0,1] avec type ∈ {num, ord, morph, phon, lex, sem, sens} et **confiance c(r)** estimée sur un jeu de validation, jamais supposée.
- **Node** v = (s, lang, len, freq, g(s) ∈ ℤ^k, p_v ∈ ℝ^D prototype, ρ_v résidu).
- **g** : Σ* → ℤ, par ex. g(s)=Σ val(cᵢ). Hypothèse H_g : I(g(s); y | features baselines) > 0 où y = représentation/étiquette sémantique. Cette hypothèse est ce qu'on teste.
- **State** S_n = (a_n, h_n) : a_n ⊆ V ensemble borné (top-K) de nœuds actifs compatibles avec le préfixe, h_n ∈ ℝ^d vecteur compact.
- **Advance** : a_{n+1} = Next(a_n, c) ∪ Spawn(c) (tronqué à K) ; h_{n+1} = Compress(Merge(h_n, Δ(c, a_{n+1}))).
  Formulations concurrentes :
  1. **Somme à oubli** : h' = λh + Σ_{v∈a'} w_v p_v (le prototype).
  2. **Récurrente apprise** : h' = GRU(h, E[c] + Σ w_v p_v).
  3. **Mélange de lois** : S = distribution sur les mots compatibles (π_w), h = Σ π_w e_w (type « beam sémantique »).
  4. **Résiduelle arborescente** : h = p_{v_courant} + ρ_{v_courant}, où ρ est un résidu par rapport au parent (delta encoding), partageant les sous-arbres.
- **Compress** : C : ℝ^d → ℝ^m, m ≪ d (low-rank, PQ, hash, sparse, codebook). Contrainte de coût : cost(C⁻¹) inclus dans le bilan.
- **ProjectToLLM** : P : ℝ^m → ℝ^{d_model} × (couches ?). Deux niveaux : (i) *une* embedding d'entrée (compatible llama.cpp via `llama_batch.embd`) ; (ii) K/V par couche (**non réaliste** sans exécuter le modèle).
- **Pertes** : semantic_loss = 1 − cos(h_n, e_LLM(préfixe)) ; predictive_loss = KL(p_LLM(·|préfixe) ‖ p_{LLM+EB}(·|préfixe)) ; reconstruction_loss = ‖Decode(h) − e‖² ; perplexity_delta, next-token acc delta, etc.
- **Théorie de l'information** : h_n de dimension m bits finis ne peut conserver qu'au plus m·b bits ; l'espace des préfixes croît exponentiellement. La compression est donc *lossy* ; ce qui compte est l'information pertinente pour p(x_{t+1}|x_{≤t}). Le LLM ne peut pas être contourné pour l'information de longue portée (accord, coréférence).

## 3. Architecture

```
saisie (char)
   │
   ▼
[normalisation NFC + segmentation graphèmes]
   │
   ▼
[trie / graphe G]──lookup O(1) amorti──► a_n (nœuds actifs)
   │                                         │
   └── Advance(S_n, c) ◄─────────────────────┘
          │
          ▼
     VectorState (h_n, a_n)
          │
   ┌──────┴────────────────────────────────┐
   ▼                                       ▼
[Draft adapter]                  [Adapter d'embedding P]
 (tokens candidats / n-gram)      (optionnel, embd d'entrée)
   │                                       │
   ▼                                       ▼
llama.cpp : speculative / lookup     llama_batch (prefill incrémental)
   │
   ▼
modèle principal (vérification + KV cache standard)
```
Vue critique : tout ce qui est à gauche de llama.cpp doit être moins coûteux que le travail qu'il économise à droite.

## 4. Algorithmes

```
Build(corpus, langues):
  pour chaque mot w, fréquence f: insérer tous les suffixes/préfixes dans le trie de sa langue
  nœud.freq += f ; nœud.g = g(sous-chaîne)
  partager les sous-arbres identiques (DAG, hash-consing)
  calibrer prototypes p_v par régression vers e_LLM(sous-chaîne dans contexte) (cf. §7)
  relations inter-langues : candidates = (alignement bilingue ∪ cognats ∪ même g)
        garder r si gain de validation > seuil ET corrigé par permutation (voir §9)

Advance(S, c):
  a' = {Next(v,c) | v ∈ S.a} ∪ {racine·c}        # O(K)
  si a' vide: repli byte-level
  Δ = Σ_{v∈a'} w_v p_v                            # O(K·d)
  h' = Compress(λ S.h + Δ)
  retourner (h', top-K(a'))

Finalize(S): normaliser, projeter, éventuellement émettre des tokens de draft
```
Coût par caractère O(K·d) ; à comparer au coût d'un token LLM (≈2·N_params FLOPs, soit ~10⁹–10¹⁰ pour 0,5–7B) : le calcul EB est négligeable, **ce n'est pas lui le facteur limitant**, c'est la valeur utile du vecteur.

## 5. Intégration llama.cpp

Points d'intervention (à vérifier dans la version ciblée du dépôt) :

| Point | Réalisme | Commentaire |
|---|---|---|
| Couche extérieure (harness) | **élevé** | pilote `llama_decode` avec batches ; ne modifie pas ggml |
| Tokenizer | moyen | remplacer par EB rompt la correspondance avec les poids du modèle |
| Embedding d'entrée (`llama_batch.embd`) | moyen | llama.cpp accepte des embeddings en entrée ; il faut un adapter entraîné ; le modèle voit un token « étranger » |
| Prefill par batch | **élevé** | prefill pendant la frappe, KV cache conservé (réel) |
| KV cache | faible | EB ne produit pas de K/V valides |
| Draft / spéculatif | **élevé** | proposeur externe de tokens (lookup/n-gram) ; vérification par le modèle principal |
| Logits | moyen | biais/mélange de logits (type « prior » additif) possible, peut dégrader la qualité |
| Graphe GGML | faible | invasif, fork nécessaire, coût de maintenance élevé |

Recommandation : couche extérieure + draft proposeur. Interface :
```cpp
struct VectorState;                      // opaque, copiable, taille bornée
class EmbedBabel {
 public:
  VectorState init() const;
  VectorState advance(const VectorState&, char32_t) const;   // pure, thread-safe
  VectorState finalize(const VectorState&) const;
  std::vector<llama_token> propose(const VectorState&, int k) const; // draft
};
```
L'interface du sujet est correcte mais insuffisante : il manque `propose()` (c'est là qu'est la valeur) et `advance` doit rester pure (copie O(taille état)) pour permettre backtracking (backspace).

## 6. Ce qui peut accélérer l'inférence

| Catégorie | Élément |
|---|---|
| Gain théorique | h compact comme substitut au contexte ; **limité** par la perte d'information |
| Gain mesurable (prouvé ailleurs, hors EB) | prefill pendant la frappe ; prefix/prompt caching ; speculative decoding avec proposeur n-gram/lookup |
| Gain probable | trie comme proposeur de drafts sur texte répétitif (code, formulaires, domaines clos) ; réduction mémoire de la table d'embeddings par hashing |
| Gain illusoire | « plus petit vecteur ⇒ moins de FLOPs » ; gématrie comme compression sémantique ; remplacement du prefill par un vecteur sans K/V |

## 7. Expérience falsifiable

1. **Données** : corpus multilingue (fr, en, de, es, ar, he, el, zh) avec dictionnaires libres ; split train/val/test par document, test jamais utilisé pour calibrer.
2. **Modèle** : un petit modèle GGUF fixe (≈0,5–1B) + sa version draft.
3. **Cibles** : y1 = embedding/dist. du LLM sur préfixe ; y2 = token suivant.
4. **Variables** : h_n sous les formulations 1–4, avec/sans g, avec/sans relations inter-langues, m ∈ {16, 32, 64, 128, 256}, K ∈ {1, 4, 16}.
5. **Mesures** : semantic/predictive loss, perplexité Δ, acc. next-token Δ, taux d'acceptation de draft, tokens/s, latence de prefill/décodage, RAM pic, VRAM, KV, FLOPs estimés (compteur analytique), énergie si disponible. Au moins 5 graines, IC 95 %.
6. **Hypothèses H0 (à rejeter)** :
   - H0_g : l'ajout de g n'améliore pas y1/y2 par rapport au hash aléatoire (test apparié, p<0,01, correction de Bonferroni).
   - H0_rel : les relations inter-langues ne battent pas des relations permutées.
   - H0_sys : tokens/s end-to-end ≤ baseline (prefix cache + draft n-gram).
7. **Critère d'échec définitif** : si le meilleur système EB est battu par la baseline 8 (petit modèle de pré-encodage à paramètres égaux) à coût égal, EB ne se justifie pas.

## 8. Baselines (contrôle paramètres et FLOPs égaux)

1 caractères seuls ; 2 byte-level ; 3 tokenizer du modèle ; 4 embeddings de sous-chaînes sans g ; 5 trie classique (comptage de préfixes) ; 6 features numériques/hash aléatoires de même dimension ; 7 embeddings appris de bout en bout ; 8 petit encodeur spécialisé ; 9 modèle seul, avec prefix cache et speculative lookup. Rapporter les courbes qualité-vs-coût (front de Pareto), pas des valeurs isolées.

## 9. Risques et objections

- **Gématrie** : dépend du système (ordinal, standard, abjad) ; collisions massives ; coïncidences numériques inévitables (problème des comparaisons multiples : avec assez de fonctions numériques, on trouve toujours des « relations »). Exiger permutation tests.
- **Inter-langues** : peu d'invariants robustes entre scripts ; arabe/hébreu (consonantique, racines), chinois (non alphabétique), allemand (composés), turc/finnois (agglutination). L'arbre de sous-chaînes ne capture pas la morphologie non concaténative.
- **Non-compositionnalité** : le sens d'un préfixe tronqué (« mai ») dépend du contexte ; l'ambiguïté croît avec la longueur de contexte.
- **Mismatch de représentation** : ProjectToLLM doit être appris par modèle ; le modèle principal n'a pas été entraîné avec ces entrées.
- **Coûts cachés** : construction du dictionnaire, cache misses du trie, transferts CPU↔GPU, synchronisations, conversion d'espaces.
- **Biais du dictionnaire** : mots rares/ouverts, noms propres, fautes de frappe (la saisie caractère par caractère comporte des corrections).
- **Réfutation principale** : le LLM calcule déjà l'information utile ; toute information de h qui est utile s'obtient plus fiablement par le prefill.

## 10. Prototype

`prototype/embedbabel_min.cpp` (C++17, compile avec `g++ -std=c++17 -O2 -Wall`) : trie de sous-chaînes, `advance` O(D) indépendant de la longueur du préfixe, `finalize`. Limites assumées : prototypes = hash pseudo-aléatoire (pas appris), pas de relations, pas de repli byte-level (reset du nœud), pas d'intégration llama.cpp, aucun benchmark. Il illustre l'interface et l'incrémentalité, pas un gain.

## 11. Recherche bibliographique (non vérifiée)

Aucune source n'a été consultée ni vérifiée ; par honnêteté, voici uniquement les mots-clés à rechercher et à vérifier avant toute citation :
speculative decoding, prompt lookup decoding, prefix/prompt caching, byte-level / token-free language models, learned compression, semantic hashing, hash embeddings, graph embeddings, hyperbolic / hierarchical embeddings, tree-structured representations, incremental/streaming encoders, multilingual representation alignment, KV-cache compression/quantization, early exit, latent-space reasoning, documentation et sources de `llama.cpp` / GGML (`examples/speculative`, `llama_batch`, `llama_kv_cache`).

## 12. Conclusion

**Plus petite expérience décisive** : sur un seul modèle GGUF et 2 langues (fr/en), construire le trie des préfixes + proposeur de drafts, et mesurer (a) le taux d'acceptation et les tokens/s contre la baseline 9 (prefix cache + lookup n-gram), puis (b) ajouter la gématrie à h et mesurer le gain de predictive_loss contre la baseline 6 (hash aléatoire) avec test de permutation. Si (a) ne gagne rien et (b) n'est pas significatif, EmbedBabel ne contient pas d'idée exploitable. Si seul (a) gagne, l'idée utile est « trie incrémental comme proposeur de drafts », sans gématrie.
