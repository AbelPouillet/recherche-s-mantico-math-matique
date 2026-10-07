# Chiffrage des tokens

Comment savoir ce que DSH consomme réellement, sur cette machine, sans rien casser.

## En une commande, en lecture seule

```bash
pip install -e ".[tokens]"          # ajoute zstandard (voir « pourquoi » plus bas)
python -m bench.selfimprove.run tokens
python -m bench.selfimprove.run tokens --json            # session par session
python -m bench.selfimprove.run tokens --log <journal>   # détail d'une session, tour par tour
```

Mesure réelle sur cette machine (9 sessions, 299 pas) :

| Compartiment | Tokens |
|---|---:|
| entrée | 832 406 |
| sortie | 388 531 |
| **lecture de cache** | **64 520 100** |
| écriture de cache | 0 |
| **total** | **65 741 037** |

**98,14 % du total est de la lecture de cache.** C'est la raison pour laquelle la ventilation par
compartiment n'est pas un détail : un total unique, dominé par le cache, ne dit rien de ce qui a été
réellement produit. Par modèle : `deepseek-flash` 64,77 M, `qwen3.8:27b` 0,97 M.

## Le format des journaux DSH — vérifié, pas supposé

Établi en décodant une session réelle de 1,67 Mo :

```
<DSH_HOME>/sessions/<slug-du-cwd>/<session-id>/session.v4.jsonl.zstd
```

- Le fichier est un conteneur de **frames zstd concaténées** (909 frames observées → 6 646 603 octets
  décompressés, 1 477 lignes). Les API zstd usuelles s'arrêtent à la première frame : il faut lire
  **au-delà** (`read_across_frames=True`). C'est ce que la documentation de `@zoytown/dsh-token`
  explique pour justifier que les API zstd de Node ne savent pas lire ces journaux.
- Contenu : JSONL `{"type", "seq", "time", "data"}`, plus une ligne d'en-tête (`id`, `cwd`,
  `createdAt`, `version`).
- Les compteurs sont sur les événements **`assistant/message`**, dans `data.usage` :
  `inputTokens`, `outputTokens`, `cacheReadTokens`, `cacheWriteTokens`, `totalTokens`.
- **Déduplication par `(turn, step)`** : un même pas émet deux rapports (pendant le streaming, puis
  sur le message final). On **remplace**, on n'accumule pas — accumuler doublerait chaque pas.

Ce module (`bench/selfimprove/tokens.py`) est un **lecteur en lecture seule** : il n'écrit dans aucun
journal, ne contacte aucun réseau, et **ne monte rien** dans DSH.

## Le plugin `@zoytown/dsh-token` — et pourquoi il n'est pas installé ici

Le plugin tiers [zoyluoblue/deepseek-harness-token](https://github.com/zoyluoblue/deepseek-harness-token)
(MIT, TypeScript, publié sous le nom `@zoytown/dsh-token`) apporte la **partie interface** : une page
**Settings → Token** avec totaux, carte de chaleur façon GitHub, séries, heure de pointe et ventilation
par modèle. Il lit exactement les mêmes journaux, ne monte aucun outil destiné au modèle et n'écrit
aucun événement de session — le monter ne coûte donc rien à la conversation.

```bash
dsh plugin --profile web add @zoytown/dsh-token      # nécessite pnpm sur le PATH
dsh plugin --profile web remove @zoytown/dsh-token   # pour revenir en arrière
```

> ### ⚠️ Pourquoi cette commande n'a **pas** été exécutée ici
>
> Le README du plugin prévient lui-même : *« An Electron shell wrapping the harness may pin bare
> module resolution to its own bundle, which makes a profile-installed plugin unresolvable — and the
> plugin tree then fails to load outright rather than degrading, **so the app will not start**.
> Do not install this plugin into such a shell's `DSH_HOME`. »*
>
> Le DSH utilisé ici est une application Electron dont le harnais est empaqueté dans
> `resources/app.asar` — c'est exactement le cas visé. Et l'installation se ferait dans le profil
> **actif** `~/.dsh/profiles/web`, celui qui sert la fenêtre en cours. Le mode d'échec annoncé n'est
> pas « le plugin ne fonctionne pas » mais « **l'application ne démarre plus** ».
>
> Décision : le chiffrage est obtenu par le lecteur Python en lecture seule ci-dessus, qui ne touche
> ni au profil ni au démarrage. L'installation du plugin reste possible, mais c'est une décision
> explicite — à prendre en connaissance de cause, et de préférence après avoir vérifié qu'un `dsh`
> autonome (et non l'application Electron) sert bien ce profil.

Pour vérifier avant de s'engager :

```bash
dsh plugin --profile web list     # quel CLI sert ce profil ?
# sauvegarde du profil avant toute modification
cp ~/.dsh/profiles/web/package.json ~/.dsh/profiles/web/package.json.bak
```

En cas d'échec au démarrage, la marche arrière est la suppression de l'entrée du bundle dans
`~/.dsh/profiles/web/package.json` et `cordis.patch.yml`, ou `dsh plugin --profile web remove`.

## Ce qui n'est pas chiffré, et ne peut pas l'être

- **Aucun coût monétaire.** Le harnais ne stocke pas de table de prix (`@deepseek-ai/dsh-llm-pi-ai`
  code `NO_COST` en dur). Tout chiffre en euros ou en dollars serait une supposition locale présentée
  comme un fait — c'est la position du plugin, et elle est reprise ici.
- **Les pas réessayés sont sous-comptés.** Quand une requête est relancée à l'intérieur d'un pas, le
  journal ne conserve que le dernier rapport d'usage ; la tentative échouée a été facturée mais ses
  chiffres ont disparu.
- **Les sessions en cours d'écriture** peuvent être lues partiellement (le lecteur s'arrête à ce qui
  est décodable).
