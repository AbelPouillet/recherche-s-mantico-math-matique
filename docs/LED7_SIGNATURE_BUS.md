# LED7 Signature Bus — Médaillon V3

## Objectif

Le médaillon possède 7 LEDs utilisées comme **canal de signature** des opérations internes.
Le but n'est pas de coder directement tout le langage lettre par lettre, mais de renvoyer un maximum d'information opérationnelle avec une interface minimale et lisible.

## Principe

- 6 LEDs périphériques = vecteur de signature (état encodé choisi par l'IA)
- 1 LED centrale = ponctuation / interface / mode

Chaque IA peut définir son mapping, mais doit respecter les contraintes de benchmark ci-dessous.

## Contraintes benchmark

1. **Décodabilité**
   - Le mapping LED -> état doit être documenté et décodable automatiquement.
2. **Stabilité**
   - Pas de clignotements chaotiques non informatifs.
3. **Densité d'information**
   - Maximiser l'information mutuelle entre état réel du pipeline et état LED.
4. **Comparabilité**
   - Fournir une baseline commune pour comparer plusieurs IA.
5. **Falsifiabilité**
   - Démontrer les gains par tests et contrôles négatifs.

## États internes recommandés à encoder

- mode pipeline (écoute / détection / traduction / restitution),
- niveau de confiance,
- temporalité sémantique (passé / présent / futur),
- ambiguïté phonétique,
- qualité du signal,
- divergence entre hypothèses concurrentes,
- état ponctuation/interaction (LED centrale).

## Deux schémas autorisés

### Schéma A — Bits de signature (interprétable)
- LEDs périphériques = 6 drapeaux binaires
- LED centre = ponctuation + bouton interface

### Schéma B — Code compact (haute densité)
- LEDs périphériques = code indexé (état parmi N classes)
- LED centre = séparateur d’événements / ACK utilisateur

## Métriques

- Information mutuelle I(LED; état interne)
- Taux d’erreur de décodage
- Latence de mise à jour
- Taux de flicker non utile
- Robustesse au bruit
- Gain utilisateur (compréhension humaine en test)

## Règle centrale

Une animation LED est valide seulement si elle permet de reconstruire utilement l’état opérationnel du médaillon mieux qu’une baseline triviale.
