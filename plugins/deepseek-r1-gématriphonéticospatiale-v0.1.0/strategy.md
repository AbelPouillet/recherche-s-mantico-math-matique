# Stratégie du plugin DeepSeek

1. Le message `go` (seul) déclenche le bench.
2. Garde anti-pollution : si l'historique contient le moindre message user/assistant/tool antérieur, le bench n'est pas lancé et le plugin demande une nouvelle conversation.
3. Le prompt envoyé est `bench/prompts/EMBEDBABEL_BENCH_V2.md` (sortie JSON stricte).

## Limites connues
- L'API de hooks du fork DeepSeek harness n'a pas été vérifiée : adapter `handle()`.
- Le seuil de pollution est strict (0 message) ; assouplir via `manifest.json`.
- Ne détecte pas la pollution via mémoire/outils externes du harness.
