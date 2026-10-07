// Prototype minimal EmbedBabel : trie de sous-chaînes + état incrémental O(1) par caractère.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

constexpr int D = 64;  // dimension de l'état

struct VectorState {
    std::array<float, D> v{};   // somme pondérée des prototypes actifs
    uint32_t node = 0;          // nœud courant du trie (0 = racine)
    uint32_t n = 0;             // nb de caractères consommés
};

struct Node {
    std::unordered_map<char32_t, uint32_t> next;
    std::array<float, D> proto{};  // prototype appris/calibré (ici : hash aléatoire déterministe)
    uint32_t gematria = 0;         // feature séparée, jamais présumée sémantique
    float freq = 0;
};

class EmbedBabel {
public:
    EmbedBabel() { nodes_.emplace_back(); }
    void insert(const std::u32string& w, float freq) {
        uint32_t cur = 0; uint32_t g = 0;
        for (char32_t c : w) {
            g += (uint32_t)c;
            auto it = nodes_[cur].next.find(c);
            if (it == nodes_[cur].next.end()) {
                uint32_t id = nodes_.size();
                nodes_[cur].next[c] = id;
                nodes_.emplace_back();
                nodes_[id].proto = hashvec(id);
                it = nodes_[cur].next.find(c);
            }
            cur = it->second;
            nodes_[cur].gematria = g;
            nodes_[cur].freq += freq;
        }
    }
    // S' = Compress(Merge(S, Delta(c, branche active))) ; coût O(D) indépendant de la longueur du préfixe
    VectorState advance(const VectorState& s, char32_t c) const {
        VectorState o = s; o.n++;
        auto it = nodes_[s.node].next.find(c);
        if (it == nodes_[s.node].next.end()) { o.node = 0; return o; } // branche inconnue : reset (à remplacer par repli byte-level)
        const Node& nd = nodes_[it->second];
        o.node = it->second;
        for (int i = 0; i < D; i++) o.v[i] = 0.9f * o.v[i] + nd.proto[i];  // Merge avec oubli
        return o;
    }
    VectorState finalize(const VectorState& s) const {
        VectorState o = s; float nr = 0;
        for (float x : o.v) nr += x * x;
        nr = std::sqrt(nr) + 1e-9f;
        for (float& x : o.v) x /= nr;
        return o;
    }
private:
    static std::array<float, D> hashvec(uint32_t seed) {
        std::array<float, D> r; uint64_t x = seed * 0x9E3779B97F4A7C15ull + 1;
        for (auto& f : r) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; f = (float)((x >> 40) & 0xFFFF) / 32768.f - 1.f; }
        return r;
    }
    std::vector<Node> nodes_;
};

int main() {
    EmbedBabel eb; eb.insert(U"maison", 1); eb.insert(U"mais", 2);
    VectorState s;
    for (char32_t c : std::u32string(U"mais")) s = eb.advance(s, c);
    s = eb.finalize(s);
    std::printf("n=%u node=%u v0=%.4f\n", s.n, s.node, s.v[0]);
}
