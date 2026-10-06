// Schéma Neo4j 5.x (édition Community suffisante : contraintes d'unicité composites).
// Exécuter avec : cypher-shell -f graph/schema.cypher   (ou via build_graph.py --apply-schema)
// Chaque instruction se termine par « ; » en fin de ligne.

// --- Nœuds -----------------------------------------------------------------
// (:Letter {lang, char, alphabet_index, gematria_systems[], gematria_values[]})   feuilles
// (:Substring {lang, text, length, gematria, ipa?})                               branches (longueur >= 2)
// (:Word {lang, text, ipa[], variety, period})                                    mots du dictionnaire
//   variety = '' pour les nœuds purement étymologiques (ex. ancêtre latin)
// (:Syllable {lang, ipa, text})                                                   uniquement si la source marque les frontières
// (:Pronunciation {lang, ipa, variety})                                           extension : une prononciation partagée par les homophones
CREATE CONSTRAINT letter_key IF NOT EXISTS FOR (n:Letter) REQUIRE (n.lang, n.char) IS UNIQUE;
CREATE CONSTRAINT substring_key IF NOT EXISTS FOR (n:Substring) REQUIRE (n.lang, n.text) IS UNIQUE;
CREATE CONSTRAINT word_key IF NOT EXISTS FOR (n:Word) REQUIRE (n.lang, n.text, n.variety) IS UNIQUE;
CREATE CONSTRAINT syllable_key IF NOT EXISTS FOR (n:Syllable) REQUIRE (n.lang, n.ipa) IS UNIQUE;
CREATE CONSTRAINT pronunciation_key IF NOT EXISTS FOR (n:Pronunciation) REQUIRE (n.lang, n.ipa, n.variety) IS UNIQUE;

// --- Index -------------------------------------------------------------------
CREATE INDEX substring_length IF NOT EXISTS FOR (n:Substring) ON (n.length);
CREATE INDEX substring_gematria IF NOT EXISTS FOR (n:Substring) ON (n.gematria);
CREATE INDEX word_text IF NOT EXISTS FOR (n:Word) ON (n.text);
CREATE INDEX word_variety IF NOT EXISTS FOR (n:Word) ON (n.variety);
CREATE INDEX letter_index IF NOT EXISTS FOR (n:Letter) ON (n.alphabet_index);

// --- Relations (documentation ; Neo4j ne déclare pas les types à l'avance) ----
// (:Letter)-[:NEXT_LETTER]->(:Letter)                      ordre alphabétique (la dernière lettre n'a pas de suivante)
// (:Substring)-[:COMPOSED_OF {pos: 0|1}]->(:Substring|:Letter)   pos 0 = s[:-1], pos 1 = s[1:] (DAG partagé)
// (:Word)-[:COMPOSED_OF {pos}]->(:Letter)                  lettres du mot dans l'ordre
// (:Letter|:Substring)-[:PREFIX_OF]->(:Word)               préfixes propres du mot
// (:Letter|:Substring)-[:SUBSTRING_OF {start}]->(:Word)    sous-chaînes contiguës propres (préfixes inclus)
// (:Word)-[:PRONOUNCED_AS {source}]->(:Pronunciation)
// (:Pronunciation)-[:HAS_SYLLABLE {pos}]->(:Syllable)      seulement si frontières explicites dans la source
// (:Word)-[:COGNATE|:BORROWING|:SOUND_CHANGE|:CONTRACTION|:CONTACT_BLEND {source, confidence, rule, template, retrieved, license}]->(:Word)
//    BORROWING / SOUND_CHANGE / CONTRACTION / CONTACT_BLEND : de l'ancêtre/de la source vers la forme dérivée ;
//    COGNATE : symétrique, stocké une seule fois (ordre lexicographique)
// (:Word {variety:''})-[:SAME_FORM]->(:Word {variety:<v>})  extension : relie un nœud étymologique à un mot du dictionnaire
