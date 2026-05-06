# Compte-rendu — TP RAG Médicaments

## Sujet choisi
**Sujet B — Assistant Médicaments**  
Système RAG permettant de répondre à des questions sur les médicaments courants à partir de données pharmaceutiques officielles.

---

## Difficultés rencontrées

### 1. Accès à l'API médicaments.gouv.fr
L'API publique `api.medicaments.gouv.fr` est parfois instable ou renvoie des réponses JSON mal structurées selon les médicaments. J'ai implémenté un **système de fallback** : si l'API échoue (timeout, réponse vide, format inattendu), le code utilise une base de données enrichie définie directement dans `indexation.py`. Cette approche garantit que l'indexation fonctionne toujours, quel que soit l'état de l'API externe, tout en intégrant les vraies données quand elles sont disponibles.

### 2. Chunking adapté aux notices médicales
Les notices sont des documents courts mais très denses. Un chunking naïf (par nombre fixe de caractères) cassait les sections importantes au milieu d'une phrase. J'ai choisi une **découpe prioritaire par sauts de ligne** avec un overlap de 80 caractères. Cela préserve la cohérence sémantique de chaque section (indications, posologie, effets indésirables…), ce qui améliore directement la précision de la recherche vectorielle.

### 3. Similarité cosinus avec FAISS
FAISS propose `IndexFlatL2` (distance euclidienne) ou `IndexFlatIP` (produit scalaire). Le produit scalaire est équivalent à la similarité cosinus uniquement si les vecteurs sont préalablement **normalisés**. J'ai donc ajouté un appel à `faiss.normalize_L2()` avant l'indexation et avant chaque recherche. Sans cette étape, les scores retournés ne seraient pas comparables entre requêtes.

### 4. Score de confiance et seuil
Choisir un seuil de confiance pertinent a nécessité plusieurs tests manuels. Un seuil trop bas (ex : 0.1) laisse passer des résultats non pertinents ; trop haut (ex : 0.6), le système refuse des questions légitimes formulées différemment. **0.35** s'est avéré être un bon compromis sur les données testées.

---

## Décisions de conception

| Décision | Justification |
|---|---|
| Modèle `paraphrase-multilingual-mpnet-base-v2` | Multilingue (français + anglais médical), 768 dims, gratuit, pas de clé API |
| `IndexFlatIP` + normalisation L2 | Équivalent exact à la similarité cosinus, cohérent avec les embeddings de sentence-transformers |
| Taille de chunk 600 / overlap 80 | Adapté aux sections de notice (paragraphes courts et denses) |
| Température LLM = 0.3 | Réponses factuelles et stables, évite les hallucinations sur les données médicales |
| Historique limité à 3 échanges | Évite de dépasser la fenêtre de contexte du modèle 8k |
| Mention de sécurité systématique | Contrainte éthique du TP, inscrite dans le prompt système (non contournable par l'utilisateur) |

---

## Ce que j'ai appris

Ce TP m'a permis de comprendre que la qualité d'un RAG dépend à **80 % de la qualité du chunking et des embeddings**, et non du LLM. Un LLM performant avec un mauvais chunking donnera de mauvaises réponses ; un chunking soigné avec un modèle modeste donne déjà d'excellents résultats. La persistance FAISS est également essentielle en pratique : l'encodage du corpus prend plusieurs minutes et ne doit pas être refait à chaque lancement.
