"""
rag_engine.py — Cerveau central du RAG Médicaments
Module unique importé par rag.py (CLI) ET dashboard.py (Streamlit).
Charge index.faiss + chunks_meta.json une seule fois.
"""

import os
import json
import re
import time
import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────

INDEX_PATH      = "data/index.faiss"
META_PATH       = "data/chunks_meta.json"
MODEL_NAME      = "paraphrase-multilingual-mpnet-base-v2"
GROQ_MODEL      = "llama-3.3-70b-versatile"
SEUIL_CONFIANCE = 0.35
TOP_K           = 4


# ───────────────────────────────────────────────────────────────────────────────
#  CLASSE PRINCIPALE
# ───────────────────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Cerveau du RAG Médicaments.
    Charge la base vectorielle FAISS une seule fois et expose
    toutes les fonctions nécessaires à rag.py et dashboard.py.
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY manquante dans le fichier .env")

        if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
            raise FileNotFoundError(
                "Base vectorielle introuvable.\n"
                "Lancez d'abord : python indexation.py"
            )

        print("⏳ Chargement de l'index FAISS...")
        self.index = faiss.read_index(INDEX_PATH)

        print("⏳ Chargement des métadonnées...")
        with open(META_PATH, "r", encoding="utf-8") as f:
            self.chunks_meta = json.load(f)

        print(f"⏳ Chargement du modèle d'embedding : {MODEL_NAME}")
        self.modele = SentenceTransformer(MODEL_NAME)

        self.client = Groq(api_key=api_key)

        print(f"✅ RAGEngine prêt — {self.index.ntotal:,} vecteurs | "
              f"{len(self.chunks_meta):,} chunks")

    # ─── Recherche vectorielle ────────────────────────────────────────────────

    def rechercher(self, question: str, k: int = TOP_K) -> list[dict]:
        """
        Encode la question et retourne les k chunks les plus similaires.
        Score = similarité cosinus (entre -1 et 1, 1 = identique).
        """
        vecteur = self.modele.encode(
            [question], convert_to_numpy=True
        ).astype(np.float32)
        faiss.normalize_L2(vecteur)

        scores, indices = self.index.search(vecteur, k)

        return [
            {
                "contenu":  self.chunks_meta[idx]["contenu"],
                "metadata": self.chunks_meta[idx]["metadata"],
                "score":    float(score),
            }
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0 and float(score) >= SEUIL_CONFIANCE
        ]

    # ─── Reformulation (Bonus C) ──────────────────────────────────────────────

    def reformuler(self, question: str) -> str:
        """
        Reformule la question en mots-clés pharmaceutiques
        pour améliorer la recherche vectorielle.
        """
        try:
            resp = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content":
                    f"Reformule en 5-8 mots-clés pharmaceutiques pour recherche documentaire.\n"
                    f"Réponds UNIQUEMENT avec les mots-clés.\n"
                    f"Question : {question}"}],
                max_tokens=60,
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return question

    # ─── Prompt système ───────────────────────────────────────────────────────

    def _prompt_systeme(self, profil: dict) -> str:
        prenom = profil.get("prenom", "le patient")
        return f"""Tu es MediAssist Pro, un assistant pharmaceutique expert et bienveillant.
Tu t'adresses à {prenom} de façon chaleureuse et claire — pas robotique.

PROFIL PATIENT :
- Prénom    : {profil.get('prenom', '?')}
- Sexe      : {profil.get('sexe', '?')}
- Âge       : {profil.get('age', '?')} ans
- Poids     : {profil.get('poids', '?')} kg
- Grossesse : {profil.get('grossesse', '?')}
- Allergies : {profil.get('allergies', 'aucune')}
- Médicaments en cours : {profil.get('medicaments', 'aucun')}
- Antécédents : {profil.get('antecedents', 'aucun')}

RÈGLES ABSOLUES :
1. Réponds UNIQUEMENT à partir du contexte fourni. Jamais d'invention.
2. Ne cite QUE les médicaments pertinents pour la question posée.
3. Tiens compte du profil patient :
   - Allergie détectée → signale-la immédiatement en ROUGE
   - Médicament en cours interagissant → mentionne-le
   - Grossesse → adapte systématiquement les mises en garde
4. Si l'information est absente du contexte → dis-le clairement.
5. Commence par t'adresser à {prenom} par son prénom.
6. Structure ta réponse : Médicament / Posologie / Précautions.
7. TOUJOURS terminer par :
⚠️ Ces informations ne remplacent pas l'avis d'un professionnel de santé.
En cas de doute, consulte ton médecin ou ton pharmacien."""

    # ─── Génération de réponse ────────────────────────────────────────────────

    def generer_reponse(
        self,
        question: str,
        chunks: list[dict],
        profil: dict,
        historique: list[dict] | None = None,
    ) -> str:
        """
        Génère une réponse via Groq en injectant le contexte RAG.
        Respecte la limite de tokens (max 5000 tokens de contexte).
        """
        if historique is None:
            historique = []

        # Construire le contexte en respectant la limite de tokens
        contexte_parts = []
        tokens_total   = 0
        MAX_TOKENS_CTX = 5000

        for i, chunk in enumerate(chunks):
            tokens_chunk = len(chunk["contenu"]) // 4
            if tokens_total + tokens_chunk > MAX_TOKENS_CTX:
                break
            denom = chunk["metadata"].get("denomination", "?")
            contexte_parts.append(
                f"[Source {i+1} — {denom} | score {chunk['score']:.2f}]\n{chunk['contenu']}"
            )
            tokens_total += tokens_chunk

        contexte = "\n\n---\n\n".join(contexte_parts)

        messages = [{"role": "system", "content": self._prompt_systeme(profil)}]
        messages += historique[-6:]  # 3 derniers échanges
        messages.append({
            "role": "user",
            "content": f"Contexte (notices médicales BDPM/ANSM) :\n\n{contexte}\n\nQuestion : {question}"
        })

        resp = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=1000,
            temperature=0.3,
        )
        return resp.choices[0].message.content

    # ─── Score de sécurité personnalisé ──────────────────────────────────────

    def evaluer_securite(self, question: str, chunks: list[dict], profil: dict) -> dict:
        """
        Analyse le profil patient vs les informations médicamenteuses.
        Retourne VERT / ORANGE / ROUGE avec les raisons.
        """
        ctx = "\n".join(c["contenu"][:400] for c in chunks[:3])
        profil_txt = (
            f"Allergies : {profil.get('allergies','aucune')}\n"
            f"Grossesse : {profil.get('grossesse','non')}\n"
            f"Médicaments en cours : {profil.get('medicaments','aucun')}\n"
            f"Antécédents : {profil.get('antecedents','aucun')}\n"
            f"Âge : {profil.get('age','?')} ans | Sexe : {profil.get('sexe','?')}"
        )

        prompt = f"""Tu es un expert en pharmacovigilance.
Analyse ce profil patient et ces informations médicamenteuses.
Réponds UNIQUEMENT en JSON valide, sans texte avant ou après.

PROFIL :
{profil_txt}

INFORMATIONS MÉDICAMENTEUSES :
{ctx}

JSON attendu :
{{
  "niveau": "VERT" ou "ORANGE" ou "ROUGE",
  "raisons": ["raison 1", "raison 2"],
  "contre_indications_detectees": [],
  "interactions_detectees": [],
  "recommandation": "phrase courte"
}}

VERT = aucune CI | ORANGE = précautions | ROUGE = CI formelle ou allergie"""

        try:
            resp = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1,
            )
            txt = resp.choices[0].message.content.strip()
            txt = txt.replace("```json", "").replace("```", "").strip()
            return json.loads(txt)
        except Exception:
            return {
                "niveau": "ORANGE",
                "raisons": ["Analyse automatique indisponible"],
                "contre_indications_detectees": [],
                "interactions_detectees": [],
                "recommandation": "Consultez un professionnel de santé.",
            }

    # ─── Détection d'hallucinations ───────────────────────────────────────────

    def detecter_hallucinations(self, reponse: str, chunks: list[dict]) -> dict:
        """
        Vérifie que les dosages mentionnés dans la réponse
        existent bien dans les chunks sources.
        Un dosage absent des sources = hallucination potentielle.
        """
        dosages = re.findall(r"\d+\s*(?:mg|g|ml|µg|UI|cp|comprimé)", reponse, re.I)
        ctx     = " ".join(c["contenu"] for c in chunks)

        valides  = [d for d in dosages if d.split()[0] in ctx]
        suspects = [d for d in dosages if d.split()[0] not in ctx]
        total    = len(dosages) or 1

        return {
            "score":    round(len(suspects) / total * 100),
            "valides":  valides,
            "suspects": suspects,
            "verdict":  "✅ Fiable" if not suspects else f"⚠️ {len(suspects)} dosage(s) à vérifier",
        }

    # ─── Pipeline complet (une seule question) ────────────────────────────────

    def pipeline(
        self,
        question: str,
        profil: dict,
        historique: list[dict] | None = None,
    ) -> dict:
        """
        Pipeline RAG complet pour une question :
        Reformulation → Recherche → Sécurité → Génération → Hallucinations
        Retourne un dict avec tous les résultats + la trace complète.
        """
        t0 = time.time()

        # 1. Reformulation
        q_ref = self.reformuler(question)

        # 2. Recherche vectorielle
        chunks = self.rechercher(q_ref)

        # 3. Score de confiance
        if not chunks:
            return {
                "succes":  False,
                "message": "Aucune information pertinente trouvée dans la base.",
                "trace":   {"query_reformulee": q_ref, "chunks_trouves": 0},
            }

        # 4. Score de sécurité
        score_secu = self.evaluer_securite(question, chunks, profil)

        # 5. Génération
        reponse = self.generer_reponse(question, chunks, profil, historique)

        # 6. Détection hallucinations
        hallu = self.detecter_hallucinations(reponse, chunks)

        dt = time.time() - t0

        # 7. Trace complète (explainability)
        trace = {
            "query_originale":   question,
            "query_reformulee":  q_ref,
            "chunks_trouves":    len(chunks),
            "best_score":        chunks[0]["score"],
            "tokens_contexte":   sum(len(c["contenu"]) // 4 for c in chunks),
            "latence":           f"{dt:.1f}s",
            "hallucinations":    hallu,
            "sources": [
                {
                    "denomination": c["metadata"].get("denomination", "?"),
                    "score":        c["score"],
                    "cis":          c["metadata"].get("cis", "?"),
                    "url":          c["metadata"].get("url_prospectus", ""),
                }
                for c in chunks
            ],
        }

        return {
            "succes":       True,
            "question":     question,
            "reponse":      reponse,
            "chunks":       chunks,
            "score_secu":   score_secu,
            "trace":        trace,
        }

    # ─── Mode comparaison (Bonus D) ───────────────────────────────────────────

    def comparer(self, med1: str, med2: str, profil: dict) -> dict:
        """Compare deux médicaments en récupérant leurs chunks respectifs."""
        chunks1 = self.rechercher(med1, k=3)
        chunks2 = self.rechercher(med2, k=3)
        tous    = chunks1 + chunks2

        question = (
            f"Compare '{med1}' et '{med2}' sur : indications, effets indésirables, "
            f"contre-indications, interactions. Tiens compte du profil patient. "
            f"Format structuré par thème."
        )
        reponse = self.generer_reponse(question, tous, profil)

        return {
            "reponse":  reponse,
            "chunks1":  chunks1,
            "chunks2":  chunks2,
        }

    # ─── Mode symptômes ───────────────────────────────────────────────────────

    def mode_symptomes(self, symptomes: str, profil: dict) -> dict:
        """
        L'utilisateur décrit ses symptômes → on cherche les médicaments
        pertinents et on les évalue selon son profil.
        """
        # Reformuler les symptômes en termes pharmaceutiques
        resp = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content":
                f"Symptômes : {symptomes}\n"
                f"Donne 6-8 mots-clés pharmaceutiques pour une recherche documentaire.\n"
                f"Réponds UNIQUEMENT avec les mots-clés."}],
            max_tokens=60,
            temperature=0.2,
        )
        mots_cles = resp.choices[0].message.content.strip()

        chunks = self.rechercher(mots_cles)
        if not chunks:
            return {"succes": False, "message": "Aucun médicament pertinent trouvé."}

        score_secu = self.evaluer_securite(symptomes, chunks, profil)

        question_orientee = (
            f"Symptômes décrits : {symptomes}\n\n"
            f"Quels médicaments de la base pourraient correspondre ? "
            f"Explique pourquoi et cite les précautions spécifiques au profil. "
            f"Ne prescris rien, oriente uniquement."
        )
        reponse = self.generer_reponse(question_orientee, chunks, profil)

        return {
            "succes":      True,
            "mots_cles":   mots_cles,
            "reponse":     reponse,
            "chunks":      chunks,
            "score_secu":  score_secu,
        }

    # ─── Mode éducation ───────────────────────────────────────────────────────

    def expliquer_simplement(self, question: str, profil: dict) -> str:
        """
        Explique une réponse médicale en langage simple
        avec des analogies du quotidien.
        """
        chunks = self.rechercher(question)
        ctx    = "\n".join(c["contenu"][:300] for c in chunks[:3])

        prompt = f"""Explique cette question de façon très simple et pédagogique,
comme si tu parlais à quelqu'un sans formation médicale.
Utilise des analogies du quotidien. Structure avec 3-4 points numérotés avec des emojis.
Termine par "💡 Ce qu'il faut retenir : ..." en une phrase.

Contexte médical : {ctx}
Question : {question}
Profil : {profil.get('prenom')}, {profil.get('age')} ans"""

        resp = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.5,
        )
        return resp.choices[0].message.content
