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

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH      = os.path.join(BASE_DIR, "data", "index.faiss")
META_PATH       = os.path.join(BASE_DIR, "data", "chunks_meta.json")
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
