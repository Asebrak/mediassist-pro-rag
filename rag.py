"""
rag.py — Interface CLI du RAG Médicaments
Importe RAGEngine depuis rag_engine.py (module central).
"""

from rag_engine import RAGEngine
from datetime import datetime

# ─── Profil patient ────────────────────────────────────────────────────────────

def construire_profil() -> dict:
    print("\n" + "═" * 60)
    print("  👤 Avant de commencer, j'ai besoin de mieux te connaître.")
    print("  (Appuie sur Entrée pour passer une question)")
    print("═" * 60 + "\n")

    questions = [
        ("prenom",      "😊 Comment tu t'appelles ? (prénom) : "),
        ("age",         "🎂 Quel est ton âge ? : "),
        ("sexe",        "🧬 Quel est ton sexe ? (homme/femme) : "),
        ("poids",       "⚖️  Ton poids approximatif (en kg) ? : "),
        ("grossesse",   "🤰 Es-tu enceinte ou susceptible de l'être ? (oui/non) : "),
        ("allergies",   "⚠️  Des allergies connues à des médicaments ? (ou 'aucune') : "),
        ("medicaments", "💊 Tu prends déjà des médicaments régulièrement ? (ou 'aucun') : "),
        ("antecedents", "🏥 Des antécédents médicaux importants ? (ou 'aucun') : "),
        ("symptomes_now","🤒 Qu'est-ce qui t'amène aujourd'hui ? : "),
    ]

    profil = {}
    for cle, question in questions:
        rep = input(question).strip()
        profil[cle] = rep if rep else "Non renseigné"

    print(f"\n✅ Merci {profil.get('prenom','')} ! J'ai noté ton profil.\n")
    return profil

# ─── Aide ──────────────────────────────────────────────────────────────────────

AIDE = """
╔══════════════════════════════════════════════════════╗
║  COMMANDES                                           ║
╠══════════════════════════════════════════════════════╣
║  symptomes        → Décris ce que tu ressens        ║
║  comparer X vs Y  → Compare deux médicaments        ║
║  profil           → Affiche ton profil patient      ║
║  aide             → Ce menu                         ║
║  quit             → Quitter                         ║
╚══════════════════════════════════════════════════════╝
"""

# ─── Affichage ─────────────────────────────────────────────────────────────────

def afficher_resultat(resultat: dict, prenom: str):
    if not resultat.get("succes"):
        print(f"\n  {resultat.get('message','Erreur inconnue')}\n")
        return

    score = resultat.get("score_secu", {})
    niveau = score.get("niveau", "?")
    icones = {"VERT": "🟢", "ORANGE": "🟡", "ROUGE": "🔴"}
    print(f"\n{icones.get(niveau,'⚪')} SÉCURITÉ : {niveau}")
    for r in score.get("raisons", []):
        print(f"   → {r}")

    ci = [x for x in score.get("contre_indications_detectees", []) if x]
    if ci:
        print(f"   ⛔ Contre-indication : {', '.join(ci)}")

    print(f"\n{'─'*60}")
    print(resultat["reponse"])
    print(f"{'─'*60}")

    trace = resultat.get("trace", {})
    sources = trace.get("sources", [])
    if sources:
        noms = list({s["denomination"][:40] for s in sources})
        print(f"\n📚 Sources : {', '.join(noms)}")

    hallu = trace.get("hallucinations", {})
    print(f"📊 Score : {trace.get('best_score', 0):.2f} | "
          f"Latence : {trace.get('latence', '?')} | "
          f"Hallucinations : {hallu.get('verdict', '?')}")

    # URLs prospectus
    urls = {s["denomination"][:40]: s["url"] for s in sources if s.get("url")}
    if urls:
        print("\n🔗 Prospectus officiels (BDPM/ANSM) :")
        for denom, url in urls.items():
            print(f"   • {denom} → {url}")
    print()

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  💊 MediAssist Pro — Assistant Médicaments")
    print("  FAISS + Groq LLaMA 3 | Données BDPM / ANSM")
    print("═" * 60)

    # Charger le moteur RAG
    try:
        engine = RAGEngine()
    except (EnvironmentError, FileNotFoundError) as e:
        print(f"\n❌ {e}")
        return

    # Profil patient
    profil = construire_profil()
    prenom = profil.get("prenom", "")

    # Symptômes initiaux
    symptomes_now = profil.get("symptomes_now", "").strip()
    if symptomes_now and symptomes_now != "Non renseigné":
        print(f"💬 Tu m'as parlé de : \"{symptomes_now}\"")
        print("   Analyse en cours...\n")
        res = engine.mode_symptomes(symptomes_now, profil)
        afficher_resultat(res, prenom)

    print(AIDE)

    historique = []

    while True:
        try:
            question = input(f"❓ {prenom}, ta question : ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\nPrends soin de toi, {prenom} ! 👋")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print(f"\nPrends soin de toi, {prenom} ! 👋")
            break
        if question.lower() == "aide":
            print(AIDE)
            continue
        if question.lower() == "profil":
            from rag_engine import RAGEngine
            print(f"\nProfil de {prenom} :")
            for k, v in profil.items():
                print(f"  {k}: {v}")
            print()
            continue

        # Mode symptômes
        if question.lower() == "symptomes":
            symptomes = input("   Décris ce que tu ressens : ").strip()
            if symptomes:
                res = engine.mode_symptomes(symptomes, profil)
                afficher_resultat(res, prenom)
            continue

        # Mode comparaison
        if question.lower().startswith("comparer ") and " vs " in question.lower():
            parties = question[len("comparer "):].split(" vs ")
            if len(parties) == 2:
                print(f"\n🔬 Comparaison : {parties[0]} vs {parties[1]}...\n")
                res = engine.comparer(parties[0].strip(), parties[1].strip(), profil)
                print(f"{'─'*60}")
                print(res["reponse"])
                print(f"{'─'*60}\n")
            continue

        # Question normale — pipeline complet
        print()
        print("  🔄 Reformulation...")
        print("  🔍 Recherche dans la base FAISS...")
        print("  🛡️  Évaluation de la sécurité...")
        print("  🤖 Génération de la réponse...\n")

        res = engine.pipeline(question, profil, historique)

        if res.get("succes"):
            historique.append({"role": "user",      "content": question})
            historique.append({"role": "assistant",  "content": res.get("reponse", "")})

        afficher_resultat(res, prenom)


if __name__ == "__main__":
    main()
