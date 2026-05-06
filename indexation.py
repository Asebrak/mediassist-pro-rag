"""
indexation.py - Pipeline d'indexation RAG Médicaments
Format réel de CIS_RCP.csv :
  En-tête : Code_CIS | RCP_html
  Le champ RCP_html est du HTML multiligne entre guillemets
  => on utilise csv.reader (gère les champs multiligne entre "")
"""

import os
import json
import zipfile
import csv
import re
import io
import numpy as np
import faiss
from html.parser import HTMLParser
from sentence_transformers import SentenceTransformer

# ─── Configuration ─────────────────────────────────────────────────────────────

CIS_RCP_ZIP = "data/CIS_RCP.zip"
INDEX_PATH  = "data/index.faiss"
META_PATH   = "data/chunks_meta.json"
MODEL_NAME  = "paraphrase-multilingual-mpnet-base-v2"

MEDICAMENTS_CIBLES = [
    "doliprane", "dafalgan", "efferalgan", "paracétamol", "paracetamol",
    "ibuprofène", "ibuprofen", "advil", "nurofen",
    "aspirine", "aspégic", "acide acétylsalicylique",
    "amoxicilline", "augmentin", "clamoxyl",
    "smecta", "diosmectite",
    "imodium", "lopéramide", "loperamide",
    "ventoline", "salbutamol",
    "oméprazole", "omeprazole", "inexium", "esoméprazole",
    "metformine", "glucophage",
    "levothyrox", "lévothyroxine",
    "xanax", "alprazolam",
]

URLS_OFFICIELLES = {
    "doliprane":    "https://base-donnees-publique.medicaments.gouv.fr/affichageDoc.php?specid=60001558&typedoc=N",
    "ibuprofène":   "https://base-donnees-publique.medicaments.gouv.fr/affichageDoc.php?specid=68009291&typedoc=N",
    "aspirine":     "https://base-donnees-publique.medicaments.gouv.fr/affichageDoc.php?specid=60001829&typedoc=N",
    "amoxicilline": "https://base-donnees-publique.medicaments.gouv.fr/affichageDoc.php?specid=60001555&typedoc=N",
    "augmentin":    "https://base-donnees-publique.medicaments.gouv.fr/affichageDoc.php?specid=60001421&typedoc=N",
    "oméprazole":   "https://base-donnees-publique.medicaments.gouv.fr/affichageDoc.php?specid=60001931&typedoc=N",
    "metformine":   "https://base-donnees-publique.medicaments.gouv.fr/affichageDoc.php?specid=60001648&typedoc=N",
    "smecta":       "https://base-donnees-publique.medicaments.gouv.fr/affichageDoc.php?specid=60001319&typedoc=N",
    "ventoline":    "https://base-donnees-publique.medicaments.gouv.fr/affichageDoc.php?specid=60002329&typedoc=N",
    "imodium":      "https://base-donnees-publique.medicaments.gouv.fr/affichageDoc.php?specid=60001459&typedoc=N",
}
URL_BDPM_RECHERCHE = "https://base-donnees-publique.medicaments.gouv.fr/recherche.php?specNom={}"


# ─── Nettoyage HTML ────────────────────────────────────────────────────────────

class RCPHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip = 0
        self.block_tags = {"p","div","br","li","h1","h2","h3","h4","tr","td","th"}
        self.skip_tags  = {"script","style","img"}

    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self._skip += 1
        if tag in self.block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.skip_tags and self._skip > 0:
            self._skip -= 1
        if tag in self.block_tags:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip == 0:
            self.parts.append(data)

    def get_text(self):
        return "".join(self.parts)


def html_vers_texte(html: str) -> str:
    if not html or "<" not in html:
        return html.strip() if html else ""
    try:
        p = RCPHTMLParser()
        p.feed(html)
        texte = p.get_text()
    except Exception:
        texte = re.sub(r"<[^>]+>", " ", html)

    for ent, car in [
        ("&nbsp;"," "),("&amp;","&"),("&lt;","<"),("&gt;",">"),
        ("&eacute;","é"),("&egrave;","è"),("&agrave;","à"),("&ccedil;","ç"),
        ("&ugrave;","ù"),("&ocirc;","ô"),("&ecirc;","ê"),("&acirc;","â"),
        ("&ucirc;","û"),("&iuml;","ï"),("&#39;","'"),("&quot;",'"'),
        ("&laquo;","«"),("&raquo;","»"),
    ]:
        texte = texte.replace(ent, car)

    texte = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", texte)
    texte = re.sub(r"[ \t]+", " ", texte)
    texte = re.sub(r"\n{3,}", "\n\n", texte)
    texte = re.sub(r" +\n", "\n", texte)
    texte = re.sub(r"\n +", "\n", texte)
    return texte.strip()


def corriger_encodage(texte: str) -> str:
    """Corrige les problèmes d'encodage Latin-1 mal interprété en UTF-8."""
    try:
        return texte.encode("latin-1").decode("utf-8")
    except Exception:
        return texte


def extraire_denomination(html: str) -> str:
    """Extrait la dénomination depuis le HTML de la notice."""
    m = re.search(r'class=["\']AmmDenomination["\'][^>]*>(.*?)</p>',
                  html, re.IGNORECASE | re.DOTALL)
    if m:
        return corriger_encodage(html_vers_texte(m.group(1)).strip())

    m = re.search(
        r'DENOMINATION DU MEDICAMENT.*?<p[^>]*>(.*?)</p>',
        html, re.IGNORECASE | re.DOTALL
    )
    if m:
        t = corriger_encodage(html_vers_texte(m.group(1)).strip())
        if t and len(t) < 300:
            return t

    m = re.search(r'class=["\'][^"\']*[Dd]enomination[^"\']*["\'][^>]*>(.*?)</p>',
                  html, re.IGNORECASE | re.DOTALL)
    if m:
        t = corriger_encodage(html_vers_texte(m.group(1)).strip())
        if t and len(t) < 300:
            return t

    return ""


# ─── Lecture du CIS_RCP.zip avec csv.reader ────────────────────────────────────

def lire_cis_rcp(chemin_zip: str) -> list[dict]:
    """
    Utilise csv.reader pour gérer correctement les champs HTML
    multiligne entre guillemets doubles.
    """
    if not os.path.exists(chemin_zip):
        raise FileNotFoundError(f"Fichier introuvable : {chemin_zip}")

    print(f"  Ouverture de {chemin_zip}...")

    with zipfile.ZipFile(chemin_zip, "r") as z:
        noms = z.namelist()
        print(f"  Fichiers dans le zip : {noms}")
        fichier = next((n for n in noms if n.lower().endswith((".csv",".txt"))), None)
        if not fichier:
            raise ValueError("Aucun .csv/.txt dans le zip")
        print(f"  Lecture de : {fichier}")
        contenu_bytes = z.read(fichier)

    # Décoder en latin-1
    contenu = contenu_bytes.decode("latin-1", errors="replace")

    # Détecter le séparateur sur la première ligne
    premiere = contenu[:500]
    sep = "\t" if premiere.count("\t") > premiere.count(";") else ";"
    print(f"  Séparateur détecté : {'TAB' if sep == chr(9) else sep!r}")

    # Augmenter la limite de taille des champs CSV (HTML peut être très long)
    csv.field_size_limit(10_000_000)

    # Parser avec csv.reader — gère les champs multiligne entre ""
    documents = []
    reader = csv.reader(io.StringIO(contenu), delimiter=sep,
                        quotechar='"', skipinitialspace=True)

    for i, row in enumerate(reader):
        # Ignorer en-tête
        if i == 0:
            print(f"  En-tête : {row[:3]}")
            continue

        if len(row) < 2:
            continue

        cis      = row[0].strip()
        rcp_html = row[1].strip()

        # Valider CIS (6-9 chiffres)
        if not re.match(r"^\d{6,9}$", cis):
            continue
        if len(rcp_html) < 50:
            continue

        documents.append({"cis": cis, "rcp_html": rcp_html})

        if i % 1000 == 0 and i > 0:
            print(f"    {i:,} lignes CSV lues, {len(documents):,} notices...", end="\r")

    print(f"\n  {len(documents):,} notices valides lues")
    return documents


# ─── Filtrage ──────────────────────────────────────────────────────────────────

def filtrer_et_enrichir(documents: list[dict], cibles: list[str]) -> list[dict]:
    cibles_norm = [c.lower() for c in cibles]
    filtres = []
    vus = set()

    print(f"  Extraction des dénominations et filtrage...")

    for i, doc in enumerate(documents):
        if i % 500 == 0:
            print(f"    {i:,}/{len(documents):,} analysés...", end="\r")

        denomination = extraire_denomination(doc["rcp_html"])
        if not denomination:
            continue

        denom_lower = denomination.lower()
        for cible in cibles_norm:
            if cible in denom_lower and doc["cis"] not in vus:
                filtres.append({
                    "cis":          doc["cis"],
                    "denomination": denomination,
                    "rcp_html":     doc["rcp_html"],
                })
                vus.add(doc["cis"])
                break

    print(f"\n  {len(filtres)} médicaments trouvés :")
    for doc in filtres:
        print(f"    ✓ [{doc['cis']}] {doc['denomination'][:70]}")

    return filtres


# ─── Construction + Chunking ──────────────────────────────────────────────────

def construire_document(doc: dict) -> dict:
    denomination = doc["denomination"]
    texte_propre = html_vers_texte(doc["rcp_html"])
    texte_final  = f"Médicament : {denomination}\n\n{texte_propre}"

    url = URL_BDPM_RECHERCHE.format(denomination.split()[0].lower())
    for cle, val in URLS_OFFICIELLES.items():
        if cle in denomination.lower():
            url = val
            break

    return {
        "id":      f"med_{doc['cis']}",
        "contenu": texte_final,
        "metadata": {
            "source":         "BDPM / ANSM — Notice officielle RCP",
            "cis":            doc["cis"],
            "medicament":     denomination.split()[0].lower(),
            "denomination":   denomination,
            "url_prospectus": url,
        }
    }


def chunker(texte: str, taille_max: int = 600, overlap: int = 80) -> list[str]:
    paragraphes = re.split(r"\n{2,}", texte)
    chunks, current = [], ""
    for para in paragraphes:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 > taille_max and current:
            chunks.append(current.strip())
            current = current[-overlap:] + "\n\n" + para
        else:
            current = (current + "\n\n" + para) if current else para
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if len(c) > 80]


# ─── Embeddings + FAISS ───────────────────────────────────────────────────────

def embedder_chunks(chunks, modele):
    print(f"  Encodage de {len(chunks)} chunks...")
    return modele.encode(
        chunks, show_progress_bar=True, batch_size=32, convert_to_numpy=True
    ).astype(np.float32)


def creer_index_faiss(vecteurs):
    dim = vecteurs.shape[1]
    faiss.normalize_L2(vecteurs)
    index = faiss.IndexFlatIP(dim)
    index.add(vecteurs)
    print(f"  Index FAISS : {index.ntotal} vecteurs (dim={dim})")
    return index


def sauvegarder_index(index, chunks_avec_meta):
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks_avec_meta, f, ensure_ascii=False, indent=2)
    print(f"  Sauvegardé → {INDEX_PATH} + {META_PATH}")


def charger_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError("Base introuvable. Lancez : python indexation.py")
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        chunks_meta = json.load(f)
    return index, chunks_meta


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Pipeline d'indexation — RAG Médicaments (BDPM réelle)")
    print("=" * 60)

    print(f"\n[1/5] Chargement du modèle : {MODEL_NAME}")
    modele = SentenceTransformer(MODEL_NAME)

    print(f"\n[2/5] Lecture de CIS_RCP.zip")
    tous_les_docs = lire_cis_rcp(CIS_RCP_ZIP)

    if not tous_les_docs:
        print("❌ Aucune notice lue. Vérifiez le fichier CIS_RCP.zip")
        return

    print(f"\n[3/5] Filtrage + extraction des dénominations")
    docs_filtres = filtrer_et_enrichir(tous_les_docs, MEDICAMENTS_CIBLES)

    if not docs_filtres:
        print("❌ Aucun médicament trouvé après filtrage.")
        return

    print(f"\n[4/5] Chunking des notices")
    chunks_avec_meta = []
    for doc in docs_filtres:
        document = construire_document(doc)
        morceaux = chunker(document["contenu"])
        print(f"  {document['metadata']['denomination'][:55]:55} → {len(morceaux)} chunks")
        for i, morceau in enumerate(morceaux):
            chunks_avec_meta.append({
                "chunk_id": f"{document['id']}_chunk{i}",
                "contenu":  morceau,
                "metadata": {**document["metadata"], "chunk_index": i},
            })

    print(f"\n  Total : {len(chunks_avec_meta)} chunks pour {len(docs_filtres)} médicaments")

    print(f"\n[5/5] Embeddings + FAISS + Sauvegarde")
    vecteurs = embedder_chunks([c["contenu"] for c in chunks_avec_meta], modele)
    index    = creer_index_faiss(vecteurs)
    sauvegarder_index(index, chunks_avec_meta)

    print(f"\n✅ Indexation terminée !")
    print(f"   {len(chunks_avec_meta)} chunks | {len(docs_filtres)} médicaments réels BDPM")
    print("   Lancez maintenant : python rag.py")


if __name__ == "__main__":
    main()
