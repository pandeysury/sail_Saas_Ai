#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sire_rules_builder.py — FINAL (drop-in)

Emits rules.yaml for downstream RAG/KG enrichment:

YAML keys:
  - global_stopwords: [..]
  - generic_terms: [..]
  - viq_rules: [{viq_no, chapter_no, patterns: [..]}]
  - synonyms: { base_term: [variants...] }   # only if maritime_sqlite supplied

CLI:
  python tools/sire_rules_builder.py \
    --sqlite "C:\\sms\\mterms\\sire_viq.sqlite" \
    --maritime_sqlite "C:\\sms\\mterms\\maritime_terms_clean.sqlite" \
    --out "C:\\sms\\rules.yaml" \
    --topk 25

 Optional tuning flags (defaults are sensible):

--max_df 25 — lower to prune more generic unigrams.

--min_phrase_ratio 0.6 — raise to force more phrases (bigrams+).

--max_unigrams 8 — cap on unigrams retained per VIQ.

This will give you a tighter, high-precision rules.yaml ready for your indexer/QA loop.
"""

import argparse, sqlite3, json, re
from collections import Counter
from pathlib import Path

# ------------------- Core lexicons -------------------
STOP = set(x.strip().lower() for x in """
a an the and or of to in on for by with without within from into over under up down is are was were be been being
shall should must may can could would will as at that this these those which who whom whose it its their there here
""".split())

# Maritime-generic terms we do NOT want as VIQ patterns (extends STOP)
GENERIC = set(x.strip().lower() for x in """
vessel ship company master officer crew personnel staff procedure procedures policy policies ensure ensured ensuring
checked verify verified compliant compliance record records document documented documentation report reporting
system systems equipment operations operation operational management meeting meetings risk risks item items type types
""".split())
GENERIC |= {"all", "where", "during", "any", "use", "using", "onboard"}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\\-_/]+")

def tokens(s: str):
    return [w.lower() for w in TOKEN_RE.findall(s or "")]

# ------------------- Pattern scoring -------------------
def top_terms(text: str, topk: int = 25):
    """Return top informative unigrams/bigrams, excluding STOP/GENERIC."""
    toks = [t for t in tokens(text) if t not in STOP]
    cnt = Counter(toks)

    # Promote informative bigrams that avoid GENERIC
    words = [w for w in toks if w not in GENERIC]
    bigrams = Counter(zip(words, words[1:]))

    scores = Counter()
    for w, c in cnt.items():
        if w in GENERIC:
            continue
        scores[w] += c
    for (a, b), c in bigrams.items():
        if a in GENERIC or b in GENERIC:
            continue
        scores[f"{a} {b}"] += c * 1.5

    # Over-sample then trim after pruning
    out = [w for w, _ in scores.most_common(max(topk * 2, topk + 10))]
    # filter again; keep len>=3 to avoid noise
    out = [p for p in out if len(p) >= 3 and p not in STOP and p not in GENERIC]
    return out

# ------------------- DB readers -------------------
def load_rows(sqlite_path: Path):
    con = sqlite3.connect(str(sqlite_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Column discovery (case-insensitive)
    cur.execute("PRAGMA table_info(sire_viq)")
    cols = {r[1].lower(): r[1] for r in cur.fetchall()}

    def col(name, fallback=None):
        return cols.get(name.lower(), fallback or name)

    CHAPTER = col("chapter_no")
    VIQ     = col("viq_no")
    REQS    = col("procedural_requirements")
    QN      = col("viq_question")

    cur.execute(f"""
        SELECT {CHAPTER} as chapter_no,
               {VIQ}     as viq_no,
               {QN}      as viq_question,
               {REQS}    as procedural_requirements
        FROM sire_viq
    """)
    rows = cur.fetchall()
    con.close()
    return rows

# ------------------- Maritime synonyms loader (tolerant) -------------------
NOISY_SYNONYM_PATTERNS = [
    r"\bup\s*to\s*date\b",
    r"\buptodate\b",
    r"\bupto\s*date\b",
    r"\bupdated?\b",
    r"\bnot\s+answered\b",
    r"\bnot\s+marked\b",
    r"\bn/?a\b",
    r"^\d+$",
    r"^[\W_]+$",
]
NOISY_RXES = [re.compile(p, re.I) for p in NOISY_SYNONYM_PATTERNS]

def is_noisy_syn(s: str) -> bool:
    s = (s or "").strip()
    if len(s) < 2:
        return True
    return any(rx.search(s) for rx in NOISY_RXES)

def load_maritime_synonyms_any_schema(db_path: str) -> dict:
    """
    Return {base_term: [synonym1, synonym2, ...]} from a variety of schemas:
      - Normalized: maritime_terms(exact_term,id) + term_synonyms(term_id,synonym)
      - Flat:       maritime_terms(term, synonym)
      - Legacy:     terms(term) + synonyms(synonym[, term_id/term])
    Also lightly expands acronyms (dotted/spaced) if no synonyms present for a base.
    """
    import sqlite3
    def norm(s): return (s or "").strip().lower()
    con = sqlite3.connect(db_path); cur = con.cursor()

    def has_table(name):
        cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND lower(name)=lower(?)", (name,))
        return cur.fetchone() is not None

    groups = {}

    # 1) Normalized
    if has_table("maritime_terms") and has_table("term_synonyms"):
        cur.execute("PRAGMA table_info(maritime_terms)")
        cols_mt = [r[1].lower() for r in cur.fetchall()]
        exact_col = "exact_term" if "exact_term" in cols_mt else ("term" if "term" in cols_mt else None)
        if exact_col:
            cur.execute(f"""
              SELECT mt.{exact_col}, ts.synonym
              FROM maritime_terms mt
              JOIN term_synonyms ts ON ts.term_id = mt.id
            """)
            for base, syn in cur.fetchall():
                b = norm(base); s = norm(syn)
                if not b or not s: continue
                groups.setdefault(b, set()).add(s)

    # 2) Flat table variant
    if not groups and has_table("maritime_terms"):
        cur.execute("PRAGMA table_info(maritime_terms)")
        cols = [r[1].lower() for r in cur.fetchall()]
        if "term" in cols and "synonym" in cols:
            cur.execute("SELECT term, synonym FROM maritime_terms")
            for term, syn in cur.fetchall():
                t = norm(term); s = norm(syn)
                if t and s: groups.setdefault(t, set()).add(s)

    # 3) Legacy two-table
    if not groups and has_table("terms") and has_table("synonyms"):
        cur.execute("SELECT * FROM terms")
        bcols = [d[0].lower() for d in cur.description]
        ib = bcols.index("term") if "term" in bcols else 0
        bases = [norm(r[ib]) for r in cur.fetchall() if norm(r[ib])]
        for b in bases: groups.setdefault(b, set())

        cur.execute("SELECT * FROM synonyms")
        scols = [d[0].lower() for d in cur.description]
        isyn = scols.index("synonym") if "synonym" in scols else 0
        for r in cur.fetchall():
            s = norm(r[isyn])
            if s:
                if "term" in scols:
                    b = norm(r[scols.index("term")])
                    if b: groups.setdefault(b, set()).add(s)
                else:
                    groups.setdefault(s, set())

    con.close()

    # 4) If still empty, try to read bases and generate dotted/space variants
    if not groups:
        try:
            con = sqlite3.connect(db_path); cur = con.cursor()
            cur.execute("PRAGMA table_info(maritime_terms)")
            cols_mt = [r[1].lower() for r in cur.fetchall()]
            if "exact_term" in cols_mt:
                cur.execute("SELECT exact_term FROM maritime_terms")
                bases = [(r[0] or "").strip().lower() for r in cur.fetchall() if (r[0] or "").strip()]
                for b in bases:
                    groups.setdefault(b, set())
        except Exception:
            pass
        finally:
            try: con.close()
            except: pass

    # Light acronym expansion: ecdis -> "e.c.d.i.s", "e c d i s"
    def expand_acronym(t):
        letters = re.sub(r"[^a-z0-9]", "", t or "")
        if len(letters) >= 3:
            return {".".join(letters), " ".join(list(letters))}
        return set()

    for b in list(groups.keys()):
        extras = expand_acronym(b)
        groups[b].update(extras)

    # Filter noisy synonyms & drop empties
    cleaned = {}
    for base, syns in groups.items():
        keep = sorted({s for s in syns if s and not is_noisy_syn(s)})
        if keep:
            cleaned[base] = keep
    return cleaned

# ------------------- DF-based pruning + phrase ratio -------------------
def build_df(viq_rows):
    """Compute DF (VIQ-level) for unigrams only."""
    df = Counter()
    for r in viq_rows:
        text = f"{(r['viq_question'] or '')}\n{(r['procedural_requirements'] or '')}"
        toks = [t for t in tokens(text) if t not in STOP and t not in GENERIC]
        df.update(set([t for t in toks if " " not in t]))
    return df

def prune_patterns(patterns, df_uni, max_df=25, min_phrase_ratio=0.6, max_unigrams=8):
    """
    - Drop unigrams whose DF > max_df (too generic globally).
    - Cap remaining unigrams to max_unigrams.
    - Ensure phrase ratio >= min_phrase_ratio by down-capping unigrams.
    """
    phrases, unigrams = [], []
    for p in patterns:
        if " " in p:
            phrases.append(p)
        else:
            unigrams.append(p)

    # DF prune unigrams and drop very short ones
    keep_uni = [u for u in unigrams if df_uni.get(u, 0) <= max_df and len(u) >= 4]

    # cap unigrams
    keep_uni = keep_uni[:max_unigrams]

    # enforce phrase ratio
    total = len(phrases) + len(keep_uni)
    if total == 0:
        return []
    ratio = len(phrases) / total
    if ratio < min_phrase_ratio:
        # target unigrams count to hit ratio
        target_uni = int((len(phrases) / min_phrase_ratio) - len(phrases))
        target_uni = max(0, min(target_uni, len(keep_uni)))
        keep_uni = keep_uni[:target_uni]

    out = phrases + keep_uni
    # stable dedup
    seen = set(); dedup = []
    for x in out:
        if x in seen:
            continue
        seen.add(x); dedup.append(x)
    return dedup

# ------------------- YAML emitter (no external deps) -------------------
def yaml_str(value, indent=0):
    sp = "  " * indent
    if isinstance(value, dict):
        lines = []
        for k, v in value.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{sp}{k}:")
                lines.append(yaml_str(v, indent + 1))
            else:
                vq = json.dumps(v, ensure_ascii=False)
                lines.append(f"{sp}{k}: {vq}")
        return "\n".join(lines)
    elif isinstance(value, list):
        lines = []
        for it in value:
            if isinstance(it, (dict, list)):
                lines.append(f"{sp}-")
                lines.append(yaml_str(it, indent + 1))
            else:
                vq = json.dumps(it, ensure_ascii=False)
                lines.append(f"{sp}- {vq}")
        return "\n".join(lines)
    else:
        return f"{sp}{json.dumps(value, ensure_ascii=False)}"

# ------------------- Main -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", required=True, help="Path to sire_viq.sqlite")
    ap.add_argument("--maritime_sqlite", default="", help="Optional maritime terms SQLite for synonyms")
    ap.add_argument("--out", default="rules.yaml", help="Output YAML path")
    ap.add_argument("--topk", type=int, default=25, help="Raw patterns per VIQ before pruning")
    ap.add_argument("--max_df", type=int, default=25, help="DF threshold for pruning unigrams")
    ap.add_argument("--min_phrase_ratio", type=float, default=0.6, help="Minimum fraction of phrases in patterns")
    ap.add_argument("--max_unigrams", type=int, default=8, help="Max unigrams to keep after DF pruning")
    args = ap.parse_args()

    rows = load_rows(Path(args.sqlite))

    # First pass rows for DF build
    raw_rows = []
    for r in rows:
        viq_no = (r["viq_no"] or "").strip()
        ch = r["chapter_no"]
        req = (r["procedural_requirements"] or "").strip()
        qn  = (r["viq_question"] or "").strip()
        if not viq_no or not (req or qn):
            continue
        raw_rows.append({
            "viq_no": viq_no,
            "chapter_no": ch,
            "viq_question": qn,
            "procedural_requirements": req
        })

    # Build unigram DF across all VIQs
    df_uni = build_df(raw_rows)

    viq_rules = []
    for r in raw_rows:
        viq_no = r["viq_no"]
        ch = r["chapter_no"]
        text = f"{r['viq_question']}\n{r['procedural_requirements']}"
        pats = top_terms(text, topk=args.topk)

        # prune & enforce phrase ratio
        pats = prune_patterns(
            patterns=pats,
            df_uni=df_uni,
            max_df=args.max_df,
            min_phrase_ratio=args.min_phrase_ratio,
            max_unigrams=args.max_unigrams
        )
        if not pats:
            continue

        viq_rules.append({
            "viq_no": viq_no,
            "chapter_no": int(ch) if ch is not None else None,
            "patterns": pats
        })

    synonyms = {}
    if args.maritime_sqlite:
        try:
            synonyms = load_maritime_synonyms_any_schema(args.maritime_sqlite)
            total_variants = sum(len(v) for v in synonyms.values())
            print(f"Loaded maritime synonyms: {total_variants} variants across {len(synonyms)} base terms")
        except Exception as e:
            print(f"⚠ Could not load maritime synonyms: {e}")
            synonyms = {}

    data = {
        "global_stopwords": sorted(list(STOP)),
        "generic_terms": sorted(list(GENERIC)),
        "viq_rules": viq_rules,
    }
    if synonyms:
        data["synonyms"] = synonyms

    out_text = yaml_str(data)
    Path(args.out).write_text(out_text, encoding="utf-8")

    # Console summary
    n_viq = len(viq_rules)
    n_syn_groups = len(synonyms) if synonyms else 0
    n_syn_variants = sum(len(v) for v in synonyms.values()) if synonyms else 0
    print(
        f"Wrote rules to {args.out} "
        f"(VIQs: {n_viq}, synonym_groups: {n_syn_groups}, synonym_variants: {n_syn_variants})"
    )

if __name__ == "__main__":
    main()
