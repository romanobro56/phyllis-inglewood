"""Builds flies/bard/words.npz: the fly's dictionary, each word's "scent", and the Shakespeare corpus.

    python flies/bard/build_words.py <complete_works.txt> <english_20k.txt>

Sources (downloaded once, not committed):
    Shakespeare: https://www.gutenberg.org/cache/epub/100/pg100.txt   (Complete Works, public domain)
    English:     https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt

The dictionary is every Shakespeare word seen at least twice, plus 20k common English words
(most of which Shakespeare never wrote: "email", "software", ...), plus punctuation.

Each word gets a fixed scent the fly can sense:
    freq   - log frequency in Shakespeare, 0..1 (0 = Shakespeare never used it)
    scent  - 5 numbers, 0..1: a summary of which words sit just left and right of it in
             Shakespeare (top components of a PPMI neighbour matrix). Words with similar jobs
             ("lord"/"king", "thou"/"he") smell alike. Words Shakespeare never used smell neutral.
The scent says nothing about which word should come next; the fly has to learn that itself.
"""
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "words.npz"
PUNCT = [",", ".", "?", "!"]
SCENT_DIMS = 5
KEEP_APOS = {"tis", "twas", "twere", "twill", "gainst", "em", "twixt", "scape"}


def shakespeare_tokens(path):
    text = Path(path).read_text(encoding="utf-8")
    text = text.split("*** START OF", 1)[1].split("\n", 1)[1].split("*** END OF", 1)[0]
    text = text.replace("’", "'").replace("‘", "'").replace("—", ", ").replace("_", "")
    text = re.sub(r"\[[^\]]*\]", " ", text)                    # stage directions
    toks = []
    for line in text.split("\n"):
        s = line.strip()
        if not s or s == s.upper():                            # speaker names, ACT/SCENE, titles
            continue
        if re.match(r"^(Enter|Exit|Exeunt|Re-enter)\b", s):
            continue
        s = s.lower().replace(";", ",").replace(":", ",")
        for m in re.finditer(r"'?[a-z]+(?:'[a-z]+)*|[,.?!]", s):
            w = m.group()
            if w.startswith("'"):
                w = w[1:]
                if w in KEEP_APOS:
                    w = "'" + w
            toks.append(w)
    return toks


def main(shakes_path, english_path):
    toks = shakespeare_tokens(shakes_path)
    counts = Counter(toks)
    english = [w.strip().lower() for w in Path(english_path).read_text().split()]
    english = [w for w in english if w.isalpha() and (len(w) > 1 or w in ("a", "i"))]
    vocab = PUNCT + sorted({w for w, c in counts.items() if c >= 2 and w not in PUNCT} | set(english))
    idx = {w: i for i, w in enumerate(vocab)}
    corpus = np.array([idx[w] for w in toks if w in idx], np.int32)
    V = len(vocab)
    count = np.bincount(corpus, minlength=V).astype(np.float64)
    print(f"{len(toks):,} Shakespeare tokens, {len(corpus):,} kept; dictionary {V:,} words "
          f"({int((count == 0).sum()):,} that Shakespeare never used)")

    # neighbour matrix: word x (left neighbour in top C, right neighbour in top C)
    C = 2000
    top = np.argsort(-count)[:C]
    col = -np.ones(V, np.int64); col[top] = np.arange(C)
    M = np.zeros((V, 2 * C))
    prev, nxt = corpus[:-1], corpus[1:]
    ok = col[prev] >= 0
    np.add.at(M, (nxt[ok], col[prev[ok]]), 1)                   # left neighbour of nxt
    ok = col[nxt] >= 0
    np.add.at(M, (prev[ok], C + col[nxt[ok]]), 1)               # right neighbour of prev
    ctx = M.sum(0) ** 0.75
    pc = ctx / ctx.sum()
    rows = M.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ppmi = np.maximum(np.log((M / np.maximum(rows, 1)) / pc), 0)
    ppmi[~np.isfinite(ppmi)] = 0
    # principal directions from frequent words, then project every word onto them
    fit = count >= 20
    X = ppmi[fit] - ppmi[fit].mean(0)
    w_fit = np.sqrt(np.log1p(count[fit]))[:, None]
    _, vecs = np.linalg.eigh((X * w_fit).T @ (X * w_fit))
    comps = vecs[:, ::-1][:, :SCENT_DIMS]
    emb = (ppmi - ppmi[fit].mean(0)) @ comps
    scent = np.full((V, SCENT_DIMS), 0.5)
    seen = count > 0
    for d in range(SCENT_DIMS):
        lo, hi = np.percentile(emb[fit, d], [2, 98])
        scent[seen, d] = np.clip((emb[seen, d] - lo) / (hi - lo), 0, 1)
    freq = np.log1p(count) / np.log1p(count.max())

    np.savez_compressed(OUT, vocab=np.array(vocab), count=count.astype(np.int32),
                        freq=freq.astype(np.float32), scent=scent.astype(np.float32), corpus=corpus)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    for w in ["thou", "thee", "lord", "king", "love", "is", "the", ",", ".", "email"]:
        if w in idx:
            i = idx[w]
            print(f"  {w:>8}  freq {freq[i]:.2f}  scent {np.round(scent[i], 2)}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
