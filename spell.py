# spell.py
from __future__ import annotations
import re
import math
from typing import Iterable, List
from lm import SmoothedNGramLanguageModel as BaseLM
import editdistance   # pip install editdistance

class SpellCorrector:
    def __init__(
        self,
        lm: BaseLM,
        max_edit_dist: int = 1,
        candidate_limit: int = 50
    ) -> None:
        """
        lm              : a trained bigram or trigram model (from lm.py)
        max_edit_dist   : maximum Levenshtein distance for candidates
        candidate_limit : cap on number of candidates per word
        """
        self.lm = lm
        # cache the sorted vocabulary list for candidate generation
        self.vocab = sorted(lm.vocabulary)
        self.max_edit = max_edit_dist
        self.candidate_limit = candidate_limit

    def _generate_candidates(self, word: str) -> Iterable[str]:
        """
        Return all vocab words within edit distance ≤ max_edit.
        We cap at candidate_limit (smallest distances first).
        """
        dists = []
        for w in self.vocab:
            d = editdistance.eval(word, w)
            if d <= self.max_edit:
                dists.append((d, w))
        # sort by distance then alphabetically, take top‑k
        dists.sort()
        return [w for _, w in dists[: self.candidate_limit]]

    def correct(self, sentence: str) -> str:
        """
        Correct a single raw sentence (preserving spacing/punctuation-ish).
        Strategy:
          1. Tokenize on whitespace/punctuation
          2. For each token that is purely alphabetic:
               - generate candidates + original
               - for each cand, compute local log‑prob with LM using
                 the surrounding context window of size n
               - pick candidate with highest score
          3. Reassemble and return corrected sentence
        """
        # split into tokens, but keep punctuation as separate tokens
        toks = re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)
        n = self.lm.n
        corrected: List[str] = []

        for i, tok in enumerate(toks):
            if not tok.isalpha():
                corrected.append(tok)
                continue

            # lowercase + no stemming here
            lower = tok.lower()
            cands = set(self._generate_candidates(lower) + [lower])

            best, best_score = tok, -math.inf
            # extract the (n−1) words before & after, in lowercase
            left = [t.lower() for t in corrected if t.isalpha()][- (n - 1) :]
            right = [
                t.lower() for t in toks[i+1 : i+1 + (n - 1)] if t.isalpha()
            ]

            for cand in cands:
                # build a list of n‑gram contexts to score:
                # we’ll score only the n‑grams that include this position.
                # e.g. for bigram n=2: score P(cand|left[-1]) + P(right[0]|cand)
                # for trigram n=3: score P(cand|left[-2],left[-1]) + P(right[0]|left[-1],cand) + P(right[1]|cand,right[0])
                score = 0.0
                # a helper to build a sliding window over [*left, cand, *right]
                window = left + [cand] + right
                for j in range(len(window) - (n - 1)):
                    ngram = window[j : j + n]
                    prob = math.exp(self.lm.get_probability([ngram]))  
                    # get_probability returns log‑prob, so exp→prob
                    score += math.log(prob + 1e-12)  # avoid log(0)
                if score > best_score:
                    best_score, best = score, cand

            # restore original casing if unchanged, else lower
            if best == lower:
                # keep original token’s casing
                corrected.append(tok)
            else:
                corrected.append(best)

        # re‑join conservatively: put spaces between alphanumerics
        out = ""
        for t in corrected:
            if re.match(r"^\w+$", t) and out and re.match(r".\w$", out[-2:]):
                out += " " + t
            else:
                out += t
        return out
