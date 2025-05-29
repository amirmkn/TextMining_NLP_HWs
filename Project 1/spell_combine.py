import re
import math
from typing import Iterable, List, Dict, Optional
from collections import Counter
from lm import SmoothedNGramLanguageModel as BaseLM
from pybktree import BKTree
import Levenshtein
from metaphone import doublemetaphone

# Peter Norvig style edit distance candidate generation
letters = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word: str) -> set:
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces   = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts    = [L + c + R     for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word: str) -> set:
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}

class SpellCorrector:
    def __init__(
        self,
        lm: Optional[BaseLM] = None,
        freq_counts: Optional[Counter] = None,
        hybrid: bool = False,
        max_edit_dist: int = 1,
        candidate_limit: int = 50,
        length_diff_limit: int = 1
    ) -> None:
        """
        lm              : trained n-gram model for contextual scoring
        freq_counts     : word frequencies for Norvig frequency prior
        hybrid          : if True, combine Norvig candidates with contextual BK-tree
        max_edit_dist   : max Levenshtein distance for BK-tree
        candidate_limit : cap on BK-tree candidates
        length_diff_limit: max allowed length difference for BK-tree filter
        """
        self.hybrid = hybrid
        if hybrid and freq_counts is None:
            raise ValueError("freq_counts required for hybrid mode")
        if not hybrid and lm is None:
            raise ValueError("lm required for contextual mode")

        self.lm = lm
        self.freq = freq_counts or Counter()
        self.N = sum(self.freq.values())
        self.vocab = sorted(lm.vocabulary) if lm else []
        self.bktree = BKTree(Levenshtein.distance, self.vocab) if lm else None
        self.max_edit = max_edit_dist
        self.candidate_limit = candidate_limit
        self.length_diff_limit = length_diff_limit
        self._candidate_cache: Dict[str, List[str]] = {}

    def _norvig_candidates(self, word: str) -> set:
        c0 = {word} if word in self.freq else set()
        if c0:
            return c0
        c1 = {w for w in edits1(word) if w in self.freq}
        if c1:
            return c1
        c2 = {w for w in edits2(word) if w in self.freq}
        return c2 or {word}

    def _generate_candidates(self, word: str) -> List[str]:
        # Use cache
        if word in self._candidate_cache:
            return self._candidate_cache[word]
        candidates = set()
        # Norvig-only if context-free or hybrid
        if self.hybrid or (self.lm is None):
            candidates |= self._norvig_candidates(word)
        # Contextual BK-tree candidates
        if self.lm is not None:
            matches = self.bktree.find(word, self.max_edit)
            matches.sort()
            candidates |= {w for _, w in matches[: self.candidate_limit]}
            # heuristics for longer words
            if len(word) >= 4:
                # phonetic
                code = doublemetaphone(word)[0]
                candidates |= {w for w in self.vocab if doublemetaphone(w)[0] == code}
                # keyboard typos & OCR
                for i, c in enumerate(word):
                    for adj in KEYBOARD_ADJ.get(c, ''):
                        v = word[:i] + adj + word[i+1:]
                        if v in self.vocab:
                            candidates.add(v)
                for src, tgt in OCR_CONFUSIONS.items():
                    if src in word:
                        v = word.replace(src, tgt)
                        if v in self.vocab:
                            candidates.add(v)
                    if tgt in word:
                        v = word.replace(tgt, src)
                        if v in self.vocab:
                            candidates.add(v)
            # length filter
            candidates = {w for w in candidates if abs(len(w) - len(word)) <= self.length_diff_limit}
        candidates.add(word)
        cand_list = list(candidates)
        self._candidate_cache[word] = cand_list
        return cand_list

    def correct(self, sentence: str) -> str:
        toks = re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)
        corrected: List[str] = []
        n = self.lm.n if self.lm else 1

        for tok in toks:
            if not tok.isalpha():
                corrected.append(tok)
                continue
            lower = tok.lower()
            cands = self._generate_candidates(lower)

            # If purely Norvig (lm None), pick highest frequency
            if self.lm is None:
                best = max(cands, key=lambda w: self.freq.get(w, 1))
                corrected.append(best.capitalize() if tok[0].isupper() else best)
                continue

            # Contextual scoring with LM
            left = [t.lower() for t in corrected if t.isalpha()][- (n - 1):]
            right = []  # no lookahead
            # score original
            orig_score = 0.0
            window = left + [lower] + right
            for j in range(len(window) - (n - 1)):
                orig_score += self.lm.get_probability([window[j:j+n]])
            best, best_score = lower, orig_score
            # score candidates
            for cand in cands:
                sc = 0.0
                win = left + [cand] + right
                for j in range(len(win) - (n - 1)):
                    sc += self.lm.get_probability([win[j:j+n]])
                # optional frequency prior
                if cand in self.freq:
                    sc += math.log((self.freq[cand]+1)/self.N)
                if sc > best_score:
                    best_score, best = sc, cand
            # accept if improved
            threshold = 1.0 + 0.1 * len(lower)
            if best_score - orig_score > threshold:
                corrected.append(best.capitalize() if tok[0].isupper() else best)
            else:
                corrected.append(tok)
        # reconstruct
        out = ""
        for t in corrected:
            if re.match(r"^\w+$", t) and out and re.match(r".\w$", out[-2:]):
                out += " " + t
            else:
                out += t
        return re.sub(r"\s+([?.!,;:])", r"\1", out)

# Ensure keyboard and OCR maps are defined
KEYBOARD_ADJ = {
    'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx',
    'e': 'wsdr', 'f': 'rtgdvc', 'g': 'tyhfvb', 'h': 'yujgbn',
    'i': 'ujko', 'j': 'uikhnm', 'k': 'ijolm', 'l': 'kop',
    'm': 'njk', 'n': 'bhjm', 'o': 'iklp', 'p': 'ol',
    'q': 'wa', 'r': 'edft', 's': 'awedxz', 't': 'rfgy',
    'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc',
    'y': 'tghu', 'z': 'asx'
}
OCR_CONFUSIONS = {
    '0': 'o', '1': 'l', '5': 's', '2': 'z', 'rn': 'm', 'cl': 'd'
}
