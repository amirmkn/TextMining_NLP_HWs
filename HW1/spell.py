import re
import math
from typing import Iterable, List, Dict
from lm import SmoothedNGramLanguageModel as BaseLM
from pybktree import BKTree
import Levenshtein
from metaphone import doublemetaphone

# Common keyboard adjacency mapping (QWERTY)
KEYBOARD_ADJ = {
    'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx',
    'e': 'wsdr', 'f': 'rtgdvc', 'g': 'tyhfvb', 'h': 'yujgbn',
    'i': 'ujko', 'j': 'uikhnm', 'k': 'ijolm', 'l': 'kop',
    'm': 'njk', 'n': 'bhjm', 'o': 'iklp', 'p': 'ol',
    'q': 'wa', 'r': 'edft', 's': 'awedxz', 't': 'rfgy',
    'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc',
    'y': 'tghu', 'z': 'asx'
}

# OCR confusion pairs
OCR_CONFUSIONS = {
    '0': 'o', '1': 'l', '5': 's', '2': 'z', 'rn': 'm', 'cl': 'd'
}

class SpellCorrector:
    def __init__(
        self,
        lm: BaseLM,
        max_edit_dist: int = 1,
        candidate_limit: int = 50,
        length_diff_limit: int = 1   # ← New: max allowed length difference
    ) -> None:
        self.lm = lm
        self.vocab = sorted(lm.vocabulary)
        self.max_edit = max_edit_dist
        self.candidate_limit = candidate_limit
        self.length_diff_limit = length_diff_limit

        # Use Levenshtein.distance directly:
        self.bktree = BKTree(Levenshtein.distance, self.vocab)
        self._candidate_cache: Dict[str, List[str]] = {}

    def _phonetic_matches(self, word: str) -> List[str]:
        code = doublemetaphone(word)[0]
        return [w for w in self.vocab if doublemetaphone(w)[0] == code]

    def _keyboard_typos(self, word: str) -> List[str]:
        variants = set()
        for i, c in enumerate(word):
            for adj in KEYBOARD_ADJ.get(c, ''):
                v = word[:i] + adj + word[i+1:]
                if v in self.vocab:
                    variants.add(v)
        return list(variants)

    def _ocr_variants(self, word: str) -> List[str]:
        variants = set()
        for src, tgt in OCR_CONFUSIONS.items():
            if src in word:
                v = word.replace(src, tgt)
                if v in self.vocab:
                    variants.add(v)
            if tgt in word:
                v = word.replace(tgt, src)
                if v in self.vocab:
                    variants.add(v)
        return list(variants)

    def _generate_candidates(self, word: str) -> Iterable[str]:
        if word in self._candidate_cache:
            return self._candidate_cache[word]

        cands = set()

        # 1) edit-distance via BK-tree
        matches = self.bktree.find(word, self.max_edit)
        matches.sort()
        cands.update(w for _, w in matches[: self.candidate_limit])

        # 2) only for longer words add phonetic/typo/OCR
        if len(word) >= 4:
            cands.update(self._phonetic_matches(word))
            cands.update(self._keyboard_typos(word))
            cands.update(self._ocr_variants(word))

        # 3) always include original
        cands.add(word)

        # 4) NEW: filter by length difference to avoid "with"→"it"
        filtered = {
            w for w in cands 
            if abs(len(w) - len(word)) <= self.length_diff_limit
        }

        self._candidate_cache[word] = list(filtered)
        return list(filtered)

    def correct(self, sentence: str) -> str:
        toks = re.findall(r"[a-z]+", sentence.lower())
        
        n = self.lm.n
        corrected: List[str] = []

        for i, tok in enumerate(toks):
            # skip non-alpha or very short words
            if not tok.isalpha() or len(tok) <= 2:
                corrected.append(tok)
                continue

            lower = tok.lower()
            candidates = set(self._generate_candidates(lower))

            # build left/right context
            left = [t.lower() for t in corrected if t.isalpha()][- (n - 1):]
            right = [t.lower() for t in toks[i+1 : i+1 + (n - 1)] if t.isalpha()]

            # score original
            orig_score = 0.0
            base_window = left + [lower] + right
            for j in range(len(base_window) - (n - 1)):
                ngram = base_window[j : j + n]
                orig_score += self.lm.get_probability([ngram])

            best, best_score = lower, orig_score

            # score each candidate
            for cand in candidates:
                sc = 0.0
                window = left + [cand] + right
                for j in range(len(window) - (n - 1)):
                    ngram = window[j : j + n]
                    sc += self.lm.get_probability([ngram])
                if sc > best_score:
                    best_score, best = sc, cand

            # accept only if improved beyond threshold
            if best_score - orig_score > 1.0:
                corrected.append(best.capitalize() if tok[0].isupper() else best)
            else:
                corrected.append(tok)

        # reassemble, then remove space before punctuation
        out = " ".join(corrected)
        out = re.sub(r"\s+([?.!,;:])", r"\1", out)
        return out
