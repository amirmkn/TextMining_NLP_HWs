import re
from collections import Counter

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
        freq_counts: Counter,  # Only the freq_counts are needed for Norvig
        use_norvig: bool = True  # Ensure we are using Norvig method
    ) -> None:
        """
        freq_counts     : Counter of word frequencies for Norvig method
        use_norvig      : if True, use Norvig's context-free corrector
        """
        self.use_norvig = use_norvig
        assert self.use_norvig, "This class is set up to only use the Norvig corrector"
        self.freq = freq_counts
        self.N = sum(freq_counts.values())

    def _norvig_candidates(self, word: str) -> List[str]:
        # Generate known candidates at edit distances 0, 1, 2
        c0 = {w for w in [word] if w in self.freq}
        if c0:
            return list(c0)
        c1 = {w for w in edits1(word) if w in self.freq}
        if c1:
            return list(c1)
        c2 = {w for w in edits2(word) if w in self.freq}
        return list(c2) or [word]

    def correct(self, sentence: str) -> str:
        """Correct a sentence using the Norvig method."""
        toks = re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)
        corrected: List[str] = []

        for tok in toks:
            if not tok.isalpha():
                corrected.append(tok)
                continue
            lower = tok.lower()

            # Use Norvig's method to find candidate corrections
            cands = self._norvig_candidates(lower)
            # Pick the highest frequency candidate
            best = max(cands, key=lambda w: self.freq.get(w, 1))
            # Preserve original case
            corrected.append(best.capitalize() if tok[0].isupper() else best)

        # Reconstruct the sentence
        out = ""
        for t in corrected:
            if re.match(r"^\w+$", t) and out and re.match(r".\w$", out[-2:]):
                out += " " + t
            else:
                out += t
        return out
