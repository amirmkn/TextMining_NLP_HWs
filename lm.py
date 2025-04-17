# lm.py
from __future__ import annotations
from collections import Counter, defaultdict
import math
import random

class SmoothedNGramLanguageModel:
    def __init__(self, n: int, k: float, threshold: int) -> None:
        """
        n: order of the model (e.g. 2 for bigram, 3 for trigram)
        k: add‑k smoothing parameter
        threshold: minimum frequency for a token to stay in-vocab; all others → <UNK>
        """
        self.n = n
        self.k = k
        self.threshold = threshold

        # to be populated in train()
        self.vocabulary: set[str] = set()
        self.V = 0                    # vocabulary size
        self.ngram_counts: Counter[tuple[str, ...]] = Counter()
        self.context_counts: Counter[tuple[str, ...]] = Counter()

    def train(self, training_sentences: list[list[str]]) -> None:
        # 1) Count raw token frequencies
        token_freq = Counter(tok for sent in training_sentences for tok in sent)

        # 2) Build vocab: tokens ≥ threshold + special tokens
        self.vocabulary = {tok for tok, cnt in token_freq.items() if cnt >= self.threshold}
        self.vocabulary.update(["<UNK>", "<s>", "</s>"])
        self.V = len(self.vocabulary)

        # 3) Process each sentence: map rare→<UNK>, pad, then update counts
        for sent in training_sentences:
            # replace rare tokens
            proc = [
                tok if tok in self.vocabulary and tok not in {"<s>","</s>"} else
                "<UNK>" if tok not in {"<s>","</s>"} else tok
                for tok in sent
            ]
            padded = self._pad_tokens(proc)
            self._update_counts(padded)

    def _pad_tokens(self, tokens: list[str]) -> list[str]:
        """Prepend (n−1) <s> and append one </s>."""
        return ["<s>"] * (self.n - 1) + tokens + ["</s>"]

    @staticmethod
    def _generate_n_grams(tokens: list[str], n: int):
        """Yield each n‑gram (as a tuple) in the token list."""
        for i in range(len(tokens) - n + 1):
            yield tuple(tokens[i:i+n])

    def _update_counts(self, sentence: list[str]) -> None:
        """Increment counts for each n‑gram and its (n−1)-gram context."""
        for ngram in self._generate_n_grams(sentence, self.n):
            context = ngram[:-1]
            self.ngram_counts[ngram] += 1
            self.context_counts[context] += 1

    def get_probability(self, sentences: list[list[str]]) -> float:
        """
        Return the total log‑probability (natural log) of the given sentences.
        """
        log_prob = 0.0
        for sent in sentences:
            # map unknowns & pad
            proc = [
                tok if tok in self.vocabulary and tok not in {"<s>","</s>"} else
                "<UNK>" if tok not in {"<s>","</s>"} else tok
                for tok in sent
            ]
            padded = self._pad_tokens(proc)

            # sum log‑probs of each n‑gram
            for ngram in self._generate_n_grams(padded, self.n):
                context = ngram[:-1]
                c_ng = self.ngram_counts.get(ngram, 0)
                c_ctx = self.context_counts.get(context, 0)
                # add‑k smoothing
                p = (c_ng + self.k) / (c_ctx + self.k * self.V)
                log_prob += math.log(p)
        return log_prob

    def get_perplexity(self, sentences: list[list[str]]) -> float:
        """
        Compute perplexity:
          PPL = exp(− (1/N) * log P(sentences) )
        where N = total # of predicted tokens (i.e. sum of (len(sent)+1) for </s>).
        """
        N = sum(len(sent) + 1 for sent in sentences)
        log_p = self.get_probability(sentences)
        return math.exp(-log_p / N)

    def sample(self, random_seed: int = None, history: list[str] = None) -> list[str]:
        """
        Generate one sentence by repeated proportional sampling.
        Stops when </s> is generated. Returns tokens (excluding <s> and </s>).
        """
        if random_seed is not None:
            random.seed(random_seed)

        # initialize history to (n−1) start‑tokens
        hist = (history[-(self.n-1):] if history and len(history)>=self.n-1
                else ["<s>"]*(self.n-1))
        result: list[str] = []

        while True:
            context = tuple(hist[-(self.n-1):]) if self.n>1 else tuple()
            # build candidate distribution
            candidates, weights = [], []
            for tok in self.vocabulary:
                ngram = context + (tok,)
                c_ng = self.ngram_counts.get(ngram, 0)
                c_ctx = self.context_counts.get(context, 0)
                p = (c_ng + self.k) / (c_ctx + self.k * self.V)
                candidates.append(tok)
                weights.append(p)

            # normalize and sample
            total = sum(weights)
            weights = [w/total for w in weights]
            next_tok = random.choices(candidates, weights=weights, k=1)[0]

            if next_tok == "</s>":
                break
            result.append(next_tok)
            hist.append(next_tok)

        return result
