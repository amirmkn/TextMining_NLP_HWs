from collections.abc import Iterable
from collections import Counter
import random
import math

class SmoothedNGramLanguageModel(object):

    def __init__(self, n: int, k: float, threshold: int) -> None:
        """
        n          : order of the n-gram (e.g., 2 for bigram, 3 for trigram)
        k          : smoothing parameter (add-k)
        threshold  : minimum frequency for a token to be in vocabulary
        """
        self.n = n
        self.k = k
        self.threshold = threshold
        self.vocabulary: set[str] = set()
        self.ngram_counts: Counter[tuple[str, ...]] = Counter()
        self.context_counts: Counter[tuple[str, ...]] = Counter()

    def train(self, training_sentences: list[list[str]]) -> None:
        # 1) Build vocabulary
        self.vocabulary = self._build_vocabulary(training_sentences)

        # 2) Initialize counts
        self.ngram_counts.clear()
        self.context_counts.clear()

        # 3) Update counts
        for sent in training_sentences:
            # replace rare tokens with <UNK>
            proc = [tok if tok in self.vocabulary else "<UNK>" for tok in sent]
            # pad sentence
            padded = self._pad_tokens(proc, self.n)
            # count n-grams and contexts
            for ngram in self._generate_n_grams(padded, self.n):
                self.ngram_counts[ngram] += 1
                context = ngram[:-1]
                self.context_counts[context] += 1

    @staticmethod
    def _pad_tokens(tokens: list[str], n: int) -> list[str]:
        # Add (n-1) start and end tokens
        return ["<s>"] * (n - 1) + tokens + ["</s>"] * (n - 1)

    @staticmethod
    def _generate_n_grams(tokens: list[str], n: int) -> Iterable[tuple[str, ...]]:
        # Yield each n-gram as a tuple
        for i in range(len(tokens) - n + 1):
            yield tuple(tokens[i : i + n])

    def _build_vocabulary(self, training_sentences: list[list[str]]) -> set[str]:
        # Count all tokens
        freq = Counter(tok for sent in training_sentences for tok in sent)
        # Keep tokens above threshold
        vocab = {tok for tok, cnt in freq.items() if cnt >= self.threshold}
        # Add special tokens
        vocab.update({"<UNK>", "<s>", "</s>"})
        return vocab

    def get_probability(self, sentences: list[list[str]]) -> float:
        """
        Compute the total log-probability of the given preprocessed sentences.
        Returns the sum of log-probabilities (natural log) of all n-grams.
        """
        total_log_prob = 0.0
        for sent in sentences:
            proc = [tok if tok in self.vocabulary else "<UNK>" for tok in sent]
            padded = self._pad_tokens(proc, self.n)
            for ngram in self._generate_n_grams(padded, self.n):
                context = ngram[:-1]
                count_ngram = self.ngram_counts.get(ngram, 0)
                count_context = self.context_counts.get(context, 0)
                # add-k smoothing
                prob = (count_ngram + self.k) / (count_context + self.k * len(self.vocabulary))
                total_log_prob += math.log(prob)
        return total_log_prob

    def get_perplexity(self, sentences: list[list[str]]) -> float:
        """
        Compute perplexity over the given preprocessed sentences.

        Perplexity = exp(-1/N * sum(log P(w_i | history)))
        where N is the total number of predicted tokens (excluding padding).
        """
        total_log_prob = 0.0
        N = 0
        for sent in sentences:
            proc = [tok if tok in self.vocabulary else "<UNK>" for tok in sent]
            padded = self._pad_tokens(proc, self.n)
            for ngram in self._generate_n_grams(padded, self.n):
                context = ngram[:-1]
                count_ngram = self.ngram_counts.get(ngram, 0)
                count_context = self.context_counts.get(context, 0)
                prob = (count_ngram + self.k) / (count_context + self.k * len(self.vocabulary))
                total_log_prob += math.log(prob)
            # count real tokens (excluding <s>, </s>)
            N += len(proc)
        # handle edge case
        if N == 0:
            return float('inf')
        return math.exp(- total_log_prob / N)

    def sample(self, random_seed: int, history: list[str] = []) -> Iterable[str]:
        """
        Generate a random sentence (or continuation) by sampling from the model.

        random_seed: seed for reproducibility
        history    : optional list of initial tokens (context)
        Returns the generated tokens (excluding <s> and </s>).
        """
        random.seed(random_seed)
        generated = []
        context = list(history[-(self.n - 1) :]) if history else []
        while True:
            # build context tuple with padding if needed
            if len(context) < (self.n - 1):
                pad = ["<s>"] * ((self.n - 1) - len(context))
                ctx = tuple(pad + context)
            else:
                ctx = tuple(context[-(self.n - 1) :])
            # build distribution over vocabulary
            vocab_list = list(self.vocabulary)
            probs = []
            for word in vocab_list:
                ngram = ctx + (word,)
                count_ngram = self.ngram_counts.get(ngram, 0)
                count_context = self.context_counts.get(ctx, 0)
                prob = (count_ngram + self.k) / (count_context + self.k * len(self.vocabulary))
                probs.append(prob)
            # normalize
            total = sum(probs)
            if total <= 0:
                break
            probs = [p / total for p in probs]
            # sample next word
            next_word = random.choices(vocab_list, probs)[0]
            if next_word == "</s>":
                break
            generated.append(next_word)
            context.append(next_word)
        return generated
