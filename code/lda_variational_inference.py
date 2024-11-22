import numpy as np
from scipy.special import digamma, gammaln


class VariationalLDA:
    def __init__(self, num_topics, alpha_prior, beta_prior, max_iters=100, tolerance=1e-3):
        self.K = num_topics
        self.alpha = alpha_prior
        self.beta = beta_prior
        self.max_iters = max_iters
        self.tol = tolerance

    def _initialize_state(self, docs, vocab_size):
        self.docs = docs
        self.V = vocab_size
        self.D = len(docs)

        self.gamma = np.random.gamma(100., 1. / 100., (self.D, self.K))
        self.phi = [np.random.dirichlet(np.ones(self.K), len(doc)) for doc in docs]
        self.lambda_ = np.random.gamma(100., 1. / 100., (self.K, self.V))

    def _update_phi(self, doc_idx):
        for word_idx, word in enumerate(self.docs[doc_idx]):
            log_phi = digamma(self.gamma[doc_idx]) + digamma(self.lambda_[:, word]) - digamma(self.lambda_.sum(axis=1))
            log_phi -= log_phi.max()
            self.phi[doc_idx][word_idx] = np.exp(log_phi)
            self.phi[doc_idx][word_idx] /= self.phi[doc_idx][word_idx].sum()

    def _update_gamma(self, doc_idx):
        self.gamma[doc_idx] = self.alpha + self.phi[doc_idx].sum(axis=0)

    def _update_lambda(self):
        self.lambda_ = self.beta + sum(
            np.array([np.bincount(doc, weights=self.phi[d_idx][:, k], minlength=self.V) for d_idx, doc in enumerate(self.docs)])
            for k in range(self.K)
        )

    def _elbo(self):
        elbo = 0
        for d_idx, doc in enumerate(self.docs):
            elbo += np.sum(self.phi[d_idx] * (digamma(self.gamma[d_idx]) - digamma(self.gamma[d_idx].sum())))
        elbo += np.sum((self.beta - 1) * (digamma(self.lambda_) - digamma(self.lambda_.sum(axis=1)[:, None])))
        elbo += np.sum((self.alpha - 1) * (digamma(self.gamma) - digamma(self.gamma.sum(axis=1))[:, None]))
        elbo -= gammaln(self.gamma).sum() + gammaln(self.gamma.sum(axis=1)).sum()
        elbo -= gammaln(self.lambda_).sum() + gammaln(self.lambda_.sum(axis=1)).sum()
        return elbo

    def fit(self, docs, vocab_size):
        self._initialize_state(docs, vocab_size)
        prev_elbo = -np.inf
        for iteration in range(self.max_iters):
            for d_idx in range(self.D):
                self._update_phi(d_idx)
                self._update_gamma(d_idx)
            self._update_lambda()
            current_elbo = self._elbo()
            if abs(current_elbo - prev_elbo) < self.tol:
                break
            prev_elbo = current_elbo

    def topic_distributions(self):
        return self.lambda_ / self.lambda_.sum(axis=1, keepdims=True)

    def document_distributions(self):
        return self.gamma / self.gamma.sum(axis=1, keepdims=True)

    def top_words(self, vocab, n_words=10):
        topics = []
        for k in range(self.K):
            top_words = self.lambda_[k].argsort()[-n_words:][::-1]
            topics.append([vocab[word] for word in top_words])
        return topics


if __name__ == "__main__":
    example_corpus = [[0, 1, 2, 3, 4], [2, 3, 5, 6], [0, 4, 5, 6], [1, 2, 3, 6]]
    example_vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
    vocab_size = len(example_vocab)

    lda_vi = VariationalLDA(num_topics=3, alpha_prior=0.1, beta_prior=0.01, max_iters=100)
    lda_vi.fit(example_corpus, vocab_size)

    print("Document-Topic Distributions:")
    print(lda_vi.document_distributions())

    print("\nTopic-Word Distributions:")
    print(lda_vi.topic_distributions())

    print("\nTop Words per Topic:")
    for idx, words in enumerate(lda_vi.top_words(example_vocab, n_words=5)):
        print(f"Topic {idx}: {', '.join(words)}")

