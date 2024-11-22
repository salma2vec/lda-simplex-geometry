import numpy as np
from scipy.special import gammaln

class TopicModelGibbs:
    def __init__(self, n_topics, dirichlet_alpha, dirichlet_beta, max_iters=1000):
        self.K = n_topics
        self.alpha = dirichlet_alpha
        self.beta = dirichlet_beta
        self.iters = max_iters

    def _initialize_state(self, documents, vocab_size):
        self.docs = documents
        self.V = vocab_size
        self.D = len(documents)

        self.topic_word = np.zeros((self.K, self.V), dtype=int)
        self.doc_topic = np.zeros((self.D, self.K), dtype=int)
        self.total_topic = np.zeros(self.K, dtype=int)
        self.word_topics = []

        for d_idx, doc in enumerate(documents):
            topics = []
            for word in doc:
                topic = np.random.randint(0, self.K)
                topics.append(topic)
                self.topic_word[topic, word] += 1
                self.doc_topic[d_idx, topic] += 1
                self.total_topic[topic] += 1
            self.word_topics.append(topics)

    def _sample_new_topic(self, word, doc_idx, current_topic):
        self.topic_word[current_topic, word] -= 1
        self.doc_topic[doc_idx, current_topic] -= 1
        self.total_topic[current_topic] -= 1

        word_given_topic = (self.topic_word[:, word] + self.beta) / (self.total_topic + self.beta * self.V)
        topic_given_doc = (self.doc_topic[doc_idx] + self.alpha) / (len(self.docs[doc_idx]) + self.alpha * self.K)
        probs = word_given_topic * topic_given_doc
        probs /= probs.sum()

        new_topic = np.random.choice(self.K, p=probs)

        self.topic_word[new_topic, word] += 1
        self.doc_topic[doc_idx, new_topic] += 1
        self.total_topic[new_topic] += 1
        return new_topic

    def _run_gibbs(self):
        for _ in range(self.iters):
            for doc_idx, doc in enumerate(self.docs):
                for word_idx, word in enumerate(doc):
                    current_topic = self.word_topics[doc_idx][word_idx]
                    new_topic = self._sample_new_topic(word, doc_idx, current_topic)
                    self.word_topics[doc_idx][word_idx] = new_topic

    def fit(self, corpus, vocab_size):
        self._initialize_state(corpus, vocab_size)
        self._run_gibbs()

    def topic_distributions(self):
        return (self.topic_word + self.beta) / (self.total_topic[:, None] + self.beta * self.V)

    def document_distributions(self):
        return (self.doc_topic + self.alpha) / (self.doc_topic.sum(axis=1)[:, None] + self.alpha * self.K)

    def top_words(self, vocab, n_words=10):
        phi = self.topic_distributions()
        topics = []
        for k in range(self.K):
            top_words = phi[k].argsort()[-n_words:][::-1]
            topics.append([vocab[word] for word in top_words])
        return topics


if __name__ == "__main__":
    example_docs = [[0, 1, 2, 3, 4], [2, 3, 5, 6], [0, 4, 5, 6], [1, 2, 3, 6]]
    example_vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]
    vocab_size = len(example_vocab)

    lda_model = TopicModelGibbs(n_topics=3, dirichlet_alpha=0.1, dirichlet_beta=0.01, max_iters=100)
    lda_model.fit(example_docs, vocab_size)

    print("Document-Topic Distributions:")
    print(lda_model.document_distributions())

    print("\nTopic-Word Distributions:")
    print(lda_model.topic_distributions())

    print("\nTop Words per Topic:")
    for idx, words in enumerate(lda_model.top_words(example_vocab, n_words=5)):
        print(f"Topic {idx}: {', '.join(words)}")
