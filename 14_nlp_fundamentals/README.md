# 14 — NLP Fundamentals

Classical and modern natural language processing applied to ecological monitoring reports, literature, and field notes. Covers the full pipeline from raw text cleaning through classification, topic discovery, embedding-based search, and named entity extraction.

---

## Notebooks

### `text_preprocessing.ipynb`
Regex-based cleaning: HTML removal, lowercasing, punctuation, normalisation. NLTK tokenisation, stopword removal, stemming (Porter) vs lemmatisation (WordNet). Vocabulary frequency analysis and Zipf's law visualisation. spaCy for POS tagging and NER on scientific text. Pitfalls: one-size preprocessing for all tasks, aggressive stemming on domain text, IDF fitted on full corpus, encoding issues, raw frequency vs TF-IDF importance.

### `tfidf_classification.ipynb`
`TfidfVectorizer` with unigrams + bigrams, min/max df, sublinear TF. Most informative features per class. Classification pipeline: Logistic Regression, Naive Bayes, Linear SVM compared via stratified CV. Confusion matrix and classification report. `GridSearchCV` tuning TF-IDF + classifier jointly. Feature importance from logistic coefficients. Pitfalls: TfidfVectorizer outside pipeline, max_features too low, MultinomialNB with TF-IDF, class imbalance, accuracy without error inspection.

### `word_embeddings.ipynb`
Word2Vec skip-gram training on ecological sentences. Cosine similarity for nearest neighbours. PCA 2D visualisation with semantic colour grouping. Document embedding via word vector averaging with OOV handling. gensim downloader for pre-trained GloVe/fastText models. Pitfalls: small training corpus, OOV in averaging, static embeddings for polysemy, cross-corpus comparison, unnormalised dot product.

### `topic_modelling.ipynb`
LDA with `CountVectorizer` (not TF-IDF). NMF with TF-IDF. Top-words display per topic. Perplexity vs k plot for topic count selection. Document-topic heatmap sorted by true topic. Dominant topic assignment. Pitfalls: perplexity alone for k selection, TF-IDF with LDA, missing domain stopwords, overlapping topics at high share, soft assignments as hard labels.

### `sentiment_ner.ipynb`
VADER lexicon-based sentiment with domain accuracy check. ML sentiment classifier (TF-IDF + Logistic Regression) trained on labelled monitoring reports. Regex-based domain NER: measurements, site codes, dates, chemical parameters. spaCy pre-trained NER on ecological text with commentary on fine-tuning needs. Pitfalls: VADER on technical text, binary sentiment for mixed documents, regex case sensitivity, general spaCy for domain entities, span-exact NER evaluation.

### `text_similarity_search.ipynb`
Jaccard similarity (word-set overlap). TF-IDF cosine similarity with ranked search. Full similarity matrix heatmap (Jaccard vs TF-IDF). Latent Semantic Analysis (TruncatedSVD) for semantic space. KMeans clustering in LSA space. Semantic search pattern: embed corpus, embed query, cosine nearest neighbours. Pointer to sentence-transformers for production semantic search. Pitfalls: TF-IDF for semantic search, unnormalised vectors, all-pairs on large corpus, LSA as interpretable topics, new vectoriser for query.

---

## Dependencies
```
pandas, numpy, matplotlib, sklearn
nltk (pip install nltk)
gensim (pip install gensim)
spacy (optional: pip install spacy && python -m spacy download en_core_web_sm)
sentence-transformers (optional, production semantic search)
```

## Data
Simulated ecological monitoring report text: water quality observations, restoration outcomes, and site condition assessments — with deliberate variation in vocabulary, sentiment, and domain-specific entities.
