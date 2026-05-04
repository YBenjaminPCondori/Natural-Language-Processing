# Preprocessing and Feature Extraction Rundown

## Preprocessing used
- HTML/entity cleanup: Week 4 lab cleanup pattern, used before tokenisation.
- Whitespace normalisation: keeps clauses readable while removing layout noise.
- Regex tokenisation: Week 2 lab `\b\w+\b` word-boundary tokenisation.
- Legal-safe lowercasing: used inside tokenisers/vectorisers, without overwriting the stored clause text.
- Negation-aware tokenisation: Week 2 lab idea, using `NOT_` on the token after `no`, `not`, or `never`.
- BPE training/encoding: Weeks 2-3 lab implementation for subword/OOV inspection.

## Feature extraction used
- Bag-of-words inspection: Week 2/3 lab idea for understanding sparse features.
- TF-IDF: Week 2 lecture and Week 3 lab term weighting.
- Unigrams and bigrams: Week 2/3 lecture/lab n-gram feature extraction.

## Deliberately not default preprocessing
- Stopword removal: skipped because legal words such as `not`, `shall`, `may`, `unless`, and `except` carry meaning.
- Logistic Regression, Linear SVM, and Naive Bayes: modelling stage, not preprocessing.
- Co-occurrence vectors, LSA, and embeddings: representation/model-analysis material, not the default clause preprocessing path.
