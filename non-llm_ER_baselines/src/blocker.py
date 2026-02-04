from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer(
                        lowercase=True,
                        analyzer='char_wb',
                        ngram_range=(2,3)
                )

def calculate_similiarity(pairs_df, compare_column1, compare_column2):

    tfidf = vectorizer.fit_transform(
        pairs_df[compare_column1].tolist() +
        pairs_df[compare_column2].tolist()
    )

    n = len(pairs_df)
    pairs_df["similarity"] = cosine_similarity(
        tfidf[:n], tfidf[n:]
    ).diagonal()

    return pairs_df