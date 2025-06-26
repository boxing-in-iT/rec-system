from fastapi import FastAPI
from book_recommender import get_recommender
from pydantic import BaseModel
from typing import List, Literal
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI()

class BookInput(BaseModel):
    id: int
    title: str
    authors: str = ''
    categories: str = ''
    description: str = ''
    average_rating: float = 0.0

class InteractionInput(BaseModel):
    book: BookInput
    interactionType: Literal['like', 'dislike']

@app.on_event("startup")
def load_model():
    get_recommender('data.csv')  # Инициализация один раз при запуске

@app.get("/recommend")
def recommend_books(title: str, count: int = 5):
    recommender = get_recommender()
    return {"recommendations": recommender.get_recommendations(title, count)}

@app.get("/recommend/category")
def recommend_by_category(category: str, count: int = 5):
    recommender = get_recommender()
    return {"recommendations": recommender.get_similar_by_category(category, count)}

@app.post("/recommend/personalized")
def recommend_personalized(interactions: List[InteractionInput], count: int = 20):
    recommender = get_recommender()

    liked_texts = [' '.join([i.book.description, i.book.categories, i.book.authors])
                   for i in interactions if i.interactionType == 'like']
    liked_clean = [recommender.clean_text(text) for text in liked_texts]

    if not liked_clean:
        return {"recommendations": []}

    tfidf = recommender.tfidf
    liked_vectors = tfidf.transform(liked_clean)
    user_profile_vector = liked_vectors.mean(axis=0).A  # Преобразуем np.matrix → array

    cosine_sim = cosine_similarity(user_profile_vector, recommender.tfidf_matrix).flatten()

    # Надёжная фильтрация: по названию (а не id)
    interacted_titles = {i.book.title.strip().lower() for i in interactions}

    indices_scores = [
        (idx, score) for idx, score in enumerate(cosine_sim)
        if recommender.df.iloc[idx]['title'].strip().lower() not in interacted_titles
    ]

    for i, (idx, score) in enumerate(indices_scores):
        popularity = recommender.df.iloc[idx]['normalized_rating']
        indices_scores[i] = (idx, score * 0.7 + popularity * 0.3)

    top_books = sorted(indices_scores, key=lambda x: x[1], reverse=True)[:count]
    indices = [i[0] for i in top_books]

    recommendations = recommender.df.iloc[indices][[
        'id', 'title', 'authors', 'categories', 'average_rating', 'published_year'
    ]].to_dict('records')

    return {"recommendations": recommendations}
