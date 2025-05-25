from fastapi import FastAPI
from book_recommender import BookRecommender

app = FastAPI()

# Загружаем рекомендательную модель при старте
recommender = BookRecommender('data.csv')

@app.get("/recommend")
def recommend_books(title: str, count: int = 5):
    result = recommender.get_recommendations(title, count)
    return {"recommendations": result}

@app.get("/recommend/category")
def recommend_by_category(category: str, count: int = 5):
    result = recommender.get_similar_by_category(category, count)
    return {"recommendations": result}
