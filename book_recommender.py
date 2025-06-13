import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import ssl
import os

# Отключаем проверку SSL (временное решение для macOS)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Загружаем необходимые NLTK данные
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Ошибка при загрузке NLTK данных: {e}")
    exit(1)

class BookRecommender:
    def __init__(self, data_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Файл {data_path} не найден")
        self.df = pd.read_csv(data_path)

        # Добавляем id, если его нет
        if 'id' not in self.df.columns:
            self.df.reset_index(inplace=True)
            self.df.rename(columns={"index": "id"}, inplace=True)

        self.stop_words = set(stopwords.words('english'))
        self.tfidf = TfidfVectorizer(max_features=5000)
        self.process_data()

    def clean_text(self, text):
        """Очистка и предобработка текста"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        try:
            tokens = word_tokenize(text)
            tokens = [word for word in tokens if word not in self.stop_words]
            return ' '.join(tokens)
        except Exception as e:
            print(f"Ошибка токенизации: {e}")
            return text

    def process_data(self):
        """Подготовка данных для рекомендаций"""
        required_columns = ['title', 'authors', 'categories', 'description', 'average_rating', 'ratings_count']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют столбцы в датасете: {missing_columns}")

        self.df['description'] = self.df['description'].fillna('')
        self.df['categories'] = self.df['categories'].fillna('')
        self.df['authors'] = self.df['authors'].fillna('')

        self.df['content'] = (self.df['description'] + ' ' +
                              self.df['categories'] + ' ' +
                              self.df['authors']).apply(self.clean_text)

        scaler = MinMaxScaler()
        self.df['normalized_rating'] = scaler.fit_transform(
            self.df[['average_rating']].fillna(self.df['average_rating'].mean())
        )

        self.tfidf_matrix = self.tfidf.fit_transform(self.df['content'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix)

    def get_recommendations(self, title, n_recommendations=5):
        """Получение рекомендаций по названию книги"""
        try:
            idx = self.df[self.df['title'].str.lower() == title.lower()].index[0]
            sim_scores = list(enumerate(self.cosine_sim[idx]))

            for i in range(len(sim_scores)):
                sim_scores[i] = (sim_scores[i][0],
                                 sim_scores[i][1] * 0.7 +
                                 self.df.iloc[sim_scores[i][0]]['normalized_rating'] * 0.3)

            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:n_recommendations + 1]
            book_indices = [i[0] for i in sim_scores]

            recommendations = self.df.iloc[book_indices][[
                'id', 'title', 'authors', 'categories', 'average_rating', 'published_year'
            ]]

            return recommendations.to_dict('records')

        except IndexError:
            return {"error": "Книга не найдена"}
        except Exception as e:
            return {"error": f"Произошла ошибка: {str(e)}"}

    def get_similar_by_category(self, category, n_recommendations=5):
        """Рекомендации по категории"""
        category_books = self.df[self.df['categories'].str.contains(category, case=False, na=False)]
        if category_books.empty:
            return {"error": "Категория не найдена"}

        sorted_books = category_books.sort_values(
            by=['average_rating', 'ratings_count'],
            ascending=False
        )

        return sorted_books[[
            'id', 'title', 'authors', 'categories', 'average_rating', 'published_year'
        ]].head(n_recommendations).to_dict('records')


# --- Глобальная инициализация ---
recommender_instance = None

def get_recommender(data_path: str = None):
    """
    Возвращает глобальный экземпляр рекомендательной системы.
    Если он ещё не создан — создаёт его на основе указанного CSV-файла.
    """
    global recommender_instance
    if recommender_instance is None:
        if data_path is None:
            raise ValueError("Рекомендательная система ещё не инициализирована. Укажи data_path.")
        recommender_instance = BookRecommender(data_path)
    return recommender_instance
