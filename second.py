# Import needed modules
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read data
df = pd.read_csv('./data.csv')

# printing the first 5 rows of the dataframe
df.head()
# Get data information
df.info()

# Selecting the relevant features for recommendation
selected_features = ['title','authors','categories','published_year']
print(selected_features)

# Replacing the null valuess with null string
for feature in selected_features:
    df[feature] = df[feature].fillna('')

# combining all the 4 selected features
combined_features = df['title'] + ' ' + df['categories'] + ' ' + df['authors'] + ' ' + f"{df['published_year']}"
combined_features

# converting the text data to feature vectors
vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)

print(feature_vectors)

# getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors, feature_vectors)

print(similarity)

# creating a list with all the book names given in the dataset

list_of_all_titles = df['title'].tolist()
print(list_of_all_titles)

# getting the book name from the user
book_name = input(' Enter your favourite book name : ') # input: Rage of angels

# finding the close match for the book name given by the user
find_close_match = difflib.get_close_matches(book_name, list_of_all_titles)
print(find_close_match)

# finding the index of the book with title
close_match = find_close_match[0]
index_of_the_book = df[df.title == close_match].index[0]

# getting a list of similar books
similarity_score = list(enumerate(similarity[index_of_the_book]))
print(similarity_score)

# sorting the books based on their similarity score
sorted_similar_books = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
print(sorted_similar_books)

top_sim = sorted_similar_books[:5]
top_sim

# print the name of similar books based on the index
i = 1

for book in sorted_similar_books:
    index = book[0]
    title_from_index = df[df.index==index]['title'].values[0]
    if (i < 6):
        print(i, '-', title_from_index)
        i += 1


book_name = input(' Enter your favourite book name : ')

list_of_all_titles = df['title'].tolist()

find_close_match = difflib.get_close_matches(book_name, list_of_all_titles)

close_match = find_close_match[0]

index_of_the_book = df[df.title == close_match].index[0]

similarity_score = list(enumerate(similarity[index_of_the_book]))

sorted_similar_books = sorted(similarity_score, key = lambda x:x[1], reverse = True) 

print('Books suggested for you : \n')

i = 1

for book in sorted_similar_books:
    index = book[0]
    title_from_index = df[df.index==index]['title'].values[0]
    if (i < 30):
        print(i, '.',title_from_index)
        i+=1