# Libraries
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

import rake_nltk
from rake_nltk import Rake
import nltk
nltk.download('stopwords')
nltk.download('punkt')

pd.set_option('display.max_columns',100)

# read data
df = pd.read_csv('Books.csv')

# column for Words
df['Bag_of_words'] = df.name + " " + df.description
df.set_index('name',inplace=True)

# countVectorizer
count = CountVectorizer()
count_matrix = count.fit_transform(df['Bag_of_words'])

# cosine similarity
cosine_sim = cosine_similarity(count_matrix, count_matrix)
indices = pd.Series(df.index)

# recommender
def recommendations(title, cosine_sim = cosine_sim):
    recommended_movies = []
    
    # getting the index of the movie that matches the title
    idx = indices[indices == title].index[0]
    print('IDX', idx)
    
    # creating a Series with the similarity scores in descending order 
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    print('Score Series: ', score_series)
    
    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)
    print(top_10_indexes)
    
    # populating the list with the titles of the best 10  matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(df.index)[i])
    
    return recommended_movies

# ---------------------------- Streamlit Workings --------------------------------

# Title
st.write("""
         # ~~~~~~Book Recommender~~~~~~""")

# Choose Book
book = st.selectbox('Choose a Book',(indices))
img_link = df.loc[book,'image_links']
image_url = "{}".format(img_link)
st.image(image_url) 
link = df.loc[book,'book_links']
st.markdown("(Your Book) {} ".format(link))

# Recommendation Generator Button
if st.button('Hit Me So I Recommend You Books!'):
    st.success("Book Recommended")
    # Page Breaker
    st.write("""
            # --------Recommended Books----------- """)

    # Get Recommendations
    recommended_books = recommendations(book)

    # Recommended Books
    count = 1
    for i in recommended_books:
        st.write(f"""
                ### Book {count}: """)
        
        # image
        img_link = df.loc[i,'image_links']
        image_url = "{}".format(img_link)
        st.image(image_url)    
        
        # Book
        st.write(f"""
                ###### Title:  """)
        rating = df.loc[i,'ratings']
        st.write(i+' (Rating-{})'.format(rating))
        
        
        # Link
        link = df.loc[i,'book_links']  
        st.markdown("Link:- {}".format(link))
        st.write("")
        count += 1


    

    
