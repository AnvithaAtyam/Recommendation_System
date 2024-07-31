#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd


# In[2]:


movies = pd.read_csv('movie.csv')


# In[3]:


movies.shape


# In[4]:


movies.isnull().sum()


# In[5]:


movies.duplicated().sum()


# In[6]:


movies.iloc[0]['genres']


# In[7]:


tags = pd.read_csv('tag.csv')


# In[8]:


movies.head


# In[9]:


tags.head


# In[10]:


''' movies['tags'] = ""
    for i in tags.index:
    a = movies.loc[movies['movieId'] == tags['movieId'][i]].index[0]
    movies['tags'][a] = movies['tags'][a] + "|" + tags['tag'][i]'''


# In[11]:


merged_df = pd.merge(movies, tags, on='movieId', how='left')


# In[12]:


grouped_df = merged_df.groupby('movieId').agg({'tag': list}).reset_index()


# In[13]:


movies = pd.merge(movies, grouped_df, on='movieId', how='left')


# In[14]:


movies['tag'].fillna(value='', inplace=True)


# In[15]:


movies


# In[16]:


movies['tag'].fillna(value=pd.NA, inplace=True)


# In[17]:


movies


# In[18]:


movies['tag'] = movies['tag'].apply(lambda x: ', '.join(map(str, x)) if x else '')


# In[19]:


movies


# In[20]:


for i in movies.index:
    if movies['tag'][i] == "nan":
        movies['tag'][i] = ""


# In[21]:


movies


# In[22]:


for i in movies.index:   
     movies['genres'][i] =  movies['genres'][i].replace('|',',')


# In[23]:


movies


# In[24]:


movies['parameters']  = movies['genres'] + movies['tag']


# In[25]:


movies


# In[26]:


'''from gensim.models import Word2Vec
import pandas as pd


# Split the 'genres' column by a comma and create lists of words
sentences = [genre.split(',') if pd.notna(genre) else [] for genre in movies['parameters']]

# Create and train the Word2Vec model
model1 = Word2Vec(sentences, vector_size=5000, window=5, min_count=1, sg=0, hs=0, negative=5, epochs=500)

# Save the model to a file
model1.save("movie_lens_500")'''


# In[27]:


#import gensim
from gensim.models import Word2Vec

# Load the Word2Vec model
model1 = Word2Vec.load("movie_lens_500")


# In[28]:


import numpy as np

def vectorize_genres(genres, model):
    # Split genres by comma and remove empty strings
    genre_list = [genre.strip() for genre in genres.split(',') if genre.strip()]

    # Initialize an empty array to store genre vectors
    genre_vectors = []

    # Iterate through genres and get vectors
    for genre in genre_list:
        if genre in model.wv:
            genre_vectors.append(model.wv[genre])

    # If there are no valid vectors, return None
    if not genre_vectors:
        return None

    # Calculate the average vector
    average_vector = np.mean(genre_vectors, axis=0)

    return average_vector


# In[29]:


movies['genre_vector'] = movies['parameters'].apply(lambda x: vectorize_genres(x, model1))


# In[30]:


movies


# In[31]:


movies['genre_vector'][0]


# In[32]:


import pandas as pd
import numpy as np
from numpy.linalg import norm


# In[33]:


def recommed(name):
    Top_movies = []
    cosine_array = []
    index = movies.loc[movies['title'] == name].index[0]
    for i in movies.index:
        A = movies['genre_vector'][i]
        B = movies['genre_vector'][index]

        # Check for missing or invalid data (NoneType or empty vectors)
        if A is not None and B is not None and len(A) > 0 and len(B) > 0:
            A = np.array(A)
            B = np.array(B)
            cos = np.dot(A, B) / (norm(A) * norm(B))
            cosine_array.append(cos)
        else:
            cosine_array.append(0.0)  # Handle missing or invalid data with a default value

    # Now cosine_array contains the cosine similarities
    #print(cosine_array)
    sorted_with_indices = sorted(enumerate(cosine_array), key=lambda x: x[1], reverse=True)
    for i in range(1,11):
        Top_movies.append(movies['title'][sorted_with_indices[i][0]])
    print(Top_movies)


# In[34]:


recommed('Harry Potter and the Deathly Hallows: Part 2 (2011)')


# In[35]:


def recommend_user(user_watch_history):
    
    movie_vectors = []
    
    for i in user_watch_history:
        index = movies.loc[movies['title'] == i].index[0]
        movie_vectors.append(movies['genre_vector'][index])
        
    average_vector = np.mean(movie_vectors, axis=0)
    
    Top_movies = []
    cosine_array = []
    
    for i in movies.index:
        A = movies['genre_vector'][i]
        B = average_vector

        # Check for missing or invalid data (NoneType or empty vectors)
        if A is not None and B is not None and len(A) > 0 and len(B) > 0:
            A = np.array(A)
            B = np.array(B)
            cos = np.dot(A, B) / (norm(A) * norm(B))
            cosine_array.append(cos)
        else:
            cosine_array.append(0.0)  # Handle missing or invalid data with a default value

    # Now cosine_array contains the cosine similarities
    #print(cosine_array)
    sorted_with_indices = sorted(enumerate(cosine_array), key=lambda x: x[1], reverse=True)
    for i in range(0,30):
        Top_movies.append(movies['title'][sorted_with_indices[i][0]])
        
    final_list = [i for i in Top_movies if i not in user_watch_history]
    print(final_list)
    


# In[36]:


movs = ['Thor (2011)', 'Harry Potter and the Deathly Hallows: Part 2 (2011)', 'Clash of the Titans (1981)', 'Harry Potter and the Deathly Hallows: Part 1 (2010)', 'Zu: Warriors from the Magic Mountain (Xin shu shan jian ke) (1983)', 'Drona (2008)', 'Web of Death, The (1976)', 'Extraordinary Adventures of Adèle Blanc-Sec, The (2010)', 'Hellboy (2004)', 'King Kong (1933)', 'Dragonball Evolution (2009)', 'Starcrash (a.k.a. Star Crash) (1978)', 'Storm Warriors, The (Fung wan II) (2009)', 'Casey Jones (2011)', 'Dragon Age: Redemption (2011)', 'Dragonphoenix Chronicles: Indomitable, The (2013)', 'Dragon Crusaders (2011)', 'Midnight Chronicles (2009)', 'Hercules and the Circle of Fire (1994)', 'Hercules (2005)', "Dragonheart 3: The Sorcerer's Curse (2015)", 'Shadow, The (1994)']
recommend_user(movs) #hard coded


# In[37]:


users = pd.read_csv('rating.csv')


# In[38]:


def get_high_rated_movies(user_id, threshold=4):
    user_ratings = users[(users['userId'] == user_id) & (users['rating'] >= threshold)]
    result = user_ratings['movieId']
    result_movies = []
    for i in result:
        index = movies.loc[movies['movieId'] == i].index[0]
        result_movies.append(movies['title'][index])
    recommend_user(result_movies)
    


# In[39]:


get_high_rated_movies(1) #give the userId as the parameter here

