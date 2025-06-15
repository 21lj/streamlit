import streamlit as st 
import pickle as pkl 
import numpy as np 
import pandas as pd 
import requests
import io

st.set_page_config(layout="wide")
st.header("Book Recommendation System")
st.markdown('''
            ### This site uses collaborative filtering to provide personalized book suggestions.
            ### Additionally, we showcase a list of the top 50 books for all users.
            ''')

def load_books_from_drive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    return pkl.load(io.BytesIO(response.content))

# importing models 

popular = pkl.load(open('popular.pkl', 'rb'))
# books = pkl.load(open('./books.pkl', 'rb'))
books = load_books_from_drive("1X4kf6vaJ9OJ7UD5QRWLL1H5U8hoxKcNl")
pt = pkl.load(open('pt.pkl', 'rb'))
similarityScores = pkl.load(open('similarityScores.pkl','rb'))

# Top 50 books

st.sidebar.title("Top 50 Books")

if st.sidebar.button("Show"):
    colsPerRow = 5
    numRows = 10
    for row in range(numRows):
        cols = st.columns(colsPerRow)
        for col in range(colsPerRow):
            bookIndex = row * colsPerRow + col
            if bookIndex < len(popular):
                with cols[col]:
                    st.image(popular.iloc[bookIndex]['Image-URL-M'])
                    st.text(popular.iloc[bookIndex]['Book-Title'])
                    st.text(popular.iloc[bookIndex]['Book-Author'])


# Similar recommendation 

def recommend(book_name):
    if book_name not in pt.index:
        return f"Oops!'{book_name}' not found in the dataset or Check the case sensitivity."
    index = np.where(pt.index == book_name)[0][0]
    similarItems = sorted(list(enumerate(similarityScores[index])), key=lambda x: x[1], reverse=True)[1:6]
    # empty list for populate with book infos
    # Book auhor Book-Title, IMG-URL
    data = []
    for i in similarItems:
        item=[]
        tempDF = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(tempDF.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(tempDF.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(tempDF.drop_duplicates('Book-Title')['Image-URL-M'].values))
        data.append(item)
    return data

bookLists = pt.index.values # returns books list

st.sidebar.title("Similar Book Suggestions")
# drop down
selectedBook = st.sidebar.selectbox("Select a book from the dropdown", bookLists)

if st.sidebar.button("Recommend"):
    bookRecommends = recommend(selectedBook)
    cols = st.columns(5)
    for colIndex in range(5):
        with cols[colIndex]:
            st.image(bookRecommends[colIndex][2])
            st.text(bookRecommends[colIndex][0])
            st.text(bookRecommends[colIndex][1])


# validate

# sBooks = pd.read_csv('./data/Books.csv')
# sRatings = pd.read_csv('./data/Ratings.csv')
# sUsers = pd.read_csv('./data/Users.csv')

# st.sidebar.title("Data Used")

# if st.sidebar.button("Show Data"):
#     st.subheader("Books")
#     st.dataframe(sBooks)
#     st.subheader("Ratings")
#     st.dataframe(sRatings)
#     st.subheader("Users")
#     st.dataframe(sUsers)


st.markdown("""<hr style="border: 0.5px solid #ccc;" />
            <div style="text-align: center; color: gray;">
            Developed by Lijo Joseph. Credits: NXTwave
            </div>
            """, unsafe_allow_html=True)
