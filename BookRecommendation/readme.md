
# Book Recommendation System

A Streamlit web application that provides personalized book recommendations using collaborative filtering and showcases the top 50 popular books.
live: https://ljbookrec.streamlit.app/


## Features

- **Top 50 Books Section**: Displays the most popular books based on ratings
- **Personalized Recommendations**: Uses collaborative filtering to suggest similar books
- **User-Friendly Interface**: Simple dropdown selection for book recommendations
- **Responsive Design**: Adapts to different screen sizes

## Technologies Used

- Python
- Streamlit (for web interface)
- Pandas (for data manipulation)
- NumPy (for numerical operations)
- Scikit-learn (for cosine similarity calculation)
- Google Drive (for model storage)

## Dataset

The system uses the [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) from Kaggle, which contains:
- Book information (title, author, publication year, etc.)
- User ratings data
- Book cover image URLs

## How It Works

1. **Popular Books Section**:
   - Displays the top 50 books based on aggregated ratings
   - Shows book cover, title, and author information

2. **Recommendation Engine**:
   - Uses collaborative filtering with cosine similarity
   - When a user selects a book, the system finds the most similar books based on user rating patterns
   - Displays 5 recommended books with covers, titles, and authors

## Installation

To run this project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/21lj/streamlit.git
   cd streamlit
   ```


2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run BookRecommendation.py
   ```

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- scikit-learn
- requests

## Hosted Version

A live version of this application is available at [GitHub Pages](https://21lj.github.io/streamlit/).

## Credits

- Developed by Lijo Joseph
- Dataset provided by NXTwave
- Inspired by various book recommendation system projects




