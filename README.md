# **Netflix Movie Recommender & Sentiment Analysis System**

### **Overview**
The **Netflix Recommender System** is a high-performance machine learning web application that suggests movies based on user preferences. By utilizing **Content-Based Filtering** and **Cosine Similarity**, the system analyzes movie metadata (genres, cast, director, keywords) to discover contextually related films. 

The application also features real-time **Sentiment Analysis** on user reviews scraped from IMDB, providing a holistic view of each movie's reception.

---

## **🚀 Key Features**
- **Precision Recommendations**: Powered by Cosine Similarity to find movies with similar metadata "DNA".
- **Dynamic UI**: A premium, responsive interface with AJAX-powered search and interactive modals.
- **Sentiment Analysis**: Uses a Naive Bayes NLP model to classify IMDB reviews as "Good" or "Bad".
- **Live Metadata**: Fetches posters, cast bios, and movie details via TMDB API.
- **Resilient Engineering**: Built-in bypasses for ISP API blocks and IMDB bot protection.

---

## **🔄 End-to-End Workflow**
1. **User Input:** The user types a movie name into the search bar (with real-time autocomplete suggestions).
2. **Recommendation Generation:** The Flask backend takes the input, vectorizes it, calculates cosine similarity against 5000+ movies, and returns the top 10 most similar titles.
3. **Data Fetching:** For the top 10 movies, the app fetches live metadata, posters, and cast information from the TMDB API.
4. **Sentiment Analysis:** Simultaneously, the app scrapes real-time user reviews for the selected movie from IMDB and runs them through a trained Naive Bayes NLP model to classify them as "Good" or "Bad".
5. **Display:** The user is presented with a comprehensive dashboard showing movie details, cast profiles, top recommendations, and a sentiment breakdown of current reviews.

---

## **🛠 Tech Stack**
- **Frontend**: HTML5, Vanilla CSS, JavaScript, AJAX, Bootstrap 4
- **Backend**: Python, Flask
- **Data Science**: Pandas, Scikit-Learn (CountVectorizer, Cosine Similarity)
- **NLP & Scraping**: BeautifulSoup4, NLTK, urllib
- **APIs**: The Movie Database (TMDB) API V3

---

## **💻 Installation & Setup**

### **1. Clone the repository**
```bash
git clone https://github.com/KannappanS2005/CONTENT-BASED-MOVIE-RECOMMENDATION-SYSTEM-WITH-SENTIMENT-ANALYSIS-.git
cd CONTENT-BASED-MOVIE-RECOMMENDATION-SYSTEM-WITH-SENTIMENT-ANALYSIS-
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the Application**
```bash
python main.py
```
*The app will be available at `http://127.0.0.1:5000/`*

---

## **🧠 How It Works**

### **1. Recommendation Engine (Content-Based)**
We use **Cosine Similarity** to measure the relationship between movies. By converting metadata (actors, director, genres) into multi-dimensional vectors, we calculate the geometric distance between them.
- **Similarity = 1**: Movies are identical in metadata.
- **Similarity = 0**: No semantic overlap.

### **2. Sentiment Analysis**
The system scrapes the latest 20 reviews for a selected movie from IMDB. A pre-trained NLP model (`nlp_model.pkl`) then classifies these reviews to give the user a quick sentiment summary.

### **3. Engineering Solutions**
- **ISP Bypass**: Reroutes TMDB API requests through a secure CORS proxy to prevent "403 Forbidden" or "Connection Timeout" issues in restrictive networks.
- **Anti-Bot Header**: Injects browser fingerprints into scraping requests to prevent IMDB from blocking the review extractor.

---

## **📂 Project Structure**
```
.
├── datasets/            # Raw movie metadata
├── data_processing/     # Scripts for data cleaning and EDA
├── static/              # CSS, JavaScript, and UI assets
├── templates/           # HTML templates for Flask
├── main.py              # Flask application logic
├── main_data.csv        # Processed production dataset
├── nlp_model.pkl        # Sentiment analysis model
├── tranform.pkl         # Metadata transformer
├── Procfile             # Deployment configuration
└── requirements.txt     # Python dependencies
```

---

## **🤝 Contributing**
Contributions are welcome! If you find a bug or have a feature request, please open an issue or submit a pull request.

## **🔗 Author**
**Kannappan S**
---
*Made with ♥ for Movie Lovers.*

---

## **10. APPENDIX: EXTENDED DOCUMENTATION**

This section provides a deep technical dive into the repository's components, dataset structure, and backend logic for technical auditors and developers.

### **10.1 File & Folder Breakdown**

#### **Core Directories**
- **`/static`**: Contains the client-side presentation layer.
  - `style.css`: The "Netflix-style" dark theme UI definitions.
  - `recommend.js`: The central engine for asynchronous API calls to TMDB and DOM rendering.
  - `autocomplete.js`: Handles real-time search suggestions.
- **`/templates`**: Contains HTML templates rendered by Flask.
  - `home.html`: The landing page with the search interface.
  - `recommend.html`: The detailed dashboard showing results, cast, and sentiment analysis.
- **`/datasets`**: A repository of raw and intermediate data.
  - `main_data.csv`: The finalized dataset used by the recommendation engine.
  - `movie_metadata.csv`: The original IMDB/TMDB combined metadata (5000+ entries).
  - `reviews.txt`: A collection of historical movie reviews used for training.
- **`/data_processing`**: Archive of the data science workflow.
  - Contains Jupyter Notebooks (`FSDS_data_*.ipynb`) documenting the Data Extraction, Cleaning, and Feature Engineering phases.

#### **Core Files**
- **`main.py`**: The heart of the application. Handles routing, recommendation logic, and web scraping.
- **`nlp_model.pkl`**: A serialized Naive Bayes Classifier trained on movie reviews.
- **`tranform.pkl`**: A serialized TfidfVectorizer/CountVectorizer used to process text for the similarity engine.
- **`requirements.txt`**: List of all Python libraries (Flask, Pandas, etc.) required to run the project.

### **10.2 Dataset Specifications**
The system relies on a curated dataset of over 5,000 films.
- **Primary Features**: `movie_title`, `genres`, `director_name`, `actor_1_name`, `actor_2_name`, `actor_3_name`.
- **Feature Engineering**: A composite column `comb` was created by merging all descriptors into a single "meta-string" for high-accuracy vectorization.
- **Integrity**: The dataset was cleaned of duplicates, handled null values in actor names, and normalized title casing for seamless search matching.

### **10.3 Backend Implementation (`main.py`)**
Below is the complete source code for the Flask backend, illustrating the integration of ML models and the recommendation logic.

```python
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import bs4 as bs
import urllib.request
import pickle
import requests

# load the nlp model and tfidf vectorizer from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl','rb'))

def create_similarity():
    data = pd.read_csv('main_data.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data,similarity

def rcmd(m):
    m = m.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        return('Sorry! try another movie name')
    else:
        i = data.loc[data['movie_title']==m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11] # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l
    
# converting list of string to list (eg. "["abc","def"]" to ["abc","def"])

def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list


# to get suggestions of movies
def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())

# Flask API

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html',suggestions=suggestions)

@app.route("/similarity",methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = rcmd(movie)
    if type(rc)==type('string'):
        return rc
    else:
        m_str="---".join(rc)
        return m_str

@app.route("/recommend",methods=["POST"])
def recommend():
    # getting data from AJAX request
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    # get movie suggestions for auto complete
    suggestions = get_suggestions()

    # call the convert_to_list function for every string that needs to be converted to list
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)
    
    # convert string to list (eg. "[1,2,3]" to [1,2,3])
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[","")
    cast_ids[-1] = cast_ids[-1].replace("]","")
    
    # rendering the string to python string
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"','\"')
    
    # combining multiple lists as a dictionary which can be passed to the html file so that it can be processed easily and the order of information will be preserved
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}

    casts = {cast_names[i]:[cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}

    cast_details = {cast_names[i]:[cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

    # web scraping to get user reviews from IMDB site
    req = urllib.request.Request('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id), headers={'User-Agent': 'Mozilla/5.0'})
    sauce = urllib.request.urlopen(req).read()
    soup = bs.BeautifulSoup(sauce,'lxml')
    soup_result = soup.find_all("div",{"class":"text show-more__control"})

    reviews_list = [] # list of reviews
    reviews_status = [] # list of comments (good or bad)
    for reviews in soup_result:
        if reviews.string:
            reviews_list.append(reviews.string)
            # passing the review to our model
            movie_review_list = np.array([reviews.string])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = clf.predict(movie_vector)
            reviews_status.append('Good' if pred else 'Bad')

    # combining reviews and comments into a dictionary
    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}     

    # passing all the data to the html file
    return render_template('recommend.html',title=title,poster=poster,overview=overview,vote_average=vote_average,
        vote_count=vote_count,release_date=release_date,runtime=runtime,status=status,genres=genres,
        movie_cards=movie_cards,reviews=movie_reviews,casts=casts,cast_details=cast_details)

if __name__ == '__main__':
    app.run(debug=True)
```
