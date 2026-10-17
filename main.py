
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
import os

# ==========================================
# LOAD NLP MODEL AND VECTORIZER
# ==========================================

filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl', 'rb'))

# Cache movie similarity data
data = None
similarity_matrix = None


# ==========================================
# MOVIE SIMILARITY
# ==========================================

def create_similarity():
    movie_data = pd.read_csv('main_data.csv')

    cv = CountVectorizer()
    count_matrix = cv.fit_transform(movie_data['comb'])
    sim_matrix = cosine_similarity(count_matrix)

    return movie_data, sim_matrix


def rcmd(m):
    global data, similarity_matrix

    m = m.lower()

    if data is None or similarity_matrix is None:
        data, similarity_matrix = create_similarity()

    if m not in data['movie_title'].unique():
        return 'Sorry! try another movie name'

    i = data.loc[data['movie_title'] == m].index[0]

    lst = list(enumerate(similarity_matrix[i]))
    lst = sorted(lst, key=lambda x: x[1], reverse=True)
    lst = lst[1:11]

    return [
        data['movie_title'][item[0]]
        for item in lst
    ]


# ==========================================
# CONVERT STRING TO LIST
# ==========================================

def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["', '')
    my_list[-1] = my_list[-1].replace('"]', '')

    return my_list


# ==========================================
# MOVIE SEARCH SUGGESTIONS
# ==========================================

def get_suggestions():
    movie_data = pd.read_csv('main_data.csv')

    return list(
        movie_data['movie_title'].str.capitalize()
    )


# ==========================================
# FLASK APPLICATION
# ==========================================

app = Flask(__name__)


@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()

    return render_template(
        'home.html',
        suggestions=suggestions
    )


@app.route("/similarity", methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = rcmd(movie)

    if isinstance(rc, str):
        return rc

    return "---".join(rc)


# ==========================================
# MOVIE RECOMMENDATION ROUTE
# ==========================================

@app.route("/recommend", methods=["POST"])
def recommend():

    # Debug: confirm that the route is called
    print("\n========== RECOMMEND ROUTE CALLED ==========",
          flush=True)
    print("Request method:", request.method, flush=True)
    print("Form fields:", list(request.form.keys()), flush=True)

    # Get movie data from AJAX request
    title = request.form.get('title', '')
    cast_ids = request.form.get('cast_ids', '[]')
    cast_names = request.form.get('cast_names', '[]')
    cast_chars = request.form.get('cast_chars', '[]')
    cast_bdays = request.form.get('cast_bdays', '[]')
    cast_bios = request.form.get('cast_bios', '[]')
    cast_places = request.form.get('cast_places', '[]')
    cast_profiles = request.form.get('cast_profiles', '[]')

    imdb_id = request.form.get('imdb_id', '')
    poster = request.form.get('poster', '')
    genres = request.form.get('genres', '')
    overview = request.form.get('overview', '')
    vote_average = request.form.get('rating', '')
    vote_count = request.form.get('vote_count', '')
    release_date = request.form.get('release_date', '')
    runtime = request.form.get('runtime', '')
    status = request.form.get('status', '')

    rec_movies = request.form.get('rec_movies', '[]')
    rec_posters = request.form.get('rec_posters', '[]')

    print("Movie title:", title, flush=True)
    print("IMDb ID received:", imdb_id, flush=True)

    # Get autocomplete suggestions
    suggestions = get_suggestions()

    # Convert string values to lists
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)

    # Convert cast IDs to list
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[", "")
    cast_ids[-1] = cast_ids[-1].replace("]", "")

    # Clean cast biographies
    for i in range(len(cast_bios)):
        cast_bios[i] = (
            cast_bios[i]
            .replace(r'\n', '\n')
            .replace(r'\"', '"')
        )

    # Build movie poster dictionary
    movie_cards = {
        rec_posters[i]: rec_movies[i]
        for i in range(
            min(len(rec_posters), len(rec_movies))
        )
    }

    # Build cast dictionary
    casts = {
        cast_names[i]: [
            cast_ids[i],
            cast_chars[i],
            cast_profiles[i]
        ]
        for i in range(
            min(
                len(cast_names),
                len(cast_ids),
                len(cast_chars),
                len(cast_profiles)
            )
        )
    }

    # Build detailed cast dictionary
    cast_details = {
        cast_names[i]: [
            cast_ids[i],
            cast_profiles[i],
            cast_bdays[i],
            cast_places[i],
            cast_bios[i]
        ]
        for i in range(
            min(
                len(cast_names),
                len(cast_ids),
                len(cast_profiles),
                len(cast_bdays),
                len(cast_places),
                len(cast_bios)
            )
        )
    }

    # ==========================================
    # FETCH MOVIE REVIEWS FROM TMDB
    # ==========================================

    tmdb_api_key = os.environ.get("TMDB_API_KEY")
    movie_reviews = {}

    print("TMDB API key loaded:",
          bool(tmdb_api_key), flush=True)

    try:
        if not tmdb_api_key:
            raise ValueError(
                "TMDB_API_KEY is missing. Set it in "
                "PyCharm Run > Edit Configurations > "
                "Environment variables."
            )

        if not imdb_id or not imdb_id.startswith("tt"):
            raise ValueError(
                f"Invalid IMDb ID received: {imdb_id!r}"
            )

        # Step 1: Find TMDB movie ID using IMDb ID
        find_url = (
            f"https://api.themoviedb.org/3/find/{imdb_id}"
        )

        print("Calling TMDB Find API...", flush=True)

        find_response = requests.get(
            find_url,
            params={
                "api_key": tmdb_api_key,
                "external_source": "imdb_id"
            },
            timeout=15
        )

        print(
            "TMDB Find API status:",
            find_response.status_code,
            flush=True
        )

        find_response.raise_for_status()

        find_data = find_response.json()
        movie_results = find_data.get(
            "movie_results", []
        )

        print(
            "TMDB movie matches:",
            len(movie_results),
            flush=True
        )

        if not movie_results:
            print(
                "No TMDB movie match found for IMDb ID:",
                imdb_id,
                flush=True
            )

        else:
            tmdb_movie_id = movie_results[0]["id"]

            print(
                "TMDB movie ID:",
                tmdb_movie_id,
                flush=True
            )

            # Step 2: Fetch TMDB reviews
            reviews_url = (
                f"https://api.themoviedb.org/3/movie/"
                f"{tmdb_movie_id}/reviews"
            )

            print(
                "Calling TMDB Reviews API...",
                flush=True
            )

            reviews_response = requests.get(
                reviews_url,
                params={
                    "api_key": tmdb_api_key,
                    "language": "en-US",
                    "page": 1
                },
                timeout=15
            )

            print(
                "TMDB Reviews API status:",
                reviews_response.status_code,
                flush=True
            )

            reviews_response.raise_for_status()

            reviews_data = reviews_response.json()
            tmdb_reviews = reviews_data.get(
                "results", []
            )

            print(
                "Reviews returned by TMDB:",
                len(tmdb_reviews),
                flush=True
            )

            # Step 3: Run sentiment analysis
            reviews_list = []
            reviews_status = []

            for item in tmdb_reviews:
                review_text = item.get(
                    "content", ""
                ).strip()

                if not review_text:
                    continue

                try:
                    movie_vector = vectorizer.transform(
                        [review_text]
                    )

                    prediction = clf.predict(
                        movie_vector
                    )[0]

                    label = str(prediction).strip().lower()

                    # Assumes model labels 1/0 mean
                    # positive/negative respectively.
                    if label in (
                        "1", "positive", "good"
                    ):
                        sentiment = "Good"

                    elif label in (
                        "0", "negative", "bad"
                    ):
                        sentiment = "Bad"

                    else:
                        sentiment = "Unknown"

                    reviews_list.append(review_text)
                    reviews_status.append(sentiment)

                except Exception as sentiment_error:
                    print(
                        "Sentiment prediction error:",
                        repr(sentiment_error),
                        flush=True
                    )

            movie_reviews = dict(
                zip(reviews_list, reviews_status)
            )

            print(
                "Reviews analyzed:",
                len(reviews_list),
                flush=True
            )

    except requests.exceptions.HTTPError as error:
        print("TMDB HTTP ERROR:", error, flush=True)

        if error.response is not None:
            print(
                "HTTP status:",
                error.response.status_code,
                flush=True
            )
            print(
                "Response body:",
                error.response.text[:500],
                flush=True
            )

    except requests.exceptions.RequestException as error:
        print(
            "TMDB CONNECTION ERROR:",
            repr(error),
            flush=True
        )

    except (ValueError, KeyError) as error:
        print(
            "TMDB CONFIGURATION/DATA ERROR:",
            repr(error),
            flush=True
        )

    except Exception as error:
        print(
            "UNEXPECTED TMDB ERROR:",
            repr(error),
            flush=True
        )

    print(
        "Reviews passed to template:",
        len(movie_reviews),
        flush=True
    )

    print(
        "========== END TMDB DEBUG ==========\n",
        flush=True
    )

    # ==========================================
    # RENDER RECOMMENDATION PAGE
    # ==========================================

    return render_template(
        'recommend.html',
        title=title,
        poster=poster,
        overview=overview,
        vote_average=vote_average,
        vote_count=vote_count,
        release_date=release_date,
        runtime=runtime,
        status=status,
        genres=genres,
        movie_cards=movie_cards,
        reviews=movie_reviews,
        casts=casts,
        cast_details=cast_details
    )


if __name__ == '__main__':
    app.run(debug=True)
