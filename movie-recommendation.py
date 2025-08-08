import pandas as pd
from surprise import Dataset, SVD
import streamlit as st


# Load MovieLens dataset

data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()

# Load movie names from file
movie_df = pd.read_csv(
    'http://files.grouplens.org/datasets/movielens/ml-100k/u.item',
    sep='|',
    encoding='latin-1',
    header=None,
    usecols=[0, 1],
    names=['movie_id', 'title']
)
movie_map = dict(zip(movie_df['movie_id'].astype(str), movie_df['title']))


# Train SVD recommender

algo = SVD()
algo.fit(trainset)

# Streamlit UI

st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🍿", layout="centered")

st.title("🍿 Collaborative Filtering Movie Recommender")
st.markdown("Enter a user ID and I'll suggest movies you might love, based on similar users’ tastes.")

user_id = st.text_input("Enter User ID", "196")

if st.button("🎯 Get Recommendations"):
    try:
        # Get all items in training set
        all_items = set(trainset.all_items())
        all_raw_items = [trainset.to_raw_iid(i) for i in all_items]

        # Get already rated items
        rated_items = set([trainset.to_raw_iid(i) for (i, _) in trainset.ur[trainset.to_inner_uid(user_id)]])

        # Unrated items
        unrated_items = set(all_raw_items) - rated_items

        # Predict scores for unrated items
        recommendations = []
        for iid in unrated_items:
            pred = algo.predict(user_id, iid)
            recommendations.append((iid, pred.est))

        # Sort top 5
        top_recs = sorted(recommendations, key=lambda x: x[1], reverse=True)[:5]

        # Display results as table
        rec_df = pd.DataFrame(top_recs, columns=['Movie ID', 'Predicted Score'])
        rec_df['Title'] = rec_df['Movie ID'].map(movie_map)
        rec_df['Predicted Score'] = rec_df['Predicted Score'].apply(lambda x: f"{x:.2f} ⭐")

        st.success(f"Top recommendations for User {user_id}:")
        st.table(rec_df[['Title', 'Predicted Score']])

    except ValueError:
        st.error("❌ Invalid User ID. Try another number from the dataset.")

