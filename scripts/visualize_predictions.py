import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def visualize_recommender_metrics(pred_df, test_df, train_users):
    #Rating distribution of recommended hotels
    test_df_with_recs = test_df[test_df['user_id'].isin(pred_df['user_id'])]
    merged_df = test_df_with_recs.merge(pred_df.explode('recommended_hotels'), on='user_id')
    rating_dist = merged_df.merge(test_df[['user_id', 'hotel_id', 'rating']], 
                                   left_on=['user_id', 'recommended_hotels'], 
                                   right_on=['user_id', 'hotel_id'], 
                                   how='left')

    plt.figure(figsize=(10, 5))
    sns.histplot(rating_dist['rating'].dropna(), kde=True, bins=20)
    plt.xlabel("Rating of Recommended Hotels")
    plt.title("Distribution of Ratings for Top-5 Recommendations")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #Cold-start user distribution
    test_users = set(test_df['user_id'])
    new_users = test_users - set(train_users)

    labels = ['Seen Users', 'Cold-Start Users']
    sizes = [len(test_users) - len(new_users), len(new_users)]
    colors = ['lightblue', 'lightcoral']

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, shadow=True)
    plt.title("User Distribution in Test Set")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    #Per-user recommendation diversity (unique hotels in top-5)
    diversity = [len(set(recs)) for recs in pred_df['recommended_hotels'] if isinstance(recs, list)]
    plt.figure(figsize=(10, 5))
    sns.histplot(diversity, bins=range(1, 7), discrete=True)
    plt.xlabel("Unique Hotels in Top-5")
    plt.ylabel("Number of Users")
    plt.title("Diversity in Recommendations (Top-5)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #Hotel popularity in top-5 recs
    all_hotels = list(itertools.chain.from_iterable(pred_df['recommended_hotels']))
    hotel_counts = Counter(all_hotels).most_common(15)
    hotel_ids, counts = zip(*hotel_counts)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(hotel_ids), y=list(counts), palette="coolwarm")
    plt.xticks(rotation=45)
    plt.xlabel("Hotel ID")
    plt.ylabel("Count in Top-5 Recs")
    plt.title("Most Frequently Recommended Hotels")
    plt.tight_layout()
    plt.show()
