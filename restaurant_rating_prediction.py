# -*- coding: utf-8 -*-
"""Restaurant Rating Prediction

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1QrJXKqk2Iqn49pp7iH1xkY9NFir8V2wk
"""

# Restaurant Rating Prediction- Task-1
# 1. Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 2. Load Dataset
df = pd.read_csv('/content/Dataset .csv')  # adjust path if needed

# 3. Data Preprocessing
# Drop irrelevant columns (e.g., restaurant name)
df.drop(['Restaurant Name'], axis=1, inplace=True, errors='ignore')

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Encode categorical variables
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# 4. Feature and Target Selection
X = df.drop('Aggregate rating', axis=1)
y = df['Aggregate rating']

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model Training
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Decision Tree Regression
dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X_train, y_train)

# 7. Evaluation Function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

# Evaluate both models
mse_lr, r2_lr = evaluate_model(lr, X_test, y_test)
mse_dtr, r2_dtr = evaluate_model(dtr, X_test, y_test)

print("Linear Regression - MSE:", mse_lr, ", R2 Score:", r2_lr)
print("Decision Tree - MSE:", mse_dtr, ", R2 Score:", r2_dtr)

# 8. Feature Importance (for Decision Tree)
feat_importances = pd.Series(dtr.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Influential Features")
plt.xlabel("Importance Score")
plt.show()

# 9. Save the model (optional)
import joblib
joblib.dump(dtr, '../model.pkl')

#task-2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import LabelEncoder
import joblib

try:
    df = pd.read_csv('/content/Dataset .csv')
    # Drop irrelevant columns
    df.drop(['Restaurant Name'], axis=1, inplace=True, errors='ignore')
    # Handle missing values
    df.fillna(method='ffill', inplace=True)
    # Encode categorical variables
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
except FileNotFoundError:
    print("Error: Dataset .csv not found. Please ensure the file path is correct.")
    df = pd.DataFrame() # Initialize an empty DataFrame to avoid errors later


if not df.empty:
    # 2. Determine Recommendation Criteria
    # 3. Implement Content-Based Filtering
    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer()

    # Fit and transform the 'Cuisines' column
    tfidf_matrix = tfidf.fit_transform(df['Cuisines'].astype(str))

    # Calculate the cosine similarity between restaurant cuisine profiles
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Create a mapping of restaurant index to the original index for easy lookup
    indices = pd.Series(df.index, index=df.index).drop_duplicates()

    def get_recommendations_from_index(index, cosine_sim=cosine_sim, df=df):
        """
        Generates restaurant recommendations based on cosine similarity of cuisine
        using the DataFrame index.
        """
        if index not in indices:
            print(f"Restaurant with index {index} not found.")
            return pd.Series()

        sim_scores = list(enumerate(cosine_sim[indices[index]]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:4]  # Get the top 3 most similar restaurants
        restaurant_indices = [i[0] for i in sim_scores]
        return df.iloc[restaurant_indices]

    # 4. Test the Recommendation System
    print("\nTesting the Recommendation System:")

    # Let's pick a sample restaurant index from your DataFrame
    sample_restaurant_index = 5  # Example index, replace with a valid index from your df

    if sample_restaurant_index in df.index:
        original_index = indices[sample_restaurant_index]
        recommendations_df = get_recommendations_from_index(sample_restaurant_index)
        print(f"Restaurants similar to the restaurant at index {sample_restaurant_index}:")
        print(recommendations_df[['Country Code', 'City', 'Cuisines', 'Average Cost for two', 'Currency', 'Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu']])
    else:
        print(f"Error: Restaurant with index {sample_restaurant_index} not found in the DataFrame.")



#task-4
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Check if you have the 'Latitude' and 'Longitude' columns in your DataFrame
if 'Latitude' in df.columns and 'Longitude' in df.columns and 'City' in df.columns:
    print("Performing geographical analysis...")

    # 1. Visualize restaurant distribution on a scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='Longitude', y='Latitude', hue='City')
    plt.title('Restaurant Distribution by Latitude and Longitude')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(title='City')
    plt.grid(True)
    plt.show()

    # 2. Analyze restaurant concentration by city
    city_counts = df['City'].value_counts().sort_values(ascending=False)
    print("\nRestaurant Concentration by City:")
    print(city_counts)

    plt.figure(figsize=(10, 6))
    city_counts.plot(kind='bar')
    plt.title('Number of Restaurants per City')
    plt.xlabel('City')
    plt.ylabel('Number of Restaurants')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # 3. Analyze average ratings by city
    average_ratings_city = df.groupby('City')['Aggregate rating'].mean().sort_values(ascending=False)
    print("\nAverage Ratings by City:")
    print(average_ratings_city)

    plt.figure(figsize=(10, 6))
    average_ratings_city.plot(kind='bar')
    plt.title('Average Restaurant Rating per City')
    plt.xlabel('City')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # 4. Analyze cuisines by city (top 5 cuisines per city)
    def top_cuisines_per_city(group):
        cuisine_counts = group['Cuisines'].value_counts().nlargest(5)
        return cuisine_counts

    top_cuisines = df.groupby('City').apply(top_cuisines_per_city)
    print("\nTop 5 Cuisines per City:")
    print(top_cuisines)

    # 5. Analyze average price range by city
    average_cost_city = df.groupby('City')['Average Cost for two'].mean().sort_values(ascending=False)
    print("\nAverage Cost for Two by City:")
    print(average_cost_city)

    plt.figure(figsize=(10, 6))
    average_cost_city.plot(kind='bar')
    plt.title('Average Cost for Two per City')
    plt.xlabel('City')
    plt.ylabel('Average Cost')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # 6. Identify interesting insights and patterns
    print("\nInteresting Insights and Patterns:")
    print("- The scatter plot shows the geographical spread of restaurants and potential clusters within cities.")
    print("- The bar chart of restaurant counts per city highlights the cities with the highest and lowest number of restaurants in the dataset.")
    print("- The average rating by city can indicate which cities have, on average, higher-rated dining experiences.")
    print("- The top cuisines per city reveal the culinary preferences or common restaurant types in different locations.")
    print("- The average cost for two by city provides insights into the general price levels of dining out in different areas.")
    print("- Further analysis could involve looking at the distribution of specific cuisines within cities or correlating location with ratings and price.")

else:
    print("Error: 'Latitude', 'Longitude', or 'City' columns not found in the DataFrame. Ensure these columns exist for geographical analysis.")