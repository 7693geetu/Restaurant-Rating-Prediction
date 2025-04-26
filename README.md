# Restaurant Rating Prediction and Recommendation System

This project encompasses several tasks related to restaurant data analysis, including rating prediction using machine learning, building a content-based restaurant recommendation system, and performing geographical analysis of restaurant distribution and characteristics.

## Overview

This notebook performs the following analyses on a restaurant dataset:

1.  **Restaurant Rating Prediction (Task 1):** Builds and evaluates machine learning models (Linear Regression and Decision Tree Regressor) to predict the aggregate rating of restaurants based on various features.
2.  **Content-Based Restaurant Recommendation System (Task 2):** Develops a recommendation system that suggests restaurants similar to a given restaurant based on the similarity of their cuisines using TF-IDF and cosine similarity.
3.  **Geographical Analysis (Task 4):** Explores the geographical distribution of restaurants, restaurant concentration per city, average ratings by city, popular cuisines per city, and average cost for two by city.

## Libraries Used

* pandas
* numpy
* matplotlib.pyplot
* seaborn
* scikit-learn (model\_selection, preprocessing, linear\_model, tree, metrics, feature\_extraction.text, metrics.pairwise)
* joblib
* warnings

## Data Source

The dataset used for this analysis is located at `/content/Dataset .csv`. **Note:** You might need to adjust the file path depending on where you have stored the dataset.

## Task 1: Restaurant Rating Prediction

### Data Preprocessing

* Irrelevant columns like 'Restaurant Name' are dropped.
* Missing values are handled using forward fill (`ffill`).
* Categorical features are encoded using Label Encoding.

### Feature and Target Selection

* Features (`X`) include all columns except 'Aggregate rating'.
* The target variable (`y`) is 'Aggregate rating'.

### Model Training and Evaluation

* The data is split into training and testing sets (80% train, 20% test) with a `random_state` of 42 for reproducibility.
* Two regression models are trained:
    * Linear Regression
    * Decision Tree Regressor
* Both models are evaluated using Mean Squared Error (MSE) and R-squared (R2) score on the test set.
* The evaluation results for both models are printed.

### Feature Importance (Decision Tree)

* For the Decision Tree Regressor, the importance of each feature in predicting the rating is calculated and visualized using a horizontal bar plot showing the top 10 influential features.

### Model Saving (Optional)

* The trained Decision Tree Regressor model is saved to a file named `model.pkl` using `joblib`.

## Task 2: Content-Based Restaurant Recommendation System

### Data Loading and Preprocessing

* The dataset is loaded and preprocessed similar to Task 1 (dropping 'Restaurant Name', handling missing values, and encoding categorical features).
* Error handling is included to manage the case where the dataset file is not found.

### Recommendation Logic

* **TF-IDF Vectorization:** The 'Cuisines' column is transformed into a TF-IDF matrix to represent the cuisine profiles of each restaurant.
* **Cosine Similarity:** The cosine similarity between all pairs of restaurant cuisine profiles is calculated.
* **Recommendation Function:** A function `get_recommendations_from_index` takes a restaurant index as input and returns the top 3 most similar restaurants based on their cuisine profiles.

### Testing the Recommendation System

* A sample restaurant index (e.g., 5) is used to demonstrate the recommendation system.
* The original details of the sample restaurant and the details (including 'Country Code', 'City', 'Cuisines', etc.) of the top 3 recommended restaurants are printed.

## Task 4: Geographical Analysis

### Prerequisites

* This task assumes the presence of 'Latitude', 'Longitude', and 'City' columns in the DataFrame.

### Analysis Performed

1.  **Restaurant Distribution on a Scatter Plot:** A scatter plot visualizes the distribution of restaurants based on their latitude and longitude, with different colors representing different cities.
2.  **Restaurant Concentration by City:** The number of restaurants in each city is calculated and displayed in a bar chart, showing the cities with the highest and lowest restaurant counts.
3.  **Average Ratings by City:** The average aggregate rating for restaurants in each city is calculated and visualized using a bar chart.
4.  **Top 5 Cuisines per City:** For each city, the top 5 most frequently listed cuisines are identified and printed.
5.  **Average Price Range by City:** The average 'Average Cost for two' for restaurants in each city is calculated and displayed in a bar chart.
6.  **Interesting Insights and Patterns:** A textual summary of potential insights observed from the geographical analysis is provided.

## Future Work

* Further exploration of other machine learning models for rating prediction.
* Incorporating more features into the recommendation system (e.g., cost, ratings).
* Developing interactive visualizations for geographical analysis.
* Deploying the trained rating prediction model and recommendation system.
* (The original notebook's Task 4 section ends abruptly, so further geographical analysis ideas could be added here, such as analyzing the relationship between location and rating, or identifying areas with specific cuisine concentrations.)
