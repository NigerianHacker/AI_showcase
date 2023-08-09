import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('IMDB.csv')
df = df.sample(frac=1).reset_index(drop=True)
df = df.head(int(len(df)*(20/100)))

reviews = df['review'].values
sentiments = df['sentiment'].values

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

parameters = {
    'tfidf__max_features': [100, 1000, 10000, 100000],
    'clf__C': [0.1,0.5,1,5]
}

grid_search = GridSearchCV(pipeline, parameters, cv=5, scoring="accuracy")
grid_search.fit(reviews, sentiments)

# Extract the results
results = grid_search.cv_results_
print(results)
mean_scores = results['mean_test_score']

# Reshape the mean scores array to match the parameter combinations
mean_scores = mean_scores.reshape(len(parameters['tfidf__max_features']), len(parameters['clf__C']))

# Plot the average accuracy scores for different max_features values
max_features_values = parameters['tfidf__max_features']
C_values = parameters['clf__C']

for i, max_features in enumerate(max_features_values):
    plt.plot(C_values, mean_scores[i], label=f"max_features={max_features}")

plt.xlabel('C')
plt.ylabel('Average Accuracy Score')
plt.title('C vs Average Accuracy Scores for Different max_features')
plt.legend()
plt.show()
