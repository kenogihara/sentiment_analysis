# Trip Advisor Sentiment Analysis

By Ken Ogihara

## Introduction
Hotels play a huge role in the traveling industry. Hotels can make or break a traveler's experience and must allocate their resources efficiently to maintain competition. 

The dataset is taken from Kaggle and contains over 20,000 reviews and their corresponding rating. In this project, I use Natural Language Tool Kit and Sentiment Intensity Analyzer to determine the sentiment behind each hotel review. I cover questions such as, "What specific attributes make a good hotel and a bad one? In what ways could hotels improve their services based on customer reviews?" Sentiment analysis is a great way for hotels to understand their strengths and weaknesses. It provides a way for not just hotels but also businesses to perfect their strengths and to see where they fall short.

This dataset contains 2 columns:

| Column Name | Description                                                   |
|-------------|---------------------------------------------------------------|
| Review      | Customer review represented as a string        |
| Rating      | Customer's rating of the hotel, on a 1-5 scale.      |

The first three rows of the dataset is shown below:

```py
reviews = pd.read_csv("/Users/kenogihara/Desktop/ALL_PROJECTS/sentiment_analysis/tripadvisor_hotel_reviews.csv")

print(reviews.head())
```

ok how about now

## Exploratory Data Analysis

Univariate plot that shows the distribution of ratings.

<iframe
  src="assets/plot1.html"
  width="700"
  height="500"
  frameborder="0"
></iframe>

