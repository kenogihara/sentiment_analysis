# Trip Advisor Sentiment Analysis

![Trip Advisor](.assets/tripadvisor-logo-circle-owl-icon-black-green-858x858.png)

By Ken Ogihara

## Introduction
Hotels play a huge role in the traveling industry. Hotels can make or break a traveler's experience and must allocate their resources efficiently to maintain competition. 

The dataset is taken from Kaggle and contains over 20,000 reviews and their corresponding rating. In this project, I use Natural Language Tool Kit and Sentiment Intensity Analyzer to determine the sentiment behind each hotel review. I cover questions such as, "What specific attributes make a good hotel and a bad one? In what ways could hotels improve their services based on customer reviews?" Sentiment analysis is a great way for hotels to understand their strengths and weaknesses. It provides a way for not just hotels but also businesses to perfect their strengths and to see where they fall short.

This dataset contains 2 columns:

| Column Name | Description                                                   |
|-------------|---------------------------------------------------------------|
| Review      | Customer review represented as a string        |
| Rating      | Customer's rating of the hotel, on a 1-5 scale.      |

The first two rows of the dataset is shown below:

```py
reviews = pd.read_csv("/Users/kenogihara/Desktop/ALL_PROJECTS/sentiment_analysis/tripadvisor_hotel_reviews.csv")
print(reviews.head(2))
```

| Review                                                                                                                         | Rating |
|--------------------------------------------------------------------------------------------------------------------------------|--------|
| nice hotel expensive parking got good deal stay hotel anniversary, arrived late evening took advice from previous reviews...    | 4      |
| ok nothing special charge diamond member hilton decided chain shot 20th anniversary seattle, start booked suite paid extra...   | 2      |


## Exploratory Data Analysis

Univariate plot that shows the distribution of ratings.

<iframe
  src="assets/plot1.html"
  width="700"
  height="500"
  frameborder="0"
></iframe>

## Tokenization

1. **Separate all reviews from texts to words** I used nltk's word_tokenize function to represent each review into words separated by commas.

```py
reviews["tokenized"] = reviews["Review"].apply(nltk.word_tokenize)
```

2. **Filter every review by removing stopwords** I used lambda function and nltk's corpus of stopwords and punctuation from the string module to remove words that do not contribute to the sentiment of a review.

```py
stopwords = nltk.corpus.stopwords.words("english")
stopwords.extend(["n't", "did"])
punctuation = set(string.punctuation)

reviews["tokenized"] = reviews["tokenized"].apply(
    lambda review: [word for word in review if word not in stopwords and word not in punctuation])
  ```

The dataframe now has a third column that contains the tokenized version:

| Review                                                                                                                                  | Rating | Tokenized                                                                                                                                                                                                                                            |
|-----------------------------------------------------------------------------------------------------------------------------------------|--------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| nice hotel expensive parking got good deal stay hotel anniversary, arrived late evening took advice from previous reviews did valet...   | 4      | [nice, hotel, expensive, parking, got, good, deal, stay, hotel, anniversary, arrived, late, evening, took, advice, previous, reviews, did, valet]                                                                                                     |
| ok nothing special charge diamond member hilton decided chain shot 20th anniversary seattle, start booked suite paid extra website...   | 2      | [ok, nothing, special, charge, diamond, member, hilton, decided, chain, shot, 20th, anniversary, seattle, start, booked, suite, paid, extra, website]                                                                                                 |
  
## Frequency Distribution and Word Cloud

1. Now I want to