
<img src="tripadvisor-logo-circle-owl-icon-black-green-858x858.png" alt="Trip Advisor" width="200"/>

By Ken Ogihara

## Introduction
Hotels play a huge role in the traveling industry. Hotels can make or break a traveler's experience and must allocate their resources efficiently to maintain competition. 

The dataset is taken from Kaggle and contains over 20,000 reviews and their corresponding rating. In this project, I use Natural Language Tool Kit and Sentiment Intensity Analyzer to determine the sentiment behind each hotel review. I cover questions such as, "What specific attributes make a good hotel and a bad one? In what ways could hotels improve their services based on customer reviews?" Sentiment analysis is a great way for hotels to understand their strengths and weaknesses. It provides a way for not just hotels but also businesses to perfect their strengths and to see where they fall short.

The following are the relevant libraries and packages I will work with throughout the project.

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
import string
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

nltk.download([
    "names",
    "stopwords",
    "wordnet",
    "vader_lexicon",
])
```

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
  
## Frequency Distribution, Lemmatizer, and Word Cloud

To analyze a frequency distribution of all words, we must store all the words in one long string. The process is simple:

- **Step 1:** Store all words in a variable called, `all_words`.

```py
all_words = " ".join([word for token_list in reviews["tokenized"] for word in token_list])
```

- **Step 2:** Tokenize all the words in the string and store it in `tokenized_all_words`.

```py
tokenized_all_words = nltk.tokenize.word_tokenize(all_words)
```

- **Step 3:** Create an instance of a frequency distribution from the list of `tokenized_all_words` and apply the instance to the tokenized column using lambda function to filter out all words that occur only *once* in the entire dataset.

```py
frequency_dist = FreqDist(tokenized_all_words)

reviews["fdist"] = reviews["tokenized"].apply(lambda text: " ".join([word for word in text if frequency_dist[word] > 1]))
```

Words that rarely appear may be less informative or relevant to the overall sentiment of the reviews. Filtering out these words reduces noise.

#### Lemmatizer

**Step 4:** Create an instance of a word net lemmatizer and use apply function to tokenized frequency distribution column to transform all words to their base form. 

```py
wordnet_lem = WordNetLemmatizer()
reviews["fdist_tokenized"] = reviews["fdist"].apply(nltk.word_tokenize)
reviews["lemmatized"] = reviews["fdist_tokenized"].apply(lambda tokens: " ".join(wordnet_lem.lemmatize(token) for token in tokens))
```

Next, I check if each column are equal to each other to ensure that the function actually worked.

```py
reviews["is_equal"] = (reviews["fdist"] == reviews["lemmatized"])
print(reviews["is_equal"] = (reviews["fdist"] == reviews["lemmatized"])
)
```

| Value  | Count |
|--------|-------|
| False  | 19921 |
| True   | 570   |

The table shows that 19,921 reviews are able to be lemmatized. In other words, these reviews contain words that are not in their base form.

#### Word Cloud

**Step 5:** In this section, I put all lemmatized reviews in a single string called, `all_words_lem` in order to generate a word cloud of the most frequently used words.

```py
all_words_lem = " ".join([word for word in reviews["lemmatized"]])
fp = '/Users/kenogihara/Desktop/ALL_PROJECTS/sentiment_analysis/assets/all_words_lem.txt'

with open(fp, 'w') as file:
    file.write(all_words_lem)

bookreader = sc.gen_stylecloud(
                          file_path = fp,
                          size = 1028,                         
                          icon_name = "fas fa-book-reader",
                          palette = 'colorbrewer.sequential.Blues_5',
                          background_color = 'white')
```
<iframe
  src="stylecloud.png"
  width="700"
  height="500"
  frameborder="0"
></iframe>



## Sentiment Intensity Analyzer
