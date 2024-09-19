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
| Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Rating |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------:|
| Nice hotel, but expensive parking. Got a good deal to stay at the hotel for an anniversary. Arrived late in the evening and took advice from previous reviews to use valet parking. Check-in was quick and easy. A little disappointed with the non-existent view. The room was clean and of nice size, and the bed was comfortable, although I woke up with a stiff neck from the high pillows. The room was not soundproof, as we heard music from the next room and loud bangs from doors opening and closing. We could also hear people talking in the hallway, which may have been due to noisy neighbors. The Aveda bath products were nice. We didn’t take advantage of the goldfish option since we were staying for a short time. Location was great, within walking distance of shopping. Overall, a nice experience, though paying $40 per night for parking was frustrating. | 4 |
| Ok, nothing special. As a Hilton Diamond member, I decided to give this chain a shot for my 20th anniversary in Seattle. I booked a suite and paid extra, but what I got was a standard hotel room with a bedroom and bathroom. I showed the printed reservation to the desk, which described amenities like a TV and couch, but the clerk apologized and said they had mixed up the suites on the website. They offered free breakfast as compensation, which was disappointing. Embassy Suites has a real suite, with a sitting room and separate bedroom, unlike what Kimpton calls a suite. During our 5-day stay, they didn’t correct their false advertising. I emailed the hotel with no response. The staff ranged from indifferent to unhelpful. When I asked for breakfast spots, they claimed there were none nearby, but one of Seattle’s best spots was only half a block away! Upon arrival, the bellman was too busy chatting on his phone to help with our bags. The room view was poor, looking out onto an alley and a high-rise building. Housekeeping was decent but unimpressive. Overall, this hotel had super high rates for what felt like a basic business hotel, and there are better chain hotels in Seattle. | 2 |


## Exploratory Data Analysis

Univariate plot that shows the distribution of ratings.

<iframe
  src="assets/plot1.html"
  width="700"
  height="500"
  frameborder="0"
></iframe>

