---
layout: post
title:      "Cleaning tweets for machine learning algorithms  "
date:       2021-02-01 21:53:43 -0500
permalink:  twitter_sentiment_analysis
---


<a href="https://imgur.com/0FZrIy7"><img src="https://i.imgur.com/0FZrIy7.png" title="source: imgur.com" /></a>

We all have strong emotions, directed to different brands. We like Apple but not Google, we like Pepsi but not Cola ... If asked we even have reasons to do so. We express them on social media and even though they are less important for the person who feels them, they can be priceless for the companies emotions are directed to.. 

The following work is part of my 4th Project with Flatiron School of Data Science to build a model that will predict if a tweet was negative, positive or neutral.

Using people to label data can be labor intensive and expensive.

A succesful model would be useful for companies looking to improve their customer service.

Negative tweets can be used to identify unhappy customers and try to understand what is the source of their unhappiness. 

Positive tweets can be used to showcase customer experience or for training purposes.

The dataset I used can be found  <a href="https://data.world/crowdflower/brands-and-product-emotions">here </a>.

> Contributors evaluated tweets about multiple brands and products. The crowd was asked if the tweet expressed positive, negative, or no emotion towards a brand and/or product. If some emotion was expressed they were also asked to say which brand or product was the target of that emotion. 

Before cleaning, the dataset is expected to be a csv file with three columns:
1. 	'Tweet_text' column contains the tweet text.
2. 	'Emotion_in_tweet_is_directed_at' column contains the product or service the tweet emotion is directed at.
3. 	'Is_there_an_emotion_directed_at_a_brand_or_product' column contains the emotion or the lack of emotion found in the tweet text.

Let's analyze the Tweet_text column, see what type of information it contains,  and decide what we want to keep, transform or get rid of?
	
`print(dataset['Tweet'][174])`

This is our test tweet. 

<a href="https://imgur.com/4uqxFbE"><img src="https://i.imgur.com/4uqxFbE.png" title="source: imgur.com" /></a>

We can see, we have: handles, emojis, punctuation, and stopwords that we need to take care.

Stopwords are considered to be unimportant words, like 'the' or 'it'. Eliminating these words allow applications to focus on the important words instead.

Let's handles emojis first:

You will need to import a helpful libraries like: re, numpy, pandas, matplotlib, and nltk.

I found this function that turns happy emojis into EMO_POS and unhappy emojis into EMO_NEG 

<a href="https://github.com/abdulfatir/twitter-sentiment-analysis/blob/master/code/preprocess.py">here </a>.

```

def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet
```


There are prefilled libraries of stopwords in nltk so let's import the list.


```
from nltk.corpus import stopwords
```

We will make a list with all the words we don't need, and we'll add punctuation signs to it.

```
STOPWORDS = stopwords.words('english')
STOPWORDS += list(string.punctuation) 
```


Turn the stopword list into a set and remove the word 'not' from it because we're also interested in catching negative emotions.

```
STOPWORDS = set(STOPWORDS)
STOPWORDS.remove("not")
```


Create a new column 'Clean_tweet' for storing tweet text after cleaning and turn any capital letter into lower case.

```
dataset['Clean_tweet'] = dataset['Tweet'].apply(lambda tweet: tweet.lower())
dataset.Clean_tweet[174]
```


We will turn emojis into either positive or negative emotion using the above function. 


```
dataset['Clean_tweet'] = dataset['Clean_tweet'].apply(lambda tweet: handle_emojis(tweet))
dataset.Clean_tweet[174] 
```

After running this piece of code our test tweet will look like this:

<a href="https://imgur.com/vdl4fUZ"><img src="https://i.imgur.com/vdl4fUZ.png" title="source: imgur.com" /></a>

Remove user handles starting with @.

```
dataset['Clean_tweet'] = dataset['Clean_tweet'].str.replace("@[\w]*","")
```

Remove special characters.

```
dataset['Clean_tweet'] = dataset['Clean_tweet'].str.replace("[^a-zA-Z' ]","")
```

Remove urls.

```
dataset['Clean_tweet'] = dataset['Clean_tweet'].replace(re.compile(r"((www\.[^\s]+)|(https?://[^\s]+))"), "")
```

Remove single characters.

```
dataset['Clean_tweet'] = dataset['Clean_tweet'].replace(re.compile(r"(^| ).( |$)"), " ")
```

Turn the text into a list of strings.

```
dataset['Clean_tweet'] = dataset['Clean_tweet'].str.split()
```
 
Remove remove unimportant words called stopwords. Check what our test tweet looks  after all the alterations we've performed.

```
dataset['Clean_tweet'] = dataset['Clean_tweet'].apply(lambda tweet: [word for word in tweet if word not in STOPWORDS])
dataset.Clean_tweet[174] 
```

<a href="https://imgur.com/ZrJtUhH"><img src="https://i.imgur.com/ZrJtUhH.png" title="source: imgur.com" /></a>

This is a function that will replace n't with not.

```
def expand_tweet(tweet):
    expanded_tweet = []
    for word in tweet:
        if re.search("n't", word):
            expanded_tweet.append(word.split("n't")[0])
            expanded_tweet.append("not")
        else:
            expanded_tweet.append(word)
    return expanded_tweet
```

We'll stemm and lemmatize next.
This is an example of what a Stemmer does vs a Lemmatizer.

<a href="https://imgur.com/lvNdvts"><img src="https://i.imgur.com/lvNdvts.png" title="source: imgur.com" /></a>

```
wordNetLemmatizer = WordNetLemmatizer()
porterStemmer = PorterStemmer()
```

Lemmanize words in tweet text.

```
dataset['Clean_tweet'] = dataset['Clean_tweet'].apply(lambda tweet: [wordNetLemmatizer.lemmatize(word) for word in tweet])
dataset.Clean_tweet[174]
```

Stemm words in tweet text.

```
dataset['Clean_tweet'] = dataset['Clean_tweet'].apply(lambda tweet: [porterStemmer.stem(word) for word in tweet])
dataset.Clean_tweet[174]
```

Turn the list of strings, now cleaned, back into tweets.

```
dataset['Clean_tweet'] = dataset['Clean_tweet'].apply(lambda tweet: ' '.join(tweet))
dataset.Clean_tweet[174]
```

Compare initial test tweet with the respective clean tweet.

```
print(dataset.Tweet[174])
print(dataset.Clean_tweet[174])
```

<a href="https://imgur.com/DZT3iIq"><img src="https://i.imgur.com/DZT3iIq.png" title="source: imgur.com" /></a>

Delete Tweet column since we have all the information we need stored in the 'Clean_tweet' column.

`cleaned_dataset = dataset.drop('Tweet', axis=1 )`

<a href="https://imgur.com/CeKYwHt"><img src="https://i.imgur.com/CeKYwHt.png" title="source: imgur.com" /></a>

To have fun we can separate positive tweets from negative tweets and find out what are  the most frequent words in both lists.

Create a list of all tweet texts that were labeled positive.

```
positive_tweets = []
for tweet in dataset_positive['Clean_tweet']:
    positive_tweets.append(tweet)
#print(positive_tweets)
```

Create bag of words for positive tweets. 
Instead of a list of tweets now we'll have a list of words.

```
positive_tweets_bag = ''.join([str(tweet) for tweet in dataset_positive['Clean_tweet']])
#print(positive_tweets_bag)
```

You can plot this using wordcloud and matplotlib.

wordcloud = WordCloud(stopwords = STOPWORDS, background_color = "white", max_words = 1000).generate(positive_tweets_bag)
plt.figure(figsize = (10, 6))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Most frequent words in positive tweets")

This is what we get from our bag of tweets.

<a href="https://imgur.com/BuPaXi9"><img src="https://i.imgur.com/BuPaXi9.png" title="source: imgur.com" /></a>




	












	 


