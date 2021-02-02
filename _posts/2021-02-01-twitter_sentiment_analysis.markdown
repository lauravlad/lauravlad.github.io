---
layout: post
title:      "Cleaning and vectorizing tweets "
date:       2021-02-01 21:53:43 -0500
permalink:  twitter_sentiment_analysis
---


<a href="https://imgur.com/0FZrIy7"><img src="https://i.imgur.com/0FZrIy7.png" title="source: imgur.com" /></a>

This work is part of my 4th Porject with Flatiron School of Data Science.

I had to build a model that will succesfuly predict if a tweet was negative, positive or neutral.
This model will save labor, therefore money in a company's journey to improve customer service therefore improving sales and customer retention.

  The dataset I used can be found 
	<a href="https://data.world/crowdflower/brands-and-product-emotions">here </a>:
	
	Before cleaning, the dataset is expected to be a csv  file with three columns:
         1.  *Tweet text* column contains tweet texts.
         2.  *Emotion in tweet is directed at* column contains a product or service tweet is referring to.
         3.  *Is there an emotion directed at a brand or product* column contains the emotion or lack of emotion in the tweet.
             Our goal here is to show how to clean and turn the tweet text column into data which classifiers will be able to digest and enjoy playing with. 
	After analyzing and cleaning the whole dataset we will go ahead and proceed to cleaning the tweet column.
	You can find more info on cleaning and analyzing this dataset  
	
	<a href="https://github.com/lauravlad/Twitter-Sentiment-Analysis-Apple-Google">here </a>.
	
	Analyze tweet column. Let's see what kind of information it contains. What do we want to keep, transform or get rid of? 
	
`print(dataset['Tweet'][174])`

This is our test tweet. 

<a href="https://imgur.com/4uqxFbE"><img src="https://i.imgur.com/4uqxFbE.png" title="source: imgur.com" /></a>

We can see we have handles, emojis, punctuation, and stopwords that we need to take care.
Stopwords are considered to be unimportant words, like 'the' or 'it'. Eliminating these words allow applications to focus on the important words instead.

Let's handles emojis first:

You will need to import a helpful library:


`import re`


This function turns happy emojis into EMO_POS and unhappy emojis into EMO_NEG

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


There are prefilled libraries of stopwords in nltk.


```
import nltk
from nltk.corpus import stopwords
```

We will make a list with all the words we don't need, and we'll add punctuation signs to it.

```
STOPWORDS = stopwords.words('english')
STOPWORDS += list(string.punctuation) 
```


Turn the stopword list into a set and remove the word 'not' from it because we're interested in catching negative emotions too.

```
STOPWORDS = set(STOPWORDS)
STOPWORDS.remove("not")
```


Create a new column 'Clean Tweet' for storing the tweet text after cleaning. and turn any capital letter into lower case.

```
dataset['Clean_tweet'] = dataset['Tweet'].apply(lambda tweet: tweet.lower())
dataset.Clean_tweet[174]
```


Turn emojis into either positive emotion or negative emotion using the above function. 


```
dataset['Clean_tweet'] = dataset['Clean_tweet'].apply(lambda tweet: handle_emojis(tweet))
dataset.Clean_tweet[174] 
```

After running this piece of code our test tweet will look like this:

<a href="https://imgur.com/vdl4fUZ"><img src="https://i.imgur.com/vdl4fUZ.png" title="source: imgur.com" /></a>

 


	












	 


