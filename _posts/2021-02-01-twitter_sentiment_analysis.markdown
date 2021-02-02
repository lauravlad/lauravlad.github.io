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

  The dataset I used can ce found here: 
	<a href="https://data.world/crowdflower/brands-and-product-emotions">here </a>
	
	Before cleaning, the dataset is expected to be a csv  file with three columns:
1.  'tweet_text ' column contains  tweet texts.
2.  'emotion_in_tweet_is_directed_at' column contains a product or service tweet is refering to.
3. ' is_there_an_emotion_directed_at_a_brand_or_product' column contains the emotion or lack of emotion in the tweet.
  
	Our goal here is to show how to clean and turn the tweet text column into data which classifiers will be able to digest and enjoy playing with. 
	After analysing and cleaning the whole dataset we will go ahead and proceed to cleaning the tweet column.
	You can find more info on cleaning and analyzing this dataset [https://github.com/lauravlad/Twitter-Sentiment-Analysis-Apple-Google](http://)
	
	<a href="https://github.com/lauravlad/Twitter-Sentiment-Analysis-Apple-Google">here </a>
	
	Analyse tweet column. Let's see what kind of information it contains. What do we want to keep and what do we want to get rid of? 
	
```print(dataset['Tweet'][174])```


	
	












	 


