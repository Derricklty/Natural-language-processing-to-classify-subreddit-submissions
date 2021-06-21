# Project 3 - Classifying subreddit submission from Google Cloud and AWS

### Problem Statement
You are a junior data scientist who was recently hired by Google to analyse big data and improve its standing and product offerings. As Cloud technology is fast arising and adopted by many companies, you decided to examine the sentiments of Google Cloud vis-a-vis that of Amazon Web Services (i.e. Google's fierce competitor in the market). To do so, you wanted to use a set of subreddit submissions prepared by your predecessor who had recently left Google for greener pastures. Unfortunately, upon inspection of the dataset, you realised that he had accidentally left out the identification tagging column. This makes it difficult to accurately classify the subreddits into Google Cloud and AWS. In order to salvage the situation and use the dataset, you would thus have to first build several Natural Language Processing (NLP) models to attempt to classify subreddit submissions (from r/googlecloud and r/aws) correctly. Following which, we will choose the best performing model using the accuracy score(i.e. percentage of observations predicted correctly).


### Executive Summary

To classify our submissions into their respective subreddits, we have deployed the techniques of Logistic Regression and Multionomial Naive Bayes. For each of these two models, we considered both the CountVectorizer and TF-IDF Vectorizer. For the former, it turns each word into a token and counts the number of times it is featured in the whole corpus. On the other hand, the TF-IDF vectorizer turns each word into a token but returns a term frequency (i.e. number of time a token appears/ number of total tokens in a document). This value is then scaled by the log(ratio of documents that include the token). The attractiveness of the TF-IDF vectorizer is mainly because it gives more predictive powers to words that occur often in one document but not many documents. To help us identify the parameters that give us the best training and test scores, we deployed all our models together with Gridsearch.

Specifically, we conducted the four models as follows:  

Logistic regression with CountVectorizer and GridSearch
Logistic regression with TF-IDF Vectorizer and GridSearch
Multinomial Naive Bayes with CountVectorizer and GridSearch
Multinomial Naive Bayes with TF-IDF Vectorizer and GridSearch

Based on the four models that we have gridsearch-ed, we first look at their individual training and test scores to identify the one model that has the highest accuracy (i.e. correctly predicting the subreddit class). 

|Model| Training Score | Test Score |
|---|---|---|
|Logistic regression with CV and GS|0.9740|0.9272|
|Logistic Regression with TV and GS| 0.9798|0.9383|
|Multinomial Naive Bayes with CV and GS|0.9633|0.9235|
|Multinomial Naive Bayes with TV and GS|0.9662|0.9309|

While our four models have relatively similar training scores, the test score for our Logistic Regression with TV and GS is the highest. This suggests that the model is best (amongst all 4 models) at correctly predicting the submissions in the test set (i.e. unseen data). As we are ultimately interested in correctly classifying the submissions into their respective subreddits so that we can conduct our sentiment analysis (i.e. our problem statement), our choice model should be that of Logistic Regression with TV and GS (best params from GS are: (lr__C = 1; lr__max_iter = 1000; lr__penalty = l2; lr__random_state = 42; tvec__max_features = 3000; tvec__ngram_range = (1, 2); tvec__stop_words = english).

From our Logistic Regression with TV and GS model, we also obtained the following evaluation metrics: 

- Accuracy: 0.9383  
- Misclassification rate: 0.0617 
- Specificity: 0.8854 
- Sensitivity aka Recall: 0.9783
- Precision: 0.9185

The interpretations of these numbers are as such:
- Our model correctly predicts 93.83% of the test observations. Accordingly, our model inaccurately predicts 6.17% of the test observations.
- Among posts that are in r/aws, our model has 88.54% of the classified accurately
- Among posts that are in r/googlecloud, our model has 97.83% of them classified accurately
- Among posts that are predicted as r/googlecloud, we have 91.85% of them classified accurately. 


### Conclusion and Recommendations

**Conclusion**

In coming up with a model to classify submissions from r/googlecloud and r/aws, we found that our Logistic Regression with TF-IDF Vectorizer and GridSearch produces the best test accuracy score of 93.93 (i.e. correctly classifying 93.83% of test observations into their rightful subreddits). This phenomenon is not entirely surprising as there could be commonly used keywords that are specific to each subreddit, thereby making it easier for our model to learn and predict their subreddits.

As set out in our problem statement, we are actually also interested in the sentiments of Google Cloud vis-a-vis AWS. Using the VADER (Valence Aware Dictionary and sEntiment reasoner), we found that Google Cloud's compound score is 0.3698 vs 0.4073 for AWS. As such, AWS has slightly more positive sentiments. Interestingly, when looking at the negative scores, we also observe that AWS has marginally higher scores, thereby suggesting that AWS'sentiments are more polarised at both ends of the spectrum while Google Cloud's are slightly more netural.


**Recommendations**

Armed with our classifier and sentiment results, we could now:

Run our classifer on the dataset left behind by our predecessor to tag each submission into their respective subreddits
Analyse sentiment results based on that particular dataset
In the interim, some prelim recommendations we can make based on the set of sentiment results to improve Google's standing and product offerings are:

Improve features of Google Cloud to make product differentiation more explicit
Improve on after-sale/troubleshooting services for better customer experience which wil tend translate to higher customer satisfaction


**Further studies**

To ensure a more holistic understanding, we could consider further extensions such as:

Supplementing existing sentiment results with comments (and not just submissions)
Analysing overall subreddits between Google and its competitors (i.e. r/google vs r/competitors)
Analysing other cloud-related subreddits (e.g. Microsoft Azure)
Analysing subreddits from other product offerings (e.g. Google Home/Pixel)


### Data Sources 

Google Cloud Subreddit: https://www.reddit.com/r/googlecloud/
AWS Subreddit: https://www.reddit.com/r/aws/