[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/mW4WPbr-)

# A06 - Deep Feelings

Implementation is in `A06 - Deep Feelings.ipynb`. Baseline is bag-of-words with CountVectorizer and LogisticRegression. Enhanced version adds stopword removal, lemmatization, and bigrams. Embeddings version averages spacy word vectors per tweet and uses the same classifier.

Results on the test set: baseline 0.6569, enhanced (all three) 0.6576, embeddings 0.5233. I also tried each change on its own — stopwords only 0.6501, bigrams only 0.6609, lemmatizer only 0.6569, stopwords + bigrams 0.6618.

## Analysis

1. Stopwords only did not beat the baseline. Why did this not help, and how could it be improved?

Removing english stopwords drops a lot of words that still carry sentiment in short tweets, like not, no, love, good, bad. On longer reviews that might matter less, but here the text is already tiny so stripping common words just removes signal. I'd probably keep negation words on a custom stopword list, or only use stopword removal together with bigrams so something like "not good" still shows up as a feature.

2. Lemmatizer only did not beat the baseline. Why did this not help, and how could it be improved?

Lemmatization mostly merges inflected forms (running -> run), but twitter text is messy anyway — slang, hashtags, typos — so a lot of tokens don't change much after lemmatizing. The vocabulary ends up basically the same as the baseline. Normalizing twitter-specific stuff first (u -> you, expanding contractions) might help more before lemmatizing, or pairing it with n-grams like the full enhanced model does.

3. Embeddings did not beat the baseline. Why did this not help, and how could it be improved?

Averaging word vectors into one vector per tweet smears out the signal when only one or two words actually matter for sentiment. A tweet that is mostly neutral with one word like "hate" or "love" loses that spike when everything gets averaged. The spacy model also isn't trained specifically for sentiment. Something like sentence-level embeddings or weighting words by tf-idf before averaging would probably work better than a plain mean.

4. Bigrams (and stopwords + bigrams) did beat the baseline. What was the magnitude of the increase?

Best run was stopwords + bigrams at 0.6618 vs baseline 0.6569, so about +0.005 accuracy or half a percentage point. Bigrams alone were 0.6609. Small gain but it was consistent. The full enhanced model with lemmatizer too was only 0.6576, so lemmatization didn't really add much on top.

5. For the improvements that worked, could the increase be due to anything besides the enhancement itself?

Maybe a little. Bigrams just create a bigger feature space, so the model has more patterns to latch onto — some of that could be noise rather than real sentiment cues. I didn't tune hyperparameters differently per model, and I used the same logistic regression setup everywhere, so the comparison is mostly fair. But the bump is small enough that I wouldn't overread it as a huge win.

## Reflection

See reflection.txt for LLM / code assist disclosure.
