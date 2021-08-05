# NLP

# Classification of proper nouns and emails into newsgroups using Perceptron and Multilayer Perceptron

We are asked to build two text classifier models the perceptron and MLP, for two different NLP tasks. The first task is to assign proper nouns to one of five classes. The data includes strings of proper nouns. The second task involves predicting the newsgroup that an email document belongs to. The data includes email headers in addition to the actual body. For the propernames dataset we experimented with n-gram character and word length features. For the newsgroup dataset we experimented with bag of words and word n-grams. We observed that the best performing model is the MLP with character N-grams for propernames with an accuracy of 85.7% and the MLP with the bag of words for the newsgroup with an accuracy of 81.7%.

# Finding similarity between two words using Word to Vector

In this assignment, we study famous embedding models to transform the English words to vectors to understand the similarity between two given words. We adopt the first 1M sentences for “1 Billion Word Language Model Benchmark” (1B) to train our model and evaluate the results by calculating the word similarity against human annotation. After trying various models and fine-tuning hyperparameters, we decided to use skip-gram model with negative resampling. Our best development score is 0.503.

# Part of Speech Tagging in Wall Street Journal Corpus

We are required to implement POS (Part of Speech) tagging. We use WSJ corpus data from the Penn Treebank. Our goal is to achieve a high F1-score in matching generated tags with the held-out tags on the test set. We focus on HMMs (Hidden Markov Models) with the Trellis structure, beam search, and the Viterbi algorithm. We experiment with bigram, trigram and four-gram Viterbi with various smoothing functions and beam search with different beam sizes. Our best results on the development set are 0.9618 for bigrams, 0.9687 on trigrams, and 0.9511, 0.9527, and 0.9525 for beam search with k=4,5,6 respectively. Our best F1 score on the test set is 0.9582 with viterbi trigram.

# Sequence to Sequence Modeling in Alchemy environment

We are required to implement a sequence to sequence model to map instructions to action in the Alchemy environment. Each example has a sequence of instruction which are mapped to action push and pop in an environment containing 7 beakers and 6 colors. Each beaker is treated as a stack. We experimented various versions of seq2seq models in terms of architecture by adding state information, attention and instruction history. We also used GRU and LSTM and experimented with various hyperparameters. The best score we get on dev set is 0.772 on instruction and 0.232 on interaction. And on test set is 0.44766 on interaction, using LSTM, with attention, world state, previous interaction history.

# Name Entity Recognition for Tweets

We are required to develop a name-entity recognizer for Twitter. Given a tweet, we need to identify sub-spans of words representing named entities with high accuracy and high speed.The data is noisy, simulating real world with emojis and slangs commonly used in conversational english. Transformer architecture based on multihead attention is the SOTA of many NLP tasks. I used pretrained transformer models trained on English tweets - BERTweet to finetune on the data provided. I tried different model architectures, training epochs, learning rates and inference methods. The best F1 score on dev set was 0.713 and test set was 0.674.