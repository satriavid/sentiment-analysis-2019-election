# 2019 Indonesian Presidential Election Tweet Sentiment Analysis

This project analyzes the sentiment of tweets related to the 2019 Indonesian Presidential Election. Using various machine learning techniques and models, we explored the general sentiment towards the presidential candidates and identified key patterns and topics within the discourse.

This project originally served as a team project (NLP B) in AI For Indonesia Bootcamp Batch 4. Team members that worked in this project including myself (Satriavi Dananjaya), Roby Koeswojo, Reinaldo Rafael, Rijal Abdulhakim, Muhammad Yatsrib. Act as the mentor for the project is Ferianda Satya.

## Overview

The aim of this project is to understand public sentiment on social media regarding the 2019 Indonesian Presidential Election. We employed several natural language processing (NLP) techniques and machine learning models, including Random Forest and LSTM, and compared these with state-of-the-art models like BERT and GPT-4 Turbo.

## Usage

To use this project simply run the iPython Notebook in Google Collab.
You may need to gain an API key yourself if you wish to run the benchmark test on GPT 4 Turbo Preview.

## Data Preparation

The dataset comprises tweets collected around the time of the 2019 Indonesian Presidential Election. Each tweet is labeled with sentiment: positive, neutral, or negative.

1. **Loading and Exploring Data**: The initial step involves loading the tweet data and conducting a preliminary analysis.
2. **Data Cleaning**: We preprocess the data, removing URLs, hashtags, and normalizing Indonesian slang words.
3. **Splitting Data**: The dataset is split into training and testing sets to prepare for model training.

## Sentiment Analysis

We conducted sentiment analysis using multiple approaches:

1. **Random Forest with TF-IDF**: Implemented Random Forest classifiers using TF-IDF vectors of the tweets.
2. **Random Forest with Word2Vec**: Applied Word2Vec embeddings with Random Forest to classify sentiments.
3. **LSTM Networks**: Utilized LSTM networks with both TF-IDF vectors and Word2Vec embeddings.
4. **Benchmark Models**: Compared our models against pretrained models such as BERT, Indonesian-roberta-base-sentiment-classifier, NusaX-senti, and GPT-4 Turbo Preview.

## Findings and Conclusion

Our analysis revealed that:
- The sentiment towards the presidential candidates was relatively balanced.
- The Random Forest model with Word2Vec embeddings performed comparably to the state-of-the-art BERT model, achieving a validation accuracy close to 63.63%.
- Pretrained models specifically tailored to Indonesian text sentiment analysis showed promising results but varied in performance based on the training data's relevance to our specific use case.
- General-purpose models like GPT-4 Turbo Preview, while powerful, did not outperform task-specific models, underscoring the importance of domain-specific training data.

## Future Work

Further improvements could include:
- Exploring deeper NLP techniques and fine-tuning models for better accuracy.
- Expanding the dataset to include more diverse sentiments and topics.
- Implementing ensemble methods to combine the strengths of various models.

## License

This project is open source and available under the MIT License.
