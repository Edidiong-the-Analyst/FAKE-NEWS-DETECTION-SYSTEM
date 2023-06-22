**Fake News Detection System**

Fake news detection detects deliberately false or fake news, recent online social media content, etc. In recent years, due to the growth of online communication, fake news for various commercial and political purposes has emerged more and more widely in the online world. Through fake news, online social users are easily infected with fake news online, which has a huge impact on the offline community.

So it is very important to find fake news and real news.
One of the best ways is to use natural language processing (NLP) to analyze the words used in articles and identify patterns associated with fake news.

**Overview**
This project uses supervised machine learning to classify data. The first step in this classification problem is the data collection process, then prioritization, using custom options, then training and testing on the data, and finally running the classification (ML algorithm).

**Preprocessed Data** 
The dataset is preprocessed so machine learning algorithms can easily detect patterns. This phase uses Pandas to manage data and Natural Language Processing Tools (NLTK) to solve NLP problems such as stop word removal, tokenization, and lemmatization.

**FEATURE EXTRACTION**
To analyze text, data must be translated. Text is converted to integer or floating-point values ​​and sent to machine learning models. The vectorization method used in the given system is the language bag method using the TF-IDF vectorizer. Term Frequency-Reverse Document Frequency (TF-IDF) is a statistic designed to show the importance of terms in a document system. The TF-IDF formulation used in this study was taken from sklearn.

**Testing and Training Separation**
Requires two datasets to run on machine learning models. The data obtained were divided into two for training and testing purposes. 80% of the obtained data is used to train the model and 20% of the remaining data is used to evaluate the accuracy of the selected model. Machine learning algorithms are trained on a labeled dataset, meaning a model is trained to predict whether an article is true or false.

**Classification Model**
The classification system consists of training and validation (testing).
Six classification models were used to find the best one for this project. The first classifier uses the GridSearchCV library to find the best parameters for different algorithms. The TF-IDF vectorizer is used to convert data to vectors. This is then transferred to the model and the specified algorithms are used to find the best fit for the model.
• Slight Gradient Boost
• Decision Trees
• K Nearest Neighbors (K-NN)
• Random Forest Classifier
• Support Vector Classifier (SVC)
• Logistic Regression


**Evaluation 4 elements Accuracy, Precision, Recovery contains, and F1**
Some models have been shown to outperform others in different parameters. For this project, are the benchmarks used as the basis of the model's best performance accurate (evaluating the overall performance of the model in classifying fake news and real news) and predicting F1_Score (how to combine fact and reuse consensus) performance negatively/positively?

**Conclusion**
Fake news detection using machine learning algorithms has received a lot of attention in recent years due to growing concerns about the spread of misinformation on social media platforms. Detecting fake news using machine learning algorithms can be a good tool to detect misinformation on social media platforms. That's why human experts need to review the flags to make sure the ordering is correct.
The information used in this report was taken from the Kaggle website. The file contains 6335 news with 4 attributes id, name, text, and tag. The database contains 3164 samples classified as fake news and 3171 samples classified as real news. There is no missing data in the dataset. The data has been done before and seen before it was decided that the newspaper column or long article was not indicative of fake news.
This report includes Light Gradient boost, Random Forest, Logistic Regression, SVC, Decision Tree, and K-Nearest Neighbor (K-NN) with 93.8%, 91.2%, 92.2%, and 92.5% accuracy, respectively. developed six prediction models. 81.
4% and 60.3%, respectively. Light Gradient Boost achieved the highest accuracy, and recommendations are based not only on the accuracy of the score but also on the model's ability to accurately identify all fake news (don't forget to remember). This is more important than the overall accuracy of the model. Since the proposed model is considered sufficient, it can be used in practical applications.
