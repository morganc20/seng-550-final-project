# Final Report Outline

## Preamble

### Description of team member contribution

Contribution:
_Morgan Chen (30116461) - 25%_

Contribution:
_Ayo Olabode (30114584) - 25%_

Contribution:
_Carter Boucher (30116690)- 25%_

Contribution:
_Maxwell Couture (30113939) - 25%_

## Abstract

### Introduction

### Problem

The core problem we are addressing in our final project is the automated identification and classification of sentiment within large-scale textual review data. Sentiment analysis, specifically binary classification into positive and negative sentiment, remains a fundamental challenge in natural language processing (NLP). In the context of massive online marketplaces, such as Amazon, the sheer volume, velocity, and variety of textual content—ranging from brief, informal comments to detailed, domain-specific critiques—poses a formidable challenge. Extracting sentiment reliably from this noisy and diverse textual input requires robust, scalable, and extensible machine learning frameworks.

Understanding customer sentiment is a critical component of maintaining a competitive edge in any consumer-driven ecosystem. For a platform hosting millions of product reviews like Amazon, accurate sentiment analysis can inform product recommendation systems, pricing strategies, inventory management, and targeted marketing. Positive sentiments correlate with consumer satisfaction, loyalty, and long-term brand health, while negative sentiments often highlight customer pain points, product flaws, and opportunities for improvement. The ability to automatically and effectively gauge sentiment at scale offers significant time and cost savings by reducing the burden on human reviewers, enabling real-time customer feedback loops, and ultimately improving the quality of goods and services offered. Thus, solving this problem is not only classification—it empowers businesses, researchers, and stakeholders to make more informed, data-driven decisions.

### What have others done in the space?

Sentiment analysis is a well-researched area within machine learning, particularly in natural language processing (NLP). Researchers have utilized various methodologies, including traditional machine learning models and deep learning approaches. Commonly used techniques include:

##### 1. Text Preprocessing:

- Techniques like tokenization, stop-word removal, stemming, and lemmatization are standard to clean and prepare textual data.

##### 2. Feature Extraction:

- Bag-of-Words (BoW) or TF-IDF (Term Frequency-Inverse Document Frequency) representations.
- Word embeddings, such as Word2Vec or GloVe, to capture semantic meaning in text.
- Pre-trained transformer-based models like BERT and GPT, which provide contextualized embeddings for superior performance.

##### 3. Machine Learning Models:

- Classical algorithms such as Naive Bayes, Support Vector Machines (SVM), and Logistic Regression for text classification.
- Ensemble methods like Random Forest and Gradient Boosting to improve prediction robustness.
- Deep learning architectures such as Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) for capturing complex patterns in text.

##### 4. Datasets:

- The Amazon Reviews dataset, Yelp reviews, and IMDB datasets are widely used benchmarks.

##### 5. Evaluation Metrics:

- Accuracy, precision, recall, F1 score, and AUC are used to measure the effectiveness of sentiment classification models.

Despite these advances, challenges such as overfitting, handling imbalanced datasets, and integrating auxiliary features for enhanced interpretability remain key areas of focus.

### What are some existing gaps that you seek to fill?

Within the machine learning space, existing gaps include over-reliance on single features for sentiment analysis, lack of integration/interprretability of auxiliary features within data and consistent reproducibility.

In terms of how our work seeks to solve the problem better and add to the existing solutions, we have implemented various changes such as text preprocessing, combination of multiple features to enrich sentiment prediction, creating robust frameworks for sentiment analysis/feature classification, allowing for extensibility etc. to build on top of and surpass existing solutions.

Our multi-algorithm solution is evaluated against rigorous performance metrics to understand its strengths, weaknesses and to ensure we are iteratively improving classification accuracy while ensuring our solution is reproducible and extendible.

### What are your data analysis questions

**What are the most frequent words, phrases, or topics within positive and negative reviews?**

**Purpose:** To identify key linguistic patterns and differentiate the vocabulary associated with positive and negative sentiment. This can inform feature engineering and provide insights into customer perspectives.

**Approach:** Use techniques like word clouds, term frequency analysis, and topic modeling (e.g., LDA) to extract prominent themes and words in each sentiment class.

**How balanced is the dataset in terms of positive and negative sentiments?**

**Purpose:** To determine whether there’s a class imbalance that could bias model training and evaluation. Imbalances might require resampling techniques like SMOTE or weighting in the loss function.

**Approach:** Perform an exploratory analysis to calculate the proportion of positive and negative reviews in the dataset. Visualize the distribution with histograms or pie charts.

**Which machine learning algorithms perform best for this dataset, and why?**

**Purpose:** To compare the performance of different algorithms and understand which one is most suitable for sentiment analysis in this context.

**Approach:** Experiment with models like logistic regression, support vector machines (SVM), random forest, and deep learning techniques. Evaluate and compare them using metrics such as accuracy, precision, recall, and F1-score on the validation/test set.

### What are you proposing?

We propose to build and test 4 different machine learning models for sentiment analysis on amazon reviews. Each model will integrate text preprocessing and feature engineering. We will validate the performance of each model using rigorous metrics including accuracy, precision, recall, F1 and AUC. At the end of this report we hope to determine the model most effective for sentiment analysis on Amazon reviews. We will use source our data from an Amazon S3 bucket and transfer the data to the model using Spark.

### Main Findings

# Everyone explain the findings of their model.

We are proposing a multi-model solution, where we each will select a machine learning classification model and compare them against each other to see which is the most effective algorithm.

Below are our main findings of each model:

#### Random Forest Regression

Random Forest Regression is a supervised ML algorithm that uses decision trees to predict target variables. Generally speaking, it involves selecting subsets of training data at random and making smaller tress from there. The smaller models are then combined to output a singular prediction value.

Pros:

- Not as prone to overfitting
- Computationally efficient
- Needs fewer fitting parameters

Cons:

- Computationally complex
- Not ideal for small datasets
- High memory usage

#### Naive Bayes

Naive Bayes is a probabilistic classifier that uses Bayes theorem to predict the probability of a given class. It assumes that the features are independent of one another.

Pros:

- Simple and easy to implement
- Efficient for large datasets
- Works well with categorical data

Cons:

- Strong independence assumptions
- Can be inaccurate if independence assumptions are violated
- Limited expressiveness

#### Logistic Regression

Logistic Regression is a supervised classification algorithm used to estimate the probability of a binary outcome. It predicts the likelihood that a given input belongs to one class versus another using the logistic function (sigmoid) to transform linear combinations of features into probabilities between 0 and 1.

**Pros:**

- Interpretable coefficients and model structure
- Fast and efficient training
- Performs well with high-dimensional data after proper regularization

**Cons:**

- Assumes a linear relationship between features and the log-odds of the outcome
- May underperform when relationships are highly non-linear
- Sensitive to imbalanced datasets without class-weight adjustments or data balancing

### BERT-Based Sentiment Classifier

The BERT-based sentiment classifier leverages a pre-trained BERT model to perform binary classification, predicting whether a given review is positive or negative.

**Pros:**

- High accuracy and robustness
- Context-aware understanding
- No feature engineering needed
- Adaptable to various tasks

**Cons:**

- Computationally intensive
- High memory usage
- Requires large datasets
- Long training time

## Methodology

### Exploration of data features and refinement of feature space

The dataset our group chose for exploration and analysis was the Amazon reviews dataset. More specifically, we used the Amazon Reviews 2023 dataset, which is based off data from May 1996 to September 2023. Below are some general details regarding the dataset.

| Year | #Review | #User  | #Item  | #R_Token | #M_Token | #Domain | Timespan        |
| ---- | ------- | ------ | ------ | -------- | -------- | ------- | --------------- |
| 2023 | 571.54M | 54.51M | 48.19M | 30.14B   | 30.78B   | 33      | May'96 - Sep'23 |

The dataset can be found and accessed at: https://amazon-reviews-2023.github.io

Additionally, each review follows the following object structure:

```
{
  "sort_timestamp": 1634275259292,
  "rating": 3.0,
  "helpful_votes": 0,
  "title": "Meh",
  "text": "These were lightweight and soft but much too small for my liking. I would have preferred two of these together to make one loc. For that reason I will not be repurchasing.",
  "images": [
    {
      "small_image_url": "https://m.media-amazon.com/images/I/81FN4c0VHzL._SL256_.jpg",
      "medium_image_url": "https://m.media-amazon.com/images/I/81FN4c0VHzL._SL800_.jpg",
      "large_image_url": "https://m.media-amazon.com/images/I/81FN4c0VHzL._SL1600_.jpg",
      "attachment_type": "IMAGE"
    }
  ],
  "asin": "B088SZDGXG",
  "verified_purchase": true,
  "parent_asin": "B08BBQ29N5",
  "user_id": "AEYORY2AVPMCPDV57CE337YU5LXA"
}
```

# Everyone explain how they refined feature space for their model

#### Random Forest Regression

In the Random Forest Regression model, the feature space was refined to off the text and title fields, and used the rating field as a binary target variable (label). With regard to the text and title fields, each field was tokenized, common words were filtered out of the data and the resulting words were mapped in numerical vectors based off of importance. The end results of both fields were then merged together and used in the model.

#### Naive Bayes

In the Naive Bayes model, the feature space was refined to include the title and text fields. The text and title fields were tokenized and transformed into numerical vectors using vectorization techniques. Common words were filtered out to enhance the relevance of features. The resulting feature vectors were merged to create the final input for the model.

#### Logistic Regression

In the Logistic Regression model, the feature space was refined to include both the `text` and `title` fields. The text and title fields were tokenized, stop words were removed and vectorized using TF-IDF after hashing with 10,000 features (5,000 feature hash for the title). The features `title` and `text` were combined using `VectorAssembler` into a single feature vector(`assembled_features`).

### BERT-Based Sentiment Classifier

In the BERT-based sentiment classifier, the feature space was refined by combining the text and title fields into a single input for the model. Tokenization was handled using the BERT tokenizer, which converts the text into subword tokens while retaining context. Common preprocessing tasks like filtering stop words were unnecessary due to BERT's ability to handle raw input effectively. The processed tokens were padded/truncated to a maximum length of 128 and then transformed into numerical input (input IDs and attention masks) that BERT can process. This refined feature space enabled the model to leverage contextual embeddings and semantic understanding for classification.

### Experiment Setup

The experiment involved training and evaluating four machine learning models: Random Forest Regression, Naive Bayes, Logistic Regression, and a BERT-based sentiment classifier. Each model was set up and trained using the same dataset to ensure consistency and comparability of results.

**Dataset:**  
The dataset consisted of user reviews with fields for the title, text, and rating. The rating field was used to create a binary target variable, with ratings ≥3 labeled as positive (1) and ratings < 3 as negative (0).

**Data Preparation:**

1. **Tokenization:** Text and title fields were tokenized for all models. BERT used its tokenizer, while traditional models used methods like TF-IDF and vectorization.
2. **Filtering:** Stop words and common words were removed for traditional models to improve feature relevance.
3. **Vectorization:** Text data was converted into numerical vectors. Traditional models used TF-IDF or similar techniques, while BERT used embeddings generated by its pre-trained transformer architecture.

**Model Training:**

1. **Random Forest Regression:** Trained using numerical vectors derived from tokenized and filtered text and title fields.
2. **Naive Bayes:** Trained on vectorized representations of tokenized and filtered text and title fields.
3. **Logistic Regression:** Used a hashed TF-IDF feature space with vector assemblage of title and text fields.
4. **BERT-Based Classifier:** Fine-tuned on the dataset using pre-trained BERT embeddings and a custom classification head.

**Evaluation Metrics:**  
All models were evaluated using common classification metrics:

- Accuracy
- Precision
- Recall
- F1 Score

The experiment was designed to compare model effectiveness based on these metrics, as well as efficiency in training and prediction. Each model's strengths and weaknesses were noted for comparative analysis.

### Experimentation factors

##### Algorithms Used:

- Random Forest Regression was selected for its robustness and ability to handle feature interactions effectively.
- Naive Bayes was chosen for its simplicity, efficiency, and effectiveness with text classification tasks.
- Logistic Regression was selected for it's simplicity and effectiveness with text classification tasks.

- BERT-Based Sentiment Classifier was selected for its state-of-the-art performance in natural language processing, ability to capture contextual relationships between words, and effectiveness in understanding complex textual data.

#### Hyperparameters Tuned:

**Random Forest**

- Number of trees (n_estimators): Determines the size of the forest.
- Maximum depth (max_depth): Limits the depth of each tree to prevent overfitting.
- Minimum samples per leaf (min_samples_leaf): Ensures each leaf node represents sufficient data.

**Logistic Regression:**  
 We tuned:

- **Regularization parameter (C):** Adjusting `C` allowed control over complexity to prevent overfitting in the high-dimensional TF-IDF space.
- **Max iterations:** Ensured proper convergence, especially when dealing with large and sparse TF-IDF vectors.

##### Training/Test/Cross-Validation Split:

- Data was split into training (80%) and testing (20%) sets.
- Cross-validation was applied to assess model performance across different data partitions.

##### Feature Space Refinement:

- The text and title fields were tokenized and transformed into numerical vectors using vectorization techniques.
- Common words (stop words) were filtered out to enhance the relevance of features.
- The resulting feature vectors were merged to create the final input for the model.
- The rating field was transformed into a binary classification label.

##### Evaluation Metrics:

- Accuracy, precision, recall, F1 score, and AUC were calculated to measure the model's effectiveness in sentiment classification.

##### Infrastructure:

- Cloud storage (Amazon S3) was used to manage the large-scale Amazon Reviews dataset. This enabled efficient loading and experimentation with high-volume data.

These factors collectively ensured a comprehensive evaluation of the machine learning algorithms and the refinement of the sentiment analysis pipeline.

### Experiment process

# Everyone does their model

#### Random Forest Regression

The experiment process for the Random Forest model entailed creating a binary target variable from the ratings field, processing the relevent feature columns (text and title), aggregating the data together, defining the classifier and pipeline, splitting the dataset for training/test/cross-validation, training the model, making predictions using the model and then finally evaluating the model's performance against specified metrics.

#### Naive Bayes

The experiment process for the Naive Bayes model involved tokenizing, vectorizing, and merging the text and title fields. The data was then split into training and testing sets. The model was trained on the training set and evaluated on the test set using accuracy, precision, recall, F1 score, and AUC metrics.

#### Logistic Regression

The experiment process for Logistic Regression involved, tokenizing, vectorizing, and merging the text and title fields. The data was then split ito train and test splits. The model was trained on the training set and evaluated on the test set using accuracy, precision, recall, F1 score, and AUC metrics.

### Performance metrics - accuracy, precision, recall, F-score etc.

# Everyone explains their model

#### Random Forest Regression

Below are the performance metrics of our Random Forest model:

```
Test AUC: 0.7386457473162675
Test Accuracy: 0.8917525773195877
Test Precision: 0.7952226591561271
Test Recall: 0.8917525773195877
Test F1 Score: 0.8407258630860418
```

#### Naive Bayes

Below are the performance metrics for the Naive Bayes mode:

```
Test AUC: 0.49456042340488093
Test Accuracy: 0.8434343434343434
Test Precision: 0.879016354016354
Test Recall: 0.8434343434343434
Test F1 Score: 0.8585264513630095
```

#### Logistic Regression

After refining features and tuning parameters:
```
- AUC: 0.7345
- Accuracy: 0.8333
- Precision: 0.8372
- Recall: 0.8333
- F1: 0.8352
```
These metrics indicate that the refined TF-IDF feature space, coupled with careful hyperparameter tuning, produced a solid baseline model for sentiment classification.

### BERT-Based Sentiment Classifier

```
- AUC: 0.9300
- Accuracy: 0.9300
- Precision: 0.9593
- Recall: 0.9593
- F1 Score: 0.9593
```

## Results

### Key findings in your exploratory data analysis and prediction. If you are trying out multiple algorithms, your results will compare them. How did you diagnose your ML model?

The key finding from our exploration of different models found that the Transformer-based model performed the best overall.

### Conclusions

The **BERT-Based Sentiment Classifier** outperformed other models, achieving the highest accuracy, precision, recall, and F1 score due to its ability to capture contextual relationships in text. However, it required more computational resources and longer training times.

Traditional models like **Random Forest**, **Naive Bayes**, and **Logistic Regression** provided solid baselines with faster training and lower complexity but fell short in performance compared to BERT.

This study highlights the trade-offs between simplicity and performance, with BERT being ideal for resource-rich environments and traditional models suitable for simpler use cases. Iterative testing and feature refinement were crucial in achieving optimal results.
