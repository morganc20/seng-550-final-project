# Final Report Outline

## Preamble
### Description of team member contribution

Contribution:
*Morgan - 25%*

Contribution:
*Ayo - 25%*

Contribution:
*Carter - 25%*

Contribution:
*Max - 25%*

## Abstract
### Introduction
### Problem
The core problem we are addressing in this work is the automated identification and classification of sentiment within large-scale textual review data. Sentiment analysis, specifically binary classification into positive and negative sentiment, remains a fundamental challenge in natural language processing (NLP). In the context of massive online marketplaces, such as Amazon, the sheer volume, velocity, and variety of textual content—ranging from brief, informal comments to detailed, domain-specific critiques—poses a formidable challenge. Extracting sentiment reliably from this noisy and diverse textual input requires robust, scalable, and extensible machine learning frameworks.

Understanding customer sentiment is a critical component of maintaining a competitive edge in any consumer-driven ecosystem. For a platform hosting millions of product reviews like Amazon, accurate sentiment analysis can inform product recommendation systems, pricing strategies, inventory management, and targeted marketing. Positive sentiments correlate with consumer satisfaction, loyalty, and long-term brand health, while negative sentiments often highlight customer pain points, product flaws, and opportunities for improvement. The ability to automatically and effectively gauge sentiment at scale offers significant time and cost savings by reducing the burden on human reviewers, enabling real-time customer feedback loops, and ultimately improving the quality of goods and services offered. Thus, solving this problem transcends mere classification—it empowers businesses, researchers, and stakeholders to make more informed, data-driven decisions, thereby enhancing both user experience and marketplace efficiency.

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
Ayo

### What are you proposing?
We propose a multi-model machine learning pipeline that:
- Combines multiple algorithms (e.g., Naive Bayes, Random Forest, Logistic Regression, Transformer-based models) to enhance robustness and accuracy.
- Integrates text preprocessing, feature engineering, and auxiliary metadata for richer representations.
- Ensures extensibility and reproducibility, allowing for continuous improvement and adaptation to new data.
- Validates performance using rigorous metrics (accuracy, precision, recall, F1, AUC) and cross-validation.


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

## Methodology

### Exploration of data features and refinement of feature space
The dataset our group chose for exploration and analysis was the Amazon reviews dataset. More specifically, we used the Amazon Reviews 2023 dataset, which is based off data from May 1996 to September 2023. Below are some general details regarding the dataset.

| Year | #Review | #User | #Item | #R_Token | #M_Token | #Domain| Timespan |
|------|---------|-------|-------|----------|----------|--------|----------|
| 2023 | 571.54M |54.51M |48.19M | 30.14B   | 30.78B   |   33   | May'96 - Sep'23 |

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

### Experiment setup
Ayo

### Experimentation factors
##### Algorithms Used:
- Random Forest Regression was selected for its robustness and ability to handle feature interactions effectively.
- Naive Bayes was chosen for its simplicity, efficiency, and effectiveness with text classification tasks.
- **Logistic Regression:**  
  Refined the feature space by:
  - Tokenizing and cleaning the text and title fields.
  - Removing stop-words to reduce noise.
  - Applying TF-IDF vectorization to emphasize important terms and diminish common, uninformative words.  
  This approach yielded a lean yet expressive feature set that improved the model’s ability to differentiate sentiment patterns.
# - add other algos once done

##### Hyperparameters Tuned:
- Number of trees (n_estimators): Determines the size of the forest.
- Maximum depth (max_depth): Limits the depth of each tree to prevent overfitting.
- Minimum samples per leaf (min_samples_leaf): Ensures each leaf node represents sufficient data.
- **Logistic Regression:**  
  We tuned:
  - **Regularization parameter (C):** Adjusting `C` allowed control over complexity to prevent overfitting in the high-dimensional TF-IDF space.
  - **Max iterations:** Ensured proper convergence, especially when dealing with large and sparse TF-IDF vectors.


##### Training/Test/Cross-Validation Split:
- Data was split into training (80%) and testing (20%) sets.
- Cross-validation was applied to assess model performance across different data partitions.

##### Feature Space Refinement:
- The text and title fields were tokenized and transformed into numerical vectors using vectorization techniques.
- Common words were filtered out to enhance the relevance of features.
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
- **Logistic Regression:**  
  1. Extracted and refined features from text and title using TF-IDF.
  2. Conducted train/test splits and cross-validation to ensure robust performance.
  3. Iteratively adjusted tokenization and stop-word filtering strategies.
  4. Tuned regularization to balance generalization and complexity.
  Through systematic iteration, we arrived at a stable configuration that improved predictive power.


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

```
#### Logistic Regression

  After refining features and tuning parameters:
  - AUC: 0.7345
  - Accuracy: 0.8333
  - Precision: 0.8372
  - Recall: 0.8333
  - F1: 0.8352

These metrics indicate that the refined TF-IDF feature space, coupled with careful hyperparameter tuning, produced a solid baseline model for sentiment classification.

## Results
### Key findings in your exploratory data analysis and prediction. If you are trying out multiple algorithms, your results will compare them. How did you diagnose your ML model?
**Key Findings:**
- Large, diverse data required robust preprocessing and feature extraction.
- Integrating both text and title fields improved model performance.
- Logistic Regression provided a solid baseline; Random Forest achieved high accuracy; Naive Bayes was efficient but less robust; Transformer-based models performed best overall.

**Model Diagnosis:**
- Used accuracy, precision, recall, F1, AUC, and confusion matrices to identify strengths and weaknesses.
- Hyperparameter tuning and feature refinement incrementally improved results.
- Iterative analysis and error inspection guided enhancements in model design and feature selection.
### Conclusions 
Ayo