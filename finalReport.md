# Final Report – SENG 550

## Preamble

### Description of Team Member Contribution
- **Morgan Chen (30116461)**: 25%  
- **Ayo Olabode (30114584)**: 25%  
- **Carter Boucher (30116690)**: 25%  
- **Maxwell Couture (30113939)**: 25%  

Each team member contributed equally.

### Declaration
We, the undersigned, declare that the above statement of contributions and the estimation of total contribution by each team member is true to the best of our knowledge and agreed upon by all group members.

**Signatures:**  
Morgan Chen _________MC__________  
Ayo Olabode _________AO___________  
Carter Boucher ________CB___________  
Maxwell Couture ______MC___________  

### Repository Link
Our code, excluding the dataset due to size and confidentiality constraints, is available at: [https://github.com/morganc20/seng-550-final-project](https://github.com/morganc20/seng-550-final-project)

---

## Abstract

This project addresses the automated sentiment classification of Amazon product reviews, focusing on accurately differentiating between favorable and unfavorable sentiment at scale. We present experiments with several machine learning algorithms—Logistic Regression, Random Forest, Naive Bayes, and a BERT-based deep learning model—using tokenized, preprocessed textual data from Amazon reviews. We explored textual features from both the review title and body. Our results show that while traditional machine learning models offer reasonable performance at lower computational costs, a fine-tuned BERT-based model provides the strongest predictive accuracy, precision, recall, and F1 score. This outcome highlights the trade-off between computational complexity and predictive performance, providing insights into model selection for large-scale industry applications.

---

## Introduction

### Problem
The core problem addressed is the automated classification of sentiment in large-scale Amazon product reviews. Online platforms generate massive textual content daily, making it challenging to manually identify patterns of satisfaction or dissatisfaction. Sentiment analysis streamlines this process, providing insights into consumer preferences, product quality, and marketplace trends.

### Importance
Accurate sentiment analysis informs data-driven decision-making. Positive reviews can guide recommendations and highlight successful product features, while negative sentiment pinpoints product shortcomings and areas for improvement. Automated solutions enable real-time feedback loops, reduced manual labor, and improved customer experiences at scale.

### What Have Others Done in this Space?
Previous work in sentiment analysis spans a broad spectrum of techniques:
- **Text Preprocessing:** Early efforts relied on bag-of-words approaches and simple text cleaning. More recent work employs sophisticated NLP pipelines, including lemmatization, entity recognition, and advanced tokenization.
- **Feature Extraction:** Traditional methods often use TF-IDF or Bag-of-Words, while more recent advancements leverage embeddings from Word2Vec, GloVe, and BERT.
- **Models and Approaches:**  
  - *Classical ML:* Naive Bayes, Logistic Regression, and SVM have been widely used due to their interpretability and efficiency.  
  - *Ensemble Methods:* Random Forest and Gradient Boosting Machines improved performance over single classifiers.  
  - *Deep Learning:* CNNs, RNNs, LSTMs, and Transformers (BERT) have pushed state-of-the-art accuracy using contextual embeddings and fine-tuning.
- **Evaluation Metrics:** Accuracy, precision, recall, F1, and AUC are standard metrics. Researchers often focus on F1 or AUC when class imbalances arise.

### Existing Gaps and Our Contribution
While much work has focused on single-feature inputs and single-model approaches, challenges remain:
- **Feature Integration:** Many approaches focus on review text alone. We incorporate both title and text, aiming to improve representation.
- **Comparative Analysis of Models:** Few studies offer a direct comparison among a variety of classical ML and advanced NLP models on the same dataset with the same preprocessing steps.
- **Resource vs. Performance Trade-off:** Many state-of-the-art models (like BERT) improve accuracy but at a higher computational cost. We provide a clear comparison to help practitioners choose models based on their resource constraints.

### Our Goals
We aim to:
1. Explore text and title features for enhanced sentiment classification.
2. Compare logistic regression, random forest, naive bayes, and a BERT-based model on a consistent dataset and pipeline.
3. Evaluate model performance using standard metrics and highlight trade-offs in accuracy, complexity, and inference time.

---

## Methodology

### Data and Preprocessing
- **Data Source:** Amazon Reviews (2023 snapshot). Fields include timestamp, rating, title, text, and additional metadata.
- **Preprocessing Steps:**
  - Tokenization of text and titles
  - Removal of stop words
  - TF-IDF feature extraction for classical ML models
  - Integration of text and title features via vector concatenation
  - For BERT: using pretrained `bert-base-uncased` tokenizer and encoding both title and text together.

### Data Analysis Questions
1. **Frequent Words in Sentiments:** Identify distinguishing terms between positive and negative sentiments.  
   - **Approach:** Use frequency counts and TF-IDF to highlight top terms in each sentiment class.  
2. **Best Performing Algorithm:** Among Logistic Regression, Random Forest, Naive Bayes, and BERT, which performs best on key metrics?  
   - **Approach:** Train, tune, and evaluate all models on the same train/test splits and compare metrics.

### Experiment Setup
- **Training/Test Split:**  
  - 80% training set, 20% test set, stratified by sentiment label when possible.
- **Infrastructure:**  
  - Spark for distributed preprocessing and model training (for classical models).  
  - PyTorch and HuggingFace Transformers for BERT model training.  
  - Data accessed via Amazon S3.

### Experimentation Factors
- **Models:** Logistic Regression, Random Forest, Naive Bayes, and a BERT-based classifier.
- **Hyperparameters:**  
  - Logistic Regression: `maxIter=20`, `regParam=0.001`  
  - Random Forest: `numTrees=100`, `maxDepth=10`  
  - Naive Bayes: `smoothing=1.0`  
  - BERT: `bert-base-uncased`, `lr=2e-5`, `epochs=3`, `batch_size=16`
- **Performance Metrics:** Accuracy, precision, recall, F1-score, and AUC.

---

## Results

### Performance Metrics Comparison

| **Model**               | **AUC** | **Accuracy** | **Precision** | **Recall** | **F1 Score** |
|--------------------------|---------|--------------|---------------|------------|--------------|
| **Random Forest**        | 0.74    | 89%          | 80%           | 89%        | 0.84         |
| **Naive Bayes**          | 0.50    | 84%          | 88%           | 84%        | 0.86         |
| **Logistic Regression**  | 0.74    | 83%          | 84%           | 83%        | 0.84         |
| **BERT-Based Classifier**| 0.93    | 93%          | 96%           | 96%        | 0.96         |

(Note: Values are approximately those observed in the notebook. Slight variations may occur depending on random seeds and data splits.)

### Key Findings
- **Feature Integration:** Including both text and title slightly improved classical model performance over text-only features.
- **Model Comparison:**  
  - Naive Bayes and Logistic Regression were efficient and reasonably accurate but did not reach the top accuracy or AUC.  
  - Random Forest improved over Naive Bayes and Logistic Regression in terms of accuracy but not AUC.  
  - The BERT-based classifier provided the highest scores across all metrics, indicating that deep contextual embeddings substantially improve sentiment classification.
- **Resource Trade-off:** BERT required significantly more computational resources (GPU, memory) and time. This suggests that the best model depends on the operational constraints.

### Model Diagnosis
- **Check for Overfitting:** The BERT model’s performance was stable across training and test sets, suggesting effective generalization.
- **Error Analysis:** Manual inspection of misclassified samples could reveal patterns, such as ambiguous sentiments, sarcasm, or lack of sufficient context.

---

## Conclusions

Our work demonstrates that while classical machine learning models are straightforward and computationally inexpensive, their performance in sentiment classification on Amazon reviews is overshadowed by a fine-tuned BERT model. The BERT-based classifier outperforms traditional models on all evaluated metrics—accuracy, precision, recall, F1, and AUC—indicating that deep learning methods leveraging pretrained transformers offer a more nuanced understanding of language.

At the same time, resource constraints and computational complexity should not be overlooked. For large-scale, real-time systems with limited computational budgets, a more lightweight model (e.g., Logistic Regression or Random Forest) might be preferred despite lower performance.

In summary, this project shows that integrating both review text and title features, experimenting with various ML algorithms, and leveraging state-of-the-art language models can significantly improve sentiment classification. These findings can guide businesses and researchers in selecting an optimal approach based on their performance requirements and resource availability.