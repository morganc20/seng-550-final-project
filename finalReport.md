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
Max
What is the problem?
Why is problem important?

### What have others done in the space?
Carter

### What are some existing gaps that you seek to fill?

Within the machine learning space, existing gaps include over-reliance on single features for sentiment analysis, lack of integration/interprretability of auxiliary features within data and consistent reproducibility. 

In terms of how our work seeks to solve the problem better and add to the existing solutions, we have implemented various changes such as text preprocessing, combination of multiple features to enrich sentiment prediction, creating robust frameworks for sentiment analysis/feature classification, allowing for extensibility etc. to build on top of and surpass existing solutions. 

Our multi-algorithm solution is evaluated against rigorous performance metrics to understand its strengths, weaknesses and to ensure we are iteratively improving classification accuracy while ensuring our solution is reproducible and extendible.

### What are your data analysis questions
Ayo

### What are you proposing?
Max

### Main Findings
Everyone explain the findings of their model.

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

Everyone explain how they refined feature space for their model
### Experiment setup
Ayo

### Experimentation factors (e.g., types of ML algorithms used, hyperparameters tuned, details on training/test/cross-validation data set etc.)
Carter
### Experiment process
Everyone does their model
### Performance metrics - accuracy, precision, recall, F-score etc.
Everyone explains their model
## Results
### Key findings in your exploratory data analysis and prediction. If you are trying out multiple algorithms, your results will compare them. How did you diagnose your ML model?
Max
### Conclusions 
Ayo