# Credit Risk Classification

![Image](CreditReport.png)

# Contents

[Outline](#outline-of-analysis)

[Results](#resultsResults)

[Summary && Recommendation]()
## Outline Of Analysis
* Credit risk poses a classification problem that’s inherently imbalanced. The reason is that healthy loans easily outnumber risky loans. 
* For this reason, a supervised learning machine learning will be used to build a model that can identify the creditworthiness of borrowers.
* Various techniques will be implemented to train and evaluate models with imbalanced classes. 
* Dataset of historical lending activity from a peer-to-peer lending services company is used for the analysis.
* Features (X) factors into account the following:
    - loan_size
    - Interest rate
    - Borrower Income
    - Debt to Income ratio
    - Number of accounts
    - Derogatory marks
    - Total Debt
    - Loan Status
* Dependent variable, label set `y` as loan amount. With `0` representing healthy loans and `1` representing default loans
* Data provided will be split into 2 parts: `Traning` and `Testing` sets
* Logistic regression model will be used to evaluate the Original Data and Resampled Training Data results

With a dataset of 77536 points, we separated labels set `y` identified as (`loan_status`) column and `features` (X) representing all other columns.
To check the balance of Labels set (`y`), `value_counts` function was used (recording 77536 data points - 750036 `0`s & 2500 `1`s)
Datasets were now split into training and testing data using `train_test_split` module from [scikit learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). `LogisticRegression` module from [scikit learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) was fitted into the training data and predictions were saved on the testing data. To evaluate performace of datasets `balanced_accuracy_score` was calculated, `confusion_matrix` was generated and `classification_report` was printed to determine if we could distinguish between healthy loans and high-risk loans.

Due to the skewed nature of the original data, a resampled data was initialized to check for better performance. Datasets were then resampled using the training data to reevaluate the model, specifically, using `RandomOverSampler`.
`RandomOverSampler` module from the [imbalanced-learn library](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html) was used to resample the data. This time round there was equal data points of 56277 each for `0`s and `1`s.
Next the `LogisticRegression` classifier was employed, resampling the data to fit the model and make predictions.
Finally, to assess the model’s performance, `balanced_accuracy_score` was calculated, `confusion_matrix` was generated and `classification_report` was printed to determine if we could distinguish between healthy and defaulted loans more accurately.

## Results
* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
ML Model 1 does a better job at predicting the False Positives, with a precision score of 100% while recording a 95% recall for False Negatives due to the unequal data points. Balanced Accuracy Score, scored a 97%. 
``` python
                   pre       rec       spe        f1       geo       iba       sup

          0       1.00      1.00      0.95      1.00      0.97      0.95     18759
          1       0.87      0.95      1.00      0.91      0.97      0.94       625

avg / total       0.99      0.99      0.95      0.99      0.97      0.95     19384
```
```python
Balanced Accuracy Score: 0.9721077669385362
```

* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
ML Model 2 gave an accurate analysis for both the False Positives, recording a 100% and False Negatives improving its results to 100% due to the uniform/equal data points using the `RandomOverSampler` simultaneously increasing the Balanced Accuracy Score to 99%.
```python
                   pre       rec       spe        f1       geo       iba       sup

          0       1.00      1.00      1.00      1.00      1.00      0.99     18759
          1       0.87      1.00      1.00      0.93      1.00      0.99       625

avg / total       1.00      1.00      1.00      1.00      1.00      0.99     19384
```
```python
Balanced Accuracy Score: 0.9959744975744975
```


## Summary
Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.

To accurately predict the credit risk for the analysis, healthy loans versus default loans, Machine Learning Model 2 (using the `RandomOver Sampler`)
performed best due to its high `balanced accuracy score, precision score and recall score`.
> False Positives, with a 100% precision allows the analyst to account for great customers without forgoing the Opportunity Cost of Revenues and Profits.

> False Negatives, with a 100% recall alerts the analyst of any defaulters which could help minimize financial loss.

> Balanced Accuracy Score, with a 99% score provided a significant insight into how the Machine Learning Model 2 correctly predicted the outcomes.  

It is imperative to predict the `1`'s and the `0`'s since analysts would like to know the healthy loan borrower in order to maximize profits and the 

high--risk loan borrowers to cut down their losses as often as possible. 