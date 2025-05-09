# College-Classification-in-R

This study analyzes a dataset to classify colleges as either private or non-private using a logistic regression model. The dataset, sourced from the ISLR R package, contains 777 observations and 18 variables that capture various characteristics of colleges such as enrollment, graduation rate, and financial metrics. The goal is to identify key predictors of college type through exploratory data analysis (EDA) and regression techniques leveraging R.

# Logistic Model was Fit for classification
<img width="250" alt="image" src="https://github.com/user-attachments/assets/62e8e2af-159f-44f3-871b-395561057b6e" />
<img width="200" alt="image" src="https://github.com/user-attachments/assets/319e357d-d4ed-4afa-af36-6d01b1b1f744" />



# Key Insights and Recommendations
1.	**Significant Predictors:** Variables such as f_undergrad, outstate, ph_d, grad_rate, and expend are critical in predicting whether a college is private.
2.	**Model Performance:** The logistic regression model explains over 92% of the variance in the data, providing reliable classifications with an accuracy of 93.59% on the test set.
3.	**Misclassification Analysis:**
•	False Positives (FP): The model only made 14 FP, which misclassified a non-private college as private. The Recall (Sensitivity) of 98% suggests a strong emphasis on minimizing False Negatives, making sure that private colleges are not misclassified as non-private.This minor error could lead to miscommunications in marketing or funding allocation.
•	False Negatives (FN): The model made 1 FNs, misclassifying private colleges as non-private. These errors may affect enrollment predictions and marketing strategies, particularly for private institutions. 


# Conclusion 
The logistic regression model effectively classifies colleges as private or non-private based on key characteristics. With a high AUC of 0.9865 and an accuracy of 93.59% on the test set, the model demonstrates strong predictive performance. While the model performs well overall, addressing False Negatives (FN), where private colleges are misclassified as non-private, could improve its real-world applicability, particularly for decision-making in areas like marketing and enrollment strategies.


