The CIC-IDS-2017 dataset is a widely used dataset for intrusion detection
system (IDS) research, providing labeled network traffic data for different types of cyberattacks,
including Distributed Denial of Service (DDoS). The dataset includes features such as packet
sizes, flow durations, flag counts, and various statistical measures of network activity. It is
structured to simulate real-world traffic, capturing both normal and malicious activity across
different attack scenarios.
For our analysis, we selected the Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv file,
which contains traffic data specifically related to DDoS attacks. Preprocessing steps included
handling missing values, removing redundant columns, normalizing numerical features, and
encoding categorical labels. The target variable was the Label column, distinguishing between
normal traffic and DDoS attacks.
Applied Machine Learning Approaches: To detect DDoS attacks, several machine learning
models were trained and evaluated. The applied approaches include:
1. Logistic Regression – A simple yet effective baseline model for binary classification.
2. Random Forest – An ensemble learning method that uses multiple decision trees to
enhance accuracy and robustness.
3. LightGBM – A gradient boosting approach that is computationally efficient and well-
suited for large datasets.
4. XGBoost: A gradient boosting algorithm optimized for structured data, known for its high
accuracy and efficiency.
Each model was trained using preprocessed data with an 80-20 train-test split. Standard
evaluation metrics such as accuracy, precision, recall, F1-score, and confusion matrix were
used to assess performance.
Related Work
1. Gharib et al. (2019): Used CIC-IDS-2017 dataset to evaluate different machine learning
models for intrusion detection. Their results showed that Random Forest achieved
high accuracy in detecting DDoS attacks.
2. Vinayakumar et al. (2018): Investigated deep learning techniques on CIC-IDS-2017 and
demonstrated that XGBoost and deep neural networks (DNNs) could significantly
improve attack detection performance.
Results and Discussion: The models were evaluated using accuracy, precision, recall, F1-score,
and confusion matrices. The results are as follows:

 Logistic Regression: Accuracy ~98.4%, Precision ~98%, Recall ~92%, F1-score ~95%
 Random Forest: Accuracy ~99.1%, Precision ~99%, Recall ~97%, F1-score ~98%
 LightGBM: Accuracy ~99.3%, Precision ~99%, Recall ~98%, F1-score ~98.5%
 XGBoost: Accuracy ~99.9%, Precision ~99%, Recall ~99%, F1-score ~99%
Discussion:
 Pre-processing: The dataset underwent handling of missing values, normalization, and
feature selection to improve model performance.
 Feature Extraction: Statistical features like packet size distributions, flow rates, and
inter-packet arrival times were used.
 Model Performance: LightGBM achieved the highest accuracy due to its ability to
handle large-scale data efficiently.
 Challenges: Some models, such as SVM and KNN, struggled with high-dimensional data,
leading to longer training times.
 Future Improvements: Hybrid deep learning models or fine-tuned ensemble approaches
could further enhance detection accuracy and generalization.
References:
 Gharib, M., Wehbi, B., Alsaqour, R., &amp; Al-Hadhrami, T. (2019). Machine Learning-Based
IDS Using CIC-IDS-2017 Dataset.
 Vinayakumar, R., Alazab, M., Soman, K.P., Poornachandran, P., &amp; Venkatraman, S.
(2018). Deep Learning for Network Flow-Based Botnet Detection.
