# Intel_OneAPI_Heart_Attack_Prediction

# INSPIRATION 💡

This project is inspired by the need for accurate prediction of heart attacks, a critical health concern worldwide. Heart disease remains a leading cause of mortality, making early detection and prevention crucial. Leveraging machine learning techniques, particularly classification algorithms, offers promising avenues for predicting the risk of heart attacks based on various physiological and lifestyle factors. Our endeavor aims to contribute to proactive healthcare by developing a predictive model that can assist in identifying individuals at higher risk of experiencing a heart attack.

# PROBLEM STATEMENT 🎯

The project addresses the challenge of accurately predicting the likelihood of a heart attack based on a set of clinical attributes and patient information. By analyzing relevant medical data, the goal is to develop a robust predictive model that healthcare professionals can use to assess an individual's risk of experiencing a heart attack. Early detection and risk stratification are essential for timely intervention and prevention of cardiovascular events.

# ABOUT THE PROJECT ❓

This project utilizes machine learning algorithms to predict the probability of a heart attack occurrence based on input features such as age, blood pressure, cholesterol levels, and other clinical parameters. By employing classification algorithms, the model learns to differentiate between individuals with a higher likelihood of experiencing a heart attack and those at lower risk. The system undergoes data preprocessing, model training, and evaluation phases to achieve accurate predictions. Ultimately, it provides a valuable tool for healthcare practitioners to assess and manage cardiovascular risk in their patients.

# HOW IT'S BUILT 🛠️

The heart attack prediction project is developed using Python programming language and popular machine learning libraries such as scikit-learn and pandas. The following steps outline the development process:

1. **Data Acquisition:** Curate a comprehensive dataset comprising clinical attributes and patient information related to heart disease.

2. **Data Preprocessing:** Employ preprocessing techniques to clean and standardize the dataset, handling missing values and ensuring data integrity.

3. **Model Development:** Utilize classification algorithms such as K Nearest Neighbors (KNN), Support Vector Machines (SVM), Decision Trees, and Random Forests to build predictive models.

4. **Model Training:** Train the machine learning models using the prepared dataset, adjusting hyperparameters and optimizing performance.

5. **Model Evaluation:** Evaluate the trained models using appropriate evaluation metrics such as accuracy, precision, recall, and F1-score to assess predictive performance.

6. **Intel OneAPI Integration:** Integrate Intel OneAPI optimizations to enhance model performance and efficiency, leveraging hardware acceleration and optimization techniques.

7. **Collaboration and Version Control:** Utilize collaborative tools like Git and GitHub for version control, enabling seamless collaboration and project management among team members.


# WHAT I LEARNED 📚


1. **Machine Learning Fundamentals:** Developed a deeper understanding of machine learning algorithms, including classification techniques and their applications in healthcare analytics.

2. **Data Preprocessing Techniques:** Acquired skills in data cleaning, standardization, and feature engineering to prepare datasets for machine learning model training.

3. **Model Development and Evaluation:** Learned to build, train, and evaluate machine learning models using popular Python libraries such as scikit-learn, and TensorFlow.

4. **Intel OneAPI Integration:** Explored the integration of Intel OneAPI for optimizing machine learning model performance, witnessing the impact of hardware acceleration on computation speed and efficiency.

5. **Collaborative Development Practices:** Utilized version control systems like Git and GitHub for collaborative development, enabling effective teamwork and project management.

6. **Healthcare Analytics and Proactive Intervention:** Recognized the importance of predictive analytics in healthcare for early detection and proactive intervention in preventing cardiovascular diseases.

# INSIGHTS & OBSERVATIONS 🔍

1. **Correlation among features:** From the correlation matrix colourmap, it can be observed that there is a moderately strong negative correlation between the features exng & oldpeak and the possibility of a heart attack, and a moderately positive correlation between chest pain & the possibility of heart attack.

![Correlation Colormap](https://github.com/HareRamaCh/Intel_OneAPI_Heart_Attack_Prediction/blob/main/Heart_Attack_Correlation.png)

2. **Model Performance Variability:** Four supervised machine learning models were used, and each had differing accuracies. kNN had the highest accuracy, with 87%, which could be attribute to its localised decision making and robustness to noise. Decision trees, on the other hand, only had an accuracy of 79%, which could be attributed to its susceptibility to overfitting and lack of ensemble learning.

3. **Intel OneAPI Optimization Impact**: Integration of Intel OneAPI and the sklearnex library has demonstrated notable improvements in computational efficiency and model performance. The optimized execution of machine learning algorithms has led to faster insights generation and enhanced predictive accuracy. The results are summarised in the table below: 

|Classifier| Execution time Without Intel OneAPI  | Execution time with Intel OneAPI | Performance Increase
|-------------| ------------- | ------------- | ------------- |
|k-Nearest Neighbours| 16.6 ms ± 8.01 ms  | 5.29 ms ± 6.08 µs  | 213.8% |
|Random Forest| 2.66 s ± 5.9 ms  | 1.67 s ± 280 ms | 59.28% |
|Decision Tree| 4.98 ms ± 891 µs  | 3.55 ms ± 2.05 µs  | 40.28% |
|Support Vector Machines| 7.21 ms ± 122 µs  | 5.61 ms ± 4.76 µs  | 28.52% |


# INTEL ONEAPI INTEGRATION 🔵

Intel OneAPI is a comprehensive suite of development tools and libraries designed to accelerate application performance across diverse hardware architectures. In the heart attack prediction project, Intel OneAPI was integrated, alongside the sklearnex package, to optimize machine learning model performance during training and inference. Intel OneAPI provides developers with optimized libraries, tools, and frameworks tailored for high-performance computing tasks. It offers a unified programming model that enables developers to write code that can seamlessly execute across various hardware architectures, including CPUs, GPUs, FPGAs, and AI accelerators.

![Intel OneAPI Logo](https://www.hearne.software/Images/oneapi-logo-resized.aspx)

The integration of Intel OneAPI and sklearnex in the heart attack prediction project offered several advantages:

**1. Optimization Capabilities:** Intel OneAPI incorporates advanced compiler optimizations and hardware-specific optimizations tuned to leverage the features and capabilities of Intel architectures. This optimization improves the performance and efficiency of machine learning computations, enhancing the overall effectiveness of the predictive models.

**2. Hardware Acceleration:** By harnessing the full potential of Intel hardware, including CPUs and GPUs, Intel OneAPI enables accelerated computations, reducing the time required for model training and inference. This acceleration enhances productivity and enables faster iteration cycles in model development.

**3. Unified Programming Model:** Intel OneAPI provides a unified programming model that simplifies development across diverse hardware architectures. Developers can write code once and deploy it across multiple hardware targets without extensive modifications, streamlining the development process and maximizing resource utilization.

The integration of Intel OneAPI and sklearnex facilitated the optimization of machine learning tasks in the heart attack prediction project, resulting in improved performance and efficiency. By leveraging optimization capabilities and hardware acceleration, Intel OneAPI contributed to the development of high-performing predictive models, empowering proactive healthcare initiatives and advancing the field of predictive analytics.

# Future Enhancements 🚀

**1. Integration of Advanced Machine Learning Techniques:** Explore the integration of advanced algorithms like gradient boosting machines (GBMs) and ensemble methods to enhance predictive accuracy and robustness. These techniques can offer better insights into complex relationships within the data, improving the model's predictive performance.

**2. Feature Engineering and Selection:** Conduct thorough feature engineering to extract more informative features from the dataset. Employ techniques such as dimensionality reduction and feature selection to focus on the most relevant predictors of heart attacks. This process can help streamline the model and improve its interpretability.

**3. Enrichment of Dataset:** Expand the dataset by incorporating additional relevant features such as genetic markers, lifestyle factors, and medical history. Increasing the diversity and depth of the dataset can provide a more comprehensive understanding of heart attack risk factors, leading to more accurate predictions.

These enhancements can further refine the heart attack prediction model, making it more accurate, interpretable, and applicable in real-world healthcare settings.
