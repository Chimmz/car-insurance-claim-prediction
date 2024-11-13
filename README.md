# Car Insurance Claim Prediction
This project focuses on predicting the likelihood of car insurance claims based on various customer, vehicle, and policy characteristics. Using a dataset of features from policyholders, vehicles, and demographic information, I developed a machine learning model with a Random Forest Classifier to classify the claim status, enabling better risk assessment and strategic premium pricing for insurers.

### Project Overview
- Objective: To build a predictive model that classifies the probability of a policyholder making an insurance claim.
- Primary Algorithm: After testing multiple models, including RandomForestClassifier, XGBoostClassifier, and Logistic Regression, the RandomForestClassifier yielded the best performance based on Accuracy (93%), Recall (92.5%), Precision (93%), F1 Score (92.69%), and has a least cost after penalizing False Negatives & False Positives
- Technologies Used: Python, Pandas, Scikit-Learn, Matplotlib, Seaborn.

### Data Description
The dataset comprises numerous predictor variables capturing key aspects of the policyholder, vehicle, and policy history. Key features include:
- Policy Information: policy_tenure, area_cluster, population_density
- Vehicle Characteristics: age_of_car, make, segment, fuel_type, max_torque, max_power, engine_type, airbags, displacement, cylinder, transmission_type, gear_box, steering_type, turning_radius, length, width, height, gross_weight.
- Safety and Security Features: is_esc, is_adjustable_steering, is_tpms, is_parking_sensors, is_parking_camera, is_front_fog_lights, is_rear_window_wiper, is_rear_window_washer, is_rear_window_defogger, is_brake_assist, is_power_door_locks, is_central_locking, is_power_steering, is_driver_seat_height_adjustable, is_day_night_rear_view_mirror, is_ecw, is_speed_alert, ncap_rating
- Demographic and Policyholder Information: age_of_policyholder, area_cluster, population_density

### Exploratory Data Analysis (EDA)

#### UNIVARIATE ANALYSIS:

![image](https://github.com/user-attachments/assets/f298f446-66f2-4b80-8b7e-4d0794c7619d?raw=true)
![image](https://github.com/user-attachments/assets/abb49c86-3cbd-439a-a073-f2862ba239c7?raw=true)
![image](https://github.com/user-attachments/assets/174f52c1-1c4b-46be-913d-ab1a4362b5e5)
![image](https://github.com/user-attachments/assets/258390f4-aabe-4b8d-9d6d-6130eead480d)


#### BIVARIATE ANALYSIS:
##### For each predictor variable, I analyzed its correlation with the likelihood of making a claim:

![image](https://github.com/user-attachments/assets/75360799-6538-4cc6-ae2c-6ac31dba5458?raw=true)
![image](https://github.com/user-attachments/assets/3b6a55fe-0932-4fef-a60d-442727f52eb8?raw=true)
![image](https://github.com/user-attachments/assets/391133b9-7095-4db7-811b-51c089187714?raw=true)
![image](https://github.com/user-attachments/assets/66b9e837-e714-453d-8a46-7a9f4bfd4dc1?raw=true)
![image](https://github.com/user-attachments/assets/a6469a5a-f0f3-4e2e-90dc-d83c1d1a67c7?raw=true)

#### MULTIVARIATE ANALYSIS:

![image](https://github.com/user-attachments/assets/da70ddb7-9ef0-46e0-86ba-1a6bda7150c9?raw=true)


### Detecting Class Imbalance:

![image](https://github.com/user-attachments/assets/2c4d250a-f88c-42d2-8b13-dea9cc5f01b3?raw=true)

### Model Selection and Evaluation
Three models were evaluated for their performance on the car insurance claim dataset: RandomForestClassifier, XGBoostClassifier, and Logistic Regression.
And the following shows the model evaluations & performances:

##### Model: Logistic Regression
##### Accuracy: 0.5794055975932173
##### Recall: 0.5970864834962198
##### Precision: 0.5714285714285714
##### F1 Score: 0.5839758329951757
##### Cost: 437048570

======================

##### Model: Random Forest
##### Accuracy: 0.9279332664782569
##### Recall: 0.9253180896182924
##### Precision: 0.92865735171648
##### F1 Score: 0.9269847134346278
##### Cost: 81007710

======================

##### Model: XGBoost
##### Accuracy: 0.9244689579724679
##### Recall: 0.8702747556702932
##### Precision: 0.9741975436061513
##### F1 Score: 0.919308497686876
##### Cost: 140702500


### Model Evaluation Comparison
#### Accuracy:

![image](https://github.com/user-attachments/assets/9bda809c-010c-4998-99e7-2ac8f5e68cff?raw=true)

#### Recall:

![image](https://github.com/user-attachments/assets/901dc390-ba92-43f6-9de3-dce23a8241c6?raw=true)

#### Precision:

![image](https://github.com/user-attachments/assets/86dd3320-e25d-4fed-b780-622a6396eaf5?raw=true)

#### F1 Score:

![image](https://github.com/user-attachments/assets/f8cc0acd-e929-4ecd-a35f-2c028afcd463)

#### Cost
cost = 10*FP + 100_000*FN, where:
- FP - False Positives
- FN - False Negatives

![image](https://github.com/user-attachments/assets/77901474-31a4-4d7a-b211-c81dc72a1d0d)


#### Area Under ROC:
![image](https://github.com/user-attachments/assets/2dc98003-25ac-451b-a38e-7a8a962e77be)


All models were assessed on various metrics, including accuracy, precision, recall, F1-score, and AUC-ROC scores, with the RandomForestClassifier emerging as the top performer.

#### Feature Importances Chart:

![image](https://github.com/user-attachments/assets/76d68e34-95cb-4bb4-85d3-e2217c2b5597)


##### A new & final RandomForestClassifier model was created with only the top 5 important features/predictors. This reduces model complexity and the risk of overfitting.


## Conclusion:
This project showcases a powerful predictive solution that leverages machine learning to support car insurance risk management. The RandomForestClassifier provides an accurate, interpretable model that helps insurers understand the impact of various factors on claim likelihood and aids in making informed decisions regarding policy pricing and risk assessment.
