# Hierarchical Multi-class and Multi-label Text Classification for Crime Report: A Traditional Machine Learning Approach
Large amounts of digital data are produced daily through society’s use of government and private companies. 
Digital transformation contributes to the increasing amount of structured and unstructured data stored in digital media. 
Organizations build centralized data repositories to store and provide information to business areas, supporting Business Intelligence solutions. 
Some databases store vast amounts of unstructured data, which must be systematized and classified to meet the data owner’s needs. 
In criminal incident report systems, each recorded incident must be classified as a specific crime, with hundreds or thousands of categories presented to the responsible officer. 
This work explores a clustering approach to group categories into a hierarchical tree of classes, enabling the use of Machine Learning (ML) models like XGBoost for automated classification of criminal incident reports narratives. 
As a case study, the Civil Police of of the State of Rio de Janeiro (SEPOL/RJ) has a database with over 6.5 million records, growing daily from Judicial Police Units (JPU) across the state. 
Each new report requires manual classification. 
A hierarchical tree of classes was developed to segment the problem, allowing various XGBoost models for automated classification. 
The proposed hierarchical model with 80 classes achieved an accuracy of 0.463, outperforming the baseline flat model which reached 0.419, along with a 25.48\% reduction in training time. 
The weighted average F1-score obtained by the hierarchical model was 0.48188, while the baseline model reached 0.44061.
The improvement was statistically validated through a Wilcoxon signed-rank test, which yielded a p-value of 0.000010.
