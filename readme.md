# Hierarchical Multi-class and Multi-label Text Classification for Crime Report: A Traditional Machine Learning Approach
Large amounts of text and digital data are produced every day by society’s use of government
and private companies. In this process, digital transformation contributes to the increasing amount of
structured and unstructured data stored in digital media. More and more data repositories, such as Data
Lakes and Data Warehouses, are being built to store and provide information to various business areas,
contributing to Business Intelligence solutions. Some databases store vast amounts of unstructured data,
which must be systematized and classified to better meet the data owner’s needs. In the context of criminal
incident report systems in law enforcement agencies, each recorded incident must be properly classified as
a specific crime. Often, hundreds or even thousands of distinct categories may be presented as options for
the police officer or security agent responsible for the task. This work explores a clustering approach to
group the different categories into a hierarchical tree of classes, making it easier to apply Machine Learning
(ML) models, such as XGBoost, for the automated classification of criminal incident report narratives. As
a case study, the Civil Police of the State of Rio de Janeiro (SEPOL/RJ) has an incident report database
with over 6.5 million records, which grows daily, fed by Judicial Police Units (JPU) spread across the
state. Each new record depends on the manual classification performed by the police officer regarding the
type of crime that occurred when registering the occurrence. A hierarchical tree of classes was developed
to segment the problem of this case study, enabling the application of various XGBoost models for the
automated classification of criminal incidents. The result obtained by the hierarchical model for 80 distinct
classes presents a final accuracy of 0.463, compared to 0.419 of the traditional model (baseline). A reduction
of 25.48% in the training time of the model was observed when transitioning from the traditional approach
to the hierarchical one.
