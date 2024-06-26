# Fetal_Health

## Overview

Improving Fetal Health with Machine Learning: Analyzing Cardiotocogram (CTG) Data
This project tackles the crucial challenge of reducing preventable child mortality.

Cardiotocograms (CTGs) are a cost-effective and widely available tool for fetal health assessment. CTGs monitor fetal heart rate (FHR), fetal movement, uterine contractions, and other vital signs, empowering healthcare professionals to make informed decisions and potentially prevent child and maternal mortality.

Data Exploration:
This dataset offers 2126 records containing features extracted from CTG exams. These features were meticulously classified into three categories by a panel of three expert obstetricians: Normal, Suspect, 
Pathological

## Exploratory Data Analysis

What's in the data?

    RangeIndex: 2126 entries, 0 to 2125
    Data columns (total 22 columns):
    #   Column                                                  Non-Null Count  Dtype  
    ---  ------                                                  --------------  -----  
    0   baseline value                                          2126 non-null   float64
    1   accelerations                                           2126 non-null   float64
    2   fetal_movement                                          2126 non-null   float64
    3   uterine_contractions                                    2126 non-null   float64
    4   light_decelerations                                     2126 non-null   float64
    5   severe_decelerations                                    2126 non-null   float64
    6   prolongued_decelerations                                2126 non-null   float64
    7   abnormal_short_term_variability                         2126 non-null   float64
    8   mean_value_of_short_term_variability                    2126 non-null   float64
    9   percentage_of_time_with_abnormal_long_term_variability  2126 non-null   float64
    10  mean_value_of_long_term_variability                     2126 non-null   float64
    11  histogram_width                                         2126 non-null   float64
    12  histogram_min                                           2126 non-null   float64
    13  histogram_max                                           2126 non-null   float64
    14  histogram_number_of_peaks                               2126 non-null   float64
    15  histogram_number_of_zeroes                              2126 non-null   float64
    16  histogram_mode                                          2126 non-null   float64
    17  histogram_mean                                          2126 non-null   float64
    18  histogram_median                                        2126 non-null   float64
    19  histogram_variance                                      2126 non-null   float64
    20  histogram_tendency                                      2126 non-null   float64
    21  fetal_health                                            2126 non-null   float64
    dtypes: float64(22)
    memory usage: 365.5 KB

![](images/correlation.png)

## Unsupervised Learning

Imports

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler


### Using clustering find the best value for k using scaled data

    # Create a list with the number of k-values from 1 to 11
    k_scaled = list(range(1, 11))

    # Create an empty list to store the inertia values
    inertia_scaled = []

    for i in k_scaled:
    model = KMeans(n_clusters=i, random_state=0)
    model.fit(un_sup_df_scaled_new)
    inertia_scaled.append(model.inertia_)



## Supervised Learning

## Deep Learning and Optimizations


### Resources
https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification/data

https://onlinelibrary.wiley.com/doi/10.1002/1520-6661(200009/10)9:5%3C311::AID-MFM12%3E3.0.CO;2-9

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10000592/#:~:text=To%20date%2C%20cardiotocography%20(CTG),a%20challenging%20signal%20processing%20task.

low variance - https://stats.stackexchange.com/questions/584174/interpretation-of-low-variance-in-pca
