## Datasets

This folder contains datasets for regression and classification tasks. The datasets are further split by dimensionality and input data type. Furthermore there are some emoticons that indicate whether the dataset is suitable for a particular task.

# table of emoticons indication complexity of the task
| Emoticon | Approx. Complexity |
|----------|---------|
| 📏 | linear |
| 🥊 | non-linear |



### Regression Datasets

- Line fitting (1d input):
    - Ismailov Dataset 🥊
    - Sinusoidal Dataset 🥊
- Regression2D Dataset (2d input)
- Real World Small (UCI)
    - Airfoil (Dua & Graff, 2017) 🥊
    - Concrete (Yeh, 1998) 🥊
    - Diabetes 📏
    - Energy Efficiency (Tsanas & Xifara, 2012) 🥊
    - Forest Fires 📏
    - Yacht (Ortigosa et al., 2007)
- Real World Medium Sized
    - Bike Sharing (OpenML, Fanaee-T, 2013) 🥊
    - Protein Structure (UCI, Dua & Graff, 2017) 🥊

### Classification Datasets

- Real World Small
    - Rice (2 classes: 1 = Cammeo, 2 = Selenio) (Kaggle) (Pretty simple task apparently) 📏
    - Australian (2 classes) (Quinlan,Ross. Statlog (Australian Credit Approval). UCI Machine Learning Repository. https://doi.org/10.24432/C59012.) 📏
    - Breast Cancer (2 classes) (Wolberg,William, Mangasarian,Olvi, Street,Nick, and Street,W.. (1995). Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B. 📏
    - Ionosphere (2 classes) (Sigillito,V., Wing,S., Hutton,L., and Baker,K.. (1989). Ionosphere. UCI Machine Learning Repository. https://doi.org/10.24432/C5W01B.) 🥊
    - Credit (Dr. Hans Hofmann, 1994, then UCI now OpenML) 📏
    - Income (OpenML: [Link](https://openml.org/search?type=data&status=active&sort=nr_of_downloads&qualities.NumberOfClasses=%3D_2&qualities.NumberOfInstances=between_1000_10000&id=1590)) 🥊
    - `diabetes_classification.data` (Note this is not usable as is, it needs to be
    preprocessed - features with linear dependence) (Kaggle)
