import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pandas.plotting import scatter_matrix

from zlib  import crc32
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.base     import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose  import ColumnTransformer
from sklearn.metrics  import mean_squared_error
from sklearn.metrics  import mean_absolute_error
from sklearn.impute   import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree         import DecisionTreeRegressor
from sklearn.ensemble     import RandomForestRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#################################################################################################################################
#                                        lOAD AND DISPLAY THE DATA                                                              #
#################################################################################################################################
HOUSING_PATH = f"C:\\Users\\manch\\OneDrive\\Documents\\DEV\\MachineLearnig\\datasets\\Housing_Data\\housing.csv"

def load_housing_data(housing_path):
    #csv_path = os.path.join(housing_path,"housing.csv")
    df=pd.read_csv(housing_path)
    return df

housing=load_housing_data(HOUSING_PATH)

print(housing.head())
print(housing.info())     #The info() method is useful to get a quick description of the data
print(housing.describe()) #The describe() method shows a summary of the numerical attributes

housing.hist(bins=50, figsize=(20,15))
plt.show()

#################################################################################################################################
#                                        SPLIT THE DATA INTO TRAINING AND TESTING SETS                                          #
#################################################################################################################################

def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)

print(len(train_set))
print(len(test_set))

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index() # adds an `index` column

train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)     #an sklearn method that splits arrays into train sets and test sets

#to avoid sampling bias label the rows in categorries that will have a normal distribution then sample training and test data according to the cartegories

housing["income_cat"] = pd.cut (
                                housing["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5]
                                )
housing["income_cat"].hist()
plt.show()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set  = housing.loc[test_index]

print(strat_test_set["income_cat"].value_counts() / len(strat_test_set),"%")
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

#################################################################################################################################
#                                        FEATURE EXTRACTION                                                                     #
#################################################################################################################################

housing = strat_train_set.copy()

housing.plot(kind="scatter", x="longitude", y="latitude",alpha=0.1)

housing.plot(   kind="scatter",
                x="longitude",
                y="latitude",
                alpha=0.4,
                s=housing["population"]/100,
                label="population",
                figsize=(10,7),
                c="median_house_value",
                cmap=plt.get_cmap("jet"),
                colorbar=True
            )
plt.legend()
plt.show()

#A correlation matrix helps you identify which values are closely related by calculating the correlation of each row with others
corr_matrix = housing.drop("ocean_proximity",axis=1).corr()     #Because ocean proximity is not a numeric value whose importance can be measured in terms of correlation

print(corr_matrix)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12, 8))

"""
housing.plot(kind="scatter", x="median_income", y="median_house_value",alpha=0.1)
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix = housing.drop("ocean_proximity",axis=1).corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
imputer.statistics_
housing_num.median().values
housing_num.info()
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,index=housing_num.index)
housing_tr.info()
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]
ordinal_encoder.categories_

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
housing_cat_1hot.toarray()
cat_encoder.categories_

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

housing.info()
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
housing.info()

num_pipeline = Pipeline([
                        ('imputer', SimpleImputer(strategy="median")),
                        ('attribs_adder', CombinedAttributesAdder()),
                        ('std_scaler', StandardScaler()),
                        ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

pd.DataFrame(data=housing_num_tr).info()

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer   ([
                                    ("num", num_pipeline, num_attribs),
                                    ("cat", OneHotEncoder(), cat_attribs),
                                    ])
housing_prepared = full_pipeline.fit_transform(housing)
pd.DataFrame(data=housing_prepared).info()
pd.DataFrame(data=housing_labels).info()

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))


lin_mse = mean_squared_error(some_labels, lin_reg.predict(some_data_prepared))
lin_rmse = np.sqrt(lin_mse)
lin_rmse

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels) #training
#Now that the model is trained, let’s evaluate it on the training set:
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_rmse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_rmse)
forest_rmse
forest_scores=cross_val_score(forest_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

param_grid =    [
                {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
                {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
                ]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
grid_search_scores=cross_val_score(grid_search, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)
grid_search_rmse_scores = np.sqrt(-grid_search_scores)
display_scores(grid_search_rmse_scores)
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) # => evaluates to 47,730.2
final_rmse

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                                loc=squared_errors.mean(),
                                scale=stats.sem(squared_errors)))

"""
def mean_absolute_percentage_error(y_true, y_pred):
    """
    This function calculates the Mean Absolute Percentage Error (MAPE) between two arrays.

    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.

    Returns:
        The MAPE value as a float.
    """
    # Avoid zero division by adding a small constant
    epsilon = 1e-8
    # Clip predicted values to avoid division by zero (optional)
    y_pred = np.clip(y_pred, epsilon, np.inf)
    # Calculate the absolute errors
    absolute_errors = np.abs(y_true - y_pred)
    # Calculate the percentage errors
    percentage_errors = (absolute_errors / y_true) * 100

    # Return the mean of the percentage errors
    return np.mean(percentage_errors)



# Assuming you have housing_labels and housing_predictions
#housing_mape = mean_absolute_percentage_error(y_test, final_predictions)
#print("Mean Absolute Percentage Error:", round(housing_mape),"%")

