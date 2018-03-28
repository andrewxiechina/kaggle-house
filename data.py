import pandas as pd
import numpy as np
import matplotlib
# Config matplotlib to use backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# sklearn
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("/input/train.csv")
test = pd.read_csv("/input/test.csv")

# Drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)



# Log transformation of target value
train["SalePrice"] = np.log1p(train["SalePrice"])

#########################
## Feature Engineering ##
#########################
# Keep n for seperating later
ntrain = train.shape[0]
# Concatenate all data into one
train_y = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)

# Handle missing data
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

COL_FILL_WITH_NONE = ['GarageType', 'GarageFinish', 
    'GarageQual', 'GarageCond', 'PoolQC', 'MiscFeature', 'Alley', 
    'Fence', 'FireplaceQu', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
    'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'Functional', 'MSSubClass']
for col in COL_FILL_WITH_NONE:
    all_data[col] = all_data[col].fillna('None')

COL_FILL_WITH_ZERO = ['GarageYrBlt', 'GarageArea', 'GarageCars',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 
    'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
for col in COL_FILL_WITH_ZERO:
    all_data[col] = all_data[col].fillna(0)

COL_FILL_WITH_MODE = ['Electrical', 'KitchenQual', 'Exterior1st',
'Exterior2nd', 'SaleType']
for col in COL_FILL_WITH_ZERO:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# Drop unuseful data
all_data.drop(['Utilities'], axis=1)

# Add new feature
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

################
## Preprocess ##
################
# Label Encode
COLS = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
for c in COLS:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# Dummies
all_data = pd.get_dummies(all_data)

# Restore data
train_X = all_data[:ntrain]
test_X = all_data[ntrain:]

# To numpy
train_X = train_X.values
test_X = test_X.values

def get_data():
    # Transform to numpy
    return train_X, train_y, test_X