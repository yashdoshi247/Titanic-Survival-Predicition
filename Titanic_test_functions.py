def test_preprocessing(data, encoder):
    import pandas as pd
    import numpy as np

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    test_df = pd.DataFrame(columns=features, data=np.array(data).reshape(1, -1))
    cat_features = ['Pclass', 'Sex', 'Embarked']

    for feature in test_df.columns.values:
        if feature not in cat_features:
            test_df[feature] = test_df[feature].astype('float')  # Converting numerical features to float data type
        if feature == "Pclass":
            test_df[feature] = test_df[feature].astype('int64')  # Converting Pclass value to int data type

    # One-Hot Encoding the categorical features
    test_encode = encoder.transform(test_df[[i for i in cat_features]])
    categories = []
    for index,feature in enumerate(encoder.categories_):
        for category in feature:
            categories.append(f"{cat_features[index]}_{category}")

    encoded_df = pd.DataFrame(data=test_encode, columns=categories)

    # Merging the one-ho-encoded values with the test_df
    merged_df = pd.concat([test_df, encoded_df], axis=1)
    # Removing the categorical features from test_df
    test_df_preprocessed = merged_df.drop(cat_features, axis=1)
    return test_df_preprocessed


def fill_na_age_func(cols, pclass_dict):
    import pandas as pd
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return round(pclass_dict[1], 2)
        elif Pclass == 2:
            return round(pclass_dict[2], 2)
        else:
            return round(pclass_dict[3], 2)
    else:
        return Age
