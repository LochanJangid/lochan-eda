import pandas as pd
from sklearn.model_selection import train_test_split
from lochan_eda.numerical import Numerical
from lochan_eda.categorical import Categorical

class AutomatedEDA():
    
    # prepare
    def prepare(self, X: pd.DataFrame, target=None, exclude=None, split=True, test_size=0.2, random_state=42, stratify=None):
        """
            prepare(
                X: pd.Dataframe -> features
                target: str(column_name)/pd.Series -> label (it will untouched :p)
                exclude: List[str]/str -> columns that you don't want to touch me
                split: bool -> do you want split into train test 
                test_size: float -> test set size in ratio
                random_state: int -> state for reproducability
                stratify: if data imbalanced
            )
            Return:
                X if split is False and target is None
                (Xtr, Xte) if split is True and target is None
                (X, y) if split is False and target is given
                (Xtr, Xte, ytr, yte) if split is True and target is given
        """
        # if target is a column 
        self.X = X
        self.exclude=exclude
        self.y = None
        if type(target) == str:
            self.y = X[target]
            self.X = X.drop(columns=target)
        # if target is a independent series
        if type(target) == pd.Series:
            self.y = target

        # if we have to split the data
        if split and self.y is not None:
            self.Xtr, self.Xte, self.ytr, self.yte = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state, stratify=stratify)
        elif split:
            self.Xtr, self.Xte = train_test_split(self.X, test_size=test_size, random_state=random_state)
        
        # preprocess
        if target and split:
            self.fit(self.Xtr, self.ytr)
            # transform Xtrain and Xtest both
            processed_data = self.transform(self.Xtr, self.ytr) + self.transform(self.Xte, self.yte)
        elif target and not split:
            self.fit(self.X, self.y)
            processed_data = self.transform(self.X, self.y)
        elif not target and split:
            self.fit(self.Xtr)
            processed_data = self.transform(self.Xtr) + self.transform(self.Xte)
        else:
            self.fit(self.X)
            processed_data = self.transform(self.X)

        return processed_data[0] if len(processed_data) == 1 else processed_data


    def fit(self, X: pd.DataFrame, y: pd.Series=None) -> None:
        """
            fit(
                X: pd.DataFrame -> features
                y: pd.Series -> labels
            )
            Work:
                analyze the data and learn about it.
            Return:
                None
        """
        # split numerical and categorical 
        num_cols = X.select_dtypes(include="number")
        cat_cols = X.select_dtypes(include=["object", "category"])

        self.categorical = Categorical()
        self.numerical = Numerical()

        self.numerical.fit(num_cols, exclude=self.exclude)
        self.categorical.fit(cat_cols, exclude=self.exclude)

        return None

    def transform(self, X: pd.DataFrame, y: pd.Series=None):
        """
            fit(
                X: pd.DataFrame -> features
                y: pd.Series -> labels
            )
            Work:
                implement the analyzed work
            Return:
                Tuple(X, y) if y available else (X,)
        """
        num_cols = X.select_dtypes(include="number")
        cat_cols = X.select_dtypes(include=["object", "category"])

        processed_num_cols = self.numerical.transform(num_cols, exclude=self.exclude)
        processed_cat_cols = self.categorical.transform(cat_cols, exclude=self.exclude)  

        X = pd.concat([processed_num_cols, processed_cat_cols. self.X[self.exclude]], axis=1)

        return (X, y) if y is not None else (X, )