import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lochan_eda.numerical import Numerical
from lochan_eda.categorical import Categorical

class AutomatedEDA():
    def __init__(self):
        self.exclude = None

    # prepare
    def prepare(self, X: pd.DataFrame, target=None, exclude=None, split=True, test_size=0.2, random_state=42, stratify=None, in_return="ndarray"):
        """
            prepare(
                X: pd.Dataframe -> features
                target: str(column_name)/pd.Series -> label (it will untouched :p)
                exclude: List[str]/str -> columns that you don't want to touch me
                split: bool -> do you want split into train test 
                test_size: float -> test set size in ratio
                random_state: int -> state for reproducability
                stratify: if data imbalanced
                in_return: ndarray/dataframe/tensor -> you return result type
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
        if self.y is not None and split:
            self.fit(self.Xtr, self.ytr)
            # transform Xtrain and Xtest both
            outXtr, outytr, outXte, outyte  = self.transform(self.Xtr, self.ytr, in_return=in_return) + self.transform(self.Xte, self.yte, in_return=in_return)
            processed_data = outXtr, outXte, outytr, outyte
        elif self.y is not None and not split:
            self.fit(self.X, self.y)
            outX, outy = self.transform(self.X, self.y, in_return=in_return)
            processed_data = outX, outy
        elif self.y is None and split:
            self.fit(self.Xtr)
            outXtr, outXte = self.transform(self.Xtr, in_return=in_return) + self.transform(self.Xte, in_return=in_return)
            processed_data = outXtr, outXte
        else:
            self.fit(self.X)
            outX = self.transform(self.X, in_return=in_return)
            processed_data = outX[0]

        return processed_data


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
        num_cols = X.select_dtypes(include=["number"])
        cat_cols = X.select_dtypes(include=["object", "category", "string"])

        self.categorical = Categorical()
        self.numerical = Numerical()

        self.numerical.fit(num_cols, exclude=self.exclude)
        self.categorical.fit(cat_cols, exclude=self.exclude)

        return None

    def transform(self, X: pd.DataFrame, y: pd.Series=None, in_return="ndarray"):
        """
            transform(
                X: pd.DataFrame -> features
                y: pd.Series -> labels
                in_return: ndarray/dataframe/tensor -> you return result type
            )
            Work:
                implement the analyzed work
            Return:
                Tuple(X, y) if y available else (X,)
        """

        num_cols = X.select_dtypes(include=["number"])
        cat_cols = X.select_dtypes(include=["object", "category", "string"])

        processed_num_cols = self.numerical.transform(num_cols, exclude=self.exclude)
        processed_cat_cols = self.categorical.transform(cat_cols, exclude=self.exclude)  

        X = pd.concat([processed_num_cols, processed_cat_cols], axis=1)

        # Change dataframe to user want (in_return) type
        if in_return=="ndarray":
            X = X.values.astype(np.float32)
            if y is not None:
                y = y.values.astype(np.float32)

        if in_return=="tensor":
            try:
                import torch
            except Exception as e:
                raise Exception(f"! Error: Install pytorch explicitly for your system. {e}")
            else:
                X = torch.from_numpy(X.values.astype(np.float32))
                if y is not None:
                    y = torch.from_numpy(y.values.astype(np.float32))

        return (X, y) if y is not None else (X, )


if __name__ == "__main__":
    eda = AutomatedEDA()
    df = pd.read_csv("dummy_cat_1000.csv")
    Xtr, Xte = eda.prepare(df, in_return="tensor")

    print(type(Xtr), type(Xte))

    print(Xtr.shape, Xte.shape) #, ytr.shape, yte.shape)
    print(Xtr.isna().sum().sum(), Xte.isna().sum().sum())
    print("Original:")
    print(df.columns.tolist())

    print("\nTrain:")
    print(Xtr.columns.tolist())

    print("\nTest:")
    print(Xte.columns.tolist())