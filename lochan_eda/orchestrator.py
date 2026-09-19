import pandas as pd
from sklearn.model_selection import train_test_split
from lochan_eda.numerical import Numerical
from lochan_eda.categorical import Categorical

class AutomatedEDA():
    def __init__(self):
        self.exclude = None

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
        if target is not None and split:
            self.fit(self.Xtr, self.ytr)
            # transform Xtrain and Xtest both
            outXtr, outytr, outXte, outyte  = self.transform(self.Xtr, self.ytr) + self.transform(self.Xte, self.yte)
            processed_data = outXtr, outXte, outytr, outyte
        elif target is not None and not split:
            self.fit(self.X, self.y)
            outX, outy = self.transform(self.X, self.y)
            processed_data = outX, outy
        elif target is None and split:
            self.fit(self.Xtr)
            outXtr, outXte = self.transform(self.Xtr) + self.transform(self.Xte)
            processed_data = outXtr, outXte
        else:
            self.fit(self.X)
            outX = self.transform(self.X)
            processed_data = outX

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
        num_cols = X.select_dtypes(include=["number"])
        cat_cols = X.select_dtypes(include=["object", "category", "string"])

        processed_num_cols = self.numerical.transform(num_cols, exclude=self.exclude)
        processed_cat_cols = self.categorical.transform(cat_cols, exclude=self.exclude)  

        X = pd.concat([processed_num_cols, processed_cat_cols], axis=1)
        if self.exclude is not None:
            X = pd.concat([X, self.X[self.exclude]], axis=1)

        return (X, y) if y is not None else (X, )


if __name__ == "__main__":
    eda = AutomatedEDA()
    df = pd.DataFrame(
        {
            "age": [22, 30, 27, 40, 35, 50],
            "income": [20000, 22000, 26000, 50000, 48000, 70000],
            "city": ["A", "B", "A", "C", "B", "A"],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    Xtr, Xte, ytr, yte = eda.prepare(df, target="target")

    print(Xtr.shape, Xte.shape, ytr.shape, yte.shape)