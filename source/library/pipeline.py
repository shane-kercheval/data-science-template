"""Contains custom classes for building sklearn pipeline."""

from typing import Callable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import OrdinalEncoder


class TransformerChooser(BaseEstimator, TransformerMixin):
    """
    Transformer that wraps another Transformer. This allows different transformer objects to be
    tuned.
    """

    def __init__(self, transformer: BaseEstimator | None = None):
        """
        Args:
            transformer:
                Transformer object (e.g. StandardScaler, MinMaxScaler).
        """
        self.transformer = transformer

    def fit(self, X, y=None):  # noqa
        """Fit implementation."""
        if self.transformer is None:
            return self

        return self.transformer.fit(X, y)

    def transform(self, X):  # noqa
        """Transform implementation."""
        if self.transformer is None:
            return X

        return self.transformer.transform(X)

    def get_feature_names_out(self, input_features=None):  # noqa
        if hasattr(self.transformer, 'get_feature_names_out'):
            return self.transformer.get_feature_names_out(input_features)
        return input_features


class CustomFunctionTransformer(FunctionTransformer):
    """
    FunctionTransformer that allows for a custom function to be used, while providing
    a get_feature_names_out method so that, for example, we can get the feature importance from
    the model (which requires us to know the final feature names).
    """

    def set_update_function(self, update_function: Callable) -> None:
        """Sets the function used to determine get_feature_names_out."""
        self.update_function = update_function

    def get_feature_names_out(self, input_features=None):  # noqa
        """
        Generates and returns the feature names for the output of transform. If the
        update_function is not set, then this method will return the input_features.
        """
        if hasattr(self, 'update_function'):
            return self.update_function(input_features)
        return input_features


class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    """First replaces missing values with '<missing>' then applies OrdinalEncoder."""

    def __init__(
            self,
            categories: str | list = 'auto',
            handle_unknown: str = 'use_encoded_value'):
        self._ordinal_encoder = OrdinalEncoder(
            categories=categories,
            handle_unknown=handle_unknown,
            unknown_value=-1,
        )
        self._missing_value = '<missing>'

    def _fill_na(self, X):  # noqa
        """Helper function that fills missing values with strings before calling OrdinalEncoder."""
        for column in X.columns.to_numpy():
            if X[column].dtype.name == 'category':
                if self._missing_value not in X[column].cat.categories:
                    X[column] = X[column].cat.add_categories(self._missing_value)
                X[column] = X[column].fillna(self._missing_value)
        return X

    def fit(self, X, y=None):  # noqa
        """Fit implementation."""
        X = self._fill_na(X)  # noqa: N806
        self._ordinal_encoder.fit(X)
        return self

    def transform(self, X):  # noqa
        """Transform implementation."""
        X = self._fill_na(X)  # noqa: N806
        return self._ordinal_encoder.transform(X)


class SavingsStatusEncoder:
    """
    Custom encoder for the savings status feature. Purely so that the graphs don't print out
    "OrdinalEncoder(categories=['no known savings', ...]".
    """

    def __init__(self):
        savings_status_order = ['no known savings', '<100', '100<=X<500', '500<=X<1000', '>=1000']
        self.encoder = OrdinalEncoder(categories=[savings_status_order])

    def fit(self, X, y=None):  # noqa
        return self.encoder.fit(X, y)

    def transform(self, X):  # noqa
        return self.encoder.transform(X)

    def fit_transform(self, X, y=None):  # noqa
        return self.encoder.fit_transform(X, y)

    def __str__(self) -> str:
        return 'SavingsStatusEncoder()'
