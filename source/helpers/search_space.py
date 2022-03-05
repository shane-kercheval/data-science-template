from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.svm import LinearSVC
from skopt.space import Categorical, Real, Integer
from xgboost import XGBClassifier

from helpsk.sklearn_pipeline import CustomOrdinalEncoder


def create_pipeline():
    pass


def create_search_space(iterations=50, random_state=42):
    return [
        (
            {
                'model':
                    Categorical(categories=(LogisticRegression(max_iter=1000, random_state=random_state))),
                'model__C':
                    Real(low=1e-06, high=100, prior='log-uniform', transform='identity'),
                'prep__numeric__imputer__transformer':
                    Categorical(
                        categories=(
                            SimpleImputer(),
                            SimpleImputer(strategy='median'),
                            SimpleImputer(strategy='most_frequent')
                        ),
                        prior=[0.5, 0.25, 0.25]
                    ),
                'prep__numeric__scaler__transformer':
                    Categorical(categories=(StandardScaler(), MinMaxScaler()), prior=[0.65, 0.35]),
                'prep__numeric__pca__transformer':
                    Categorical(categories=(None, PCA(n_components='mle'))),
                'prep__non_numeric__encoder__transformer':
                    Categorical(
                        categories=(
                            OneHotEncoder(handle_unknown='ignore'),
                            CustomOrdinalEncoder()
                        ),
                        prior=[0.65, 0.35]
                    )
            },
            iterations
        ),
        (
            {
                'model':
                    Categorical(categories=(LogisticRegression(max_iter=1000, random_state=random_state))),
                'prep__numeric__imputer__transformer':
                    Categorical(categories=(SimpleImputer())),
                'prep__numeric__scaler__transformer':
                    Categorical(categories=(StandardScaler())),
                'prep__numeric__pca__transformer':
                    Categorical(categories=None),
                'prep__non_numeric__encoder__transformer':
                    Categorical(categories=(OneHotEncoder(handle_unknown='ignore')))
            },
            1
        ),
        (
            {
                'model':
                    Categorical(categories=(LinearSVC(random_state=random_state))),
                'model__C':
                    Real(low=1e-06, high=100, prior='log-uniform', transform='identity'),
                'prep__numeric__imputer__transformer':
                    Categorical(
                        categories=(
                            SimpleImputer(),
                            SimpleImputer(strategy='median'),
                            SimpleImputer(strategy='most_frequent')
                        ),
                        prior=[0.5, 0.25, 0.25]
                    ),
                'prep__numeric__scaler__transformer':
                    Categorical(categories=(StandardScaler(), MinMaxScaler()), prior=[0.65, 0.35]),
                'prep__numeric__pca__transformer':
                    Categorical(categories=(None, PCA(n_components='mle'))),
                'prep__non_numeric__encoder__transformer':
                    Categorical(
                        categories=(
                            OneHotEncoder(handle_unknown='ignore'),
                            CustomOrdinalEncoder()
                        ),
                        prior=[0.65, 0.35]
                    )
            },
            iterations
        ),
        (
            {
                'model':
                    Categorical(categories=(LinearSVC(random_state=random_state))),
                'prep__numeric__imputer__transformer':
                    Categorical(categories=(SimpleImputer())),
                'prep__numeric__scaler__transformer':
                    Categorical(categories=(StandardScaler())),
                'prep__numeric__pca__transformer':
                    Categorical(categories=None),
                'prep__non_numeric__encoder__transformer':
                    Categorical(categories=(OneHotEncoder(handle_unknown='ignore')))
            },
            1
        ),
        (
            {
                'model':
                    Categorical(categories=(
                        ExtraTreesClassifier(bootstrap=True, n_estimators=500, random_state=random_state)
                    )),
                'model__max_features':
                    Real(low=0.01, high=0.95, prior='uniform', transform='identity'),
                'model__max_depth':
                    Integer(low=1, high=100, prior='uniform', transform='identity'),
                'model__n_estimators':
                    Integer(low=500, high=2000, prior='uniform', transform='identity'),
                'model__min_samples_split':
                    Integer(low=2, high=50, prior='uniform', transform='identity'),
                'model__min_samples_leaf':
                    Integer(low=1, high=50, prior='uniform', transform='identity'),
                'model__max_samples':
                    Real(low=0.5, high=1.0, prior='uniform', transform='identity'),
                'model__criterion':
                    Categorical(categories=('gini', 'entropy')),
                'prep__numeric__imputer__transformer':
                    Categorical(
                        categories=(
                            SimpleImputer(),
                            SimpleImputer(strategy='median'),
                            SimpleImputer(strategy='most_frequent')
                        ),
                        prior=[0.5, 0.25, 0.25]
                    ),
                'prep__numeric__scaler__transformer':
                    Categorical(categories=None),
                'prep__numeric__pca__transformer':
                    Categorical(categories=(None, PCA(n_components='mle'))),
                'prep__non_numeric__encoder__transformer':
                    Categorical(
                        categories=(
                            OneHotEncoder(handle_unknown='ignore'),
                            CustomOrdinalEncoder()
                        ),
                        prior=[0.65, 0.35]
                    )
            },
            iterations
        ),
        (
            {
                'model':
                    Categorical(categories=(
                        ExtraTreesClassifier(bootstrap=True, n_estimators=500, random_state=random_state)
                    )),
                'prep__numeric__imputer__transformer':
                    Categorical(categories=(SimpleImputer())),
                'prep__numeric__scaler__transformer':
                    Categorical(categories=None),
                'prep__numeric__pca__transformer':
                    Categorical(categories=None),
                'prep__non_numeric__encoder__transformer':
                    Categorical(categories=(OneHotEncoder(handle_unknown='ignore')))
            },
            1
        ),
        (
            {
                'model':
                    Categorical(categories=(RandomForestClassifier(n_estimators=500, random_state=random_state))),
                'model__max_features':
                    Real(low=0.01, high=0.95, prior='uniform', transform='identity'),
                'model__max_depth':
                    Integer(low=1, high=100, prior='uniform', transform='identity'),
                'model__n_estimators':
                    Integer(low=500, high=2000, prior='uniform', transform='identity'),
                'model__min_samples_split':
                    Integer(low=2, high=50, prior='uniform', transform='identity'),
                'model__min_samples_leaf':
                    Integer(low=1, high=50, prior='uniform', transform='identity'),
                'model__max_samples':
                    Real(low=0.5, high=1.0, prior='uniform', transform='identity'),
                'model__criterion':
                    Categorical(categories=('gini', 'entropy')),
                'prep__numeric__imputer__transformer':
                    Categorical(
                        categories=(
                            SimpleImputer(),
                            SimpleImputer(strategy='median'),
                            SimpleImputer(strategy='most_frequent')),
                        prior=[0.5, 0.25, 0.25]),
                'prep__numeric__scaler__transformer':
                    Categorical(categories=None),
                'prep__numeric__pca__transformer':
                    Categorical(categories=(None, PCA(n_components='mle'))),
                'prep__non_numeric__encoder__transformer':
                    Categorical(
                        categories=(
                            OneHotEncoder(handle_unknown='ignore'),
                            CustomOrdinalEncoder()
                        ),
                        prior=[0.65, 0.35]
                    )
            },
            iterations
        ),
        (
            {
                'model':
                    Categorical(categories=(RandomForestClassifier(n_estimators=500, random_state=random_state))),
                'prep__numeric__imputer__transformer':
                    Categorical(categories=(SimpleImputer())),
                'prep__numeric__scaler__transformer':
                    Categorical(categories=None),
                'prep__numeric__pca__transformer':
                    Categorical(categories=None),
                'prep__non_numeric__encoder__transformer':
                    Categorical(categories=(OneHotEncoder(handle_unknown='ignore')))
            },
            1
        ),
        (
            {
                'model':
                    Categorical(categories=(
                        XGBClassifier(
                            n_estimators=500,
                            eval_metric='logloss',
                            use_label_encoder=False,
                            random_state=random_state,
                        )
                    )),
                'model__max_depth':
                    Integer(low=1, high=20, prior='log-uniform', transform='identity'),
                'model__learning_rate':
                    Real(low=0.01, high=0.3, prior='log-uniform', transform='identity'),
                'model__n_estimators':
                    Integer(low=500, high=2000, prior='uniform', transform='identity'),
                'model__min_child_weight':
                    Integer(low=1, high=50, prior='log-uniform', transform='identity'),
                'model__subsample':
                    Real(low=0.5, high=1, prior='uniform', transform='identity'),
                'model__colsample_bytree':
                    Real(low=0.5, high=1, prior='uniform', transform='identity'),
                'model__colsample_bylevel':
                    Real(low=0.5, high=1, prior='uniform', transform='identity'),
                'model__reg_alpha':
                    Real(low=0.0001, high=1, prior='log-uniform', transform='identity'),
                'model__reg_lambda':
                    Real(low=1, high=4, prior='log-uniform', transform='identity'),
                'prep__numeric__imputer__transformer':
                    Categorical(
                        categories=(
                            SimpleImputer(),
                            SimpleImputer(strategy='median'),
                            SimpleImputer(strategy='most_frequent')),
                        prior=[0.5, 0.25, 0.25]
                    ),
                'prep__numeric__scaler__transformer':
                    Categorical(categories=None),
                'prep__numeric__pca__transformer':
                    Categorical(categories=(None, PCA(n_components='mle'))),
                'prep__non_numeric__encoder__transformer':
                    Categorical(
                        categories=(
                            OneHotEncoder(handle_unknown='ignore'),
                            CustomOrdinalEncoder()),
                        prior=[0.65, 0.35]
                    )
            },
            iterations
        ),
        (
            {
                'model':
                    Categorical(categories=(
                        XGBClassifier(
                            n_estimators=500,
                            eval_metric='logloss',
                            use_label_encoder=False,
                            random_state=random_state,
                        )
                    )),
                'prep__numeric__imputer__transformer':
                    Categorical(categories=(SimpleImputer())),
                'prep__numeric__scaler__transformer':
                    Categorical(categories=None),
                'prep__numeric__pca__transformer':
                    Categorical(categories=None),
                'prep__non_numeric__encoder__transformer':
                    Categorical(categories=(OneHotEncoder(handle_unknown='ignore')))
            },
            1
        ),
    ]
