import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.feature_selection import chi2, f_classif
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class TwoStageFeatureSelector:

    def __init__(
        self,
        num_univariate_features=50,
        cv=5,
        max_iter=1000,
        random_state=9,
        n_jobs=-1,
    ):
        """Two-stage feature selection pipeline:

        Stage 1: Composite univariate statistical ranking (t-test, Chi-square,
        ANOVA).
        Stage 2: LassoCV regularized regression on the filtered features.
        """
        self.num_univariate_features = num_univariate_features
        self.cv = cv
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.univariate_features_ = []  # Features retained after Stage 1
        self.selected_features_ = []  # Final features with non-zero coef in Stage 2
        self.lasso_model_ = None
        self.lasso_coef_ = None

    def fit(self, X, y):
        # Ensure input data is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        feature_names = X.columns

        # =========================================================================
        # Stage 1: Univariate Statistical Filtering (Composite Ranking)
        # =========================================================================

        # 1. Two-sided independent two-sample t-test (|t-value|)
        t_scores = []
        for column in feature_names:
            group1 = X[column][y == 0]
            group2 = X[column][y == 1]
            t_stat, _ = ttest_ind(group1, group2, equal_var=False)
            t_scores.append(np.abs(t_stat))

        # 2. Chi-square scoring (requires Min-Max normalization into non-negative range)
        scaler = MinMaxScaler()
        X_minmax = scaler.fit_transform(X)
        chi2_scores, _ = chi2(X_minmax, y)

        # 3. ANOVA F-statistics (computed on original continuous features)
        anova_scores, _ = f_classif(X, y)

        # 4. Convert statistical scores to rank positions (Rank 1 = Highest Score)
        ranking_df = pd.DataFrame(index=feature_names)
        ranking_df['t_rank'] = (
            pd.Series(t_scores, index=feature_names)
            .rank(ascending=False, method='min')
        )
        ranking_df['chi2_rank'] = (
            pd.Series(chi2_scores, index=feature_names)
            .rank(ascending=False, method='min')
        )
        ranking_df['anova_rank'] = (
            pd.Series(anova_scores, index=feature_names)
            .rank(ascending=False, method='min')
        )

        # 5. Composite Ranking: Average the ranking positions
        ranking_df['composite_rank'] = ranking_df[
            ['t_rank', 'chi2_rank', 'anova_rank']
        ].mean(axis=1)

        # Select the top num_univariate_features
        ranking_df = ranking_df.sort_values(by='composite_rank', ascending=True)
        self.univariate_features_ = ranking_df.index[
            : self.num_univariate_features
        ].tolist()

        print('==============================================================')
        print(
            f'Stage 1 (Univariate Filter) Completed: {len(self.univariate_features_)} features retained.'
        )

        # =========================================================================
        # Stage 2: LassoCV Feature Selection on Filtered Features
        # =========================================================================
        X_stage1 = X[self.univariate_features_]

        # Standardize features before fitting Lasso to avoid scale bias
        zscore = StandardScaler()
        X_stage1_scaled = zscore.fit_transform(X_stage1)

        # Define alpha search space
        alphas = np.logspace(-3, 2, 100)

        self.lasso_model_ = LassoCV(
            alphas=alphas,
            cv=self.cv,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        ).fit(X_stage1_scaled, y)

        # Extract non-zero coefficients
        coef_series = pd.Series(
            self.lasso_model_.coef_, index=self.univariate_features_
        )
        self.lasso_coef_ = coef_series[coef_series != 0]
        self.selected_features_ = self.lasso_coef_.index.tolist()

        print(
            f'Stage 2 (LassoCV) Completed: Optimal Alpha = {self.lasso_model_.alpha_:.6f}'
        )
        print(
            f'Final Non-zero Features Selected: {len(self.selected_features_)}'
        )
        print(self.selected_features_)
        print('==============================================================')

        return self

    def transform(self, X):
        """Returns the dataset restricted to the final selected features."""
        return X[self.selected_features_]

    def fit_transform(self, X, y):
        """Fits the two-stage selector and transforms the input data."""
        self.fit(X, y)
        return self.transform(X)