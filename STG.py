import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_poisson_deviance
from sklearn.metrics import mean_gamma_deviance
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

#MAE por debajo de 0.05
class ExperimentData:
    def apply(self, data_raw, output_feature, sample=1):
        metadataExperiment = {}
        if sample != 1:
            data = data_raw.sample(frac=sample, random_state=17)
        else:
            data = data_raw
        X = data.drop(output_feature, axis=1)
        Y = data[output_feature]
        X_Norm_Train, X_Norm_Test, Y_Train, Y_Test = train_test_split(
            X, Y, test_size=0.25, train_size=0.75, random_state=17
        )
        metadataExperiment["input_features_train"] = X_Norm_Train
        metadataExperiment["input_features_test"] = X_Norm_Test
        metadataExperiment["output_feature_train"] = Y_Train
        metadataExperiment["output_feature_test"] = Y_Test
        return metadataExperiment

class Metrics_Regression:
    def preprocess_output(self, output_feature_test, predicted_output_feature):
        MN = MinMaxScaler(feature_range=(0, 1))
        Y_Test_R = output_feature_test.values
        Y_Test_R = np.asarray(Y_Test_R)
        Y_merge = np.vstack([Y_Test_R, predicted_output_feature]).T
        for _, n in enumerate(np.arange(0, Y_merge.shape[0])):
            if Y_merge[n, 0] == 0:
                Y_merge[n, 0] = 0.0001
            elif Y_merge[n, 1] == 0:
                Y_merge[n, 1] = 0.0001
        return Y_merge
    def calculate_metric(self, LR, input_features_test, output_feature_test, metric):
        try:
            if metric == "explained_variance_score":
                predicted_output_feature = LR.predict(input_features_test)
                result = explained_variance_score(
                    output_feature_test, predicted_output_feature
                )
            elif metric == "mean_absolute_error":
                predicted_output_feature = LR.predict(input_features_test)
                result = mean_absolute_error(output_feature_test, predicted_output_feature)
            elif metric == "mean_poisson_deviance":
                predicted_output_feature = LR.predict(input_features_test)
                preprocess_output_ = self.preprocess_output(
                    output_feature_test, predicted_output_feature
                )
                result = mean_poisson_deviance(
                    preprocess_output_[:, 0], preprocess_output_[:, 1]
                )
            elif metric == "mean_gamma_deviance":
                predicted_output_feature = LR.predict(input_features_test)
                preprocess_output_ = self.preprocess_output(
                    output_feature_test, predicted_output_feature
                )
                result = mean_gamma_deviance(
                    preprocess_output_[:, 0], preprocess_output_[:, 1]
                )
            else:
                result = cross_val_score(
                    LR, input_features_test, output_feature_test, cv=5, scoring=metric
                ).mean()
        
            return result
        except:
            return np.nan
        
    def getMetricDataFrame(
        self,
        description,
        mse,
        r2,
        explained_variance_score_,
        mean_absolute_error_,
        mean_poisson_deviance_,
        mean_gamma_deviance_,
        is_linear_reg=False
    ):
        if is_linear_reg:
            record = [
                description,
                mse,
                r2,
                explained_variance_score_,
                mean_absolute_error_,
                mean_poisson_deviance_,
                mean_gamma_deviance_
            ]
        else:
            record = [
                description,
                mse,
                np.nan,
                explained_variance_score_,
                mean_absolute_error_,
                mean_poisson_deviance_,
                mean_gamma_deviance_
            ]

        columnsNames = [
            "description",
            "mse",
            "r2",
            "explained_variance_score_",
            "mean_absolute_error_",
            "mean_poisson_deviance_",
            "mean_gamma_deviance_"
        ]
        df = pd.DataFrame(data=[record], columns=columnsNames)
        return df
    def get_model_metadata(self, LR, input_features_test, output_feature_test, description, is_linear_regression=False):
        mse = self.calculate_metric(
            LR,
            input_features_test,
            output_feature_test,
            metric="neg_mean_squared_error",
            )
        r2 = self.calculate_metric(LR, input_features_test, output_feature_test, metric="r2")
        explained_variance_score_ = self.calculate_metric(
            LR,
            input_features_test,
            output_feature_test,
            metric="explained_variance_score",
            )
        mean_absolute_error_ = self.calculate_metric(
            LR, input_features_test, output_feature_test, metric="mean_absolute_error"
            )
        mean_poisson_deviance_ = self.calculate_metric(
            LR, input_features_test, output_feature_test, metric="mean_poisson_deviance"
            )
        mean_gamma_deviance_ = self.calculate_metric(
            LR, input_features_test, output_feature_test, metric="mean_gamma_deviance"
            )
        df = self.getMetricDataFrame(
                description,
                mse,
                r2,
                explained_variance_score_,
                mean_absolute_error_,
                mean_poisson_deviance_,
                mean_gamma_deviance_,
                is_linear_regression
            )
        return df

class LinearRegressionPreprocessing:
    def get_influencias(self, data, output_feature):
        input_features, output_features = data.drop(output_feature, axis=1), data[output_feature]
        output_features_R = np.asarray(output_features.values)
        transformer = sm.OLS(output_features_R, input_features.assign(const=1))
        output_data_raw = transformer.fit()
        output_data_process = OLSInfluence(output_data_raw)
        th = 0.9*output_data_process.hat_matrix_diag.max()
        output_data_process=output_data_process.cooks_distance[0]
        samples_filtered = [distance < th for distance in output_data_process]
        input_features_processed = input_features.loc[samples_filtered]
        output_features_processed = pd.DataFrame(output_features_R, columns=[output_feature]).loc[
            samples_filtered
        ]
        return pd.concat(
            [
                input_features_processed, 
                output_features_processed
            ], 
            axis=1).reset_index().drop("index", axis=1)

class RawLinearRegression(Metrics_Regression):
    def trainLinearRegression(self, input_features_train, output_feature_train):
        LR = LinearRegression()
        LR.fit(input_features_train, output_feature_train)
        return LR

    def get_feature_weights(self, LR, input_features_train):
        score_df = pd.DataFrame(
            LR.coef_, index=input_features_train.columns, columns=["Score"]
        )
        return score_df

    def apply(self, data_raw, output_feature, description, sample=1):
        metadata = {}
        metadataExperiment = ExperimentData().apply(data_raw, output_feature, sample)
        LR = self.trainLinearRegression(
            metadataExperiment["input_features_train"],
            metadataExperiment["output_feature_train"]
            )
        metric_information = self.get_model_metadata(
                LR, 
                metadataExperiment["input_features_test"], 
                metadataExperiment["output_feature_test"], 
                description,
                is_linear_regression=True
            )
        metadata["regressor"] = LR 
        metadata["metric_information"] = metric_information
        return metadata

class RobustLinearRegression(Metrics_Regression):
    def trainRobustLinearRegression(
        self,
        input_features_train,
        output_feature_train,
        max_trials=1000,
        residual_threshold=0.2,
        stop_probability=0.99,
    ):
        LR = LinearRegression()
        LR.fit(input_features_train, output_feature_train)
        MR = RANSACRegressor(
            base_estimator=LR,
            max_trials=max_trials,
            residual_threshold=residual_threshold,
            stop_probability=stop_probability,
        )
        MR.fit(input_features_train, output_feature_train)
        return MR

    def apply(self, data_raw, output_feature, description, sample=1):
        metadata = {}
        metadataExperiment = ExperimentData().apply(data_raw, output_feature, sample)
        LR = self.trainRobustLinearRegression(
            metadataExperiment["input_features_train"],
            metadataExperiment["output_feature_train"]
            )
        metric_information = self.get_model_metadata(
                LR, 
                metadataExperiment["input_features_test"], 
                metadataExperiment["output_feature_test"], 
                description
            )
        metadata["regressor"] = LR 
        metadata["metric_information"] = metric_information
        return metadata

class stackingRegressor(Metrics_Regression):
    def trainRobustLinearRegression(self, input_features_train, output_feature_train):
        RF_1 = RandomForestRegressor(
            n_estimators=200,
            criterion="absolute_error",
            max_depth=7,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            n_jobs=-1,
            ccp_alpha=0.0,
        )
        RF_2 = RandomForestRegressor(
            n_estimators=150,
            criterion="squared_error",
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            n_jobs=-1,
            ccp_alpha=0.0,
        )
        LR = LinearRegression()
        models = {("Random Forest 1", RF_1), ("Random Forest 2", RF_2)}
        SCR = StackingRegressor(models, final_estimator=LR, cv=2, n_jobs=-1)
        SCR.fit(input_features_train, output_feature_train)
        return SCR

    def apply(self, data_raw, output_feature, description, sample=1):
        metadata = {}
        metadataExperiment = ExperimentData().apply(data_raw, output_feature, sample)
        LR = self.trainRobustLinearRegression(
                metadataExperiment["input_features_train"],
                metadataExperiment["output_feature_train"]
            )
        metric_information = self.get_model_metadata(
                LR, 
                metadataExperiment["input_features_test"], 
                metadataExperiment["output_feature_test"], 
                description
            )
        metadata["regressor"] = LR 
        metadata["metric_information"] = metric_information
        return metadata

class svmRegressor(Metrics_Regression):
    def __init__(self, kernel="rbf", gamma="scale", coef0=0.0, C=1.0):
        self.kernel=kernel
        self.gamma=gamma
        self.coef0=coef0 
        self.C=C

    def trainSVMLinearRegression(self, input_features_train, output_feature_train, kernel, gamma, coef0, C):
        regressor = SVR(kernel = kernel, gamma=gamma, coef0=coef0, C=C)
        regressor.fit(input_features_train, output_feature_train)
        return regressor

    def apply(self, data_raw, output_feature, description, sample=1):
        metadata = {}
        metadataExperiment = ExperimentData().apply(data_raw, output_feature, sample)
        LR = self.trainSVMLinearRegression(
                metadataExperiment["input_features_train"],
                metadataExperiment["output_feature_train"],
                kernel=self.kernel,
                gamma=self.gamma,
                coef0=self.coef0,
                C=self.C
            )
        metric_information = self.get_model_metadata(
                LR, 
                metadataExperiment["input_features_test"], 
                metadataExperiment["output_feature_test"], 
                description
            )
        metadata["regressor"] = LR 
        metadata["metric_information"] = metric_information
        return metadata

class mlpRegressor(Metrics_Regression):

    def trainSVMLinearRegression(self, input_features_train, output_feature_train):
        regressor = MLPRegressor(random_state=1)
        regressor.fit(input_features_train, output_feature_train)
        return regressor

    def apply(self, data_raw, output_feature, description, sample=1):
        metadata = {}
        metadataExperiment = ExperimentData().apply(data_raw, output_feature, sample)
        LR = self.trainSVMLinearRegression(
                metadataExperiment["input_features_train"],
                metadataExperiment["output_feature_train"]
            )
        metric_information = self.get_model_metadata(
                LR, 
                metadataExperiment["input_features_test"], 
                metadataExperiment["output_feature_test"], 
                description
            )
        metadata["regressor"] = LR 
        metadata["metric_information"] = metric_information
        return metadata

class rfRegressor(Metrics_Regression):
    def trainSVMLinearRegression(self, input_features_train, output_feature_train):
        regressor = RandomForestRegressor(random_state=1)
        regressor.fit(input_features_train, output_feature_train)
        return regressor

    def apply(self, data_raw, output_feature, description, sample=1):
        metadata = {}
        metadataExperiment = ExperimentData().apply(data_raw, output_feature, sample)
        LR = self.trainSVMLinearRegression(
                metadataExperiment["input_features_train"],
                metadataExperiment["output_feature_train"]
            )
        metric_information = self.get_model_metadata(
                LR, 
                metadataExperiment["input_features_test"], 
                metadataExperiment["output_feature_test"], 
                description
            )
        metadata["regressor"] = LR 
        metadata["metric_information"] = metric_information
        return metadata

class dtRegressor(Metrics_Regression):
    def trainSVMLinearRegression(self, input_features_train, output_feature_train):
        regressor = DecisionTreeRegressor(random_state=1)
        regressor.fit(input_features_train, output_feature_train)
        return regressor

    def apply(self, data_raw, output_feature, description, sample=1):
        metadata = {}
        metadataExperiment = ExperimentData().apply(data_raw, output_feature, sample)
        LR = self.trainSVMLinearRegression(
                metadataExperiment["input_features_train"],
                metadataExperiment["output_feature_train"]
            )
        metric_information = self.get_model_metadata(
                LR, 
                metadataExperiment["input_features_test"], 
                metadataExperiment["output_feature_test"], 
                description
            )
        metadata["regressor"] = LR 
        metadata["metric_information"] = metric_information
        return metadata
