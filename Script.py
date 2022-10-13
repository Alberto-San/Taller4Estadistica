from public.categoricalPreprocessing import *
from public.kernelPCA import *	 
from public.spearmanCorrelation import *
from public.common import *		       
from public.Kmeans import *	     
from public.statistic import *
from public.dimensionalReductionCommon import *  
from public.pca import *
from public.ica import *			       
from public.removeOutliersLOF import *
from pandas.api.types import is_numeric_dtype
import pandas as pd
import pickle

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

from Taller4Estadistica.STG import *


def imputation_data(data_processed):
  columns = data_processed.columns
  for column in columns:
    median = data_processed[column].median()
    data_processed[column] = data_processed[column].fillna(median)
  return data_processed

def executeOutlierAnalisis(CHECKPOINT_LOF):
    if not CHECKPOINT_LOF:
        df = pd.read_csv('household_power_consumption.txt', sep=';')
        df_raw = df.replace("?", np.nan)
        data_transformed = df_raw.copy()
        data_transformed["Date"] = (
            data_transformed["Date"]
            .apply(lambda x: x.split("/"))
            .apply(lambda x: "{}{}{}".format(x[2], x[1], x[0]))
            .astype(np.int)
        )
        data_transformed["Time"] = (
            data_transformed["Time"]
            .apply(lambda x: x.split(":"))
            .apply(lambda x: int(x[0])*60*60 + int(x[1])*60 + int(x[2]))
        )
        data_transformed = data_transformed.astype(float)
        data_imputed = imputation_data(data_transformed)
        output_feature = "Global_intensity"
        summary_df, data_dic_collection = (
            Remove_Outliers_LOF()
            .process_outliers_non_labeled(
                    data_imputed, 
                    regression=True, 
                    feature=output_feature
                )
            )
        
        with open('data_dic_collection.pickle', 'wb') as f:
            pickle.dump(data_dic_collection, f)
        f.close()
        with open('data_imputed.pickle', 'wb') as f:
            pickle.dump(data_imputed, f)
        f.close()
    else:    
        file = open('data_dic_collection.pickle', 'rb')
        data_dic_collection = pickle.load(file)
        file.close()
        file = open('data_imputed.pickle', 'rb')
        data_imputed = pickle.load(file)
        file.close()
    return data_imputed, data_dic_collection

def executeOutlierAnalisisWithoutTime(CHECKPOINT_LOF):
    if not CHECKPOINT_LOF:
        df = pd.read_csv('household_power_consumption.txt', sep=';')
        df_raw = df.replace("?", np.nan)
        data_transformed = df_raw.copy()
        data_transformed = data_transformed.drop("Date", axis=1).drop("Time", axis=1)
        data_transformed = data_transformed.astype(float)
        data_imputed = imputation_data(data_transformed)
        output_feature = "Global_intensity"
        summary_df, data_dic_collection = (
            Remove_Outliers_LOF()
            .process_outliers_non_labeled(
                    data_imputed, 
                    regression=True, 
                    feature=output_feature
                )
            )
        
        with open('data_dic_collection_without_time.pickle', 'wb') as f:
            pickle.dump(data_dic_collection, f)
        f.close()
        with open('data_imputed_without_time.pickle', 'wb') as f:
            pickle.dump(data_imputed, f)
        f.close()
    else:    
        file = open('data_dic_collection_without_time.pickle', 'rb')
        data_dic_collection = pickle.load(file)
        file.close()
        file = open('data_imputed_without_time.pickle', 'rb')
        data_imputed = pickle.load(file)
        file.close()
    return data_imputed, data_dic_collection

def getDicScalerData(data_imputed, data_dic_collection, output_feature = "Global_intensity"):
    data_dic = {}
    data_dic[""] = get_scale_data_regression(
        data_imputed, 
        data_dic_collection, 
        scaler="", 
        feature=output_feature
        )
    data_dic["robust"] = get_scale_data_regression(
        data_imputed, 
        data_dic_collection, 
        scaler="robust", 
        feature=output_feature
        )
    data_dic["StandardScaler"] = get_scale_data_regression(
        data_imputed, 
        data_dic_collection, 
        scaler="StandardScaler", 
        feature=output_feature
        )
    data_dic["mixMax"] = get_scale_data_regression(
        data_imputed, 
        data_dic_collection, 
        scaler="mixMax", 
        feature=output_feature
        )
    return data_dic

def get_summary_algorithms(data_dic, scaler, flag_first=True, summary_dim_reduction=None, regression=False, output=None):
  algoritms = [PCA_(),
               ICA_()]
  for algorimth in algoritms:
    data = data_dic[scaler]
    if regression:
      X, Y = data.drop(output, axis=1), data[output]
      algorimth.apply(X, scaler)
    else:
      algorimth.apply(data, scaler)
    if flag_first:
      summary_dim_reduction = algorimth.metadata["summary"]
      flag_first = False
    else:
      df = algorimth.metadata["summary"]
      summary_dim_reduction = concat_vertical(summary_dim_reduction, df)
  return summary_dim_reduction, flag_first

def get_summary_scaler(data_dic, regression=False, output=None):
  scalers = list(data_dic.keys())
  flag_first = True
  summary_dim_reduction = None
  for index in range(len(scalers)):
    scaler = scalers[index]
    summary_dim_reduction, flag_first = get_summary_algorithms(data_dic, scaler, flag_first, summary_dim_reduction, regression=regression, output=output)
  columns = ["Tipo de Escalador", "Metodo", "No Componentes (98% Varianza)", "Error Cuadratico Medio Escalado"]
  summary_dim_reduction = summary_dim_reduction[columns]
  summary_dim_reduction["Numero de Componentes Optimo"] = summary_dim_reduction["No Componentes (98% Varianza)"]
  summary_dim_reduction = summary_dim_reduction.drop("No Componentes (98% Varianza)", axis=1)
  return summary_dim_reduction

def executeRegressors(data_dic_pre, output_feature, name, sample=1):
  metadata = {}
  
  scalers = list(data_dic_pre.keys())
  algoritms = {
    "pca": PCA_(),
    "ica": ICA_()
  }
  linear_regressors = {
    "regresion lineal": RawLinearRegression(),
    "regresion robusta": RobustLinearRegression(),
    "regresion stacking": stackingRegressor()
  }
  influencias = {
    "": None,
    "Con Influencias": LinearRegressionPreprocessing()
  }
  summary = None
  flagSummary = True
  for scaler in scalers:
    for method in algoritms.keys():
      for influencia in influencias.keys():
        if influencia!="":
          raw_data = influencias[influencia].get_influencias(
            data_dic_pre[scaler],
            output_feature
          )
        else:
          raw_data = data_dic_pre[scaler]
        for regressor in linear_regressors.keys():
          method_ = algoritms[method]
          regressor_ =  linear_regressors[regressor]
          description = "scaler: {} - dim red: {} - regressor: {} - influencia: {}".format(scaler, method, regressor, influencia)
          print(description)
          method_.apply(raw_data.drop(output_feature, axis=1), scaler)
          datos_reduce = pd.concat(
            [
              method_.metadata["transformed_data"],
              raw_data[output_feature]
            ],
            axis=1
          )
          if regressor=="regresion stacking":
            sample = 0.001
          else:
            sample = 1
          metadata_ = regressor_.apply(
              datos_reduce, 
              output_feature, 
              description=description, 
              sample=sample
            )
          metadata[description] = metadata_["regressor"]
          if flagSummary:
            flagSummary = False
            summary = metadata_["metric_information"]
            summary.to_csv(name, index=False)
            #mode='a'
          else:
            summary = metadata_["metric_information"]
            summary.to_csv(name, index=False, mode='a', header=False)
  return summary, metadata
      

# CHECKPOINT = False
# output_feature = "Global_intensity"
# executions = {
#   "with Time": [CHECKPOINT, "withTime.csv", 1],
#   "without Time": [CHECKPOINT, "withoutTime.csv", 1]
# }

# for execution in executions.keys():
#   CHECKPOINT_LOF, name, sample = executions[execution]
#   if execution == "with Time":
#     data_imputed, data_dic_collection = executeOutlierAnalisis(CHECKPOINT_LOF)
#   else:
#     data_imputed, data_dic_collection = executeOutlierAnalisisWithoutTime(CHECKPOINT_LOF)
#   data_dic = getDicScalerData(data_imputed, data_dic_collection)
#   summary, metadata = executeRegressors(data_dic, output_feature, name=name, sample=sample)
