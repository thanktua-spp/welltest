from metaflow import FlowSpec, step
from utility import slice_date
class TrainModels(FlowSpec):
    """_summary_
    """
    
    @step 
    def start(self):
        # bring in all data
        from metaflow import Flow
        import pandas as pd
        run = Flow('PreprocessData').latest_run
        self.X_df = run['data_processing_split'].task.data.X
        self.y_water_df = run['data_processing_split'].task.data.y_water
        self.y_oil_df = run['data_processing_split'].task.data.y_oil
        self.y_liqrate_df = run['data_processing_split'].task.data.liqrate

        
        # self.X_train = run['data_processing_split'].task.data.X_train
        # self.X_test = run['data_processing_split'].task.data.X_test
        # self.y_train_water = run['data_processing_split'].task.data.y_train_water
        # self.y_test_water = run['data_processing_split'].task.data.y_test_water
        # self.y_train_oil = run['data_processing_split'].task.data.y_train_oil
        # self.y_test_oil = run['data_processing_split'].task.data.y_test_oil
        
        self.df_oil = pd.concat([self.X_df, self.y_oil_df], axis=1)
        self.df_water = pd.concat([self.X_df, self.y_water_df], axis=1)
        self.df_liqrate = pd.concat([self.X_df, self.y_liqrate_df], axis=1)
        
        # Filter rows based on the date range
        start_date = pd.to_datetime('2017-04-11')
        end_date = pd.to_datetime('2022-01-01')
        self.df_oil_train = slice_date(self.df_oil, start_date, end_date)
        self.df_water_train = slice_date(self.df_water, start_date, end_date)
        self.df_liqrate_train = slice_date(self.df_liqrate, start_date, end_date)

        self.next(self.train_prophet)
        
    @step
    def train_prophet(self):
        from prophet import Prophet
        self.model_name = "Prophet"

        self.prophet_oilmodel = Prophet()
        self.prophet_oilmodel.fit(self.df_oil_train)

        self.prophet_watermodel = Prophet()
        self.prophet_watermodel.fit(self.df_water_train)
        
        self.prophet_liqratemodel = Prophet()
        self.prophet_liqratemodel.fit(self.df_liqrate_train)
        self.next(self.end)
        
    #@step
    def train_randomforest(self):
        from sklearn.ensemble import RandomForestRegressor
        self.model_name = "RandomForest"

        self.regr_oil = RandomForestRegressor(max_depth=2,
                                          random_state=0)
        self.regr_water = RandomForestRegressor(max_depth=2,
                                          random_state=0)        
        self.regr_liqrate = RandomForestRegressor(max_depth=2,
                                          random_state=0)
        self.regr_oil.fit(self.X_train_transformed, self.y_train_or)
        self.regr_water.fit(self.X_train_transformed, self.y_train_water)

        self.trained_models = {'oil model': self.regr_oil,
                        'water model': self.regr_water}
        self.next(self.join_models)
                
    #@step
    def train_xgboost(self):
        import xgboost as xg     
        self.model_name = "XGBoost"
          
        self.xgb_oil = xg.XGBRegressor(objective ='reg:squarederror',
                        n_estimators = 10, seed = 123)
        self.xgb_water = xg.XGBRegressor(objective ='reg:squarederror',
                        n_estimators = 10, seed = 123)
        self.xgb_oil.fit(self.X_train_transformed, self.y_train_or)
        self.xgb_water.fit(self.X_train_transformed, self.y_train_water)
        self.trained_models = {'oil model': self.xgb_oil,
                               'water model': self.xgb_water}
        self.next(self.join_models)
        
    #@step
    def join_models(self, modeling_tasks):
        self.model = [model.trained_models for model in modeling_tasks]
        self.next(self.end)
        
    @step
    def end(self):
        print('End: Train model workflow complete')            
    
if __name__ == '__main__':
    TrainModels()