from metaflow import FlowSpec, step, IncludeFile, Parameter
from utility import get_data, data_processing, data_split


data_path = '/home/tegbe/2023 dev projects/well-test/data_set/INHOUSE DATA/Well 4SS x Inhouse Data.xlsx'

class PreprocessData(FlowSpec):
    welltest_data = IncludeFile("data", default=data_path)

    @step
    def start(self):
        """_summary_
        """
        self.welltest_df = get_data(data_path)
        self.next(self.data_processing_split)

    @step
    def data_processing_split(self):
        """_summary_
        """
        self.X, self.y_oil, self.y_water, self.liqrate = data_processing(self.welltest_df)

        self.X_train, self.X_test, \
        self.y_train_oil, self.y_test_oil, \
        self.y_train_water, self.y_test_water = data_split(self.X, self.y_water, self.y_oil)
        
        assert self.X.isna().sum().sum() == 0
        
        self.next(self.end) 
        
    @step 
    def end(self):
        print('Train workflow completed')
        
        
if __name__ == '__main__':
    PreprocessData()