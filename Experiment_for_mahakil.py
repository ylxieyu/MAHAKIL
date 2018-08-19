from Tools.DataSetTool import DataSetTool
from MAHAKIL.mahakil import MAHAKIL

data_list, label_list = DataSetTool.init_data("D:\\data_for_where\\", 20, False, False)
data = data_list[0]
label = label_list[0]
MAHAKIL().fit_sample(data, label)
