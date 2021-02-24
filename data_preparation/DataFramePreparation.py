from typing import List

from loading.LoadingUtils import LoadedData
from loading.Preprocessing import dataToDataFrame, dataFrameToCSV
from utils.Data import MultiLabelData
from utils.DataProvider import DataProvider
import pandas as pd
from utils.DataSelector import DataSelector


class DataFramePreparation(DataProvider):

    def execute(self, rawData: LoadedData, test_ids: List[str], data_selector: DataSelector) -> MultiLabelData:

        # try to load existing data file; if no data can be found, create new file

        try:
            dataFrame = pd.read_csv("./Data.csv")
        except FileNotFoundError:
            dataFrame = dataToDataFrame(rawData, test_ids)
            dataFrameToCSV(dataFrame)

        train_data = dataFrame.loc[dataFrame["inTestIDs"] == bool(False)].drop(["SentenceNr", "inTestIDs"], axis=1)
        test_data = dataFrame.loc[dataFrame["inTestIDs"] == bool(True)].drop(["SentenceNr", "inTestIDs"], axis=1)

        x_train = train_data[["Token", "POS_tag"]].to_dict('records')
        y_train = train_data.drop(["Token", "POS_tag"], axis=1).values.tolist()

        x_test = test_data[["Token", "POS_tag"]].to_dict('records')
        y_test = test_data.drop(["Token", "POS_tag"], axis=1).values.tolist()

        return MultiLabelData(x_train, y_train, x_test, y_test)
