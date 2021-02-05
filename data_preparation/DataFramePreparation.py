from typing import List

from loading.LoadingUtils import LoadedData
from utils.Data import DataFrameData
from utils.DataProvider import DataProvider
import pandas as pd

from utils.DataSelector import DataSelector


class DataFramePreparation(DataProvider):

    def execute(self, rawData: LoadedData, test_ids: List[str], data_selector: DataSelector):
        dataFrame = pd.read_csv("./Data.csv")

        # Transform raw data to pandas DataFrame including POS-tags and save as CSV-file
        # dataFrame = dataToDataFrame(rawData, test_ids)
        # dataFrameToCSV(dataFrame)

        # Add TestIDs to already existing CSV-file (not necessary anymore, because they now get implemented, too)
        # addTestIDs(raw_data, test_ids, "./Data.csv")

        train_data = dataFrame.loc[dataFrame["inTestIDs"] == bool(False)].drop(["SentenceNr", "inTestIDs"], axis=1)
        test_data = dataFrame.loc[dataFrame["inTestIDs"] == bool(True)].drop(["SentenceNr", "inTestIDs"], axis=1)

        x_train = train_data[["Token", "POS_tag"]].to_dict('records')
        y_train = train_data.drop(["Token", "POS_tag"], axis=1)

        x_test = test_data[["Token", "POS_tag"]].to_dict('records')
        y_test = test_data.drop(["Token", "POS_tag"], axis=1)


        return DataFrameData(x_train, y_train, x_test, y_test)
