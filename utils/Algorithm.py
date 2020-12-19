from utils.Data import Data
from utils.Result import Result


class Algorithm:
    def execute(self, data: Data) -> Result:
        """Executes the specific algorithm and runs analytics"""
        raise Exception("Not implemented")
        pass

    # noinspection PyMethodMayBeStatic
    def get_name(self) -> str:
        """Returns the name of the given method"""
        raise Exception("Not implemented")
        pass
