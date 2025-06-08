from pathlib import Path

from v2.data.data_provider_impl import DataProvider


class DatabentoFileProvider(DataProvider):
    """
    DatabentoFileProvider is a subclass of DataProvider that provides
    functionality to read data from a file.
    """

    def __init__(self, path: Path):
        """
        Initializes the DatabentoFileProvider with the given file path.

        :param file_path: The path to the file containing the data.
        """
        pass

    def read_data(self):
        """
        Reads data from the specified file.

        :return: The data read from the file.
        """
        with open(self.file_path, 'r') as file:
            return file.read()