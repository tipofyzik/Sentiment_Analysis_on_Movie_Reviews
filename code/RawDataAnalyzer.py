import pandas as pd



class RawDataAnalyzer:
    """
    Analyzes the raw dataset to give a hint where the preprocess should start first.

    Attributes: 
        __dataset (pd.DataFrame): The raw dataset to be analyzed.
        __dataset_name (str): The name of the provided dataset.
    """

    def __init__(self, dataset: pd.DataFrame, dataset_name: str) -> None:
        """
        Initializes the RawDataAnalyzer with the provided dataset and its name.

        Args:
            dataset (pd.DataFrame): The raw dataset to be analyzed.
            dataset_name (str): The name of the provided dataset.
        """
        self.__dataset = dataset
        self.__dataset_name = dataset_name

    def __get_dataset_shape(self) -> pd.DataFrame.shape:
        """
        Retrieves the shape of the dataset. 

        Returns:
            (pd.DataFrame.shape): The shape of the dataset.
        """
        return self.__dataset.shape

    def print_dataset_shape(self) -> None:
        """
        Prints the shape of the dataset.
        """
        print(f"The shape of the \'{self.__dataset_name}\' is {self.__get_dataset_shape()}")

    def __check_null_values(self) -> bool:
        """
        Sums up all the null values found in the table and calculates 
        the boolean value that reflects the presence of null values.

        Returns:
            (bool): The value that represents whether there are null values in the dataset.
        """
        self.__null_sum = self.__dataset.isnull().sum().sum()
        self.__have_null = bool(self.__null_sum)
        return self.__have_null
    
    def print_have_null(self) -> None:
        """
        Prints whether the dataset has null values or not.
        """
        self.__check_null_values()
        if self.__have_null:
            print(f"There are {self.__null_sum} of null values in the \'{self.__dataset_name}\'.")
        else:
            print(f"The \'{self.__dataset_name}\' does not have any null values.")

    def print_column_names(self) -> None:
        """
        Prints the list that contains sheet's column names.
        """
        print(self.__dataset.columns)