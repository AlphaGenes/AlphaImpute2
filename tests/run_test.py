import os
import shutil


def read_file(file_path, **kwargs):
    """
    INPUT:
    file_path: str, the path of the file to be read
    decimal_place(optional): if provided, round the data with the given
    decimal places
    OUTPUT:
    values: 2d list of str, store the values of the records
    """
    with open(file_path, "r") as file:
        values = [line.strip().split() for line in file]

    if "decimal_place" in kwargs.keys():
        # round the data if data and rounding decimal place exist
        values = [
            [line[0]]
            + [round(float(data), kwargs["decimal_place"]) for data in line[1:]]
            if line
            else line
            for line in values
        ]
    else:
        # convert data to float for comparison if data exists
        values = [
            [line[0]] + [float(data) for data in line[1:]] if line else line
            for line in values
        ]

    return values


def read_and_sort_file(file_path, id_list=None, **kwargs):
    """
    INPUT:
    file_path: str, the path of the file to be read
    id_list: None or list of int, the ids of the records to be read
    OUTPUT:
    values: 2d list of str, store the sorted values of the (selected) records
    """

    values = read_file(file_path, **kwargs)

    if id_list is not None:
        # consider only the entries with id in id_list
        values = [row for row in values if row[0] in id_list]

    # remove the empty strings
    values = list(filter(None, values))

    # sort according to the id
    values.sort(key=lambda row: row[0])

    return values


def delete_columns(two_d_list, col_del):
    """
    Delete the columns in col_del of two_d_list
    """
    for n in range(len(col_del)):
        for row in two_d_list:
            del row[col_del[n] - n - 1]


class TestClass:
    path = os.path.join("tests")
    command = "AlphaImpute2 "
    test_cases = None
    test_file_depend_on_test_cases = None

    # all the input file options except the binary file
    files_to_input = ["genotypes", "pedigree"]
    # all the output files except the binary file and the parameter files
    files_to_check = ["true_genotypes"]

    def mk_output_dir(self):
        """
        Prepare a empty folder at the input path
        """
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)

        os.mkdir(self.output_path)

    def generate_command(self):
        """
        generate the command for the test
        """
        for file in self.input_files:
            if (
                (self.test_cases is not None)
                and (self.input_file_depend_on_test_cases is not None)
                and (file in self.input_file_depend_on_test_cases)
            ):
                self.command += f"-{file} {os.path.join(self.path, f'{file}-{self.test_cases}.txt')} "
            else:
                self.command += f"-{file} {os.path.join(self.path, f'{file}.txt')} "

        for key, value in self.arguments.items():
            if value is not None:
                self.command += f"-{key} {value} "
            else:
                self.command += f"-{key} "

        self.command += (
            f"-out {os.path.join(self.output_path, self.output_file_prefix)}"
        )

    def prepare_path(self):
        """
        Initialize the paths for the test
        """
        self.path = os.path.join(self.path, self.test_name)
        self.output_path = os.path.join(self.path, "outputs")
        self.mk_output_dir()

    def check_files(self):
        """
        Run the full algorithm which runs both population and pedigree based imputation.
        """

        def check(file_type):
            return os.path.exists(
                os.path.join(
                    self.output_path, f"{self.output_file_prefix}.{file_type}.txt"
                )
            )

        files = [
            "genotypes",
        ]
        return [check(file) for file in files]

    def test_one(self):
        """
        Running population and pedigree based imputation
        """
        self.test_name = "test_one"
        self.prepare_path()

        self.input_files = ["genotypes", "pedigree"]
        self.arguments = {
            "maxthreads": "4",
        }
        self.output_file_prefix = "files"
        self.output_file_to_check = "genotypes"

        self.generate_command()
        print(self.command)
        os.system(self.command)

        self.output_file_path = os.path.join(
            self.output_path,
            f"{self.output_file_prefix}.{self.output_file_to_check}",
        )
        self.expected_file_path = os.path.join(
            self.path, f"true_{self.output_file_to_check}.txt"
        )

        self.output = read_and_sort_file(self.output_file_path)
        self.expected = read_and_sort_file(self.expected_file_path)

        assert self.output == self.expected
