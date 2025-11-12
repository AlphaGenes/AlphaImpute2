import os


def read_file(file_path, test_alt_allele_prob=False, **kwargs):
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
    if test_alt_allele_prob:
        MF = values[0]
        values.pop(0)

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
    if test_alt_allele_prob:
        return values, MF
    else:
        return values


def test_1():
    os.system(
        "AlphaImpute2 -genotypes tests/functional_tests/data/genotypes.txt -pedigree tests/functional_tests/data/pedigree.txt -ped_only -phase_output -seg_output -out tests/functional_tests/outputs/test_1"
    )
    assert os.path.exists("tests/functional_tests/outputs/test_1.genotypes")
    assert os.path.exists("tests/functional_tests/outputs/test_1.haplotypes")
    assert os.path.exists("tests/functional_tests/outputs/test_1.segregation")

    genotypes = read_file("tests/functional_tests/outputs/test_1.genotypes")
    expected_genotypes = read_file("tests/functional_tests/data/true_genotypes.txt")
    assert genotypes == expected_genotypes
