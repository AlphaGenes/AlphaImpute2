import os
import subprocess
import numpy as np


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
    """basic functionality test with pedigree only mode"""
    os.system(
        "AlphaImpute2 -genotypes tests/functional_tests/test_1/genotypes.txt -pedigree tests/functional_tests/test_1/pedigree.txt -ped_only -phase_output -seg_output -out tests/functional_tests/outputs/test_1"
    )
    assert os.path.exists("tests/functional_tests/outputs/test_1.genotypes")
    assert os.path.exists("tests/functional_tests/outputs/test_1.haplotypes")
    assert os.path.exists("tests/functional_tests/outputs/test_1.segregation")

    genotypes = read_file("tests/functional_tests/outputs/test_1.genotypes")
    expected_genotypes = read_file("tests/functional_tests/test_1/true_genotypes.txt")
    assert genotypes == expected_genotypes


def test_2():
    """check if genotype and phase outputs match for pedigree imputation, and error handling for insufficient HD individuals for population imputation"""
    os.system(
        "AlphaImpute2 -genotypes tests/functional_tests/test_2/genotypes.txt -pedigree tests/functional_tests/test_2/pedigree.txt -ped_only -phase_output -out tests/functional_tests/outputs/test_2"
    )
    assert os.path.exists("tests/functional_tests/outputs/test_2.genotypes")
    assert os.path.exists("tests/functional_tests/outputs/test_2.haplotypes")

    genotypes = read_file("tests/functional_tests/outputs/test_2.genotypes")
    genotypes = np.array(genotypes, dtype=int)

    haplotypes = read_file("tests/functional_tests/outputs/test_2.haplotypes")
    haplotypes = np.array(haplotypes, dtype=int)

    for ind, genotype in enumerate(genotypes):
        hap_0 = haplotypes[ind * 2][1:]
        hap_1 = haplotypes[ind * 2 + 1][1:]
        assert np.all(genotype[1:] == hap_0 + hap_1)

    error_message = "Too few HD individuals were found for population imputation (the filter is set at 10 HD individuals, but you likely need way more!). Population imputation cannot proceed. Consider using -ped_only imputation or reducing the -hd_threshold parameter."

    pipes_1 = subprocess.Popen(
        [
            "AlphaImpute2",
            "-genotypes",
            "tests/functional_tests/test_2/genotypes.txt",
            "-pedigree",
            "tests/functional_tests/test_2/pedigree.txt",
            "-phase_output",
            "-out",
            "tests/functional_tests/outputs/test_2",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _, std_err = pipes_1.communicate()
    decoded_std_err = std_err.decode("utf-8")
    assert error_message in decoded_std_err

    pipes_2 = subprocess.Popen(
        [
            "AlphaImpute2",
            "-genotypes",
            "tests/functional_tests/test_2/genotypes.txt",
            "-pedigree",
            "tests/functional_tests/test_2/pedigree.txt",
            "-pop_only",
            "-phase_output",
            "-out",
            "tests/functional_tests/outputs/test_2",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _, std_err = pipes_2.communicate()
    decoded_std_err = std_err.decode("utf-8")
    assert error_message in decoded_std_err
