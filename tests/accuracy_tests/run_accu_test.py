import os
import shutil
import numpy as np
import warnings
import pytest


@pytest.fixture(scope="session")
def sim_path():
    return os.path.join("tests", "accuracy_tests", "sim_for_alphapeel_accu_test")


def prepare_path(output_path):
    """
    Prepare an empty output folder
    """
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    os.mkdir(output_path)


def generate_output_path(name):
    return os.path.join(
        "tests",
        "accuracy_tests",
        "outputs",
        name,
    )


def generate_command(
    sim_path,
    method,
    output_path,
):
    command = "AlphaImpute2 "
    input_file = ["pedigree"]
    arguments = {
        "cycles": "5",
        "maxthreads": "6",
        "phase_output": None,
        "seg_output": None,
    }

    input_file.append("genotypes")

    if method in ["pop_only", "combined"]:
        arguments["hd_threshold"] = "0.8"

    if method in ["pop_only", "ped_only"]:
        command += f"-{method} "

    for file in input_file:
        command += f"-{file} {os.path.join(sim_path, f'{file}.txt')} "

    for key, value in arguments.items():
        if value is not None:
            command += f"-{key} {value} "
        else:
            command += f"-{key} "

    command += f"-out {output_path}{os.sep}"

    return command


def make_directory(path):
    """
    Prepare a empty folder at the input path
    """
    if os.path.exists(path):
        shutil.rmtree(path)

    os.mkdir(path)


def get_marker_accu(output, real):
    """
    Get marker accuracy between the output and the real data
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        accus = np.array(
            [np.corrcoef(real[:, i], output[:, i])[0, 1] for i in range(real.shape[1])]
        )
        return round(np.nanmean(accus), 3)


def get_ind_accu(output, real, nIndPerGen, n_row_per_ind, gen=None):
    """
    Get individual accuracy between the output and the real data
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        accus = np.array(
            [np.corrcoef(real[i, :], output[i, :])[0, 1] for i in range(real.shape[0])]
        )
        if type(gen) == int:
            accus = accus[
                gen
                * (nIndPerGen * n_row_per_ind) : (gen + 1)
                * (nIndPerGen * n_row_per_ind)
            ]
        return round(np.nanmean(accus), 3)


def assess_peeling(sim_path, get_params, output_path, name, method):
    """
    Assess the performance of the peeling
    """

    file_to_check = [
        "genotypes",
        "haplotypes",
    ]
    if method in ["ped_only", "combined"]:
        file_to_check.append("segregation")

    nGen = int(get_params["nGen"])
    nIndPerGen = int(get_params["nInd"] / nGen)
    nLociAll = int(get_params["nLociAll"])

    print(" ")
    print(f"Test: {name}")

    for file in file_to_check:
        if file == "genotypes":
            n_row_per_ind = 1
        elif file == "segregation":
            n_row_per_ind = 4
        elif file == "haplotypes":
            n_row_per_ind = 2

        file_path = os.path.join(output_path, f".{file}")

        true_path = os.path.join(sim_path, f"true-{file}.txt")

        new_file = np.loadtxt(file_path, usecols=np.arange(1, nLociAll + 1))
        true_file = np.loadtxt(true_path, usecols=np.arange(1, nLociAll + 1))

        print(f"File: {file}")

        Marker_accu = [str(get_marker_accu(new_file[:, 1:], true_file[:, 1:]))]
        for gen in range(nGen):
            Marker_accu.append(
                str(
                    get_marker_accu(
                        new_file[
                            gen
                            * (nIndPerGen * n_row_per_ind) : (gen + 1)
                            * (nIndPerGen * n_row_per_ind)
                        ],
                        true_file[
                            gen
                            * (nIndPerGen * n_row_per_ind) : (gen + 1)
                            * (nIndPerGen * n_row_per_ind)
                        ],
                    )
                )
            )

        print("Marker_accuracies", " ".join(Marker_accu))

        Ind_accu = [
            str(get_ind_accu(new_file[:, 1:], true_file[:, 1:], nIndPerGen, None))
        ]
        for gen in range(nGen):
            Ind_accu.append(
                str(
                    get_ind_accu(
                        new_file[:, 1:],
                        true_file[:, 1:],
                        nIndPerGen,
                        n_row_per_ind,
                        gen,
                    )
                )
            )

        print("Individual_accuracies", " ".join(Ind_accu))


@pytest.mark.parametrize(
    "method",
    [
        ("pop_only"),
        ("ped_only"),
        ("combined"),
    ],
)
def test_accu(
    get_params,
    method,
    sim_path,
    benchmark,
):
    name = "_".join(
        [
            param
            for param in filter(
                lambda param: True if param else False,
                [
                    method,
                ],
            )
        ]
    )
    output_path = generate_output_path(name)
    prepare_path(output_path)

    command = generate_command(
        sim_path,
        method,
        output_path,
    )

    benchmark(os.system, command)

    assess_peeling(sim_path, get_params, output_path, name, method)
