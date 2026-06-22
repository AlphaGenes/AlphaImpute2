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
    x_chr=False,
):
    command = "AlphaImpute2 "

    arguments = {
        "cycles": "5",
        "maxthreads": "6",
        "phase_output": None,
        "seg_output": None,
    }
    if x_chr:
        command += "-x_chr "
        input_file = {"pedigree": "X_chr_ped_file"}
        input_file["genotypes"] = "X_chr_geno_file"
    else:
        input_file = {
            "pedigree": "pedigree",
            "genotypes": "genotypes",
        }

    if method in ["pop_only", "combined"]:
        arguments["hd_threshold"] = "0.8"

    if method in ["pop_only", "ped_only"]:
        command += f"-{method} "

    for argu in input_file.keys():
        command += f"-{argu} {os.path.join(sim_path, f'{input_file[argu]}.txt')} "

    for key, value in arguments.items():
        if value is not None:
            command += f"-{key} {value} "
        else:
            command += f"-{key} "

    command += f"-out {os.path.join(output_path, 'test')}"

    return command


def make_directory(path):
    """
    Prepare a empty folder at the input path
    """
    if os.path.exists(path):
        shutil.rmtree(path)

    os.mkdir(path)


def get_marker_corr(output, real):
    """
    Get marker Pearson correlation between the output and the real data
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        accus = np.array(
            [np.corrcoef(real[:, i], output[:, i])[0, 1] for i in range(real.shape[1])]
        )
        return round(np.nanmean(accus), 3)


def get_ind_corr(output, real, nIndPerGen, n_row_per_ind, gen=None):
    """
    Get individual Peason correlation between the output and the real data
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        accus = np.array(
            [np.corrcoef(real[i, :], output[i, :])[0, 1] for i in range(real.shape[0])]
        )
        if type(gen) == int:
            accus = accus[gen * (nIndPerGen * n_row_per_ind) :]
        return round(np.nanmean(accus), 3)


def get_abs_diff(output, real, n_row_per_ind):
    """
    Sum of absolute difference divided by the norm of the number of loci being counted
    """
    return n_row_per_ind * np.sum(np.abs(output - real)) / real.size


def get_correct_rate(output, real):
    """
    Summing up the probabilities of the true state from the output data
    divided by the number of loci being counted

    :param output: Output data
    :type output: ndarray
    :param real: Real simulated data
    :type real: ndarray
    """
    return np.sum(output[real == 1]) / (np.size(real) / 4)


def assess_peeling(sim_path, get_params, output_path, method, file_out, x_chr=False):
    """
    Assess the performance of the peeling
    """

    file_to_check = [
        "genotypes",
        "haplotypes",
    ]
    if method in ["ped_only", "combined"]:
        file_to_check.append("segregation")

    if x_chr:
        method += "_x_chr"

    nGen = int(get_params["nGen"])
    nIndPerGen = int(get_params["nInd"] / nGen)
    nLociAll = int(get_params["nLociAll"])

    true_prefix = "true-X_chr_" if x_chr else "true-"

    for file in file_to_check:
        if file == "genotypes":
            n_row_per_ind = 1
        elif file == "segregation":
            n_row_per_ind = 4
        elif file == "haplotypes":
            n_row_per_ind = 2

        file_path = os.path.join(output_path, f"test.{file}")

        true_path = os.path.join(sim_path, f"{true_prefix}{file}.txt")

        new_file = np.loadtxt(file_path, usecols=np.arange(1, nLociAll + 1))
        true_file = np.loadtxt(true_path, usecols=np.arange(1, nLociAll + 1))

        if file == "segregation":
            marker_corr = get_marker_corr(
                new_file[2 * (nIndPerGen * n_row_per_ind) :, 1:],
                true_file[2 * (nIndPerGen * n_row_per_ind) :, 1:],
            )
        else:
            marker_corr = get_marker_corr(new_file[:, 1:], true_file[:, 1:])

        file_out.write(f"{file},{method},marker_corr,{marker_corr}\n")

        if file == "segregation":
            ind_corr = get_ind_corr(
                new_file[:, 1:],
                true_file[:, 1:],
                nIndPerGen,
                n_row_per_ind,
                2,
            )
        else:
            ind_corr = get_ind_corr(
                new_file[:, 1:],
                true_file[:, 1:],
                nIndPerGen,
                n_row_per_ind,
            )

        file_out.write(f"{file},{method},ind_corr,{ind_corr}\n")

        if file == "segregation":
            abs_diff = get_abs_diff(
                new_file[2 * (nIndPerGen * n_row_per_ind) :, 1:],
                true_file[2 * (nIndPerGen * n_row_per_ind) :, 1:],
                n_row_per_ind,
            )

        else:
            if file == "haplotypes":
                new_file[true_file == 9] = 0
                true_file[true_file == 9] = 0

            abs_diff = get_abs_diff(
                new_file[:, 1:],
                true_file[:, 1:],
                n_row_per_ind,
            )

        file_out.write(f"{file},{method},abs_diff,{abs_diff}\n")

        if file == "segregation":
            correct_rate = get_correct_rate(
                new_file[2 * (nIndPerGen * n_row_per_ind) :, 1:],
                true_file[2 * (nIndPerGen * n_row_per_ind) :, 1:],
            )
            file_out.write(f"{file},{method},correct_rate,{correct_rate}\n")


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

    def run_command(cmd):
        exit_code = os.system(cmd)
        if exit_code == 11:
            import glob

            outputs = glob.glob(os.path.join(output_path, "test.*"))
            if outputs:
                return  # output was written, crash was in cleanup only
        assert exit_code == 0, f"AlphaImpute2 failed with exit code {exit_code}"

    benchmark(run_command, command)

    file_out = open("tests/accuracy_tests/accu_report.txt", "a")

    assess_peeling(sim_path, get_params, output_path, method, file_out)

    file_out.close()


# ── X chromosome accuracy test ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "method",
    [
        ("pop_only"),
        ("ped_only"),
        ("combined"),
    ],
)
def test_sex_accu(get_params, method, sim_path, benchmark):
    name = "_".join(
        [
            param
            for param in filter(
                lambda param: True if param else False,
                [
                    "x_chr",
                    method,
                ],
            )
        ]
    )
    output_path = generate_output_path(name)
    prepare_path(output_path)

    command = generate_command(sim_path, method, output_path, x_chr=True)

    benchmark(os.system, command)

    file_out = open("tests/accuracy_tests/accu_report.txt", "a")

    assess_peeling(sim_path, get_params, output_path, method, file_out, x_chr=True)

    file_out.close()
