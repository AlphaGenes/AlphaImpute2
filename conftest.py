import pytest
import os
import shutil
import warnings
import numpy as np


@pytest.fixture(scope="session")
def get_params():
    param_file = os.path.join("tests", "accuracy_tests", "simulation_parameters.txt")
    with open(param_file, "r") as file:
        sim_params = [line.strip().split() for line in file]

    params = {}
    for param_name, param_value in sim_params:
        params[param_name] = float(param_value)

    return params


@pytest.fixture(scope="session")
def sim_path():
    return os.path.join("tests", "accuracy_tests", "sim_for_alphapeel_accu_test")


def pytest_configure(config):
    """
    Prepare path and report file for accuracy tests and functional tests
    """
    accu_output_path = os.path.join("tests", "accuracy_tests", "outputs")
    if os.path.exists(accu_output_path):
        shutil.rmtree(accu_output_path)
    os.mkdir(accu_output_path)

    func_output_path = os.path.join("tests", "functional_tests", "outputs")
    if os.path.exists(func_output_path):
        shutil.rmtree(func_output_path)
    os.mkdir(func_output_path)

    report_path = os.path.join("tests", "accuracy_tests", "accu_report.txt")
    f = open(report_path, "w")
    f.close()


@pytest.hookimpl()
def pytest_terminal_summary(terminalreporter):
    try:
        data = np.genfromtxt(
            "tests/accuracy_tests/accu_report.txt",
            delimiter=",",
            names=["file", "method", "metric", "value"],
            dtype=[
                ("file", "U20"),
                ("method", "U20"),
                ("metric", "U15"),
                ("value", float),
            ],
        )
    except ValueError:
        warnings.warn(
            "Some outputs may be missing for the accuracy report. You can check tests/accuracy_tests/accu_report.txt for the recorded accuracies or rerun the tests. "
        )
        return

    terminalreporter.write_sep("=", " Accuracy")

    files = np.unique(data["file_"])
    metrics = np.unique(data["metric"])

    for metric in metrics:
        terminalreporter.write_sep("~", metric)
        metric_data = data[data["metric"] == metric]

        if metric == "abs_diff":
            terminalreporter.write_line(
                "The metric abs_diff is the sum of absolute difference divided by the norm of the number of loci being counted. "
            )
            terminalreporter.write_line("The lower the value, the better the accuracy.")
        elif metric == "marker_corr":
            terminalreporter.write_line(
                "Pearson correlation evaluated at markers, ranged from 0 to 1."
            )
        elif metric == "ind_corr":
            terminalreporter.write_line(
                "Pearson correlation evaluated at individuals, ranged from 0 to 1."
            )
        elif metric == "correct_rate":
            terminalreporter.write_line(
                "Summing up the probabilities of the true state from the output data divided by the number of loci being counted."
            )

        bar_char = "#"
        empty_char = "."

        for file in files:
            file_data = metric_data[metric_data["file_"] == file]
            if len(file_data) != 0:
                terminalreporter.write_sep("-", file)
                terminalreporter.write_line("{:<20} {:<10}".format("Method", "Value"))
                terminalreporter.write_sep("-")

                values = file_data["value"]
                max_value = np.max(values)
                min_value = np.min(values)
                cols, _ = shutil.get_terminal_size()
                max_length = int(cols * 0.7)

                for row in file_data:
                    value = row["value"]
                    if max_value == min_value:
                        bar_length = max_length
                    else:
                        bar_length = int(
                            max_length * (value - min_value) / (max_value - min_value)
                        )
                    bar = bar_char * bar_length + empty_char * (max_length - bar_length)
                    terminalreporter.write_line(
                        "{:<20} {:.3f} | {} |".format(row["method"], value, bar)
                    )
