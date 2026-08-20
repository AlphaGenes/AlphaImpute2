import glob
import os
import subprocess

import pytest

from accu_test_utils import (
    assess_peeling,
    generate_command,
    generate_output_path,
    prepare_path,
)


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

    def run_command(cmd):
        exit_code = subprocess.run(cmd, shell=True).returncode
        outputs = glob.glob(os.path.join(output_path, "test.*"))
        assert exit_code == 0, (
            f"AlphaImpute2 failed with exit code {exit_code}; "
            f"command: {cmd}; outputs: {outputs}"
        )

    benchmark(run_command, command)

    file_out = open("tests/accuracy_tests/accu_report.txt", "a")

    assess_peeling(sim_path, get_params, output_path, method, file_out, x_chr=True)

    file_out.close()
