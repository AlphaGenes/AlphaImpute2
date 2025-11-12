import os
import shutil


def pytest_configure(config):
    """
    Prepare path and report file for functional tests
    """
    func_output_path = os.path.join("tests", "functional_tests", "outputs")
    if os.path.exists(func_output_path):
        shutil.rmtree(func_output_path)
    os.mkdir(func_output_path)
