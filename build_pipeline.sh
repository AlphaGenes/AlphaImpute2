rm -r build
rm -r dist
python setup.py bdist_wheel
pip uninstall TinyPeel -y
pip install dist/TinyPeel-0.0.1-py3-none-any.whl
