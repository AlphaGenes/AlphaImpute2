command=$1
if [ $# -eq 0 ] ; then 
    command=none
fi

if [[ ! -f src/alphaimpute2/tinyhouse/Pedigree.py ]] ; then
    echo Pedigree.py file not found. Check that the tinyhouse submodule is up to date
    exit 
fi

# Create python wheel.
rm -r build
rm -r dist
python setup.py bdist_wheel

if [ $command == "install" ] ; then
    pip uninstall AlphaImpute2 -y
    pip install dist/AlphaImpute2-0.0.1-py3-none-any.whl
fi

#Compile manual
 ( cd docs; make latexpdf )


target=AlphaImpute2
rm -rf $target
mkdir $target
cp dist/* $target
cp docs/build/latex/AlphaImpute2.pdf $target
cp -r example $target
zip -r $target.zip $target

