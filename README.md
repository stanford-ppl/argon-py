# argon-py

# Installation
1. In the root directory, run:

```shell
python -m pip install .
```

2. If you are getting ModuleNotFoundError for `argon`, then add this to the `sys.path` directory list. (This should probably be done using a virtual environment.)

```shell
# On MacOSX
vim ~/.bash_profile
# Then, add: 
#      export PYTHONPATH="<root-directory>/argon-py/src"
source ~/.bash_profile
```