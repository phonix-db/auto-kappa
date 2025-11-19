
python setup.py sdist
# Install auto-kappa and its dependencies in a virtual environment
# Install matbench-discovery
git clone https://github.com/janosh/matbench-discovery --depth 1
uv pip install -e ./matbench-discovery
# Install MLIPs
# fairchem-core-1.10.0
uv pip install -e packages/fairchem-core
# you need to setup the eSEN model path in auto-kappa/auto_kappa/calculators/mlips.py

# mace
# Note: if you try to use mace, please install it.
# Install auto-kappa
uv pip install dist/auto_kappa-1.1.0.tar.gz

rm -r dist

