# Reproducibility

### Clone the repository

```
git clone https://github.com/uzh-rpg/scene_graph_3d.git
cd scene_graph_3d
```

### Set up the Python environment 

macOS
```
python3 -m venv venv
source venv/bin/activate
pip install -r docs/dependencies/requirements-mac.txt
```

Linux
```
python3 -m venv venv
source venv/bin/activate
pip install -r docs/dependencies/requirements-linux.txt
```

Windows (untested)
```
virtualenv --python C:\Path\To\Python\python.exe venv
venv\Scripts\activate
pip install -r docs/dependencies/requirements.txt
```

GPU cluster SNAGA 
```
python3 -m venv venv
source venv/bin/activate
pip install -r docs/dependencies/requirements-snaga.txt
```

To import the virtual environment in a jupyter notebook
```
python -m ipykernel install --name=venv
```

### Run experiment

Create a new configuration file (```myconfig.json```) with the same fields as in config/config.json file provided. 

```
python3 main.py --c myconfig.json
```
