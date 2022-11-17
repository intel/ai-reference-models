rm -rf env_inc
python -m venv env_inc 

source env_inc/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
env_inc/bin/python -m ipykernel install --user --name=env_inc

echo "Running environment env_inc is created, enable it by:\nsource env_inc/bin/activate"
