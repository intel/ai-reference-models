mkdir -p tf_2012_val

#copy ImageNet validation (tf record) to tf_2012_val

rm -rf env_inc
python -m venv env_inc 

source env_inc/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
runipy inc_for_tensorflow.ipynb

if [ $? != 0 ]; then
  echo "ut is wrong!"
  deactivate
  exit 1
else
  echo "ut is passed!"
  deactivate
  exit 0
fi

