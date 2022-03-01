mkdir -p tf_2012_val

#copy ImageNet validation (tf record) to tf_2012_val
#cd tf_2012_val
#rm tf_2012_val/val*
#cd -
rm -rf vlpot
python -m venv vlpot 

source vlpot/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
runipy lpot_for_tensorflow.ipynb
#runipy ut.ipynb

if [ $? != 0 ]
then
  echo "ut is wrong!"
  deactivate
  exit 1
else
  echo "ut is passed!"
  deactivate
  exit 0
fi
