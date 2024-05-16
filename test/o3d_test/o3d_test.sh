rm -r test/o3d_test/data
source venv/bin/activate
python3 test/o3d_test/o3d_test.py
tensorboard --logdir test/o3d_test/data/ --load_fast=false --port 6090

