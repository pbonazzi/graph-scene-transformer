rm -r test/o3d_test/data
python3 test/o3d_test/o3d_test_mesh_align.py
tensorboard --logdir test/o3d_test/data/ --load_fast=false