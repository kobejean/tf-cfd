TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
# g++ -std=c++11 -shared roll_op.cc -o roll_op.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2
g++ -std=c++11 -shared roll_op.cc -o roll_op.so -undefined dynamic_lookup -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2
