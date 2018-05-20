#TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_CFLAGS=-I/home/wb/tf/tf/lib/python3.5/site-packages/tensorflow/include -I/home/wb/tf/tf/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -I/home/wb/eigen -I/opt/intel/mkl/include -D_GLIBCXX_USE_CXX11_ABI=0
#LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TF_LFLAGS=-L/home/wb/tf/tf/lib/python3.5/site-packages/tensorflow -ltensorflow_framework -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64 -lmkl_rt

#zero_out.so:zero_out.cc
#	g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC ${TF_CFLAGS} ${TF_LFLAGS} `pkg-config --cflags --libs opencv` -O2 -w
#	cp ./zero_out.so ~/project/autoMatting/zero_out.so

matting.so:matting.cc
	g++ -std=c++11 -shared matting.cc -o matting.so -fPIC ${TF_CFLAGS} ${TF_LFLAGS} `pkg-config --cflags --libs opencv` -O2 -w
	cp ./matting.so ~/project/autoMatting/matting.so

loss.so:loss.cc
	g++ -std=c++11 -shared loss.cc -o loss.so -fPIC ${TF_CFLAGS} ${TF_LFLAGS} `pkg-config --cflags --libs opencv` -O2 -w
	cp ./loss.so ~/project/autoMatting/loss.so
