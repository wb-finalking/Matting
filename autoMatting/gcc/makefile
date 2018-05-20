TF_CFLAGS=-I/home/wb/eigen

TF_LFLAGS=


mian:main.cpp
	g++ -std=c++11 main.cpp -o main ${TF_CFLAGS}  `pkg-config --cflags --libs opencv` -O2 -w
