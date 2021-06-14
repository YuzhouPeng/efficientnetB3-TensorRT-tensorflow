# efficientnetB3-TensorRT-tensorflow
This is TensorRT version of efficientnetB3, based on trained tensorflow 2.x version<br>

# Install
TensorRT 7.2.3<br>
cmake>3.2<br>
tensorflow>2.0<br>
keras efficientnet 1.1<br>
cuda 11.0<br>

# generate tf wts
1.Install all required package<br>
2.Run efficientnet.py (you will get a pretrained model and generate weights to wts file (NCHW) format)<br>
3.config paths in cmakelists.txt
4.gen a fir using 
'''
mkdir build &&cd build
'''
and run 
'''
cmake ..&&make
'''
to gen exec.
5. generate engine, using
'''
./effficientnet -s
'''
command.
6. test random generated imgs, using
'''
./efficientnet -d
'''
