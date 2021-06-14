# efficientnetB3-TensorRT-tensorflow
This is TensorRT version of efficientnetB3, support trained tensorflow 2.x weights<br>

# Install
TensorRT 7.2.3<br>
cmake>3.2<br>
tensorflow>2.0<br>
keras efficientnet 1.1<br>
cuda 11.0<br>

# Generate wts based on tensorflow2.X
1.Install all required package<br>
2.Run efficientnet.py (you will get a pretrained model and generate weights to wts file (NCHW) format)<br>
3.config paths in cmakelists.txt

# Generate trt engine and infer
1.gen a fir using 
"""
mkdir build &&cd build
"""
and run 
"""
cmake ..&&make
"""
to gen exec.
2. generate engine, using
"""
./effficientnet -s
"""
command.
3. test random generated imgs, using
"""
./efficientnet -d
"""
