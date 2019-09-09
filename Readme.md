## A few model training examples with tensorflow serving

### Dependencies:
Docker
Python Dependencies:  pip install -r requirements.txt
### Pretrained Resnet:
1. mkdir resnet
2. Get Resnet: "cd resnet && curl -s https://storage.googleapis.com/download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC_jpg.tar.gz | tar --strip-components=2 -C resnet -xvz"
3. Get Tensorflow Serving Docker Image by running: "docker pull tensorflow/serving"
4. Start server by running: "docker run -p 8501:8501 --name tfserving_resnet --mount type=bind,source=<path to resnet folder>/resnet,target=/models/resnet -e MODEL_NAME=resnet -t tensorflow/serving > resnet_log.txt &"
5. python3 resnet/resnet_client.py


### Train Fashion Mnist:
1. mkdir fashion_mnist && cd fashion_mnist
2. Train Fashion Mnist by running "python3 train_fashion_mnist.py"
3. Get Tensorflow Serving Docker Image by running: "docker pull tensorflow/serving"
4. Start server by running: "docker run -p 8501:8501 --name tfserving_fmnist --mount type=bind,source=/Users/julianweisbord/Documents/ai/tensorflow_serving/serving_examples/fashion_mnist/fmnist_model,target=/models/fmnist_model -e MODEL_NAME=fmnist_model -t tensorflow/serving > fmnist_log.txt &"
5. python3 fashion_mnist/fashion_mnist_client.py
