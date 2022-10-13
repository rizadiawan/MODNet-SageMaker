# Deploy [MODNet](https://github.com/ZHKKKe/MODNet) as SageMaker Endpoint

Tested with SageMaker Studio using Kernel PyTorch 1.10 Python 3.8 GPU Optimized and studio instance ml.g4dn.xlarge.

Deploy MODNet. High level steps:

1. Convert inference file

2. Test inference file before deploying (saves time instead of testing on deployed endpoint)

```
cd code
pip install numpy sagemaker Pillow torch torchvision --no-cache-dir
python test.py
```

3. Run `Untitled.ipynb` for deployment instructions

References:
- [Deploy a Trained PyTorch Model](https://sagemaker-examples.readthedocs.io/en/latest/frameworks/pytorch/get_started_mnist_deploy.html)
- SageMaker SDK [PyTorch Model](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/sagemaker.pytorch.html#pytorch-model)
- SageMaker SDK [PyTorch Predictor](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/sagemaker.pytorch.html#pytorch-predictor)