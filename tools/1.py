import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
data = torch.randn([8, 8, 256, 256], dtype=torch.float, device='cuda', requires_grad=True)
net = torch.nn.Conv2d(8, 1, kernel_size=[3, 3], padding=[1, 1], stride=[1, 1], dilation=[1, 1], groups=1)
net = net.cuda().float()
out = net(data)
out.backward(torch.randn_like(out))
torch.cuda.synchronize()

# ConvolutionParams 
#     data_type = CUDNN_DATA_FLOAT
#     padding = [1, 1, 0]
#     stride = [1, 1, 0]
#     dilation = [1, 1, 0]
#     groups = 1
#     deterministic = false
#     allow_tf32 = true
# input: TensorDescriptor 0x559693d27950
#     type = CUDNN_DATA_FLOAT
#     nbDims = 4
#     dimA = 8, 8, 256, 256, 
#     strideA = 524288, 65536, 256, 1, 
# output: TensorDescriptor 0x559693d1bb90
#     type = CUDNN_DATA_FLOAT
#     nbDims = 4
#     dimA = 8, 1, 256, 256, 
#     strideA = 65536, 65536, 256, 1, 
# weight: FilterDescriptor 0x7f6124148110
#     type = CUDNN_DATA_FLOAT
#     tensor_format = CUDNN_TENSOR_NCHW
#     nbDims = 4
#     dimA = 1, 8, 3, 3, 
# Pointer addresses: 
#     input: 0x7f5df07d2c00
#     output: 0x7f5e10a00000
#     weight: 0x7f62d55f0800