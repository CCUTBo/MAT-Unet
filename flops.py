from MTUNet.model.MTUNet import MTUNet
# from HiFormer.models.HiFormer import HiFormer
from DFormer.model import DFormer3D


import torch
# from MTUNet.model.MTUNet import MTUNet
from thop import profile
# from project_TransUNet.TransUNet.networks.vit_seg_modeling import VisionTransformer
# 创建MTUNet模型实例
model = MTUNet()  # 请替换your_num_classes为实际的类别数
# model = VisionTransformer(config=)  # 请替换your_num_classes为实际的类别数
# model = DFormer3D()
model = model.to("cuda:0")  # 将模型移到GPU上

# 统计模型参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters in MTUNet: {total_params}")



# 创建模拟输入
input_data = torch.randn(1, 3, 224, 224)  # 替换为实际的输入尺寸
input_data = input_data.to("cuda:0")  # 将输入数据移到GPU上

# 使用thop进行模型FLOPs计算
flops, params = profile(model, inputs=(input_data,))
print(f"FLOPs: {flops}, Parameters: {params}")
# def get_n_params(model):
#     pp=0
#     for p in list(model.parameters()):
#         nn=1
#         for s in list(p.size()):
#             nn = nn*s
#         pp += nn
#     return pp
# if __name__ == "__main__":
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Total parameters in MTUNet: {total_params}")
#     print(get_n_params(model))