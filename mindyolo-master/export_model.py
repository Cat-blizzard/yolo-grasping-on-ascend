# export_model.py
import os
import numpy as np
from mindspore import context, load_checkpoint, load_param_into_net, export, Tensor, set_device
import mindspore as ms

# ✅ 正确导入
from mindyolo.models.yolov8 import YOLOv8
from mindyolo.utils.config import load_config, Config

# 设置设备
set_device("CPU")
context.set_context(mode=context.GRAPH_MODE)

# ===================================================================
# ✅ 步骤1: 加载 YAML 配置文件
# ===================================================================
yaml_path = "/home/HwHiAiUser/Downloads/mindyolo-master/configs/yolov8/yolov8s.yaml"
cfg_dict, _, _ = load_config(yaml_path)
cfg = Config(cfg_dict)
print(f"✅ 成功加载配置: {yaml_path}")

# ===================================================================
# ✅ 关键修复：提升 network 下所有关键字段到顶层
# ===================================================================
if hasattr(cfg, 'network'):
    network_cfg = cfg.network
    fields_to_promote = [
        'depth_multiple', 'width_multiple', 'stride', 'reg_max',
        'max_channels', 'backbone', 'head'
    ]
    for key in fields_to_promote:
        if hasattr(network_cfg, key):
            setattr(cfg, key, getattr(network_cfg, key))
            print(f"✅ 提升 {key} 到顶层")
else:
    raise ValueError("❌ cfg 中没有 'network' 字段！")

# 提升 nc
if hasattr(cfg, 'data') and hasattr(cfg.data, 'nc'):
    cfg.nc = cfg.data.nc
    print(f"✅ 提升 nc = {cfg.nc} 到顶层")
else:
    cfg.nc = 80
    print(f"✅ 手动设置 nc = {cfg.nc}")

# ===================================================================
# ✅ 步骤2: 实例化 YOLOv8 模型
# ===================================================================
net = YOLOv8(
    cfg=cfg,
    in_channels=3,
    num_classes=80,
    sync_bn=False
)
net.set_train(False)

# ===================================================================
# ✅ 步骤3: 加载 .ckpt 检查点
# ===================================================================
ckpt_path = "/home/HwHiAiUser/Downloads/mindyolo-master/yolov8-s_500e_mAP446-3086f0c9.ckpt"
if os.path.exists(ckpt_path):
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    print(f"✅ 成功加载检查点: {ckpt_path}")
else:
    raise FileNotFoundError(f"❌ 未找到 .ckpt 文件: {ckpt_path}")

# ===================================================================
# ✅ 步骤4: 构造输入并导出为 MINDIR 和 AIR
# ===================================================================
input_tensor = Tensor(np.random.uniform(0, 1, size=[1, 3, 640, 640]).astype(np.float32))
print(f"✅ 输入 Tensor 形状: {input_tensor.shape}, 类型: {input_tensor.dtype}")

# 导出为 MINDIR
export(net, input_tensor, file_name='yolov8s_coco', file_format='MINDIR')
print("🎉 模型已成功导出为: yolov8s_coco.mindir")

# ✅ 新增：导出为 AIR
export(net, input_tensor, file_name='yolov8s_coco', file_format='AIR')
print("🎉 模型已成功导出为: yolov8s_coco.air")