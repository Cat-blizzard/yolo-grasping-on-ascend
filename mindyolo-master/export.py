import os
import argparse
import mindspore as ms
import numpy as np
from mindyolo.models import create_model
from mindyolo.utils.config import load_config

class AttrDict(dict):
    """支持属性访问和字典访问的配置类，兼容 deepcopy() 与缺省字段"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = AttrDict(v)
            elif isinstance(v, list):
                self[k] = [AttrDict(x) if isinstance(x, dict) else x for x in v]

    def __getattr__(self, item):
        # 🧩 屏蔽系统特殊方法
        if item in ("__deepcopy__", "__getstate__", "__setstate__"):
            raise AttributeError(f"{item} not found")

        # ⚙️ 缺省属性返回 None（避免 KeyError）
        return self.get(item, None)

    def __setattr__(self, key, value):
        self[key] = value



def export_model(args):
    print("[INFO] Starting MindIR export...")

    # 设置 Ascend 环境
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_device("Ascend", device_id)

    # 加载配置文件
    cfg_raw = load_config(args.config)
    cfg = cfg_raw[0] if isinstance(cfg_raw, tuple) else cfg_raw
    cfg = AttrDict(cfg)

    # 创建模型
    network = create_model(
        model_name=cfg.network.model_name,
        model_cfg=cfg.network,
        num_classes=cfg.data.nc,
        checkpoint_path=args.weight,
    )
    network.set_train(False)

    # 构造伪输入
    dummy_input = ms.Tensor(np.ones((1, 3, args.img_size, args.img_size)), ms.float32)

    # 导出目录
    output_dir = os.path.join(os.getcwd(), "runs_export")
    os.makedirs(output_dir, exist_ok=True)
    export_path = os.path.join(output_dir, f"{cfg.network.model_name}_Ascend.mindir")

    # 导出模型
    ms.export(network, dummy_input, file_name=export_path, file_format=args.file_format)
    print(f"[✅] Model exported successfully to: {export_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weight", type=str, required=True)
    parser.add_argument("--device_target", type=str, default="Ascend")
    parser.add_argument("--file_format", type=str, default="MINDIR")
    parser.add_argument("--img_size", type=int, default=640)
    args = parser.parse_args()

    export_model(args)
