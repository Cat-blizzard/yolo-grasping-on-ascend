import argparse
import ast
import math
import os
import sys
import time
import cv2
import numpy as np
import yaml
from datetime import datetime

import mindspore as ms
from mindspore import Tensor, nn

from mindyolo.data import COCO80_TO_COCO91_CLASS
from mindyolo.models import create_model
from mindyolo.utils import logger
from mindyolo.utils.config import parse_args
from mindyolo.utils.metrics import non_max_suppression, scale_coords, xyxy2xywh, process_mask_upsample, scale_image
from mindyolo.utils.utils import draw_result, set_seed


# ==============================================================
#                参数与上下文初始化
# ==============================================================

def get_parser_infer(parents=None):
    parser = argparse.ArgumentParser(description="MindYOLO NPU Infer", parents=[parents] if parents else [])
    parser.add_argument("--task", type=str, default="detect", choices=["detect", "segment"])
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend/GPU/CPU")
    parser.add_argument("--ms_mode", type=int, default=0, help="Graph/Pynative")
    parser.add_argument("--ms_amp_level", type=str, default="O0", help="O0/O1/O2")
    parser.add_argument("--precision_mode", type=str, default=None)
    parser.add_argument("--weight", type=str, required=True, help=".ckpt 或 .mindir 模型路径")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--exec_nms", type=ast.literal_eval, default=True)
    parser.add_argument("--conf_thres", type=float, default=0.25)
    parser.add_argument("--iou_thres", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default="./runs_infer")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--save_result", type=ast.literal_eval, default=True)
    return parser


def set_default_infer(args):
    """设置推理环境"""
    ms.set_context(mode=args.ms_mode, device_target="Ascend")
    ms.set_recursion_limit(2000)
    if args.precision_mode is not None:
        ms.device_context.ascend.op_precision.precision_mode(args.precision_mode)
    if args.ms_mode == 0:
        ms.set_context(jit_config={"jit_level": "O2"})

    args.rank, args.rank_size = 0, 1

    timestamp = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
    args.save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(args.save_dir, exist_ok=True)

    with open(os.path.join(args.save_dir, "cfg.yaml"), "w") as f:
        yaml.dump(vars(args), f, sort_keys=False)

    logger.setup_logging("MindYOLO", "INFO", rank_id=args.rank, device_per_servers=args.rank_size)
    logger.setup_logging_file(os.path.join(args.save_dir, "logs"))
    print(f"[INFO] ✅ 推理环境初始化完成，输出目录: {args.save_dir}")


# ==============================================================
#                      推理函数定义
# ==============================================================

def detect(network, img, conf_thres=0.25, iou_thres=0.65, exec_nms=True, img_size=640, stride=32, num_class=80):
    """检测任务"""
    h_ori, w_ori = img.shape[:2]
    r = img_size / max(h_ori, w_ori)
    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w_ori * r), int(h_ori * r)), interpolation=interp)
    h, w = img.shape[:2]
    if h < img_size or w < img_size:
        new_h, new_w = math.ceil(h / stride) * stride, math.ceil(w / stride) * stride
        dh, dw = (new_h - h) / 2, (new_w - w) / 2
        img = cv2.copyMakeBorder(img, int(dh), int(dh), int(dw), int(dw), cv2.BORDER_CONSTANT, value=(114, 114, 114))

    img = img[:, :, ::-1].transpose(2, 0, 1) / 255.0
    imgs_tensor = Tensor(img[None], ms.float32)

    # 进行推理
    _t = time.time()
    if isinstance(network, (ms.nn.GraphCell, ms.Model)):
        out = network(imgs_tensor)
    else:
        out, _ = network(imgs_tensor)
        out = out[-1] if isinstance(out, (tuple, list)) else out
    infer_time = time.time() - _t

    out = out.asnumpy()
    t = time.time()
    out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, need_nms=exec_nms)
    nms_time = time.time() - t

    # 格式化输出结果
    result = {"category_id": [], "bbox": [], "score": []}
    for pred in out:
        if len(pred) == 0:
            continue
        predn = np.copy(pred)
        scale_coords(img.shape[1:], predn[:, :4], (h_ori, w_ori))
        box = xyxy2xywh(predn[:, :4])
        box[:, :2] -= box[:, 2:] / 2
        for p, b in zip(pred.tolist(), box.tolist()):
            result["category_id"].append(int(round(p[5])))
            result["bbox"].append([round(x, 3) for x in b])
            result["score"].append(round(p[4], 5))

    logger.info(f"Predict result is: {result}")
    logger.info(f"Speed: {infer_time*1e3:.1f}/{nms_time*1e3:.1f}/{(infer_time+nms_time)*1e3:.1f} ms total")
    return result


# ==============================================================
#                      主推理流程
# ==============================================================

def infer(args):
    set_seed(args.seed)
    set_default_infer(args)

    # 1️⃣ 加载模型
    if args.weight.endswith(".mindir"):
        print(f"[INFO] 🔄 加载 MindIR 模型：{args.weight}")
        graph = ms.load_mindir(args.weight)
        network = ms.nn.GraphCell(graph)
        print("[INFO] ✅ MindIR 模型加载完成（Ascend NPU 已启用）")

        # 预热触发图编译
        dummy = Tensor(np.ones((1, 3, args.img_size, args.img_size)), ms.float32)
        try:
            _ = network(dummy)
            print("[INFO] 🔥 图结构预热完成，NPU 编译优化已就绪")
        except Exception as e:
            print(f"[WARNING] MindIR 预热失败: {e}")

    else:
        print(f"[INFO] 使用 Checkpoint 模型：{args.weight}")
        network = create_model(
            model_name=args.network.model_name,
            model_cfg=args.network,
            num_classes=args.data.nc,
            sync_bn=False,
            checkpoint_path=args.weight,
        )
        network.set_train(False)
        ms.amp.auto_mixed_precision(network, amp_level=args.ms_amp_level)

    # 2️⃣ 加载图片
    if isinstance(args.image_path, str) and os.path.isfile(args.image_path):
        img = cv2.imread(args.image_path)
    else:
        raise ValueError("❌ Detect: input image file not available.")

    # 3️⃣ 推理
    result_dict = detect(network, img,
                         conf_thres=args.conf_thres,
                         iou_thres=args.iou_thres,
                         exec_nms=args.exec_nms,
                         img_size=args.img_size,
                         stride=max(max(args.network.stride), 32),
                         num_class=args.data.nc)

    # 4️⃣ 保存结果
    if args.save_result:
        save_path = os.path.join(args.save_dir, "detect_results")
        draw_result(args.image_path, result_dict, args.data.names, is_coco_dataset=False, save_path=save_path)

    logger.info("✅ 推理完成")


# ==============================================================
#                         主入口
# ==============================================================

if __name__ == "__main__":
    parser = get_parser_infer()
    args = parse_args(parser)
    infer(args)
