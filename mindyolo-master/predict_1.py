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


def get_parser_infer(parents=None):
    parser = argparse.ArgumentParser(description="Infer", parents=[parents] if parents else [])
    parser.add_argument("--task", type=str, default="detect", choices=["detect", "segment"])
    parser.add_argument("--device_target", type=str, default="Ascend", help="device target, Ascend/GPU/CPU")
    parser.add_argument("--ms_mode", type=int, default=0, help="train mode, graph/pynative")
    parser.add_argument("--ms_amp_level", type=str, default="O0", help="amp level, O0/O1/O2")
    parser.add_argument(
        "--ms_enable_graph_kernel", type=ast.literal_eval, default=False, help="use enable_graph_kernel or not"
    )
    parser.add_argument(
        "--precision_mode", type=str, default=None, help="set accuracy mode of network model"
    )
    parser.add_argument("--weight", type=str, default="yolov7_300.ckpt", help="model.ckpt path(s)")
    parser.add_argument("--img_size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument(
        "--single_cls", type=ast.literal_eval, default=False, help="train multi-class data as single-class"
    )
    parser.add_argument("--exec_nms", type=ast.literal_eval, default=True, help="whether to execute NMS or not")
    parser.add_argument("--nms_time_limit", type=float, default=60.0, help="time limit for NMS")
    parser.add_argument("--conf_thres", type=float, default=0.25, help="object confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.65, help="IOU threshold for NMS")
    parser.add_argument(
        "--conf_free", type=ast.literal_eval, default=False, help="Whether the prediction result include conf"
    )
    parser.add_argument("--seed", type=int, default=2, help="set global seed")
    parser.add_argument("--log_level", type=str, default="INFO", help="save dir")
    parser.add_argument("--save_dir", type=str, default="./runs_infer", help="save dir")

    parser.add_argument("--image_path", type=str, help="path to image")
    parser.add_argument("--save_result", type=ast.literal_eval, default=True, help="whether save the inference result")

    return parser


def set_default_infer(args):
    # Set Context
    ms.set_context(mode=args.ms_mode)
    ms.set_recursion_limit(2000)
    if args.precision_mode is not None:
        ms.device_context.ascend.op_precision.precision_mode(args.precision_mode)
    if args.ms_mode == 0:
        ms.set_context(jit_config={"jit_level": "O2"})
    if args.device_target == "Ascend":
        ms.set_device("Ascend", int(os.getenv("DEVICE_ID", 0)))
    args.rank, args.rank_size = 0, 1
    # Set Data
    args.data.nc = 1 if args.single_cls else int(args.data.nc)  # number of classes
    args.data.names = ["item"] if args.single_cls and len(args.names) != 1 else args.data.names  # class names
    assert len(args.data.names) == args.data.nc, "%g names found for nc=%g dataset in %s" % (
        len(args.data.names),
        args.data.nc,
        args.config,
    )
    # Directories and Save run settings
    platform = sys.platform
    if platform == "win32":
        args.save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))
    else:
        args.save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.rank % args.rank_size == 0:
        with open(os.path.join(args.save_dir, "cfg.yaml"), "w") as f:
            yaml.dump(vars(args), f, sort_keys=False)
    # Set Logger
    logger.setup_logging(logger_name="MindYOLO", log_level="INFO", rank_id=args.rank, device_per_servers=args.rank_size)
    logger.setup_logging_file(log_dir=os.path.join(args.save_dir, "logs"))


def detect(
    network: nn.Cell,
    img: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.65,
    conf_free: bool = False,
    exec_nms: bool = True,
    nms_time_limit: float = 60.0,
    img_size: int = 640,
    stride: int = 32,
    num_class: int = 80,
    is_coco_dataset: bool = False,  # ✅ 改默认为 False
):
    """标准检测推理"""
    h_ori, w_ori = img.shape[:2]
    r = img_size / max(h_ori, w_ori)
    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w_ori * r), int(h_ori * r)), interpolation=interp)
    h, w = img.shape[:2]
    if h < img_size or w < img_size:
        new_h, new_w = math.ceil(h / stride) * stride, math.ceil(w / stride) * stride
        dh, dw = (new_h - h) / 2, (new_w - w) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # Transpose + Norm
    img = img[:, :, ::-1].transpose(2, 0, 1) / 255.0
    imgs_tensor = Tensor(img[None], ms.float32)

    _t = time.time()
    out, _ = network(imgs_tensor)
    out = out[-1] if isinstance(out, (tuple, list)) else out
    infer_times = time.time() - _t

    t = time.time()
    out = out.asnumpy()
    out = non_max_suppression(
        out, conf_thres=conf_thres, iou_thres=iou_thres,
        conf_free=conf_free, multi_label=True, time_limit=nms_time_limit,
        need_nms=exec_nms,
    )
    nms_times = time.time() - t

    # ✅ 修正结果生成逻辑
    result_dict = {"category_id": [], "bbox": [], "score": [], "label_name": []}

    for si, pred in enumerate(out):
        if len(pred) == 0:
            continue

        predn = np.copy(pred)
        scale_coords(img.shape[1:], predn[:, :4], (h_ori, w_ori))
        box = xyxy2xywh(predn[:, :4])
        box[:, :2] -= box[:, 2:] / 2

        for p, b in zip(pred.tolist(), box.tolist()):
            raw_cls = int(round(p[5]))  # 模型输出类
            label_name = "unknown"
            if hasattr(network, "names"):  # 如果模型定义了 names
                label_list = getattr(network, "names")
                if raw_cls < len(label_list):
                    label_name = label_list[raw_cls]
            # ✅ 不再映射到 COCO91
            result_dict["category_id"].append(raw_cls)
            result_dict["bbox"].append([round(x, 3) for x in b])
            result_dict["score"].append(round(p[4], 5))
            result_dict["label_name"].append(label_name)

    logger.info(f"Predict result is: {result_dict}")
    logger.info(
    f"Speed: {infer_times * 1e3:.1f}/{nms_times * 1e3:.1f}/{(infer_times + nms_times) * 1e3:.1f} "
    f"ms inference/NMS/total per {img_size}x{img_size} image;"
)

    return result_dict


def segment(
    network: nn.Cell,
    img: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.65,
    conf_free: bool = False,
    nms_time_limit: float = 60.0,
    img_size: int = 640,
    stride: int = 32,
    num_class: int = 80,
    is_coco_dataset: bool = False,  # ✅ 同样改默认 False
):
    """语义分割推理"""
    h_ori, w_ori = img.shape[:2]
    r = img_size / max(h_ori, w_ori)
    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w_ori * r), int(h_ori * r)), interpolation=interp)
    h, w = img.shape[:2]
    if h < img_size or w < img_size:
        new_h, new_w = math.ceil(h / stride) * stride, math.ceil(w / stride) * stride
        dh, dw = (new_h - h) / 2, (new_w - w) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    img = img[:, :, ::-1].transpose(2, 0, 1) / 255.0
    imgs_tensor = Tensor(img[None], ms.float32)

    _t = time.time()
    out, (_, _, prototypes) = network(imgs_tensor)
    infer_times = time.time() - _t

    _c = num_class + 4 if conf_free else num_class + 5
    out = out.asnumpy()
    bboxes, mask_coefficient = out[:, :, :_c], out[:, :, _c:]
    t = time.time()
    out = non_max_suppression(
        bboxes, mask_coefficient,
        conf_thres=conf_thres, iou_thres=iou_thres, conf_free=conf_free,
        multi_label=True, time_limit=nms_time_limit,
    )
    nms_times = time.time() - t

    prototypes = prototypes.asnumpy()
    result_dict = {"category_id": [], "bbox": [], "score": [], "label_name": [], "segmentation": []}

    for si, (pred, proto) in enumerate(zip(out, prototypes)):
        if len(pred) == 0:
            continue
        pred_masks = process_mask_upsample(proto, pred[:, 6:], pred[:, :4], shape=imgs_tensor[si].shape[1:])
        pred_masks = scale_image(pred_masks.transpose(1, 2, 0), (h_ori, w_ori))
        predn = np.copy(pred)
        scale_coords(img.shape[1:], predn[:, :4], (h_ori, w_ori))
        box = xyxy2xywh(predn[:, :4])
        box[:, :2] -= box[:, 2:] / 2

        for ii, (p, b) in enumerate(zip(pred.tolist(), box.tolist())):
            raw_cls = int(round(p[5]))
            label_name = "unknown"
            if hasattr(network, "names"):
                label_list = getattr(network, "names")
                if raw_cls < len(label_list):
                    label_name = label_list[raw_cls]
            result_dict["category_id"].append(raw_cls)
            result_dict["bbox"].append([round(x, 3) for x in b])
            result_dict["score"].append(round(p[4], 5))
            result_dict["label_name"].append(label_name)
            result_dict["segmentation"].append(pred_masks[:, :, ii])

    logger.info(f"Predict result is: {result_dict}")
    logger.info(f"Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image;" %
                tuple(x * 1e3 for x in (infer_times, nms_times, infer_times + nms_times)) + (img_size, img_size))
    logger.info("[设备类型]", ms.context.get_context("device_target"  "[模式]", ms.context.get_context("mode")))
    return result_dict


def infer(args):
    # Init
    set_seed(args.seed)
    set_default_infer(args)

    # Create Network
    network = create_model(
        model_name=args.network.model_name,
        model_cfg=args.network,
        num_classes=args.data.nc,
        sync_bn=False,
        checkpoint_path=args.weight,
    )
    network.set_train(False)
    ms.amp.auto_mixed_precision(network, amp_level=args.ms_amp_level)

    # Load Image
    if isinstance(args.image_path, str) and os.path.isfile(args.image_path):
        import cv2
        img = cv2.imread(args.image_path)
    else:
        raise ValueError("Detect: input image file not available.")

    # Detect
    is_coco_dataset = "coco" in args.data.dataset_name
    if args.task == "detect":
        result_dict = detect(
            network=network,
            img=img,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            conf_free=args.conf_free,
            exec_nms=args.exec_nms,
            nms_time_limit=args.nms_time_limit,
            img_size=args.img_size,
            stride=max(max(args.network.stride), 32),
            num_class=args.data.nc,
            is_coco_dataset=is_coco_dataset,
        )
        if args.save_result:
            save_path = os.path.join(args.save_dir, "detect_results")
            draw_result(
    args.image_path,
    result_dict,
    args.data.names,
    is_coco_dataset=False,  # ✅ 强制关闭 COCO91 映射
    save_path=save_path
)

    elif args.task == "segment":
        result_dict = segment(
            network=network,
            img=img,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            conf_free=args.conf_free,
            nms_time_limit=args.nms_time_limit,
            img_size=args.img_size,
            stride=max(max(args.network.stride), 32),
            num_class=args.data.nc,
            is_coco_dataset=is_coco_dataset,
        )
        if args.save_result:
            save_path = os.path.join(args.save_dir, "segment_results")
            draw_result(
    args.image_path,
    result_dict,
    args.data.names,
    is_coco_dataset=False,  # ✅ 强制关闭 COCO91 映射
    save_path=save_path
)


    logger.info("Infer completed.")


if __name__ == "__main__":
    parser = get_parser_infer()
    args = parse_args(parser)
    infer(args)
