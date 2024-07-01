import configparser
import argparse
import json
import sys, os
import torch
from torch import nn
import torch.nn.functional as F
import onnx
import subprocess
try:
    import sentry_sdk

    sentry_sdk.init(
        dsn="https://d5afa0d557b2491a2aa9e34e1c803cf2@o4507524861853696.ingest.de.sentry.io/4507524873977936",
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
        traces_sample_rate=1.0,
    # Set profiles_sample_rate to 1.0 to profile 100%
    # of sampled transactions.
    # We recommend adjusting this value in production.
        profiles_sample_rate=1.0,
)
except Exception as e:
    print("远程日志已经禁用")
sys.path.append(os.getcwd())
from yiku.nets.model.Labs.labs import Labs
from pathlib import Path
import platform
from yiku.utils.download import download_from_url

MACOS, LINUX, WINDOWS = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows'])
from yiku.PATH import ASSETS, WTS_STORAGE_DIR


def export_ncnn(net, f, imgsz=512, fp16=True, **kwargs):
    batch = kwargs.get("batch", 1)
    device = kwargs.get("device", "cpu")
    if device.isdigit():
        device = "gpu"
    try:
        import ncnn
    except ImportError:
        return
    name = Path('pnnx.exe' if WINDOWS else 'pnnx')
    pnnx = WTS_STORAGE_DIR / name if (WTS_STORAGE_DIR / name).is_file() else ASSETS / "exec" / name
    if not pnnx.is_file():
        print(
            f'WARNING ⚠️ PNNX not found. Attempting to download binary file from '
            f'https://github.com/pnnx/pnnx/.\nNote PNNX Binary file must be placed in {WTS_STORAGE_DIR} directory '
            f'or in {ASSETS / "exec"}. See PNNX repo for full installation instructions.')
        pnnx = WTS_STORAGE_DIR / name
        if WINDOWS:
            download_from_url("https://github.com/VIRobotics/hgnetv2-deeplabv3/releases/download/v0.0.2-beta/pnnx.exe",
                              WTS_STORAGE_DIR)
        elif LINUX:
            download_from_url("https://github.com/VIRobotics/hgnetv2-deeplabv3/releases/download/v0.0.2-beta/pnnx",
                              WTS_STORAGE_DIR)
            os.chmod(pnnx, 0o755)

    ts = export_torchscript(net, f=f, fimgsz=imgsz, **kwargs)
    fo = Path(str(f).replace(f.suffix, f'_ncnn_model{os.sep}'))
    os.makedirs(fo, exist_ok=True)
    ncnn_args = [
        f'ncnnparam={fo / "model.ncnn.param"}',
        f'ncnnbin={fo / "model.ncnn.bin"}',
        f'ncnnpy={fo / "model_ncnn.py"}', ]

    pnnx_args = [
        f'pnnxparam={fo / "model.pnnx.param"}',
        f'pnnxbin={fo / "model.pnnx.bin"}',
        f'pnnxpy={fo / "model_pnnx.py"}',
        f'pnnxonnx={fo / "model.pnnx.onnx"}', ]

    if kwargs.get("no_pre", False):
        inshape = [batch, 3, imgsz, imgsz]
    else:
        inshape = [batch, imgsz, imgsz, 3]
    cmd = [
        str(pnnx),
        str(ts),
        *ncnn_args,
        *pnnx_args,
        f'fp16={int(fp16)}',
        f'device={device}',
        f'inputshape="{inshape}"', ]
    subprocess.run(cmd, check=True)
    pnnx_files = [x.split('=')[-1] for x in pnnx_args]
    for f_debug in ('debug.bin', 'debug.param', 'debug2.bin', 'debug2.param', *pnnx_files):
        Path(f_debug).unlink(missing_ok=True)
    return fo


def export_onnx(net, f: Path, imgsz=512, **kwargs):
    d = kwargs.get("device", torch.device("cpu"))
    f = f.with_suffix(".onnx")
    if kwargs.get("no_pre", False):
        f = Path(str(f).replace(f.stem, f.stem + "+no_pre"))
    if kwargs.get("no_post", False):
        f = Path(str(f).replace(f.stem, f.stem + "+no_post"))
        kwargs["include_resize"] = False
    batch = kwargs["batch"]
    if batch <= 0:
        batch = 1
        dynamic = {'images': {0: 'batch'}}
        dynamic['output0'] = {0: 'batch'}
    else:
        dynamic = False
    if kwargs.get("no_pre", False):
        im = torch.zeros(batch, 3, imgsz, imgsz).to(d)
    else:
        im = torch.zeros(batch, imgsz, imgsz, 3).to(d)

    if "include_resize" in kwargs.keys() and kwargs.get("include_resize", False):
        input_layer_names = ["images", 'nh', 'nw']
        im = (im, torch.Tensor([100]), torch.Tensor([100]))
    else:
        input_layer_names = ["images"]
    output_layer_names = ["output"]

    class Net_with_resize(nn.Module):
        def __init__(self, m, inputsize=(512, 512)):
            super().__init__()
            self.m = m
            self.input_shape = inputsize

        def forward(self, x, nh, nw):

            if not kwargs.get("no_pre", False):
                x = x.permute(0, 3, 1, 2)
                x = x / 255.0
            pr = self.m(x)
            if not kwargs.get("no_post", False):
                pr = F.softmax(pr, dim=1)
                pr = pr.argmax(dim=1, keepdim=True)
                pr = pr.to(torch.uint8)
                pr = pr[:, :, int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                     int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
                pr = pr.permute(0, 2, 3, 1)
            return pr

    class Net_with_post(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            if not kwargs.get("no_pre", False):
                x = x.permute(0, 3, 1, 2)
                x = x / 255.0
            pr = self.m(x)
            if not kwargs.get("no_post", False):
                # pr = F.softmax(pr, dim=1)
                pr = pr.argmax(dim=1, keepdim=True)
                pr = pr.permute(0, 2, 3, 1)
                pr = pr.to(torch.int)
            return pr

    if "include_resize" in kwargs.keys() and kwargs.get("include_resize", False):
        net = Net_with_resize(net, inputsize=(imgsz, imgsz))
    else:
        net = Net_with_post(net)

    print(f'Starting export with onnx {onnx.__version__}.')
    torch.onnx.export(net,
                      im,
                      f=f,
                      verbose=False,
                      opset_version=15,
                      training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=True,
                      input_names=input_layer_names,
                      output_names=output_layer_names,
                      dynamic_axes=dynamic or None)

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    if dynamic:
        b = -1
    else:
        b = batch
    di = {
        "imgsz": [imgsz, imgsz],
        "batch": b,
        "names": kwargs["names"],
        "num_classes": kwargs["num_classes"]
    }
    for k, v in di.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    with open(f.with_suffix(".json"), mode="w") as f_json:
        json.dump(di, f_json, indent=4)

    # Simplify onnx
    try:
        import onnxsim
        print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
        model_onnx, check = onnxsim.simplify(
            model_onnx,
            dynamic_input_shape=False,
            input_shapes=None)
        assert check, 'assert check failed'
    except ImportError:
        print(f'onnx-simplifier not installed. SKIP')
    onnx.save(model_onnx, f)
    return f


def export_openvino(net, f, imgsz=512, fp16=True, **kwargs):
    import openvino.runtime as ov  # noqa
    from openvino.tools import mo  # noqa
    fo = f + "_openvino"
    onnx_path = export_onnx(net, f, imgsz, **kwargs)
    print("Start OpenVINO model export")
    ov_model = mo.convert_model(onnx_path,
                                model_name="Labs",
                                framework='onnx',
                                compress_to_fp16=fp16)
    ov.serialize(ov_model, os.path.join(fo, "model.xml"))
    print("OpenVINO model export at %s" % os.path.join(fo, "model.xml"))
    return os.path.join(fo, "model.xml")


def export_torchscript(net, f, imgsz=512, **kwargs):
    f = f.with_suffix(".torchscript")
    if kwargs.get("no_pre", False):
        f = Path(str(f).replace(f.stem, f.stem + "+no_pre"))
    if kwargs.get("no_post", False):
        f = Path(str(f).replace(f.stem, f.stem + "+no_post"))
        kwargs["include_resize"] = False
    batch = kwargs["batch"]
    if batch <= 0: batch = 1
    if kwargs.get("no_pre", False):
        im = torch.zeros(batch, 3, imgsz, imgsz).to('cpu')
    else:
        im = torch.zeros(batch, imgsz, imgsz, 3).to('cpu')

    if "include_resize" in kwargs.keys() and kwargs.get("include_resize", False):
        input_layer_names = ["images", 'nh', 'nw']
        im = (im, torch.Tensor([100]), torch.Tensor([100]))
    else:
        input_layer_names = ["images"]
    output_layer_names = ["output"]

    class Net_with_post(nn.Module):
        def __init__(self, m, **kwargs):
            super().__init__()
            self.m = m

        def forward(self, x):
            if not kwargs.get("no_pre", False):
                x = x.permute(0, 3, 1, 2)
                x = x / 255.0
            pr = self.m(x)
            if not kwargs.get("no_post", False):
                # pr = F.softmax(pr, dim=1)
                pr = pr.argmax(dim=1, keepdim=True)
                pr = pr.permute(0, 2, 3, 1)
                pr = pr.to(torch.uint8)
            return pr

    net = Net_with_post(net, no_post=kwargs.get("no_post", False), no_pre=kwargs.get("no_pre", False))

    ts = torch.jit.trace(net, im, strict=False)
    ts.save(str(f))
    return f


def export_paddle(net, f, imgsz=512, **kwargs):
    d = kwargs.get("device", torch.device("cpu"))
    import x2paddle  # noqa
    from x2paddle.convert import pytorch2paddle  # noqa
    fo = f + "_paddle"
    batch = kwargs["batch"]
    if batch > 0:
        im = torch.zeros(batch, imgsz, imgsz, 3).to(d)
    else:
        im = torch.zeros(1, imgsz, imgsz, 3).to(d)
    print("Start Paddle model export")
    pytorch2paddle(module=net, save_dir=fo, jit_type='trace', input_examples=[im])
    print("Paddle model export at %s" % fo)
    return fo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default="config.ini")
    parser.add_argument('-n', '--names', type=str)
    parser.add_argument('-f', '--format', nargs="*", default=["onnx"], help="format to export (onnx,openvino)")
    parser.add_argument('--half', action='store_true', help="set this flag to export fp16 ov model")
    parser.add_argument("-b", '--batch', type=int, default=1, help="batch,set -1 for dynamic batch")
    parser.add_argument("-m", '--model', default=None, help=".pth model path to override config file")
    parser.add_argument('--no-pre', action='store_true', help="Skip Preproccess")
    parser.add_argument('--no-post', action='store_true', help="Skip Postproccess")
    parser.add_argument('--include-resize', action='store_true', help="Add Resize")
    parser.add_argument('--device', default="cpu", help="set this flag to select device from cpu or gpu")
    config = configparser.ConfigParser()
    args = parser.parse_args()
    if os.path.exists(args.config):
        config.read(args.config, encoding="utf-8")
    else:
        config["base"] = {}
        config["advance"] = {}
    CONFIG_DIR = os.path.dirname(os.path.abspath(args.config))
    IMGSZ = config["base"].getint("image_size", 512)
    DATASET_PATH = config["base"].get("dataset_path", 'VOCdevkit')
    if not os.path.isabs(DATASET_PATH):
        DATASET_PATH = os.path.join(CONFIG_DIR, DATASET_PATH)
    SAVE_PATH = config["base"].get("save_path", "save")
    if not os.path.isabs(SAVE_PATH):
        SAVE_PATH = os.path.join(CONFIG_DIR, SAVE_PATH)
    BACKBONE = config["base"].get("backbone", "hgnetv2l")
    NUM_CLASSES = config["base"].getint("num_classes", 21)
    PP = config["base"].get("header", "transformer")

    if "advance" not in config:
        config["advance"] = {}

    DOWNSAMPLE_FACTOR = config["advance"].getint("downsample_factor", 16)
    ARCH = config["base"].get("arch", "lab")
    FORMATS = args.format
    model_path = os.path.join(SAVE_PATH, "best.pth")
    if args.model and os.path.isfile(str(args.model)):
        model_path = str(args.model)

    if ARCH.lower() == "unet":
        from nets.model.UNet import UNet
        net = UNet(num_classes=NUM_CLASSES, pretrained=False, backbone=BACKBONE)
    elif ARCH.lower() == "pspnet":
        from nets.model.PSPNet import pspnet
        net = pspnet(num_classes=NUM_CLASSES, backbone=BACKBONE, downsample_factor=DOWNSAMPLE_FACTOR,
                     pretrained=False)
    elif ARCH.lower() == "segformer":
        from nets.model.SegFormer import SegFormer
        net = SegFormer(num_classes=NUM_CLASSES, backbone=BACKBONE,
                        pretrained=False)
    else:
        net = Labs(num_classes=NUM_CLASSES, backbone=BACKBONE,
                   downsample_factor=DOWNSAMPLE_FACTOR, pretrained=False, header=PP, img_sz=[IMGSZ, IMGSZ])
    if (str(args.device).isdigit() or "cuda" in str(args.device)) and torch.cuda.is_available():
        if str(args.device).isdigit():
            device = torch.device(f'cuda:{args.device}')
        else:
            device = torch.device(f'{args.device}')
    else:
        device = torch.device('cpu')
    net.to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.eval()
    if hasattr(net, "fuse"):
        net.fuse()
    mod = sys.modules[__name__]
    # if args.names and os.path.isfile(args.names):
    #     with open(args.names,mode="r")as f:
    #         lines = [line.strip() for line in f.readlines() if line.strip() != ""]
    #         if len(lines) > 0:
    #             NUM_CLASSES=len(lines)
    for format in FORMATS:
        func = getattr(mod, "export_" + format)
        func(net, Path(SAVE_PATH) / Path(model_path).name, IMGSZ, fp16=args.half, batch=args.batch,
             no_pre=args.no_pre, no_post=args.no_post, include_resize=args.include_resize,
             device=device, num_classes=NUM_CLASSES, names=["?"] * NUM_CLASSES
             )


if __name__ == "__main__":
    main()
