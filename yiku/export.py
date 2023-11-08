import configparser
import argparse
import sys,os
import torch
from torch import nn
import torch.nn.functional as F
import onnx
sys.path.append(os.getcwd())
from nets.labs import Labs
from pathlib import Path

def export_onnx(net,f:Path,imgsz=512,**kwargs):
    f=f.with_suffix(".onnx")
    if kwargs.get("no_pre",False):
        f=Path(str(f).replace(f.stem,f.stem+"+no_pre"))
    if kwargs.get("no_post",False):
        f=Path(str(f).replace(f.stem,f.stem+"+no_post"))
    batch=kwargs["batch"]
    if batch<=0:batch=1
    if kwargs.get("no_pre", False):
        im = torch.zeros(batch, 3, imgsz,imgsz).to('cpu')
    else:
        im = torch.zeros(batch, imgsz, imgsz, 3).to('cpu')
    input_layer_names = ["images"]
    output_layer_names = ["output"]

    class Net_with_post(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            if not kwargs.get("no_pre", False):
                x = x.permute(0, 3, 1, 2)
                x = x / 255.0
            pr = self.m(x)
            if not kwargs.get("no_ost", False):
                pr = F.softmax(pr.permute(0, 2, 3, 1), dim=-1)
                pr = torch.argmax(pr, -1)
            return pr
    net=Net_with_post(net)
    if batch<=0:
        dynamic = {'images': {0: 'batch'}}
        dynamic['output0'] = {0: 'batch'}
    else:
        dynamic=False

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

    # Simplify onnx
    try:
        import onnxsim
        print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
        model_onnx, check = onnxsim.simplify(
            model_onnx,
            dynamic_input_shape=False,
            input_shapes=None)
        assert check, 'assert check failed'
        onnx.save(model_onnx, f)
    except ImportError:
        print(f'onnx-simplifier not installed. SKIP')
    return f

def export_openvino(net,f,imgsz=512,fp16=True,**kwargs):
    import openvino.runtime as ov  # noqa
    from openvino.tools import mo  # noqa
    fo=f+"_openvino"
    onnx_path=export_onnx(net,f,imgsz,**kwargs)
    print("Start OpenVINO model export")
    ov_model = mo.convert_model(onnx_path,
                                model_name="Labs",
                                framework='onnx',
                                compress_to_fp16=fp16)
    ov.serialize(ov_model, os.path.join(fo,"model.xml"))
    print("OpenVINO model export at %s"%os.path.join(fo,"model.xml"))
    return os.path.join(fo,"model.xml")

def export_paddle(net,f,imgsz=512,**kwargs):
    import x2paddle  # noqa
    from x2paddle.convert import pytorch2paddle  # noqa
    fo = f + "_paddle"
    batch = kwargs["batch"]
    if batch > 0:
        im = torch.zeros(batch, imgsz, imgsz, 3).to('cpu')
    else:
        im = torch.zeros(1, imgsz, imgsz, 3).to('cpu')
    print("Start Paddle model export")
    pytorch2paddle(module=net, save_dir=fo, jit_type='trace', input_examples=[im])
    print("Paddle model export at %s" % fo)
    return fo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',default="config.ini")
    parser.add_argument('-f', '--format',nargs="*" ,default=["onnx"],help="format to export (onnx,openvino)")
    parser.add_argument('--half', action='store_true',help="set this flag to export fp16 ov model")
    parser.add_argument("-b",'--batch', type=int,default=1,help="batch,set -1 for dynamic batch")
    parser.add_argument("-m", '--model', default=None, help=".pth model path to override config file")
    parser.add_argument('--no-pre', action='store_true', help="Skip Preproccess")
    parser.add_argument('--no-post', action='store_true', help="Skip Postproccess")
    config = configparser.ConfigParser()
    args = parser.parse_args()
    if os.path.exists(args.config):
        config.read(args.config,encoding="utf-8")
    else:
        config["base"]={}
        config["advance"] = {}
    CONFIG_DIR=os.path.dirname(os.path.abspath(args.config))
    IMGSZ= config["base"].getint("image_size",512)
    DATASET_PATH = config["base"].get("dataset_path", 'VOCdevkit')
    if not os.path.isabs(DATASET_PATH):
        DATASET_PATH = os.path.join(CONFIG_DIR,DATASET_PATH)
    SAVE_PATH = os.path.join(CONFIG_DIR,config["base"].get("save_path"))
    if not os.path.isabs(SAVE_PATH):
        SAVE_PATH=os.path.join(CONFIG_DIR,SAVE_PATH)
    BACKBONE = config["base"].get("backbone","hgnetv2l")
    NUM_CLASSES = config["base"].getint("num_classes",21)
    PP = config["base"].get("header", "transformer")

    if "advance" not in config:
        config["advance"] = {}

    DOWNSAMPLE_FACTOR = config["advance"].getint("downsample_factor",16)
    ARCH = config["base"].get("arch", "lab")
    FORMATS=args.format
    model_path = os.path.join(SAVE_PATH, "best_epoch_weights.pth")
    if args.model and os.path.isfile(str(args.model)):
        model_path=str(args.model)

    if ARCH.lower()=="unet":
        from nets.third_party.UNet import UNet
        net=UNet(num_classes=NUM_CLASSES,pretrained=False)
    else:
        net = Labs(num_classes=NUM_CLASSES, backbone=BACKBONE,
               downsample_factor=DOWNSAMPLE_FACTOR, pretrained=False, header=PP, img_sz=[IMGSZ,IMGSZ])
    device = torch.device('cpu')
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.eval()
    mod = sys.modules[__name__]
    for format in FORMATS:
        func = getattr(mod, "export_"+format)
        func(net,Path(SAVE_PATH)/Path(model_path).name,IMGSZ,fp16=args.half,batch=args.batch,
             no_pre=args.no_pre,no_post=args.no_post
             )

if __name__ == "__main__":
    main()