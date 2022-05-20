## 라이브러리 추가하기
import argparse

from train import *
## Parser 생성하기
# parser = argparse.ArgumentParser(description="Train the Unet", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser = argparse.ArgumentParser(description="Train the CycleGAN")

parser.add_argument("--port", default=35943)

parser.add_argument("--lr", default=2e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=16, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=30, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default='./datasets', type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default='./checkpoint', type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default='./log/new2', type=str, dest="log_dir")
parser.add_argument("--info_dir", default='/checkpoint', type=str, dest="info_dir")
parser.add_argument("--result_dir", default='./result', type=str, dest="result_dir")

parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

parser.add_argument("--task", default="pix2pix", choices=["DCGAN", "pix2pix", "cyclegan"], type=str, dest="task")
parser.add_argument("--opts", nargs='+', default=["direction", 0], dest="opts")

parser.add_argument("--ny", default=128, type=int, dest="ny")
parser.add_argument("--nx", default=128, type=int, dest="nx")
parser.add_argument("--nch", default=1, type=int, dest="nch")
parser.add_argument("--nker", default=64, type=int, dest="nker")

parser.add_argument("--wgt_cycle", default=1e1, type=float, dest="wgt_cycle")
parser.add_argument("--wgt_ident", default=5e-1, type=float, dest="wgt_ident")
parser.add_argument("--norm", default='inorm', type=str, dest="norm")

parser.add_argument("--network", default="Unet", choices=["DCGAN", "pix2pix", "cyclegan", "Unet"], type=str, dest="network")
parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")
args = parser.parse_args()

if args.mode == "client":
    args.mode = "train"

if __name__ == "__main__":
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
        