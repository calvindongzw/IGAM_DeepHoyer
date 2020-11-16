import configargparse
import pdb

def pair(arg):
    return [float(x) for x in arg.split(',')]

def get_args():
    parser = configargparse.ArgParser(default_config_files=[])
    #parser.add("--config", type=str, is_config_file=True, help="You can store all the config args in a config file and pass the path here")
    parser.add("--model_dir", type=str, default="models/", help="Path to save/load the checkpoints, default=models/model")
    parser.add("--data_path", type=str, default="datasets/cifar100", help="Path to load datasets from, default=datasets")
    parser.add("--target_class", type=int, default=100, help="Number of target class, default=100")
    parser.add("--lr", type=float, default=0.1, help="Learning rate, default=0.1")
    parser.add("--momentum", type=float, default=0.9, help="The momentum parameter, default=0.9")
    parser.add("--weight_decay", "-w", type=float, default=0.0002, help="The weight decay parameter, default=0.0002")
    parser.add("--lr_boundaries", type=int, default=[40000,60000], help="The boundaries of learning rate decay, default=[5000,7000]")
    parser.add("--lr_decay", type=float, default=0.1, help="Learning rate decay value, default=0.1")
    parser.add("--batch_size", type=int, default=128, help="The training batch size, default=128")
    parser.add("--image_size", type=int, default=32, help="One dimension of image size (due to squared image), default=32")
    parser.add("--flip_rate", type=float, default=1.0, help="The rate of raw image horizontally flip, default=1.0")
    parser.add("--train_steps", type=int, default=80000, help="Maximum number of training steps, default=80000")
    parser.add("--summary_steps", type=int, default=500, help="Number of summary steps, default=500") 
    parser.add("--beta", type=float, default=1, help="Weight of input gradient regularization, default=1")
    parser.add("--gamma", type=float, default=0.001, help="Weight of disc xent term on encoder opt, default=1")

    args = parser.parse_args()
    return args


if __name__ == "__main__": 
    print(get_args())
    pdb.set_trace()