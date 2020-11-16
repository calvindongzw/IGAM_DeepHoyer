import configargparse
import pdb

def pair(arg):
    return [float(x) for x in arg.split(',')]

def get_args():
    parser = configargparse.ArgParser(default_config_files=[])
    #parser.add("--config", type=str, is_config_file=True, help="You can store all the config args in a config file and pass the path here")
    parser.add("--model_dir", type=str, default="./models/", help="Path to save/load the checkpoints, default=models/model")
    parser.add("--data_path", type=str, default="datasets/cifar10", help="Path to load datasets from, default=datasets")
    parser.add("--output_class", type=int, default=10, help="Number of output class, default=10")
    parser.add("--lr", type=float, default=0.1, help="Learning rate, default=0.1")
    parser.add("--momentum", type=float, default=0.9, help="The momentum parameter, default=0.9")
    parser.add("--weight_decay", "-w", type=float, default=0.0002, help="The weight decay parameter, default=0.0002")
    parser.add("--lr_boundaries", type=int, default=[50,75], help="The boundaries of learning rate decay, default=[5000,7000]")
    parser.add("--lr_decay", type=float, default=0.1, help="Learning rate decay value, default=0.1")
    parser.add("--batch_size", type=int, default=128, help="The training batch size, default=128")
    parser.add("--image_size", type=int, default=32, help="One dimension of image size (due to squared image), default=32")
    parser.add("--attack_steps", "-k", type=int, default=7, help="Number of steps to PGD attack, default=7")
    parser.add("--epsilon", "-e", type=float, default=0.3, help="Epsilon (Lp Norm distance from the original image) for generating adversarial examples, default=8.0")
    parser.add("--step_size", "-s", type=float, default=2/225, help="Step size in PGD attack for generating adversarial examples in each step, default=2.0")
    parser.add("--img_rand_pert", dest="img_rand_pert", action="store_true", help="Random start image pertubation augmentation default=False")
    parser.set_defaults(img_rand_pert=False)
    
    parser.add("--do_advtrain", dest="do_advtrain", action="store_true", help="Do adversarial training default=False")
    parser.set_defaults(do_advtrain=True)

    parser.add("--random_start", dest="random_start", action="store_true", help="Random start for PGD attack default=True")
    #parser.add("--no-random_start", dest="random_start", action="store_false", help="No random start for PGD attack default=True")
    parser.set_defaults(random_start=True)

    parser.add("--normalize_zero_mean", dest="normalize_zero_mean", action="store_true", help="Normalize classifier input to zero mean default=True")
    #parser.add("--no-normalize_zero_mean", dest="normalize_zero_mean", action="store_false", help="Normalize classifier input to zero mean default=True")
    parser.set_defaults(normalize_zero_mean=True)

    parser.add("--steps_before_adv_training", type=int, default=0, help="Number of training steps to wait before pgd adv training, default=0")
    parser.add("--train_steps", type=int, default=100, help="Maximum number of training steps, default=200")
    parser.add("--summary_steps", type=int, default=10, help="Number of summary steps, default=20") 
    parser.add("--flip_rate", type=float, default=1.0, help="The rate of raw image horizontally flip, default=1.0")
    # input grad generation param
    args = parser.parse_args()
    return args


if __name__ == "__main__": 
    print(get_args())
    pdb.set_trace()