import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

def parse_args():
    parser = argparse.ArgumentParser(description='Intelligence Poem and Lyric Writer.')

    help_ = 'choose to train or generate.'
    parser.add_argument('--train', dest='train', action='store_true', help=help_)
    parser.add_argument('--no-train', dest='train', action='store_false', help=help_)
    parser.set_defaults(train=True)

    args_ = parser.parse_args()
    return args_

if __name__ == '__main__':
    args = parse_args()
    from inference import tang_poems
    if args.train:
        tang_poems.main(True)
    else:
        tang_poems.main(False)




