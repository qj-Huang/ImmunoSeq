import sys
import argparse

class Args:
    def __init__(self):
        self.args = {}

    def add(self, key, val):
        self.args[key] = val

    def digest(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
        for key, val in self.args.items():
            parser.add_argument('--' + key, type=type(val), default=val)
        conf = parser.parse_args(sys.argv[1:])        
        return conf

args = Args()
args.add('infer_round', 1)           # iteration round
args.add('min_mer', 8)               # prepare k-mer files
args.add('max_mer', 13)              # prepare k-mer files
conf = args.digest()