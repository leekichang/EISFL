import sys
sys.path.insert(0, '/disk2/Kichang/EISFL')

class NoDefense(object):
    def __init__(self, args):
        self.args = args
    
    def __call__(self, model, *args):
        pass