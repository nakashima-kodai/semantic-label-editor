from options.base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        ### for display ###
        self.parser.add_argument('--save_epoch_freq', type=int, default=1)
        self.parser.add_argument('--print_freq', type=int, default=100)

        ### for training ###
        self.parser.add_argument('--phase', default='train')

        self.isTrain = True
