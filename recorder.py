import matplotlib.pyplot as plt
import numpy as np
import os.path


class Recorder:

    def __init__(self, base_path, has_validation=False):
        self.base_path = base_path
        self.has_validation = has_validation
        self.record = {'class_acc': [],
                       'loss_rpn_cls': [],
                       'loss_rpn_regr': [],
                       'loss_class_cls': [],
                       'loss_class_regr': [],
                       'curr_loss': [],
                       'elapsed_time': [],
                       'loss_rpn_cls_val': [],
                       'loss_rpn_regr_val': [],
                       'loss_class_cls_val': [],
                       'loss_class_regr_val': [],
                       'class_acc_val': [],
                       'curr_loss_val': [],
                       'best_loss_val': []}

    def _create_graphs(self, save=False):
        num_epochs = len(self.record['class_acc'])

        # Training
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(0, num_epochs), self.record['loss_rpn_cls'], 'r')
        plt.title('loss_rpn_cls')
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(0, num_epochs), self.record['loss_rpn_regr'], 'r')
        plt.title('loss_rpn_regr')
        if save:
            plt.savefig(os.path.join(self.base_path, 'loss_rpn_cls&loss_rpn_regr.png'))
        else:
            plt.show()

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(0, num_epochs), self.record['loss_class_cls'], 'r')
        plt.title('loss_class_cls')
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(0, num_epochs), self.record['loss_class_regr'], 'r')
        plt.title('loss_class_regr')
        if save:
            plt.savefig(os.path.join(self.base_path, 'loss_class_cls&loss_class_regr.png'))
        else:
            plt.show()

        plt.figure(figsize=(15, 5))
        plt.plot(np.arange(0, num_epochs), self.record['curr_loss'], 'r')
        plt.title('total_loss')
        if save:
            plt.savefig(os.path.join(self.base_path, 'total_loss.png'))
        else:
            plt.show()

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(0, num_epochs), self.record['curr_loss'], 'r')
        plt.title('total_loss')
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(0, num_epochs), self.record['elapsed_time'], 'r')
        plt.title('elapsed_time')
        if save:
            plt.savefig(os.path.join(self.base_path, 'total_loss&elapsed_time.png'))
        else:
            plt.show()

        plt.figure(figsize=(15, 5))
        plt.title('loss')
        plt.plot(np.arange(0, num_epochs), self.record['loss_rpn_cls'], 'b')
        plt.plot(np.arange(0, num_epochs), self.record['loss_rpn_regr'], 'g')
        plt.plot(np.arange(0, num_epochs), self.record['loss_class_cls'], 'r')
        plt.plot(np.arange(0, num_epochs), self.record['loss_class_regr'], 'c')
        if save:
            plt.savefig(os.path.join(self.base_path, 'loss.png'))
        else:
            plt.show()

        # Validation
        if self.has_validation:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            plt.plot(np.arange(0, num_epochs), self.record['loss_rpn_cls_val'], 'r')
            plt.title('loss_rpn_cls_val')
            plt.subplot(1, 2, 2)
            plt.plot(np.arange(0, num_epochs), self.record['loss_rpn_regr_val'], 'r')
            plt.title('loss_rpn_regr_val')
            if save:
                plt.savefig(os.path.join(self.base_path, 'loss_rpn_cls_val&loss_rpn_regr_val.png'))
            else:
                plt.show()

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            plt.plot(np.arange(0, num_epochs), self.record['loss_class_cls_val'], 'r')
            plt.title('loss_class_cls_val')
            plt.subplot(1, 2, 2)
            plt.plot(np.arange(0, num_epochs), self.record['loss_class_regr_val'], 'r')
            plt.title('loss_class_regr_val')
            if save:
                plt.savefig(os.path.join(self.base_path, 'loss_class_cls_val&loss_class_regr_val.png'))
            else:
                plt.show()

            plt.figure(figsize=(15, 5))
            plt.plot(np.arange(0, num_epochs), self.record['curr_loss_val'], 'r')
            plt.title('total_loss_val')
            if save:
                plt.savefig(os.path.join(self.base_path, 'total_loss_val.png'))
            else:
                plt.show()

            plt.figure(figsize=(15, 5))
            plt.plot(np.arange(0, num_epochs), self.record['best_loss_val'], 'r')
            plt.title('best_loss_val')
            if save:
                plt.savefig(os.path.join(self.base_path, 'best_loss_val.png'))
            else:
                plt.show()

            plt.figure(figsize=(15, 5))
            plt.title('loss_val')
            plt.plot(np.arange(0, num_epochs), self.record['loss_rpn_cls_val'], 'b')
            plt.plot(np.arange(0, num_epochs), self.record['loss_rpn_regr_val'], 'g')
            plt.plot(np.arange(0, num_epochs), self.record['loss_class_cls_val'], 'r')
            plt.plot(np.arange(0, num_epochs), self.record['loss_class_regr_val'], 'c')
            if save:
                plt.savefig(os.path.join(self.base_path, 'loss_val.png'))
            else:
                plt.show()

    def add_new_entry_with_validation(self, class_acc, loss_rpn_cls, loss_rpn_regr, loss_class_cls, loss_class_regr, curr_loss,
                      elapsed_time, class_acc_val, loss_rpn_cls_val, loss_rpn_regr_val, loss_class_cls_val,
                      loss_class_regr_val, curr_loss_val, best_loss_val):
        self.record['class_acc'].append(class_acc)
        self.record['loss_rpn_cls'].append(loss_rpn_cls)
        self.record['loss_rpn_regr'].append(loss_rpn_regr)
        self.record['loss_class_cls'].append(loss_class_cls)
        self.record['loss_class_regr'].append(loss_class_regr)
        self.record['curr_loss'].append(curr_loss)
        self.record['elapsed_time'].append(elapsed_time)
        self.record['loss_rpn_cls_val'].append(loss_rpn_cls_val)
        self.record['loss_rpn_regr_val'].append(loss_rpn_regr_val)
        self.record['loss_class_cls_val'].append(loss_class_cls_val)
        self.record['loss_class_regr_val'].append(loss_class_regr_val)
        self.record['class_acc_val'].append(class_acc_val)
        self.record['curr_loss_val'].append(curr_loss_val)
        self.record['best_loss_val'].append(best_loss_val)

    def add_new_entry(self, class_acc, loss_rpn_cls, loss_rpn_regr, loss_class_cls, loss_class_regr, curr_loss,
                      elapsed_time):
        self.record['class_acc'].append(class_acc)
        self.record['loss_rpn_cls'].append(loss_rpn_cls)
        self.record['loss_rpn_regr'].append(loss_rpn_regr)
        self.record['loss_class_cls'].append(loss_class_cls)
        self.record['loss_class_regr'].append(loss_class_regr)
        self.record['curr_loss'].append(curr_loss)
        self.record['elapsed_time'].append(elapsed_time)

    def show_graphs(self):
        self._create_graphs(False)

    def save_graphs(self):
        if not os.path.exists(self.base_path):
            os.mkdir(self.base_path)
        self._create_graphs(True)
