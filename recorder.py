import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd


class Recorder:

    def __init__(self, base_path, has_validation=False):
        self.base_path = base_path
        self.has_validation = has_validation
        self.csv_path = base_path + ".csv"
        try:
            self.record_df = pd.read_csv(self.csv_path)
        except Exception as _:
            self.record_df = pd.DataFrame(columns=['class_acc',
                                                   'loss_rpn_cls',
                                                   'loss_rpn_regr',
                                                   'loss_class_cls',
                                                   'loss_class_regr',
                                                   'curr_loss',
                                                   'elapsed_time',
                                                   'loss_rpn_cls_val',
                                                   'loss_rpn_regr_val',
                                                   'loss_class_cls_val',
                                                   'loss_class_regr_val',
                                                   'class_acc_val',
                                                   'curr_loss_val',
                                                   'best_loss_val'])

    def _create_graphs(self, save=False):
        num_epochs = len(self.record_df['class_acc'])

        # Training
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(1, num_epochs + 1), self.record_df['loss_rpn_cls'], 'r')
        plt.title('loss_rpn_cls')
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(1, num_epochs + 1), self.record_df['loss_rpn_regr'], 'r')
        plt.title('loss_rpn_regr')
        if save:
            plt.savefig(os.path.join(self.base_path, 'loss_rpn_cls&loss_rpn_regr.png'))
        else:
            plt.show()

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(1, num_epochs + 1), self.record_df['loss_class_cls'], 'r')
        plt.title('loss_class_cls')
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(1, num_epochs + 1), self.record_df['loss_class_regr'], 'r')
        plt.title('loss_class_regr')
        if save:
            plt.savefig(os.path.join(self.base_path, 'loss_class_cls&loss_class_regr.png'))
        else:
            plt.show()

        plt.figure(figsize=(15, 5))
        plt.plot(np.arange(1, num_epochs + 1), self.record_df['curr_loss'], 'r')
        plt.title('total_loss')
        if save:
            plt.savefig(os.path.join(self.base_path, 'total_loss.png'))
        else:
            plt.show()

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(1, num_epochs + 1), self.record_df['curr_loss'], 'r')
        plt.title('total_loss')
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(1, num_epochs + 1), self.record_df['elapsed_time'], 'r')
        plt.title('elapsed_time')
        if save:
            plt.savefig(os.path.join(self.base_path, 'total_loss&elapsed_time.png'))
        else:
            plt.show()

        plt.figure(figsize=(15, 5))
        plt.title('loss')
        plt.plot(np.arange(1, num_epochs + 1), self.record_df['loss_rpn_cls'], 'b')
        plt.plot(np.arange(1, num_epochs + 1), self.record_df['loss_rpn_regr'], 'g')
        plt.plot(np.arange(1, num_epochs + 1), self.record_df['loss_class_cls'], 'r')
        plt.plot(np.arange(1, num_epochs + 1), self.record_df['loss_class_regr'], 'c')
        if save:
            plt.savefig(os.path.join(self.base_path, 'loss.png'))
        else:
            plt.show()

        # Validation
        if self.has_validation:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            plt.plot(np.arange(1, num_epochs + 1), self.record_df['loss_rpn_cls_val'], 'r')
            plt.title('loss_rpn_cls_val')
            plt.subplot(1, 2, 2)
            plt.plot(np.arange(1, num_epochs + 1), self.record_df['loss_rpn_regr_val'], 'r')
            plt.title('loss_rpn_regr_val')
            if save:
                plt.savefig(os.path.join(self.base_path, 'loss_rpn_cls_val&loss_rpn_regr_val.png'))
            else:
                plt.show()

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            plt.plot(np.arange(1, num_epochs + 1), self.record_df['loss_class_cls_val'], 'r')
            plt.title('loss_class_cls_val')
            plt.subplot(1, 2, 2)
            plt.plot(np.arange(1, num_epochs + 1), self.record_df['loss_class_regr_val'], 'r')
            plt.title('loss_class_regr_val')
            if save:
                plt.savefig(os.path.join(self.base_path, 'loss_class_cls_val&loss_class_regr_val.png'))
            else:
                plt.show()

            plt.figure(figsize=(15, 5))
            plt.plot(np.arange(1, num_epochs + 1), self.record_df['curr_loss_val'], 'r')
            plt.title('total_loss_val')
            if save:
                plt.savefig(os.path.join(self.base_path, 'total_loss_val.png'))
            else:
                plt.show()

            plt.figure(figsize=(15, 5))
            plt.plot(np.arange(1, num_epochs + 1), self.record_df['best_loss_val'], 'r')
            plt.title('best_loss_val')
            if save:
                plt.savefig(os.path.join(self.base_path, 'best_loss_val.png'))
            else:
                plt.show()

            plt.figure(figsize=(15, 5))
            plt.title('loss_val')
            plt.plot(np.arange(1, num_epochs + 1), self.record_df['loss_rpn_cls_val'], 'b')
            plt.plot(np.arange(1, num_epochs + 1), self.record_df['loss_rpn_regr_val'], 'g')
            plt.plot(np.arange(1, num_epochs + 1), self.record_df['loss_class_cls_val'], 'r')
            plt.plot(np.arange(1, num_epochs + 1), self.record_df['loss_class_regr_val'], 'c')
            if save:
                plt.savefig(os.path.join(self.base_path, 'loss_val.png'))
            else:
                plt.show()

    def add_new_entry_with_validation(self, class_acc, loss_rpn_cls, loss_rpn_regr, loss_class_cls, loss_class_regr,
                                      curr_loss,
                                      elapsed_time, class_acc_val, loss_rpn_cls_val, loss_rpn_regr_val,
                                      loss_class_cls_val,
                                      loss_class_regr_val, curr_loss_val, best_loss_val):
        new_row_val = {
            'class_acc': class_acc,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_regr': loss_rpn_regr,
            'loss_class_cls': loss_class_cls,
            'loss_class_regr': loss_class_regr,
            'curr_loss': curr_loss,
            'elapsed_time': elapsed_time,
            'loss_rpn_cls_val': loss_rpn_cls_val,
            'loss_rpn_regr_val': loss_rpn_regr_val,
            'loss_class_cls_val': loss_class_cls_val,
            'loss_class_regr_val': loss_class_regr_val,
            'class_acc_val': class_acc_val,
            'curr_loss_val': curr_loss_val,
            'best_loss_val': best_loss_val
        }
        self.record_df = self.record_df.append(new_row_val, ignore_index=True)
        self.record_df.to_csv(self.csv_path, index=0)

    def add_new_entry(self, class_acc, loss_rpn_cls, loss_rpn_regr, loss_class_cls, loss_class_regr, curr_loss,
                      elapsed_time):
        new_row = {
            'class_acc': class_acc,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_regr': loss_rpn_regr,
            'loss_class_cls': loss_class_cls,
            'loss_class_regr': loss_class_regr,
            'curr_loss': curr_loss,
            'elapsed_time': elapsed_time
        }
        self.record_df = self.record_df.append(new_row, ignore_index=True)
        self.record_df.to_csv(self.csv_path, index=0)

    def show_graphs(self):
        self._create_graphs(False)

    def save_graphs(self):
        if not os.path.exists(self.base_path):
            os.mkdir(self.base_path)
        self._create_graphs(True)
