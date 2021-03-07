import pandas as pd
import ast


if __name__ == '__main__':
    imgs_record_df = pd.read_csv('./logs/model6000_val_epoch - imgs.csv')
    last_row = imgs_record_df.tail(1)
    test_imgs_temp = ast.literal_eval(last_row['train'].tolist()[0])
    test_imgs = []
    for img_dict in test_imgs_temp:
        print(img_dict)
        # print(img_dict['bboxes'][0]['class'])
        test_imgs.append(img_dict['filepath'])

    # print(test_imgs)

