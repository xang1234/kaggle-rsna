import os
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES']='0'

def run_ensemble():

    file=[
      #replace with retrain   
        # '../input/5resnext50-4efficientnetb3-3inceptionv4/model001_fold0_ep2_test_tta5.csv', #0.72
        '../input/5resnext50-4efficientnetb3-3inceptionv4/model001_fold1_ep2_test_tta5.csv', #0.70
        '../input/5resnext50-4efficientnetb3-3inceptionv4/model001_fold2_ep2_test_tta5.csv', #0.71
      #  '../input/5resnext50-4efficientnetb3-3inceptionv4/model001_fold3_ep2_test_tta5.csv',  #0.72
        '../input/5resnext50-4efficientnetb3-3inceptionv4/model001_fold4_ep2_test_tta5.csv',  #0.71
        '../input/5resnext50-4efficientnetb3-3inceptionv4/model003_2_fold3_ep2_test_tta5.csv',#0.71
     #   '../input/5resnext50-4efficientnetb3-3inceptionv4/model003_fold0_ep1_test_tta5.csv', #0.72
        '../input/5resnext50-4efficientnetb3-3inceptionv4/model003_fold2_ep2_test_tta5.csv', #0.70
        '../input/5resnext50-4efficientnetb3-3inceptionv4/model003_fold4_ep2_test_tta5.csv', #0.71
     #   '../input/5resnext50-4efficientnetb3-3inceptionv4/model004_fold0_ep2_test_tta5.csv', #0.73
        '../input/5resnext50-4efficientnetb3-3inceptionv4/model004_fold1_ep2_test_tta5.csv',#0.69
        '../input/5resnext50-4efficientnetb3-3inceptionv4/model004_fold2_ep2_test_tta5.csv', #0.71
        

    ]


    ############################################################




    csv_file='ensemble071..csv'
    label = 0
    for f in file:
        #print(f)
        df = pd.read_csv(f)
        df = df.sort_values('ID')

        id = df['ID'].values
        label += df['Label'].values

    label = label/len(file)
    df = pd.DataFrame(zip(id, label), columns=['ID', 'Label'])
    df.to_csv(csv_file, index=False)
    return df
