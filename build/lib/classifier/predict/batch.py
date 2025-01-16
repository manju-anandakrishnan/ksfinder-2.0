import pandas as pd
import os
from datetime import datetime

from util import data_util,constants
from classifier.predict.generator import predict
from itertools import product

kinases = data_util.get_kg_kinases()
substrate_motifs = data_util.get_kg_substrate_motifs()
df_l_kinases, df_l_substrate_motifs = zip(*product(kinases, substrate_motifs))
total_cnt = len(df_l_kinases)

print('Total records to process::',len(df_l_kinases),len(df_l_substrate_motifs))

def batch_process_pred(df_l_kinases, df_l_sms, batch_size=500000):
    for i in range(0,len(df_l_kinases),batch_size):
        batch_pred_df = pd.DataFrame()
        batch_pred_df['head']= df_l_kinases[i:i+batch_size]
        batch_pred_df['tail']= df_l_sms[i:i+batch_size]
        print(f'Initial size::{batch_pred_df.shape[0]}',end='|')
        batch_pred_df['ksf_pred'] = predict(batch_pred_df,include_label=True)
        yield i, batch_pred_df

output_dir = constants.DIR_KSF2_PREDICTIONS_BATCH

processed_cnt = 0
for i, batch_pred_df in batch_process_pred(df_l_kinases, df_l_substrate_motifs):
    batch_pred_df.dropna(axis=0,how='any',inplace=True)
    batch_pred_df['ksf_pred'] = batch_pred_df['ksf_pred'].apply(lambda x:round(x,3))
    file_name = f'pred_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    processed_cnt = processed_cnt+batch_pred_df.shape[0]
    print(f'Remaining count:{total_cnt-processed_cnt}')
    data_util.process_prediction_output(batch_pred_df).to_csv(os.path.join(output_dir,file_name),sep='|',index=False)

print('Completed processing all')