import argparse
import pandas as pd
import os 
from datetime import datetime

from util import constants, data_util
from classifier.predict.generator import predict

def batch_process_pred(df_l_kinases, df_l_sms, batch_size=10000):
    for i in range(0,len(df_l_kinases),batch_size):
        batch_pred_df = pd.DataFrame()
        batch_pred_df['head']= df_l_kinases[i:i+batch_size]
        batch_pred_df['tail']= df_l_sms[i:i+batch_size]
        print(f'Initial size::{batch_pred_df.shape[0]}',end='|')
        batch_pred_df['ksf_pred'] = predict(batch_pred_df,include_label=True)
        yield i, batch_pred_df

def _process_data(kinase=None,substrate=None,motif=None):
    kinases = [kinase] if kinase else data_util.get_kg_kinases()
    substrate_motifs = [substrate+'_'+motif] if substrate and motif else []
    if (not substrate) and motif:        
        substrate_motifs = [x for x in data_util.get_kg_substrate_motifs() if x[x.index('_')+1:] == motif]
    elif (not motif) and substrate:
        substrate_motifs = [x for x in data_util.get_kg_substrate_motifs() if x[:x.index('_')] == substrate]
    elif (not substrate) and (not motif):
        substrate_motifs = [x for x in data_util.get_kg_substrate_motifs()]
    df_l_kinases = kinases * len(substrate_motifs)
    df_l_substrate_motifs = substrate_motifs * len(kinases)
    for i, batch_pred_df in batch_process_pred(df_l_kinases, df_l_substrate_motifs):
        batch_pred_df.dropna(axis=0,how='any',inplace=True)
        batch_pred_df['ksf_pred'] = batch_pred_df['ksf_pred'].apply(lambda x:round(x,3))
        yield i, data_util.process_prediction_output(batch_pred_df)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--k',default='',help='Kinase UniProt ID',required=False)
    parser.add_argument('--sp',default='',help='Substrate Protein UniProt ID',required=False)
    parser.add_argument('--sm',default='',help='Substrate Motif (-/+ 4 mer)',required=False)

    args = parser.parse_args()
    kinase = args.k
    substrate = args.sp
    motif = args.sm
    output_dir = constants.DIR_KSF2_PREDICTIONS

    if (kinase) or (substrate) or (motif):
        for i, result_df in _process_data(kinase,substrate,motif):
            file_name = f'pred_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            result_df.to_csv(os.path.join(output_dir,file_name),index=False,sep='|')
    else:
        print('''Either kinase or substrate or motif information is required. 
              Pass argument --k, --sp, --sm for kinase, susbtrate protein and motif respectively.
               If you need all prediction, download from KSFinder 2.0 repository.''')
        

