import random
from util import constants
import pandas as pd

random.seed(13)

def normalize_data_cnt(test_df):
    label_cnts = test_df['label'].value_counts()
    pos_label_cnt, neg_label_cnt = label_cnts[1], label_cnts[0]
    if pos_label_cnt > neg_label_cnt:
        select_indices = random.sample(test_df[test_df['label']==1].index.to_list(),neg_label_cnt)
        select_indices.extend(test_df[test_df['label']==0].index.to_list())
        test_df = test_df[test_df.index.isin(select_indices)].copy()
    elif neg_label_cnt > pos_label_cnt:
        select_indices = random.sample(test_df[test_df['label']==0].index.to_list(),pos_label_cnt)
        select_indices.extend(test_df[test_df['label']==1].index.to_list())
        test_df = test_df[test_df.index.isin(select_indices)].copy()
    return test_df

def get_kg_substrate_motifs():
    substrate_motif = pd.read_csv(constants.CSV_SUBSTRATES_MOTIF,sep='|')
    substrate_motif['substrate_motif']=substrate_motif.apply(lambda x:x['Seq_Substrate']+'_'+x['Motif'],axis=1)
    substrate_motifs = substrate_motif['substrate_motif'].unique()
    return substrate_motifs

def process_prediction_output(df):
    df.rename({'head':'kinase','tail':'substrate_motif','ksf_pred':'ksf2_pred'},axis=1,inplace=True)
    df = df[['kinase','substrate_motif','ksf2_pred']].copy()
    df.drop_duplicates(inplace=True)
    return df

def get_kg_kinases():
    kinase_df = pd.read_csv(constants.CSV_KG_KINASES,sep='|')
    return kinase_df['Kinase'].to_list()