import pandas as pd
import random

from classifier.predict.generator import predict
from util.metrics import Curve, Score
from util import constants
from util import data_util

random.seed(13)

def load_data():
    train_df = pd.read_csv(constants.CSV_CLF_TRAIN_DATA,sep='|')
    train_df['motif'] = train_df['tail'].apply(lambda x:x[x.find('_')+1:])
    train_df['substrate'] = train_df['tail'].apply(lambda x:x[:x.find('_')])
    train_df['km'] = train_df.apply(lambda x:(x['head'],x['motif']),axis=1)
    tr_pos_data = train_df[train_df['label']==1].copy()
    tr_pos_data_km = tr_pos_data['km'].unique()

    test_df = pd.read_csv(constants.CSV_CLF_TEST_D2,sep='|')
    test_df['motif'] = test_df['tail'].apply(lambda x:x[x.find('_')+1:])
    test_df['substrate'] = test_df['tail'].apply(lambda x:x[:x.find('_')])
    test_df['km'] = test_df.apply(lambda x:(x['head'],x['motif']),axis=1)
    test_pos_data = test_df[test_df['label']==1].copy()
    test_pos_data_km = test_pos_data['km'].unique()

    tr_pos_data_km = set(tr_pos_data_km)
    tr_pos_data_km.update(set(test_pos_data_km))

    return tr_pos_data_km


if __name__ == '__main__':

    tr_pos_data_km = load_data()
    test_df = pd.read_csv(constants.CSV_PHOSFORMER_PREDICTIONS,sep='|')
    pos_test_df = test_df[test_df['label']==1].copy()
    neg_test_df = test_df[test_df['label']==0].copy()
    
    print("Filtering of 300 serine-threonine kinases used in Phosformer-ST's work")
    phos_comp_kinases = None
    with open(constants.PHOS_ST_KINASES) as ip_f:
        phos_comp_kinases =  [x.strip() for x in ip_f]
    if phos_comp_kinases:
        test_df = test_df[test_df['head'].isin(phos_comp_kinases)].copy()

    pos_test_df = test_df[test_df['label']==1].copy()
    neg_test_df = test_df[test_df['label']==0].copy()
    print('Total test sample count::',pos_test_df.shape, neg_test_df.shape)

    test_df = pd.concat([pos_test_df,neg_test_df])
    test_df = test_df[['head','tail','label','motif','substrate','motif_15mer','phosST_pred']].copy()

    na_df = test_df[test_df.isnull().any(axis=1)].copy()
    
    test_df.drop_duplicates(inplace=True)
    test_df.dropna(axis=0,how='any',inplace=True)

    test_df['km'] = test_df.apply(lambda x:(x['head'],x['motif']),axis=1)
    pos_data = test_df[test_df['label']==1].copy()
    pos_data_km = pos_data['km'].unique()

    neg_data = test_df[test_df['label']==0].copy()
    neg_km_indices = neg_data[(neg_data['km'].isin(pos_data_km))
                          | (neg_data['km'].isin(tr_pos_data_km))].index.to_list()
    neg_data = neg_data[~neg_data.index.isin(neg_km_indices)].copy()
    
    test_df = pd.concat([pos_data,neg_data])
    test_df.drop_duplicates(inplace=True)
    test_df.dropna(axis=0,how='any',inplace=True)
    test_df.reset_index(drop=True,inplace=True)    

    ksf_pred = predict(test_df)
    test_df['ksf_pred'] = ksf_pred
    test_df.dropna(axis=0,how='any',inplace=True)

    # Testing dataset 2
    test_df2 = test_df.copy()

    # Testing dataset 1
    test_df1 = data_util.normalize_data_cnt(test_df)
    print(test_df1['label'].value_counts().to_dict())
    
    y_true = test_df1['label'].to_list()
    phosST_pred = test_df1['phosST_pred'].to_list()
    ksf_pred = test_df1['ksf_pred'].to_list()

    roc_curve = Curve.get_roc_curves([y_true, y_true],
                                     [phosST_pred,ksf_pred],
                                     ['blue','magenta'],)
    pr_curve = Curve.get_pr_curves([y_true, y_true],[phosST_pred,ksf_pred],
                                   ['blue','magenta'],)
    
    roc_score, _,_,_ = Score.get_roc_score(y_true,ksf_pred)
    pr_score, _,_,_ = Score.get_pr_score(y_true,ksf_pred)
    print(f'KSFinder 2.0:: ROC-AUC: {roc_score} | PR-AUC: {pr_score}')
    
    roc_score, _,_,_ = Score.get_roc_score(y_true,phosST_pred)
    pr_score, _,_,_ = Score.get_pr_score(y_true,phosST_pred)
    print(f'Phosformer-ST:: ROC-AUC: {roc_score} | PR-AUC: {pr_score}')

    roc_curve.savefig(constants.KSF2_PHOS_ST_ASSESS1_ROC_CURVES)
    pr_curve.savefig(constants.KSF2_PHOS_ST_ASSESS1_PR_CURVES)

    # Testing dataset 2 evaluation
    print(test_df2['label'].value_counts().to_dict())
    
    y_true = test_df2['label'].to_list()
    phosST_pred = test_df2['phosST_pred'].to_list()
    ksf_pred = test_df2['ksf_pred'].to_list()

    roc_curve = Curve.get_roc_curves([y_true, y_true],
                                     [phosST_pred,ksf_pred],
                                     ['blue','magenta'],)
    pr_curve = Curve.get_pr_curves([y_true, y_true],[phosST_pred,ksf_pred],
                                   ['blue','magenta'],)
    
    roc_score, _,_,_ = Score.get_roc_score(y_true,ksf_pred)
    pr_score, _,_,_ = Score.get_pr_score(y_true,ksf_pred)
    print(f'KSFinder 2.0:: ROC-AUC: {roc_score} | PR-AUC: {pr_score}')
    
    roc_score, _,_,_ = Score.get_roc_score(y_true,phosST_pred)
    pr_score, _,_,_ = Score.get_pr_score(y_true,phosST_pred)
    print(f'Phosformer-ST:: ROC-AUC: {roc_score} | PR-AUC: {pr_score}')

    roc_curve.savefig(constants.KSF2_PHOS_ST_TD2_ASSESS1_ROC_CURVES)
    pr_curve.savefig(constants.KSF2_PHOS_ST_TD2_ASSESS1_PR_CURVES)