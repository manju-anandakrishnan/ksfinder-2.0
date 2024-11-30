import pandas as pd
import random

from classifier.predict.generator import predict
from util.metrics import Curve, Score
from util import constants
from util import data_util

random.seed(13)

if __name__ == '__main__':

    pred_df = pd.read_csv(constants.CSV_LINKPHINDER_PREDICTIONS,sep='\t')
    pred_df = pred_df[['ProteinKinase_ID','ProteinSubstrate_ID','Site','Score']]
    
    test_df = pd.read_csv(constants.CSV_CLF_TEST_D2,sep='|')
    test_df['motif'] = test_df['tail'].apply(lambda x:x[x.find('_')+1:])
    test_df['substrate'] = test_df['tail'].apply(lambda x:x[:x.find('_')])

    substrates_df = pd.read_csv(constants.CSV_SUBSTRATES_MOTIF,sep='|')
    test_df = test_df.merge(substrates_df,how='left',left_on=['substrate','motif'],right_on=['Seq_Substrate','Motif'])

    test_df = test_df[['head','tail','label','substrate','Site']]
    test_df.dropna(how='any',axis=0,inplace=True)
    test_df.drop_duplicates(inplace=True)

    test_df = test_df.merge(pred_df,how='left',left_on=['head','substrate','Site'],right_on=['ProteinKinase_ID','ProteinSubstrate_ID','Site'])
    test_df.dropna(axis=0,how='any',inplace=True)
    test_df = test_df[['head','tail','label','Score']].copy()
    test_df.drop_duplicates(inplace=True)
    
    neg_test_df = test_df[test_df['label']==0].copy()
    
    test_df = data_util.normalize_data_cnt(test_df)    
    print(test_df['label'].value_counts().to_dict())

    y_true = test_df['label'].to_list()
    linkphinder_pred = test_df['Score'].to_list()
    test_df.dropna(axis=0,how='any',inplace=True)
    
    ksf_pred = predict(test_df)
    test_df['ksf_pred'] = ksf_pred
    test_df.dropna(axis=0,how='any',inplace=True)
    print(test_df['label'].value_counts())

    roc_curve = Curve.get_roc_curves([y_true, y_true],
                                     [linkphinder_pred,ksf_pred],
                                     ['blue','magenta'],)
    pr_curve = Curve.get_pr_curves([y_true, y_true],[linkphinder_pred,ksf_pred],
                                   ['blue','magenta'],)
    
    roc_score, _,_,_ = Score.get_roc_score(y_true,ksf_pred)
    pr_score, _,_,_ = Score.get_pr_score(y_true,ksf_pred)
    print(f'KSFinder 2.0:: ROC-AUC: {roc_score} | PR-AUC: {pr_score}')
    
    roc_score, _,_,_ = Score.get_roc_score(y_true,linkphinder_pred)
    pr_score, _,_,_ = Score.get_pr_score(y_true,linkphinder_pred)
    print(f'LinkPhinder:: ROC-AUC: {roc_score} | PR-AUC: {pr_score}')

    roc_curve.savefig(constants.KSF2_LINKPHINDER_ROC_CURVES)
    pr_curve.savefig(constants.KSF2_LINKPHINDER_PR_CURVES)