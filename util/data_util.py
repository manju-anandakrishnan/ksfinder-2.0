import random

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