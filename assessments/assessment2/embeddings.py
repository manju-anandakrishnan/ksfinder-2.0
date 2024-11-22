import pandas as pd
import numpy as np

import traceback
from util import constants

np.random.seed(13)

class DataLoader:

    def __init__(self):
        pass

    def preprocess(self,df):
        df['kinase1'] = df['head'].copy()
        df['substrate1'] = df['tail'].apply(lambda x:x[:x.find('_')])
        df['motif1'] = df['tail'].apply(lambda x:x[x.find('_')+1:])
        df['kinase_substrate'] = df.apply(lambda x:(x['kinase1'],x['substrate1']),axis=1)
        df['kinase_motif'] = df.apply(lambda x:(x['kinase1'],x['motif1']),axis=1)
        return df

    def get_training_data(self,train_data):
        df = pd.read_csv(train_data,sep='|')
        df = self.preprocess(df)
        df = df.sample(random_state=13,frac=1).reset_index(drop=True)    
        return df
    
    def get_testing_data(self,test_data):
        df = pd.read_csv(test_data,sep='|')
        df = self.preprocess(df)
        df = df.sample(random_state=13,frac=1).reset_index(drop=True)    
        return df

class ESM2DataLoader(DataLoader):

    def __init__(self):
        super().__init__()    
    

class ESM3DataLoader(DataLoader):
    
    def __init__(self):
        super().__init__()

class ProtT5DataLoader(DataLoader):
    def __init__(self):
        super().__init__()

    def preprocess(self,df):
        df['kinase1'] = df['head'].copy()
        df['substrate1'] = df['tail'].apply(lambda x:x[:x.find('_')])
        df['motif1'] = df['tail'].apply(lambda x:x[x.find('_')+1:])
        df['substrate_motif'] = df['tail'].copy()
        df['kinase_substrate'] = df.apply(lambda x:(x['kinase1'],x['substrate1']),axis=1)
        df['kinase_motif'] = df.apply(lambda x:(x['kinase1'],x['motif1']),axis=1)
        return df

class RandomDataLoader(DataLoader):
    def __init__(self):
        super().__init__()

class KSF2DataLoader(DataLoader):
    def __init__(self):
        super().__init__()

class KSMEmbeddings:

    def __init__(self,emb_path):
        self.entity_embedding_dict = self._load_embeddings(emb_path)

    def _load_embeddings(self,emb_path):
        emb_df = pd.read_csv(emb_path,sep='|')
        emb_df.set_index('entity', inplace=True)
        entity_embedding_dict = emb_df.apply(lambda row: row.values, axis=1).to_dict()
        return entity_embedding_dict
    
    def _map_embeddings(self,df):
        for df_col in df.columns:
            if df_col == 'label': continue
            df[df_col] = df[df_col].apply(lambda x:self.entity_embedding_dict.get(x))
        df.dropna(axis=0,how='any',inplace=True)
        df_kinase = pd.DataFrame(df['kinase1'].to_list())
        df_substrate = pd.DataFrame(df['substrate1'].to_list())
        df_motif = pd.DataFrame(df['motif1'].to_list())
        df_label = pd.DataFrame(df['label'].to_list())            
        return df_kinase.to_numpy(), df_substrate.to_numpy(), df_motif.to_numpy(), df_label.to_numpy()

    def get_training_data(self,raw_training_data):        
        self._training_data = raw_training_data
        training_data = self._training_data[['kinase1','substrate1','motif1','label']].copy()
        training_data_k, training_data_s, training_data_m, training_data_l = self._map_embeddings(training_data)
        print('Training data count::',training_data_l.shape[0],
              '| Positive data count::',training_data_l[training_data_l[:,-1]==1].shape[0],
              '| Negative data count::',training_data_l[training_data_l[:,-1]==0].shape[0])
        return (training_data_k, training_data_s, training_data_m, training_data_l)
    
    def get_testing_data(self,raw_testing_data):        
        self._testing_data = raw_testing_data
        testing_data = self._testing_data[['kinase1','substrate1','motif1','label']].copy()
        testing_data_k, testing_data_s, testing_data_m, testing_data_l = self._map_embeddings(testing_data)
        print('Testing data count::',testing_data_l.shape[0],
            '| Positive data count::',testing_data_l[testing_data_l[:,-1]==1].shape[0],
            '| Negative data count::',testing_data_l[testing_data_l[:,-1]==0].shape[0])
        return (testing_data_k, testing_data_s, testing_data_m, testing_data_l)
    
    def get_data_embedding(self,kinase,substrate, motif):
        try:
            k_emb = self.entity_embedding_dict.get(kinase).reshape(1,-1)
            s_emb = self.entity_embedding_dict.get(substrate).reshape(1,-1)
            m_emb = self.entity_embedding_dict.get(motif).reshape(1,-1)
            return (k_emb, s_emb, m_emb)
        except:
            print(traceback.print_exc())
            return None

class ESM2Embeddings(KSMEmbeddings):

    def __init__(self):
        super().__init__(constants.CSV_ESM2_EMB)

class ESM3Embeddings(KSMEmbeddings):

    def __init__(self):
        super().__init__(constants.CSV_ESM3_EMB)

class RandomEmbeddings(KSMEmbeddings):

    def __init__(self):
        super().__init__(constants.CSV_RANDOM_EMB)

class KSF2Embeddings(KSMEmbeddings):

    def __init__(self):
        super().__init__(constants.CSV_TRANSE_EMB)

class ProtT5Embeddings(KSMEmbeddings):

    def __init__(self):
        super().__init__(constants.CSV_PROTT5_EMB)

    def _map_embeddings(self,df):
        for df_col in df.columns:
            if df_col == 'label': continue
            df[df_col] = df[df_col].apply(lambda x:self.entity_embedding_dict.get(x))
        df.dropna(axis=0,how='any',inplace=True)
        df_kinase = pd.DataFrame(df['kinase1'].to_list())
        df_substrate = pd.DataFrame(df['substrate1'].to_list())
        df_motif = pd.DataFrame(df['substrate_motif'].to_list())
        df_label = pd.DataFrame(df['label'].to_list())            
        return df_kinase.to_numpy(), df_substrate.to_numpy(), df_motif.to_numpy(), df_label.to_numpy()

    def get_training_data(self,raw_training_data):
        self._training_data = raw_training_data
        training_data = self._training_data[['kinase1','substrate1','substrate_motif','label']].copy()
        training_data_k, training_data_s, training_data_m, training_data_l = self._map_embeddings(training_data)
        print('Training data count::',training_data_l.shape[0],
              '| Positive data count::',training_data_l[training_data_l[:,-1]==1].shape[0],
              '| Negative data count::',training_data_l[training_data_l[:,-1]==0].shape[0])
        return (training_data_k, training_data_s, training_data_m, training_data_l)
    
    def get_testing_data(self,raw_testing_data):
        self._testing_data = raw_testing_data
        testing_data = self._testing_data[['kinase1','substrate1','substrate_motif','label']].copy()
        testing_data_k, testing_data_s, testing_data_m, testing_data_l = self._map_embeddings(testing_data)
        print('Testing data count::',testing_data_l.shape[0],
            '| Positive data count::',testing_data_l[testing_data_l[:,-1]==1].shape[0],
            '| Negative data count::',testing_data_l[testing_data_l[:,-1]==0].shape[0])
        return (testing_data_k, testing_data_s, testing_data_m, testing_data_l)