import pdb
import config
import pandas as pd

class featuredictionary(object):
    def __init__(self, traindata, testdata):
        self.numeric_cols = config.numeric_cols
        self.ignore_cols = config.ignore_cols

        df = pd.concat([traindata, testdata], sort=False)
        self.feat_dict = {}; tc = 0

        for col in df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                self.feat_dict[col] = tc
                tc += 1
            else:
                us = df[col].unique()
                self.feat_dict[col] = dict(zip(us,range(tc,len(us)+tc)))
                tc += len(us)

        self.feat_dim = tc


class dataparser(object):
    def __init__(self,feat_dict):
        self.feat_dict = feat_dict

    def parse(self, df, has_label):
        dfi = df.copy()
        if has_label:
            y = dfi['target'].values.tolist()
            dfi.drop(['id','target'],axis=1,inplace=True)
        else:
            ids = dfi['id'].values.tolist()
            dfi.drop(['id'],axis=1,inplace=True)
        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            if col in self.feat_dict.numeric_cols:
                dfi[col] = self.feat_dict.feat_dict[col]
            else:
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                dfv[col] = 1.

        xi = dfi.values.tolist()
        xv = dfv.values.tolist()
        if has_label:
            return xi, xv, y
        else:
            return xi, xv, ids