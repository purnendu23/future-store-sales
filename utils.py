import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, FunctionTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator


replacements = {'Accessories - PS2':'Accessories',
       'Accessories - PS3':'Accessories', 'Accessories - PS4':'Accessories',
       'Accessories - PSP':'Accessories','Accessories - PSVita':'Accessories',
       'Accessories - XBOX 360':'Accessories', 'Accessories - XBOX ONE':'Accessories',
       
       'Game Consoles - PS2':'Game Consoles', 'Game Consoles - PS3':'Game Consoles',
       'Game Consoles - PS4':'Game Consoles', 'Gaming Consoles - PSP':'Game Consoles',
       'Game Consoles - PSVita':'Game Consoles', 'Gaming Consoles - XBOX 360':'Game Consoles',
       'Gaming Consoles - XBOX ONE':'Game Consoles', 'Game Consoles - Other':'Game Consoles',
                
       'Games - PS2':'Games-digital', 'Games - PS3':'Games-digital', 'Games - PS4':'Games-digital',
       'Games - PSP':'Games-digital', 'Games - PSVita':'Games-digital', 'Games - XBOX 360':'Games-digital',
       'Games - XBOX ONE':'Games-digital',

       'Games - Accessories for games': 'Game-Accessories',
       'Android Games - Digital':'Android Games', 'MAC Games - Digit':'MAC Games', 
       
       'PC Games - Additional Editions':'PC Games',
       'PC Games - Collectible Editions':'PC Games', 'PC Games - Standard Editions':'PC Games',
       'PC Games - Digital':'PC Games',
       'Payment cards (Cinema, Music, Games)':'Payment cards',
       'Payment Cards - Live!':'Payment cards', 'Payment Cards - Live! (Numeral)':'Payment cards',
       'Payment Cards - PSN':'Payment cards', 'Payment Cards - Windows (Digital)':'Payment cards',
       
       'Cinema - Blu-Ray':'Movies', 'The Movie - Blu-Ray 3D':'Movies',
       'Cinema - Blu-Ray 4K':'Movies', 'Cinema - DVD':'Movies', 'Cinema - Collection':'Movies',
       
       'Books - Artbooks, encyclopedias':'Books', 'Books - Audiobooks':'Books',
       'Books - Audiobooks (Figure)':'Books', 'Books - Audiobooks 1C':'Books',
       'Books - Business Literature':'Books', 'Books - Comics, Manga':'Books',
       'Books - Computer Literature':'Books', 'Books - Methodical materials 1C':'Books',
       'Books - Postcards':'Books', 'Books - Cognitive Literature':'Books',         
       'Книги - Путеводители':'Books', 'Books - Fiction':'Books', 'Books - The Figure':'Books',
                
       'Music - Local Production CD':'Music', 'Music - CD branded production':'Music',
       'Music - MP3':'Music', 'Music - Vinyl':'Music', 'Music - Music Video':'Music',
       'Music - Gift Edition':'Music',
       
       'Gifts - Attributes':'Gifts',
       'Gifts - Gadgets, Robots, Sports':'Gifts', 'Gifts - Soft Toys':'Gifts',
       'Gifts - Board Games':'Gifts', 'Gifts - Board Games (Compact)':'Gifts',
       'Gifts - Cards, stickers':'Gifts', 'Gifts - Development':'Gifts',
       'Gifts - Certificates, Services':'Gifts', 'Gifts - Souvenirs':'Gifts',
       'Gifts - Souvenirs (in a hitch)':'Gifts',
       'Gifts - Bags, Albums, Mouse pads':'Gifts', 'Gifts - Figures':'Gifts',
                
       'Programs - 1C: Enterprise 8':'Programs', 'Programs - MAC (Digit)':'Programs',
       'Programs - Home and Office':'Programs',
       'Programs - Home and Office (Digital)':'Programs', 'Programs - Educational':'Programs',
       'Programs - Educational (Figure)':'Programs',
       'Service Tickets':'Service',
                
       'Clean media (spire)':'Clean', 'Clean Media (Piece)':'Clean'}


def get_dates_in_month(year, month, time_zone):
    num_days = monthrange(year, month)[1]
    first_date_of_month = datetime.datetime(year,month,1, tzinfo=time_zone)
    last_date_of_month =  datetime.datetime(year,month,num_days, tzinfo=time_zone)
    return get_dates_inrange(first_date_of_month, last_date_of_month)

def get_dates_inrange(date1, date2):
    if (not isinstance(date1, datetime.datetime)) | (not isinstance(date2, datetime.datetime)):
        return "date1 and date2 should be of type datetime.date"
    num_days = (date2 - date1).days + 1
    date_list = [date1 + datetime.timedelta(days=x) for x in range(0, num_days)]
    return date_list

def remove_nonAlphaNumeric(s: str):
    return re.sub(r'\W+', '', s)

def tokenize(raw_text: str):
    lower_tokens = [w.lower() for w in word_tokenize(raw_text)]
    refined_tokens = [remove_nonAlphaNumeric(w) for w in lower_tokens]
    return list(filter(None, refined_tokens))

def w2v_vectorize(raw_text: str, model):
    words = tokenize(raw_text)
    word_vectors = model.wv
    vector = [0] * model.wv.vector_size
    for word in words:
        if word in word_vectors.vocab:
            vector += word_vectors.get_vector(word)
    vector = [np.round(d, 4) for d in vector]
    return vector

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)
    
class LogFeaturizer(BaseEstimator,TransformerMixin):
    '''
    Log1p transforms inputs, filling NAs with zeroes
    '''
    def fit(self, X,y=None):
        return self
    
    def transform(self,X):
        res= np.log1p(X.fillna(0)).values
        return pd.DataFrame(res, columns= [i+'_log' for i in X.columns])

class ClipFeaturizer(BaseEstimator,TransformerMixin):
    '''
    Clips input values below min_value to min_value, and/or
    max_value to max value.
    '''
    def __init__(self, min_value=None, max_value=None):
        self.min_value=min_value
        self.max_value=max_value
        
    def fit(self, X,y=None):
        return self
    
    def transform(self,X):
        X= X.copy()
        if self.min_value is not None:
            X[X< self.min_value]= self.min_value
        if self.max_value is not None:
            X[X>self.max_value]= self.max_value   
        return X
    
class LocationExtractor(BaseEstimator,TransformerMixin):
    '''
    Extracts location from the Reserve and Sales_Floor_Location column
    Currently does not enforce the columns to be strings
    Also untested when providing just one column (might break if DataFrameSelector 
    on a single column returns a Series instead of a DataFrame)
    '''
    def fit(self, X,y=None):
        return self
    
    def transform(self,X):
        newdf= {}
        for col in X.columns:
            extract= X.loc[:,col].str.extract('(\w+)-.*').iloc[:,0]
            extract= extract.fillna(value=f'na_{col}')
            newdf[col+'_proc']= extract
        return pd.DataFrame(newdf)

class TimeExtractor(BaseEstimator,TransformerMixin):
    '''
    Extracts location from the Reserve and Sales_Floor_Location column
    Currently does not enforce the columns to be strings
    Also untested when providing just one column (might break if DataFrameSelector 
    on a single column returns a Series instead of a DataFrame)
    '''
    def fit(self, X,y=None):
        return self
    
    def transform(self,X):
        newdf= {}
        for col in X.columns:
            extract= X.loc[:,col].apply(lambda x: x.weekday())
            newdf[col+'_proc']= extract
        return pd.DataFrame(newdf)
        
class CategoryFeaturizer(BaseEstimator,TransformerMixin):
    '''
    Returns Dummy variables of categorical inputs (assumes that they are categorical for now)
    Accepts strings and integers
    Important: Will work even if the testing dataset that the object is transforming has fewer 
    categories than the fitted dataset, and so will have the same number of columns as the latter
    '''
    def __init__(self):
        self.onehot_enc= OneHotEncoder(sparse=False,dtype='int', handle_unknown='ignore') 

    def fit(self, X,y=None):
        self.onehot_enc.fit(X)
        self.colnames=[]
        for i,col in enumerate(X.columns):
            for level in self.onehot_enc.categories_[i]:
                self.colnames.append(col+'_'+str(level))
        return self
    
    def transform(self,X):
        res= self.onehot_enc.transform(X)
        return pd.DataFrame(res, columns= self.colnames)
        

class ColumnMerge(BaseEstimator, TransformerMixin):
    '''
    Like scikit-learn's FeatureUnion but dataframe aware
    '''
    def __init__(self,transformer_list, n_jobs=None, transformer_weights=None):
        self.tf_list= transformer_list
        
    def fit(self, X, y=None):
        for tf_name,tf in self.tf_list:
            tf.fit(X)
        return self
    
    def transform(self, X):
        res=[]
        for tf_name,tf in self.tf_list:
            res.append(tf.transform(X).reset_index(drop=True))
        res= pd.concat(res, axis=1)
        return res
    
class ModelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        self.model.fit(X)
        return self

    def transform(self, X, **transform_params):
        df =  pd.DataFrame(self.model.predict(X), columns=['result']).reset_index(drop=True)
        df.index = list(df.index)
        return df
    
class MinMaxScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.min_max_scalar= MinMaxScaler()
    
    def fit(self, X, y=None):
        self.min_max_scalar.fit(X)
        return self
    
    def transform(self, X):
        arr = self.min_max_scalar.transform(X)
        return pd.DataFrame(arr, columns=list(X.columns))

    
    
def nearest_smaller(x, arr):
    if len(arr)>1:
        mid = int(len(arr)/2)
        if arr[mid] == x :
            if mid-1>=0:
                return arr[mid-1] 
        elif (arr[mid]>x) :
            return nearest_smaller(x, arr[:mid])
        elif (arr[mid]<x):
            return nearest_smaller(x, arr[mid:])
        else:
            return arr[mid] 
    elif arr[0] < x:
        return arr[0]
    else:
        return -1