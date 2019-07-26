from utils import nearest_smaller
import numpy as np


def lagged_shop_item_sales(train_test_set, lag):    
    feature_name = "shop_item_sales_lag" + str(lag)
    train_test_set_c = train_test_set[['shop_id', 'item_id', 'date_block_num', 'item_cnt_month']].copy()
    train_test_set_c.date_block_num = train_test_set_c.date_block_num + lag
    train_test_set_c.rename(columns={'item_cnt_month': feature_name}, inplace=True)
    train_test_set = train_test_set.merge(train_test_set_c, on=['item_id', 'shop_id', 'date_block_num'], how='left').fillna(0)
    return train_test_set, feature_name


def lagged_item_sales(train_test_set, lag):    
    feature_name = "item_sales_lag" + str(lag)
    train_test_set_c = train_test_set.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month':'sum'}).reset_index()
    train_test_set_c.rename(columns={'item_cnt_month': feature_name}, inplace=True)
    train_test_set_c.date_block_num = train_test_set_c.date_block_num + lag
    train_test_set = train_test_set.merge(train_test_set_c, on=['item_id', 'date_block_num'], how='left').fillna(0)
    return train_test_set, feature_name


def lagged_shop_item_price(train_test_set, lag):
    feature_name = "shop_item_price_lag" + str(lag)
    train_test_set_c = train_test_set.groupby(['date_block_num', 'item_id', 'shop_id']).agg({'avg_item_price':'mean'}).reset_index()
    train_test_set_c.rename(columns={'avg_item_price': feature_name}, inplace=True)
    train_test_set_c.date_block_num = train_test_set_c.date_block_num + lag
    train_test_set = train_test_set.merge(train_test_set_c, on=['shop_id', 'item_id', 'date_block_num'], how='left').fillna(0)
    return train_test_set, feature_name


def lagged_item_price(train_test_set, lag):
    feature_name = "item_price_lag" + str(lag)
    train_test_set_c = train_test_set.groupby(['date_block_num', 'item_id']).agg({'avg_item_price':'mean'}).reset_index()
    train_test_set_c.rename(columns={'avg_item_price': feature_name}, inplace=True)
    train_test_set_c.date_block_num = train_test_set_c.date_block_num + lag
    train_test_set = train_test_set.merge(train_test_set_c, on=['item_id', 'date_block_num'], how='left').fillna(0)
    return train_test_set, feature_name
    

# def months_from_lastsale_item(data):
#     tmp = data[data.item_cnt_month>0].groupby(['item_id']).agg(list_sale_months = ('date_block_num', set)).reset_index()
#     tmp['list_sale_months'] = tmp['list_sale_months'].apply(list).apply(np.sort)
#     tmp = data[['date_block_num', 'item_id']].merge(tmp, on=['item_id'], how='left')
#     tmp.drop_duplicates(['date_block_num', 'item_id'], keep='first', inplace=True)
#     tmp['lastsale_month'] = \
#     tmp.apply(lambda row: nearest_smaller(row['date_block_num'], row['list_sale_months']) if row['list_sale_months'] is not np.NaN  else -1, axis=1 )
#     tmp['months_from_lastsale_item'] = tmp.apply(lambda r: r['date_block_num']-r['lastsale_month'] if r['lastsale_month'] > -1 else -1, axis=1 )
#     tmp.drop(columns=['lastsale_month', 'list_sale_months'], inplace = True)
#     data = data.merge(tmp, on=['date_block_num','item_id'], how='left')
#     return data


# def months_from_lastsale_shop_item(data):    
#     tmp = data[data.item_cnt_month>0].groupby(['shop_id', 'item_id']).agg(list_sale_months = ('date_block_num', list)).reset_index()
#     data = data.merge(tmp, on=['shop_id', 'item_id'], how='left')
#     data['lastsale_month_shop_item'] = \
#     data.apply(lambda row: nearest_smaller(row['date_block_num'], row['list_sale_months']) if row['list_sale_months'] is not np.NaN  else -1, axis=1 )

#     data['months_from_lastsale_shop_item'] = \
#     data.apply(lambda r: r['date_block_num']-r['lastsale_month_shop_item'] if r['lastsale_month_shop_item'] > -1 else -1, axis=1 )

#     data.drop(columns=['list_sale_months', 'lastsale_month_shop_item'], inplace= True)
#     return data



def months_from_last_shopitem_sale(data, lags):
    data_c = data[data.item_cnt_month>0].groupby(['date_block_num','shop_id', 'item_id']).agg(count = ('item_id', 'count')).reset_index()
    feat_names = []
    for lag in lags:
        feature_name = "pair_sale_" + str(lag) + "_months_ago"
        feat_names.append(feature_name)
        data_c_lagged = data_c.copy()
        data_c_lagged.date_block_num = data_c_lagged.date_block_num + lag
        data_c_lagged.rename(columns={'count': feature_name}, inplace=True)
        data_c_lagged[feature_name] = 1
        data = data.merge(data_c_lagged, on=['item_id', 'date_block_num', 'shop_id'], how='left').fillna(0)
    return data, feat_names



def months_from_last_item_sale(data, lags):
    data_c = data[data.item_cnt_month>0].groupby(['date_block_num', 'item_id']).agg(count = ('item_id', 'count')).reset_index()
    feat_names = []
    for lag in lags:
        feature_name = "sold_" + str(lag) + "_months_ago"
        feat_names.append(feature_name)
        data_c_lagged = data_c.copy()
        data_c_lagged.date_block_num = data_c_lagged.date_block_num + lag
        data_c_lagged.rename(columns={'count': feature_name}, inplace=True)
        data_c_lagged[feature_name] = 1
        data = data.merge(data_c_lagged, on=['item_id', 'date_block_num'], how='left').fillna(0)
    return data, feat_names
