from utils import nearest_smaller
import numpy as np


def lagged_shop_item_sales(train_test_set, lag):    
    feature_name = "shop_item_sales_lag" + str(lag)
    train_test_set_c = train_test_set[['shop_id', 'item_id', 'date_block_num', 'item_cnt_month']]
    train_test_set_c.date_block_num = train_test_set_c.date_block_num + lag
    train_test_set_c.rename(columns={'item_cnt_month': feature_name}, inplace=True)
    train_test_set = train_test_set.merge(train_test_set_c, on=['item_id', 'shop_id', 'date_block_num'], how='left').fillna(0)
    return train_test_set



def lagged_item_sales(train_test_set, lag):    
    feature_name = "item_sales_lag" + str(lag)
    train_test_set_c = train_test_set.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month':'sum'}).reset_index()
    train_test_set_c.rename(columns={'item_cnt_month': feature_name}, inplace=True)
    train_test_set_c.date_block_num = train_test_set_c.date_block_num + lag
    train_test_set = train_test_set.merge(train_test_set_c, on=['item_id', 'date_block_num'], how='left').fillna(0)
    return train_test_set



def lagged_shop_item_price(train_test_set, lag):
    feature_name = "shop_item_price_lag" + str(lag)
    train_test_set_c = train_test_set.groupby(['date_block_num', 'item_id', 'shop_id']).agg({'avg_item_price':'mean'}).reset_index()
    train_test_set_c.rename(columns={'avg_item_price': feature_name}, inplace=True)
    train_test_set_c.date_block_num = train_test_set_c.date_block_num + lag
    train_test_set = train_test_set.merge(train_test_set_c, on=['shop_id', 'item_id', 'date_block_num'], how='left').fillna(0)
    return train_test_set



def lagged_item_price(train_test_set, lag):
    feature_name = "item_price_lag" + str(lag)
    train_test_set_c = train_test_set.groupby(['date_block_num', 'item_id']).agg({'avg_item_price':'mean'}).reset_index()
    train_test_set_c.rename(columns={'avg_item_price': feature_name}, inplace=True)
    train_test_set_c.date_block_num = train_test_set_c.date_block_num + lag
    train_test_set = train_test_set.merge(train_test_set_c, on=['item_id', 'date_block_num'], how='left').fillna(0)
    return train_test_set


def months_from_last_shopitem_sale(data, lags):
    data_c = data[data.item_cnt_month>0].groupby(['date_block_num','shop_id', 'item_id']).agg(count = ('item_id', 'count')).reset_index()
    feat_names = []
    for lag in lags:
        feature_name = "pair_sale_" + str(lag) + "_months_ago"
        feat_names.append(feature_name)
        data_c.date_block_num = data_c.date_block_num + lag
        data_c.rename(columns={'count': feature_name}, inplace=True)
        data_c[feature_name] = 1
        data = data.merge(data_c, on=['item_id', 'date_block_num', 'shop_id'], how='left').fillna(0)
        data_c.drop(columns = feature_name, inplace=True)
    return data



def months_from_last_item_sale(data, lags):
    data_c = data[data.item_cnt_month>0].groupby(['date_block_num', 'item_id']).agg(count = ('item_id', 'count')).reset_index()
    feat_names = []
    for lag in lags:
        feature_name = "sold_" + str(lag) + "_months_ago"
        feat_names.append(feature_name)
        data_c.date_block_num = data_c.date_block_num + lag
        data_c.rename(columns={'count': feature_name}, inplace=True)
        data_c[feature_name] = 1
        data = data.merge(data_c, on=['item_id', 'date_block_num'], how='left').fillna(0)
        data_c.drop(columns = feature_name, inplace=True)
    return data