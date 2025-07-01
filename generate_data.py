import pandas as pd
import numpy as np
import polars as pl
from random import seed,sample
from datetime import timedelta
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name' ,type=str ,default='swdpf' ,help='dataset name')
parser.add_argument('--data' ,type=str ,default="data/raw/wtbdata_245days.csv" ,help='data path')
parser.add_argument('--locations' ,type=str ,default="data/raw/turb_location.csv" ,help='locations file path')
parser.add_argument('--n_targets' ,type=int ,default=1 ,help='number of target time-steps')
parser.add_argument('--n_hist_steps' ,type=int ,default=12 ,help='number of historical features time-steps')
parser.add_argument('--size_train', type=int, default=30, help='size train in terms of days')
parser.add_argument('--size_test', type=int, default=7, help='size test in terms of days')
parser.add_argument('--targetcol', type=str, default="patv_target", help='name column target')
parser.add_argument('--id_key', type=str, default="turbid", help='name id key')
args = parser.parse_args()


def add_features(df, key, window_size, features, type='lag'):
    if type == 'lag':
        for i in np.arange(1, window_size + 1):
            for feature in features:
                df = df.with_columns(
                    pl.col(feature).shift(i).over(key).alias(feature.replace("_target", "") + '_' + str(i)))
    if type == 'lead':
        for i in np.arange(1, window_size + 1):
            for feature in features:
                df = df.with_columns(pl.col(feature).shift(-(i - 1)).over(key).alias(feature + '_' + str(i)))
    return df


def prep_data_sdwpf(args):
    print("preprocessing")
    data = pd.read_csv(args.data)
    data.columns = [col.lower() for col in data.columns]
    data['start_time'] = data['day'].astype(str) + ' ' + data['tmstamp'].astype(str)
    data[['patv']] = data.groupby(['turbid'])[['patv']].apply(lambda x: x.ffill().bfill()).reset_index(drop=True)
    locations = pl.read_csv(args.locations, new_columns=[args.id_key, "lat", "long"])
    data = data[[args.id_key, "day", "tmstamp", "patv"]]
    data['date'] = pd.to_datetime('2022-01-01') + pd.to_timedelta(data['day'] - 1, unit='D')
    data['date'] = pd.to_datetime(data['date'].astype(str) + ' ' + data['tmstamp'])
    data = pl.from_pandas(data)
    data = data.rename({"patv": "patv_target"})
    data = data.join(locations, on="turbid").drop("tmstamp")
    seed(1)
    sequence = [i for i in np.arange(30, 245)]
    subset = sample(sequence, 8)

    return subset,data



subset, data = prep_data_sdwpf(args)
print("Subset created")

for i,elem in enumerate(subset):
    print(f"Create fold: {i}")
    dataset = add_features(df=data, key=[args.id_key], window_size=args.n_hist_steps, features=[args.targetcol], type='lag')
    dataset = add_features(df=dataset, key=[args.id_key], window_size=args.n_targets, features=[args.targetcol], type='lead').drop_nulls()
    dataset = dataset.drop([args.targetcol])

    train = dataset.filter((pl.col("day") <= elem) & (pl.col("day") >= elem-args.size_train)).drop(["day"])
    test = dataset.filter((pl.col("day") > elem) & (pl.col("day") <= elem + args.size_test)).drop(["day"])

    outputpath = f"./data/{args.dataset_name}/T{args.n_targets}H{args.n_hist_steps}/fold{i}/"

    isExist = os.path.exists(outputpath)
    if not isExist:
        os.makedirs(outputpath)

    train.write_csv(outputpath + "train.csv")
    test.write_csv(outputpath + "test.csv")