import os
import scipy.linalg
import pandas as pd
import numpy as np
from os.path import join
from argparse import ArgumentParser
from sklearn.metrics.pairwise import cosine_similarity

REG_PARAM = 1e-2

columns1 = ["Plate", "Well", "Treatment", "Replicate", "broad_sample"]
columns2 = [str(i) for i in range(672)]

class WhiteningNormalizer(object):
    def __init__(self, controls, reg_param=1e-6):
        # Whitening transform on population level data
        self.mu = controls.mean()
        self.whitening_transform(controls - self.mu, reg_param, rotate=True)
        print(self.mu.shape, self.W.shape)
        
    def whitening_transform(self, X, lambda_, rotate=True):
        C = (1/X.shape[0]) * np.dot(X.T, X)
        s, V = scipy.linalg.eigh(C)
        D = np.diag( 1. / np.sqrt(s + lambda_) )
        W = np.dot(V, D)
        if rotate:
            W = np.dot(W, V.T)
        self.W = W

    def normalize(self, X):
        return np.dot(X - self.mu, self.W)

def load_similarity_matrix(filename):
    # Load matrix in triplet format and reshape
    cr_mat = pd.read_csv(filename)
    X = cr_mat.pivot(index="Var1", columns="Var2", values="value").reset_index()
    
    # Identify annotations
    Y = cr_mat.groupby("Var1").max().reset_index()
    Y = Y[~Y["Metadata_moa.x"].isna()].sort_values(by="Var1")
    
    # Make sure the matrix is sorted by treatment
    X = X.loc[X.Var1.isin(Y.Var1), ["Var1"] + list(Y.Var1)].sort_values("Var1")
    
    return X, Y

def single_cell(config, meta):
    features = []
    for i in meta.index:
        filename = config["paths"]["features"] + "{}/{}/{}.npz"
        filename = filename.format(
            meta.loc[i, "Metadata_Plate"], 
            meta.loc[i, "Metadata_Well"], 
            meta.loc[i, "Metadata_Site"]
        )
        if os.path.isfile(filename):
            with open(filename, "rb") as data:
                info = np.load(data)
                cells = np.array(np.copy(info["features"]))
                cells_f = cells[~np.isnan(cells).any(axis=1)]
                features.append(cells_f)
        else:
            features.append([])

    total_single_cells = 0
    for i in range(len(features)):
        if len(features[i]) > 0:
            total_single_cells += features[i].shape[0]

    total_images = len(features)
    print("Processed single-cell data.")
    print(f"Total images: {total_images}")
    print(f"Total single cells: {total_single_cells}")
    
    return features

def site_level(meta, features, dataset):
    site_level_data = []
    site_level_features = []

    if dataset == 'BBBC037':
        pert_col_name = 'pert_name'
        replicate_col_name = 'pert_name_replicate'
    elif dataset == 'BBBC022':
        pert_col_name = 'Treatment'
        replicate_col_name = 'broad_sample_Replicate'

    for plate in meta["Metadata_Plate"].unique():
        m1 = meta["Metadata_Plate"] == plate
        wells = meta[m1]["Metadata_Well"].unique()
        for well in wells:
            result = meta.query("Metadata_Plate == {} and Metadata_Well == '{}'".format(plate, well))
            for i in result.index:
                if len(features[i]) == 0:
                    continue
                mean_profile = np.median(features[i], axis=0)
                pert_name = result[pert_col_name].unique()
                replicate = result[replicate_col_name].unique()
                if len(pert_name) > 1:
                    print(pert_name)
                site_level_data.append(
                    {
                        "Plate": plate,
                        "Well": well,
                        "Treatment": pert_name[0],
                        "Replicate": replicate[0],
                        "broad_sample": result["broad_sample"].unique()[0] if dataset == 'BBBC037' else pert_name[0].split("@")[0]
                    }
                )
                site_level_features.append(mean_profile)

    sites1 = pd.DataFrame(columns=columns1, data=site_level_data)
    sites2 = pd.DataFrame(columns=columns2, data=site_level_features)
    sites = pd.concat([sites1, sites2], axis=1)

    print("Processed site-level data.")

    return sites

def well_level(meta, config, sites):
    wells = sites.groupby(["Plate", "Well", "Treatment"]).mean(numeric_only=True).reset_index()
    tmp = meta.groupby(["Metadata_Plate", "Metadata_Well", config['dataset']['metadata']['label_field'], "broad_sample"])["DNA"].count().reset_index()
    wells = pd.merge(wells, tmp, how="left", left_on=["Plate", "Well", "Treatment"], right_on=["Metadata_Plate", "Metadata_Well", config['dataset']['metadata']['label_field']])
    wells = wells[columns1 + columns2]
    print("Processed well-level data.")

    return wells

def sphering(config, wells):
    whN = WhiteningNormalizer(wells.loc[wells["Treatment"] == config['dataset']['metadata']['control_value'], columns2], reg_param=REG_PARAM)
    whD = whN.normalize(wells[columns2])
    wells[columns2] = whD
    wells.to_csv(join(config["paths"]["results"], f"{config['experiment_name']}_well.csv"), index=False)
    print(f"Saved well-level features.")

    return wells

def treatment_level(config, wells):
    profiles = wells.groupby("Treatment").mean(numeric_only=True).reset_index()
    tmp = wells.groupby(["Treatment", "broad_sample"])["Replicate"].count().reset_index()
    profiles = pd.merge(profiles.reset_index(), tmp, on="Treatment", how="left")
    profiles = profiles[["Treatment", "broad_sample"] + columns2]
    Y = pd.read_csv(join(config['paths']['metadata'], config['dataset']['metadata']['moa_metadata']))
    profiles = pd.merge(profiles, Y, left_on="broad_sample", right_on="Var1")
    profiles = profiles[["Treatment", "broad_sample", "Metadata_moa.x"] + columns2].sort_values(by="broad_sample")

    return profiles

def similarity_matrix(config, profiles):
    COS = cosine_similarity(profiles[columns2], profiles[columns2])
    df = pd.DataFrame(data=COS, index=list(profiles.broad_sample), columns=list(profiles.broad_sample))
    df = df.reset_index().melt(id_vars=["index"])

    df2 = pd.merge(
        df, 
        profiles[["broad_sample", "Metadata_moa.x"]], 
        how="left", 
        left_on="index",
        right_on="broad_sample"
    ).drop("broad_sample",axis=1)

    df2 = pd.merge(
        df2, profiles[["broad_sample", "Metadata_moa.x"]],
        how="left", 
        left_on="variable",
        right_on="broad_sample"
    ).drop("broad_sample",axis=1)

    df2.columns = ["Var1", "Var2", "value", "Metadata_moa.x", "Metadata_moa.y"]
    df2.to_csv(join(config["paths"]["results"], f"{config['experiment_name']}_matrix.csv"), index=False)
    print("Saved COS matrix.")