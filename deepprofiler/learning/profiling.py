import importlib
import os

import numpy as np
import tensorflow as tf
import pandas as pd
import scipy.linalg

from deepprofiler.dataset.utils import tic, toc

tf.compat.v1.disable_v2_behavior()
tf.config.run_functions_eagerly(False)

class Profile(object):
    
    def __init__(self, config, dset):
        self.config = config
        self.dset = dset
        self.num_channels = len(self.config["dataset"]["images"]["channels"])
        self.crop_generator = importlib.import_module(
            "plugins.crop_generators.{}".format(config["train"]["model"]["crop_generator"])
        ).GeneratorClass

        self.profile_crop_generator = importlib.import_module(
            "plugins.crop_generators.{}".format(config["train"]["model"]["crop_generator"])
        ).SingleImageGeneratorClass

        self.config["num_classes"] = self.dset.targets[0].shape[1]

        self.dpmodel = importlib.import_module(
            "plugins.models.{}".format(config["train"]["model"]["name"])
        ).model_factory(self.config, dset, self.crop_generator, self.profile_crop_generator, is_training=False)

        self.profile_crop_generator = self.profile_crop_generator(config, dset)

    def configure(self):        
        # Main session configuration
        self.profile_crop_generator.start(tf.compat.v1.keras.backend.get_session())
        
        # Create feature extractor
        if self.config["profile"]["checkpoint"] != "None":
            checkpoint = self.config["paths"]["checkpoints"]+"/"+self.config["profile"]["checkpoint"]
            try:
                self.dpmodel.feature_model.load_weights(checkpoint)
            except ValueError:
                print("Loading weights without classifier (different number of classes)")
                self.dpmodel.feature_model.layers[-1]._name = "classifier"
                self.dpmodel.feature_model.load_weights(checkpoint, by_name=True)

        self.dpmodel.feature_model.summary()
        self.feat_extractor = tf.compat.v1.keras.Model(
            self.dpmodel.feature_model.inputs, 
            self.dpmodel.feature_model.get_layer(self.config["profile"]["feature_layer"]).output
        )
        print("Extracting output from layer:", self.config["profile"]["feature_layer"])

    def check(self, meta):
        output_file = self.config["paths"]["features"] + "/{}/{}/{}.npz"
        output_file = output_file.format( meta["Metadata_Plate"], meta["Metadata_Well"], meta["Metadata_Site"])

        # Check if features were computed before
        if os.path.isfile(output_file):
            print("Already done:", output_file)
            return False
        else:
            return True
    
    # Function to process a single image
    def extract_features(self, key, image_array, meta):  # key is a placeholder
        start = tic()
        output_file = self.config["paths"]["features"] + "/{}/{}/{}.npz"
        output_file = output_file.format( meta["Metadata_Plate"], meta["Metadata_Well"], meta["Metadata_Site"])
        os.makedirs(self.config["paths"]["features"] + "/{}/{}".format(meta["Metadata_Plate"], meta["Metadata_Well"]), exist_ok=True)

        batch_size = self.config["profile"]["batch_size"]
        image_key, image_names, outlines = self.dset.get_image_paths(meta)
        crop_locations = self.profile_crop_generator.prepare_image(
                                   tf.compat.v1.keras.backend.get_session(),
                                   image_array,
                                   meta,
                                   False
                            )
        total_crops = len(crop_locations)
        if total_crops == 0:
            print("No cells to profile:", output_file)
            return
        repeats = self.config["train"]["model"]["crop_generator"] in ["repeat_channel_crop_generator", "individual_channel_cropgen"]
        
        # Extract features
        crops = next(self.profile_crop_generator.generate(tf.compat.v1.keras.backend.get_session()))[0]  # single image crop generator yields one batch
        
        # Ablation study - Replace nan values with 0 - otherwise features result all in nan values
        crops = np.nan_to_num(crops)
        
        feats = self.feat_extractor.predict(crops, batch_size=batch_size)
        
        while len(feats.shape) > 2:  # 2D mean spatial pooling
            feats = np.mean(feats, axis=1)
        
        if repeats:
            feats = np.reshape(feats, (self.num_channels, total_crops, -1))
            feats = np.concatenate(feats, axis=-1)
            
        # Save features
        key_values = {k:meta[k] for k in meta.keys()}
        key_values["Metadata_Model"] = self.config["train"]["model"]["name"]
        np.savez_compressed(output_file, features=feats, metadata=key_values, locations=crop_locations)
        toc(image_key + " (" + str(total_crops) + " cells)", start)

class WhiteningNormalizer(object):
    def __init__(self, controls, reg_param=1e-2):
        # Whitening transform on population level data
        self.mu = controls.mean()
        self.whitening_transform(controls - self.mu, reg_param, rotate=True)
        
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
        
def profile(config, dset):
    profile = Profile(config, dset)
    profile.configure()
    dset.scan(profile.extract_features, frame="all", check=profile.check)
    print("Profiling: Done")

def process_features(config, dset, mode="site_well"):

    meta = dset.meta.data
    experiment_name = dset.config["experiment_name"]
    dataset = dset.config["paths"]['root'].replace("/", "")
    columns1 = ["Plate", "Well", "Treatment", "Replicate", "broad_sample"]
    columns2 = list(map(str, range(672)))

    channel = config["experiment_name"].split("_")[-1]
        
    # Single cell
    features = []
    for i in meta.index:
        plate, well, site = meta.loc[i, "Metadata_Plate"], meta.loc[i, "Metadata_Well"], meta.loc[i, "Metadata_Site"]
        features_file = f"{dataset}/outputs/{experiment_name}/features/{plate}/{well}/{site}.npz"
        if os.path.isfile(features_file):
            try:
                f = np.load(features_file)
                features.append(f["features"])
            except:
                print(features_file)
                try:
                    f = np.load(features_file, pickle=True)
                    features.append(f["features"])
                except:
                    features.append([])    
        else:
            features.append([])
    
    # Site-level
    site_level_data = []
    site_level_features = []

    for plate in meta["Metadata_Plate"].unique():
        plate_meta = meta["Metadata_Plate"] == plate
        wells = meta[plate_meta]["Metadata_Well"].unique()
        
        for well in wells:
            result = meta.query(f"Metadata_Plate == {plate} and Metadata_Well == '{well}'")
            
            for i in result.index:
                if len(features[i]) > 0:
                    profile_mean = np.median(features[i], axis=0)
                    pert_name = result["Treatment"].unique()
                    replicate = result["Replicate"].unique()

                    site_level_features.append(profile_mean)
                    site_level_data.append(
                        {
                            "Plate": plate,
                            "Well": well,
                            "Treatment": pert_name[0],
                            "Replicate": replicate[0],
                            "broad_sample": pert_name[0].split("@")[0]
                        }
                    )

    sites1 = pd.DataFrame(columns=columns1, data=site_level_data)
    sites2 = pd.DataFrame(columns=columns2, data=site_level_features)
    sites = pd.concat([sites1, sites2], axis=1)

    # Well-level
    wells = sites.groupby(["Plate", "Well", "Treatment"]).mean().reset_index()
    tmp = meta.groupby(["Metadata_Plate", "Metadata_Well", "Treatment", "broad_sample"])["DNA"].count().reset_index()
    wells = pd.merge(wells, tmp, how="left", left_on=["Plate", "Well", "Treatment"], right_on=["Metadata_Plate", "Metadata_Well", "Treatment"])
    wells = wells[columns1 + columns2]

    # Sphering
    whN = WhiteningNormalizer(wells.loc[wells["Treatment"] == "Negative@0", columns2])
    whD = whN.normalize(wells[columns2])

    # Save data
    process_features_dir = f"{dset.config['paths']['results']}/features_processed"
    os.makedirs(process_features_dir, exist_ok=True)
    
    # Site-level
    sites.to_csv(f"{process_features_dir}/{channel}_sites.csv", index=False)

    # Well-level
    wells[columns2] = whD
    wells.to_csv(f"{process_features_dir}/{channel}_wells.csv", index=False)

    # Info
    total_single_cells = sum([features[i].shape[0] if len(features[i]) > 0 else 0 for i in range(len(features))])
    print(f"\tTotal images:", len(features))
    print(f"\tTotal single cells:", total_single_cells)