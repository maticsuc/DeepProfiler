import importlib
import os

import numpy as np
import tensorflow as tf
import pandas as pd
import scipy.linalg

from deepprofiler.dataset.utils import tic, toc, set_seed

import deepprofiler.learning.feature_processing

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
            except tf.errors.InvalidArgumentError as e:
                print("Loading weights without classifier (different number of classes)")
                self.dpmodel.feature_model.layers[-1]._name = "classifier"
                self.dpmodel.feature_model.load_weights(checkpoint, by_name=True)
            except Exception as e:
                print("An unexpected error occurred while loading weights.")
                print(f"Details: {e}")

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

        # check image size matches config
        if (self.config["dataset"]["images"]["width"] != image_array.shape[1] or
            self.config["dataset"]["images"]["height"] != image_array.shape[0]):
            config_shape = (self.config["dataset"]["images"]["width"],
                            self.config["dataset"]["images"]["height"])
            im_shape = (image_array.shape[1], image_array.shape[0])
            raise ValueError("Loaded image shape WxH " + str(im_shape) +
                             " != configured image shape WxH " + str(config_shape))

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

    # Fixed seed
    if "seed" in config:
        set_seed(seed=config["seed"])
        print(f"profiling.py: Fixed seed to {config['seed']}.")

    profile = Profile(config, dset)
    profile.configure()
    dset.scan(profile.extract_features, frame="all", check=profile.check)
    print("Profiling: Done")

def process_features(config, dset):

    meta = dset.meta.data
    dataset = config['profile']['process_features']

    if dataset == 'BBBC022':
        meta["broad_sample"] = meta["Treatment"].str.split("@", expand=True)[0]

    # Single cell
    features = deepprofiler.learning.feature_processing.single_cell(config, meta)
        
    # Site-level
    sites = deepprofiler.learning.feature_processing.site_level(meta, features, dataset)

    # Well-level
    wells = deepprofiler.learning.feature_processing.well_level(meta, config, sites)

    # Sphering
    wells = deepprofiler.learning.feature_processing.sphering(config, wells)

    # Treatment-level
    profiles = deepprofiler.learning.feature_processing.treatment_level(config, wells)

    # Similarity matrix
    deepprofiler.learning.feature_processing.similarity_matrix(config, profiles)