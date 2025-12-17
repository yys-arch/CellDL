from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy
import anndata
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tf_keras.optimizers as opt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_mutual_info_score
from scipy.stats import spearmanr
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')
from tf_keras import layers, models, losses, callbacks, initializers
Model = models.Model
Input = layers.Input
Dense = layers.Dense
Activation = layers.Activation
BatchNormalization = layers.BatchNormalization
Lambda = layers.Lambda
PReLU = layers.PReLU
EarlyStopping = callbacks.EarlyStopping
MeanSquaredError = losses.MeanSquaredError
load_model = models.load_model


# ==============================================================================
# Distribution Mean Functions
# ==============================================================================

@tf.function
def rna_Negbinom_pmf(inputs):
    """Mean of Negative Binomial (r=dispersion, theta=prob)."""
    r, theta = inputs
    nb = tfd.NegativeBinomial(total_count=r, probs=theta)
    return nb.mean()


def rna_Inflatednegbinom_pmf(inputs):
    """Mean of Zero-Inflated Negative Binomial."""
    r, theta, inflated_loc_prob = inputs
    zinb = tfd.ZeroInflatedNegativeBinomial(total_count=r, probs=theta, inflated_loc_probs=inflated_loc_prob)
    return zinb.mean()


def rna_Inflatedpoisson(inputs):
    """Mean of Zero-Inflated Poisson."""
    lambda_, inflated_loc_prob = inputs
    poissonb = tfd.Poisson(lambda_)
    zip_dist = tfd.Inflated(distribution=poissonb, inflated_loc_probs=inflated_loc_prob)
    return zip_dist.mean()


def rna_Indinflatedpoisson(inputs):
    """Mean of Independent Zero-Inflated Poisson."""
    lambda_, inflated_loc_prob = inputs
    poissonb = tfd.Poisson(lambda_)
    ind_zip = tfd.Independent(
        distribution=tfd.Inflated(distribution=poissonb, inflated_loc_probs=inflated_loc_prob),
        reinterpreted_batch_ndims=0
    )
    return ind_zip.mean()


def rna_Mixpoissonnb(inputs):
    """Mean of Mixture (Poisson + Negative Binomial)."""
    lambda_, r, theta, cat = inputs
    poisson = tfd.Poisson(lambda_)
    nb = tfd.NegativeBinomial(total_count=r, probs=theta)
    mixpoissonnb = tfd.Mixture(
        cat=tfd.Categorical(tf.stack([cat, 1 - cat], axis=-1)),
        components=[poisson, nb]
    )
    return mixpoissonnb.mean()


def rna_zindmixpoissonnb(inputs):
    """Mean of Zero-Inflated Mixture (Poisson + NB)."""
    lambda_, r, theta, cat, inflated_loc_prob = inputs
    poisson = tfd.Poisson(lambda_)
    nb = tfd.NegativeBinomial(total_count=r, probs=theta)
    mixpoissonnb = tfd.Mixture(
        cat=tfd.Categorical(tf.stack([cat, 1 - cat], axis=-1)),
        components=[poisson, nb]
    )
    zindmixpoissonnb = tfd.Inflated(distribution=mixpoissonnb, inflated_loc_probs=inflated_loc_prob)
    return zindmixpoissonnb.mean()


def rna_Mixpoissonlognormal(inputs):
    """Mean of Mixture (Poisson + LogNormal)."""
    lambda_, loc, scale, cat = inputs
    poisson = tfd.Poisson(lambda_)
    lognormal = tfd.LogNormal(loc=loc, scale=scale)
    mixpoissonlognormal = tfd.Mixture(
        cat=tfd.Categorical(tf.stack([cat, 1 - cat], axis=-1)),
        components=[poisson, lognormal]
    )
    return mixpoissonlognormal.mean()


def rna_zindmixpoissonlognormal(inputs):
    """Mean of Zero-Inflated Mixture (Poisson + LogNormal)."""
    lambda_, loc, scale, cat, inflated_loc_prob = inputs
    poisson = tfd.Poisson(lambda_)
    lognormal = tfd.LogNormal(loc=loc, scale=scale)
    mixpoissonlognormal = tfd.Mixture(
        cat=tfd.Categorical(tf.stack([cat, 1 - cat], axis=-1)),
        components=[poisson, lognormal]
    )
    zindmixpoissonlognormal = tfd.Inflated(distribution=mixpoissonlognormal, inflated_loc_probs=inflated_loc_prob)
    return zindmixpoissonlognormal.mean()


def rna_indzindmixpoissonlognormal(inputs):
    """Mean of Independent Zero-Inflated Mixture (Poisson + LogNormal)."""
    lambda_, loc, scale, cat, inflated_loc_prob = inputs
    poisson = tfd.Poisson(lambda_)
    lognormal = tfd.LogNormal(loc=loc, scale=scale)
    mixpoissonlognormal = tfd.Mixture(
        cat=tfd.Categorical(tf.stack([cat, 1 - cat], axis=-1)),
        components=[poisson, lognormal]
    )
    zindmixpoissonlognormal = tfd.Inflated(distribution=mixpoissonlognormal, inflated_loc_probs=inflated_loc_prob)
    ind_zind = tfd.Independent(distribution=zindmixpoissonlognormal, reinterpreted_batch_ndims=0)
    return ind_zind.mean()


def rna_indzindmixnblognormal(inputs):
    """Mean of Independent Zero-Inflated Mixture (NB + LogNormal)."""
    r, theta, loc, scale, cat, inflated_loc_prob = inputs
    nb = tfd.NegativeBinomial(total_count=r, probs=theta)
    lognormal = tfd.LogNormal(loc=loc, scale=scale)
    mixnblognormal = tfd.Mixture(
        cat=tfd.Categorical(tf.stack([cat, 1 - cat], axis=-1)),
        components=[nb, lognormal]
    )
    zindmixnblognormal = tfd.Inflated(distribution=mixnblognormal, inflated_loc_probs=inflated_loc_prob)
    ind_zind = tfd.Independent(distribution=zindmixnblognormal, reinterpreted_batch_ndims=0)
    return ind_zind.mean()


# ==============================================================================
# Reconstruction Layers
# ==============================================================================

def NB_reconstruct(input_dim_rna, h_rna_decoder_z, inikernel):
    """Output layer for NB distribution."""
    NorAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 3, 1e10)
    rna_r = Dense(input_dim_rna, kernel_initializer=inikernel, activation=NorAct, name="rna_r")(h_rna_decoder_z)
    rna_theta = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_theta")(
        h_rna_decoder_z)
    rna_mean = Lambda(rna_Negbinom_pmf, output_shape=(input_dim_rna,), name="rna_denoised")([rna_r, rna_theta])
    return rna_mean


def ZINB_reconstruct(input_dim_rna, h_rna_decoder_z, inikernel):
    """Output layer for ZINB distribution."""
    NorAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 3, 1e10)
    rna_r = Dense(input_dim_rna, kernel_initializer=inikernel, activation=NorAct, name="rna_r")(h_rna_decoder_z)
    rna_theta = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_theta")(
        h_rna_decoder_z)
    rna_zerorate = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_zerorate")(
        h_rna_decoder_z)
    rna_mean = Lambda(rna_Inflatednegbinom_pmf, output_shape=(input_dim_rna,), name="rna_denoised")(
        [rna_r, rna_theta, rna_zerorate])
    return rna_mean


def ZIP_reconstruct(input_dim_rna, h_rna_decoder_z, inikernel):
    """Output layer for ZIP distribution."""
    rna_lambda_ = Dense(input_dim_rna, kernel_initializer=inikernel, activation="relu", name="rna_lambda_")(
        h_rna_decoder_z)
    rna_zerorate = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_zerorate")(
        h_rna_decoder_z)
    rna_mean = Lambda(rna_Inflatedpoisson, output_shape=(input_dim_rna,), name="rna_denoised")(
        [rna_lambda_, rna_zerorate])
    return rna_mean


def IZIP_reconstruct(input_dim, h_rna_decoder_z, inikernel):
    """Output layer for IZIP distribution."""
    rna_lambda_ = Dense(input_dim, kernel_initializer=inikernel, activation="relu", name="rna_lambda_")(h_rna_decoder_z)
    rna_zerorate = Dense(input_dim, kernel_initializer=inikernel, activation="sigmoid", name="rna_zerorate")(
        h_rna_decoder_z)
    rna_mean = Lambda(rna_Indinflatedpoisson, output_shape=(input_dim,), name="rna_denoised")(
        [rna_lambda_, rna_zerorate])
    return rna_mean


def Mix_P_NB_reconstruct(input_dim_rna, h_rna_decoder_z, inikernel):
    """Output layer for Mixture (Poisson + NB)."""
    NorAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 3, 1e10)
    rna_lambda_ = Dense(input_dim_rna, kernel_initializer=inikernel, activation="relu", name="rna_lambda_")(
        h_rna_decoder_z)
    rna_r = Dense(input_dim_rna, kernel_initializer=inikernel, activation=NorAct, name="rna_r")(h_rna_decoder_z)
    rna_theta = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_theta")(
        h_rna_decoder_z)
    rna_cat = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_cat")(h_rna_decoder_z)
    rna_mean = Lambda(rna_Mixpoissonnb, output_shape=(input_dim_rna,), name="rna_denoised")(
        [rna_lambda_, rna_r, rna_theta, rna_cat])
    return rna_mean


def ZIMix_P_NB_reconstruct(input_dim_rna, h_rna_decoder_z, inikernel):
    """Output layer for Zero-Inflated Mixture (Poisson + NB)."""
    NorAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 3, 1e10)
    rna_lambda_ = Dense(input_dim_rna, kernel_initializer=inikernel, activation="relu", name="rna_lambda_")(
        h_rna_decoder_z)
    rna_r = Dense(input_dim_rna, kernel_initializer=inikernel, activation=NorAct, name="rna_r")(h_rna_decoder_z)
    rna_theta = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_theta")(
        h_rna_decoder_z)
    rna_cat = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_cat")(h_rna_decoder_z)
    rna_zerorate = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_zerorate")(
        h_rna_decoder_z)
    rna_mean = Lambda(rna_zindmixpoissonnb, output_shape=(input_dim_rna,), name="rna_denoised")(
        [rna_lambda_, rna_r, rna_theta, rna_cat, rna_zerorate])
    return rna_mean


def Mix_P_logNormal_reconstruct(input_dim_rna, h_rna_decoder_z, inikernel):
    """Output layer for Mixture (Poisson + LogNormal)."""
    rna_lambda_ = Dense(input_dim_rna, kernel_initializer=inikernel, activation="relu", name="rna_lambda_")(
        h_rna_decoder_z)
    rna_loc = Dense(input_dim_rna, kernel_initializer=inikernel, activation="relu", name="rna_loc")(h_rna_decoder_z)
    rna_scale = Dense(input_dim_rna, kernel_initializer=inikernel, activation="linear", name="rna_scale")(
        h_rna_decoder_z)
    rna_cat = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_cat")(h_rna_decoder_z)
    rna_mean = Lambda(rna_Mixpoissonlognormal, output_shape=(input_dim_rna,), name="rna_denoised")(
        [rna_lambda_, rna_loc, rna_scale, rna_cat])
    return rna_mean


def ZIMix_P_logNormal_reconstruct(input_dim_rna, h_rna_decoder_z, inikernel):
    """Output layer for Zero-Inflated Mixture (Poisson + LogNormal)."""
    rna_lambda_ = Dense(input_dim_rna, kernel_initializer=inikernel, activation="relu", name="rna_lambda_")(
        h_rna_decoder_z)
    rna_loc = Dense(input_dim_rna, kernel_initializer=inikernel, activation="relu", name="rna_loc")(h_rna_decoder_z)
    rna_scale = Dense(input_dim_rna, kernel_initializer=inikernel, activation="linear", name="rna_scale")(
        h_rna_decoder_z)
    rna_cat = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_cat")(h_rna_decoder_z)
    rna_zerorate = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_zerorate")(
        h_rna_decoder_z)
    rna_mean = Lambda(rna_zindmixpoissonlognormal, output_shape=(input_dim_rna,), name="rna_denoised")(
        [rna_lambda_, rna_loc, rna_scale, rna_cat, rna_zerorate])
    return rna_mean


def IZIMix_P_logNormal_reconstruct(input_dim_rna, h_rna_decoder_z, inikernel):
    """Output layer for Independent Zero-Inflated Mixture (Poisson + LogNormal)."""
    rna_lambda_ = Dense(input_dim_rna, kernel_initializer=inikernel, activation="relu", name="rna_lambda_")(
        h_rna_decoder_z)
    rna_loc = Dense(input_dim_rna, kernel_initializer=inikernel, activation="relu", name="rna_loc")(h_rna_decoder_z)
    rna_scale = Dense(input_dim_rna, kernel_initializer=inikernel, activation="linear", name="rna_scale")(
        h_rna_decoder_z)
    rna_cat = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_cat")(h_rna_decoder_z)
    rna_zerorate = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_zerorate")(
        h_rna_decoder_z)
    rna_mean = Lambda(rna_indzindmixpoissonlognormal, output_shape=(input_dim_rna,), name="rna_denoised")(
        [rna_lambda_, rna_loc, rna_scale, rna_cat, rna_zerorate])
    return rna_mean


def IZIMix_NB_logNormal_reconstruct(input_dim_rna, h_rna_decoder_z, inikernel):
    """Output layer for Independent Zero-Inflated Mixture (NB + LogNormal)."""
    NorAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 3, 1e10)
    rna_r = Dense(input_dim_rna, kernel_initializer=inikernel, activation=NorAct, name="rna_r")(h_rna_decoder_z)
    rna_theta = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_theta")(
        h_rna_decoder_z)
    rna_loc = Dense(input_dim_rna, kernel_initializer=inikernel, activation="relu", name="rna_loc")(h_rna_decoder_z)
    rna_scale = Dense(input_dim_rna, kernel_initializer=inikernel, activation="linear", name="rna_scale")(
        h_rna_decoder_z)
    rna_cat = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_cat")(h_rna_decoder_z)
    rna_zerorate = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_zerorate")(
        h_rna_decoder_z)
    rna_mean = Lambda(rna_indzindmixnblognormal, output_shape=(input_dim_rna,), name="rna_denoised")(
        [rna_r, rna_theta, rna_loc, rna_scale, rna_cat, rna_zerorate])
    return rna_mean


# ==============================================================================
# Core Functions
# ==============================================================================

def load_data(filepath, donor_id, assay, gene_mean_min, gene_mean_max, gene_disp_min):
    """Loads and preprocesses data (Legacy wrapper). See data_preprocessing."""
    scobj = sc.read_h5ad(filepath)
    scobj = scobj[scobj.obs['donor_id'] == donor_id, :]
    scobj = scobj[scobj.obs['assay'] == assay, :]
    if scobj.raw.X is not None:
        scobj.X = scobj.raw.X
    scobj.var_names_make_unique()
    scobj.var.index = pd.Index(scobj.var['feature_name'].values)
    sc.pp.log1p(scobj)
    sc.pp.highly_variable_genes(scobj, min_mean=gene_mean_min, max_mean=gene_mean_max, min_disp=gene_disp_min)
    scobj = scobj[:, scobj.var["highly_variable"]]
    scobj.obsm["rna_nor"] = scobj.X.toarray()
    scobj.obsm["X_input"] = 1 + scobj.obsm["rna_nor"]
    scaler = StandardScaler()
    scobj.obsm["X_input"] = scaler.fit_transform(scobj.obsm["X_input"])
    return scobj


def build_model(scobj, seed=100, bottle_dim=512, mode='IZIP_mode'):
    """
    Builds the CellDL model.
    Modes: 'IZIP_mode' (default), 'NB_mode', 'ZINB_mode', 'ZIP_mode', 'Mix_P_NB_mode', etc.
    """
    inikernel = initializers.glorot_uniform(seed=seed)
    if "X_input" not in scobj.obsm:
        raise ValueError("scobj.obsm['X_input'] missing. Run data_preprocessing first.")
    input_dim = scobj.obsm["X_input"].shape[1]
    input_data = Input(shape=(input_dim,), name='X_input')

    # Encoder
    h = input_data
    for units in [2048, 1024]:
        h = Dense(units, kernel_initializer=inikernel)(h)
        h = BatchNormalization()(h)
        h = PReLU()(h)

    # Bottleneck
    h = Dense(bottle_dim, kernel_initializer=inikernel, name="rna_features")(h)
    h = Activation("relu")(h)

    # Decoder
    h = Dense(input_dim, kernel_initializer=inikernel, name="rec_dim")(h)
    h = Activation("relu")(h)

    # Distribution Heads
    if mode == 'IZIP_mode':
        rna_mean = IZIP_reconstruct(input_dim, h, inikernel)
    elif mode == 'NB_mode':
        rna_mean = NB_reconstruct(input_dim, h, inikernel)
    elif mode == 'ZINB_mode':
        rna_mean = ZINB_reconstruct(input_dim, h, inikernel)
    elif mode == 'ZIP_mode':
        rna_mean = ZIP_reconstruct(input_dim, h, inikernel)
    elif mode == 'Mix_P_NB_mode':
        rna_mean = Mix_P_NB_reconstruct(input_dim, h, inikernel)
    elif mode == 'ZIMix_P_NB_mode':
        rna_mean = ZIMix_P_NB_reconstruct(input_dim, h, inikernel)
    elif mode == 'Mix_P_logNormal_mode':
        rna_mean = Mix_P_logNormal_reconstruct(input_dim, h, inikernel)
    elif mode == 'ZIMix_P_logNormal_mode':
        rna_mean = ZIMix_P_logNormal_reconstruct(input_dim, h, inikernel)
    elif mode == 'IZIMix_P_logNormal_mode':
        rna_mean = IZIMix_P_logNormal_reconstruct(input_dim, h, inikernel)
    elif mode == 'IZIMix_NB_logNormal_mode':
        rna_mean = IZIMix_NB_logNormal_reconstruct(input_dim, h, inikernel)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    model = Model(inputs=input_data, outputs=rna_mean)
    return model


def train_model(model, scobj, lr=0.001, batch_size=32, epochs=3000):
    """Trains the model using RMSprop and EarlyStopping."""
    optimizer = opt.RMSprop(learning_rate=lr, clipvalue=5)
    model.compile(optimizer=optimizer, loss=MeanSquaredError())
    callbacks_list = [EarlyStopping(monitor="loss", patience=15, verbose=2)]

    history = model.fit(
        x=scobj.obsm["X_input"], y=scobj.obsm["rna_nor"],
        epochs=epochs, callbacks=callbacks_list,
        batch_size=batch_size, shuffle=True, verbose=1
    )
    return history


def denoise_data(model, scobj):
    """Denoises data by calculating the expected value of the learned distribution."""
    temp_denoised_rna = Model(inputs=model.inputs, outputs=model.get_layer("rna_denoised").output).predict(
        [scobj.obsm["X_input"]])
    scobj.obsm["rna_denoised"] = temp_denoised_rna
    scobj_denoised = sc.AnnData(
        X=temp_denoised_rna, obs=scobj.obs, var=scobj.var,
        obsm=scobj.obsm, layers=scobj.layers, uns=scobj.uns, varm=scobj.varm
    )
    return scobj_denoised


def calculate_spearman_correlation(scobj):
    """Calculates mean Spearman correlation between denoised and raw data."""
    temp_denoised_rna = scobj.obsm["rna_denoised"]
    corr_list = [spearmanr(x, y)[0] for x, y in zip(temp_denoised_rna, scobj.obsm["rna_nor"])]
    return np.mean(corr_list)


def save_trained_model(model, filepath):
    """Saves the trained model to a file."""
    model.save(filepath)


def load_trained_model(filepath):
    """Loads a trained model with custom distribution layers."""
    custom_objects = {
        'rna_Negbinom_pmf': rna_Negbinom_pmf,
        'rna_Inflatednegbinom_pmf': rna_Inflatednegbinom_pmf,
        'rna_Inflatedpoisson': rna_Inflatedpoisson,
        'rna_Indinflatedpoisson': rna_Indinflatedpoisson,
        'rna_Mixpoissonnb': rna_Mixpoissonnb,
        'rna_zindmixpoissonnb': rna_zindmixpoissonnb,
        'rna_Mixpoissonlognormal': rna_Mixpoissonlognormal,
        'rna_zindmixpoissonlognormal': rna_zindmixpoissonlognormal,
        'rna_indzindmixpoissonlognormal': rna_indzindmixpoissonlognormal,
        'rna_indzindmixnblognormal': rna_indzindmixnblognormal,
    }
    return load_model(filepath, custom_objects=custom_objects, safe_mode=False)


def generate_sc_synthetic_data(model, scobj, num_samples=1, deviation_scale=0.1):
    """Generates synthetic cells by perturbing learned distribution parameters."""
    X_input = scobj.obsm["X_input"]
    num_cells, num_genes = X_input.shape

    lambda_layer = model.get_layer("rna_lambda_")
    zerorate_layer = model.get_layer("rna_zerorate")

    lambda_model = Model(inputs=model.inputs, outputs=lambda_layer.output)
    zerorate_model = Model(inputs=model.inputs, outputs=zerorate_layer.output)

    lambda_values = lambda_model.predict(X_input)
    zerorate_values = zerorate_model.predict(X_input)

    mean_values = (1 - zerorate_values) * lambda_values
    mean_values_repeated = np.repeat(mean_values, num_samples, axis=0)

    noise = np.random.uniform(-deviation_scale, deviation_scale, size=mean_values_repeated.shape) * mean_values_repeated
    synthetic_data = mean_values_repeated + noise

    synthetic_data = np.clip(synthetic_data, a_min=0, a_max=None)


    synthetic_obs = pd.DataFrame(np.repeat(scobj.obs.values, num_samples, axis=0), columns=scobj.obs.columns)
    synthetic_obs.reset_index(drop=True, inplace=True)
    synthetic_obs['original_cell_index'] = np.repeat(np.arange(num_cells), num_samples)   # ?索引也要添加原始索引吧


    synthetic_scobj = anndata.AnnData(
        X=synthetic_data,
        obs=synthetic_obs,
        var=scobj.var.copy()
    )

    return synthetic_scobj


def data_preprocessing(scobj, assay=None, ID=None, gene_mean_min=0.0125, gene_mean_max=3, gene_disp_min=0.5):
    """
    Preprocess single-cell data for CellDL: filter, normalize, and select HVGs.

    Args:
        scobj: AnnData object.
        assay: (Optional) Filter by 'assay' column.
        ID: (Optional) Filter by 'donor_id' column.
        gene_mean_min/max, gene_disp_min: Thresholds for Highly Variable Genes.

    Returns:
        AnnData object with prepared input in `.obsm['X_input']`.
    """
    # Use raw counts if available
    if scobj.raw is not None:
        scobj.X = scobj.raw.X
    scobj.var_names_make_unique()

    # Filter by assay or ID if specified and columns exist
    if assay is not None:
        if 'assay' in scobj.obs.columns:
            scobj = scobj[scobj.obs['assay'] == assay].copy()
        else:
            warnings.warn(f"'assay' column missing; skipping filter assay='{assay}'.")

    if ID is not None:
        if 'donor_id' in scobj.obs.columns:
            scobj = scobj[scobj.obs['donor_id'] == ID].copy()
        else:
            warnings.warn(f"'donor_id' column missing; skipping filter ID='{ID}'.")

    # Use feature names if available
    if 'feature_name' in scobj.var.columns:
        scobj.var.index = pd.Index(scobj.var['feature_name'].values)

    # Standard preprocessing
    sc.pp.log1p(scobj)
    sc.pp.highly_variable_genes(scobj, min_mean=gene_mean_min, max_mean=gene_mean_max, min_disp=gene_disp_min)
    scobj = scobj[:, scobj.var["highly_variable"]].copy()

    # Prepare dense input for model (handle sparse matrices)
    scobj.obsm["rna_nor"] = scobj.X.toarray() if scipy.sparse.issparse(scobj.X) else scobj.X

    # Scale data (StandardScaler)
    scaler = StandardScaler()
    scobj.obsm["X_input"] = scaler.fit_transform(1 + scobj.obsm["rna_nor"])

    return scobj
