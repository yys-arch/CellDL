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

# NB
# r 是离散度参数，theta 是成功概率
@tf.function
def rna_Negbinom_pmf(inputs):
    r, theta = inputs
    nb = tfd.NegativeBinomial(total_count=r, probs=theta)
    return nb.mean()


# ZINB
def rna_Inflatednegbinom_pmf(inputs):
    r, theta, inflated_loc_prob = inputs
    zinb = tfd.ZeroInflatedNegativeBinomial(total_count=r, probs=theta, inflated_loc_probs=inflated_loc_prob)
    return zinb.mean()

# ZIP
def rna_Inflatedpoisson(inputs):
    lambda_, inflated_loc_prob = inputs
    poissonb = tfd.Poisson(lambda_)
    zip = tfd.Inflated(
        distribution=poissonb,
        inflated_loc_probs=inflated_loc_prob
    )
    return zip.mean()

# IZIP
def rna_Indinflatedpoisson(inputs):
    lambda_, inflated_loc_prob = inputs
    poissonb = tfd.Poisson(lambda_)
    ind_zip = tfd.Independent(
        distribution=tfd.Inflated(
            distribution=poissonb,
            inflated_loc_probs=inflated_loc_prob
        ),
        reinterpreted_batch_ndims=0
    )
    return ind_zip.mean()

# 混合泊松分布与负二项分布版的RNA分布的平均数推断
def rna_Mixpoissonnb(inputs):
    lambda_, r, theta, cat = inputs
    poisson = tfd.Poisson(lambda_)
    nb = tfd.NegativeBinomial(total_count=r, probs=theta)
    mixpoissonnb = tfd.Mixture(
        cat=tfd.Categorical(tf.stack([cat, 1-cat], axis=-1)), # 混合系数
        components=[poisson, nb] # 分布类型列表
    )
    return mixpoissonnb.mean()

# 零膨胀的混合泊松分布与负二项分布版的RNA分布的平均数推断
def rna_zindmixpoissonnb(inputs):
    lambda_, r, theta, cat, inflated_loc_prob = inputs
    poisson = tfd.Poisson(lambda_)
    nb = tfd.NegativeBinomial(total_count=r, probs=theta)
    mixpoissonnb = tfd.Mixture(
        cat=tfd.Categorical(tf.stack([cat, 1-cat], axis=-1)), # 混合系数
        components=[poisson, nb] # 分布类型列表
    )
    zindmixpoissonnb=tfd.Inflated(
        distribution=mixpoissonnb,
        inflated_loc_probs=inflated_loc_prob
    )
    return zindmixpoissonnb.mean()


# 混合泊松分布与对数正态分布版的RNA分布的平均数推断
def rna_Mixpoissonlognormal(inputs):
    lambda_, loc, scale, cat = inputs
    poisson = tfd.Poisson(lambda_)
    lognormal = tfd.LogNormal(loc=loc, scale=scale)
    mixpoissonlognormal = tfd.Mixture(
        cat=tfd.Categorical(tf.stack([cat, 1-cat], axis=-1)), # 混合系数
        components=[poisson, lognormal] # 分布类型列表
    )
    return mixpoissonlognormal.mean()


# 零膨胀的混合泊松分布与对数正态分布版的RNA分布的平均数推断
def rna_zindmixpoissonlognormal(inputs):
    lambda_, loc, scale, cat, inflated_loc_prob= inputs
    poisson = tfd.Poisson(lambda_)
    lognormal = tfd.LogNormal(loc=loc, scale=scale)
    mixpoissonlognormal = tfd.Mixture(
        cat=tfd.Categorical(tf.stack([cat, 1-cat], axis=-1)), # 混合系数
        components=[poisson, lognormal] # 分布类型列表
    )
    zindmixpoissonlognormal=tfd.Inflated(
        distribution=mixpoissonlognormal,
        inflated_loc_probs=inflated_loc_prob
    )
    return zindmixpoissonlognormal.mean()


# 独立的零膨胀的混合泊松分布与对数正态分布版的RNA分布的平均数推断
def rna_indzindmixpoissonlognormal(inputs):
    lambda_, loc, scale, cat, inflated_loc_prob= inputs
    poisson = tfd.Poisson(lambda_)
    lognormal = tfd.LogNormal(loc=loc, scale=scale)
    mixpoissonlognormal = tfd.Mixture(
        cat=tfd.Categorical(tf.stack([cat, 1-cat], axis=-1)), # 混合系数
        components=[poisson, lognormal] # 分布类型列表
    )
    zindmixpoissonlognormal=tfd.Inflated(
        distribution=mixpoissonlognormal,
        inflated_loc_probs=inflated_loc_prob
    )
    ind_zind = tfd.Independent(
            distribution=zindmixpoissonlognormal,
            reinterpreted_batch_ndims=0
    )
    return ind_zind.mean()

# 独立的零膨胀的混合负二项分布与对数正态分布版的RNA分布的平均数推断
def rna_indzindmixnblognormal(inputs):
    r, theta, loc, scale, cat, inflated_loc_prob= inputs
    nb = tfd.NegativeBinomial(total_count=r, probs=theta)
    lognormal = tfd.LogNormal(loc=loc, scale=scale)
    mixnblognormal = tfd.Mixture(
        cat=tfd.Categorical(tf.stack([cat, 1-cat], axis=-1)), # 混合系数
        components=[nb, lognormal] # 分布类型列表
    )
    zindmixnblognormal=tfd.Inflated(
        distribution=mixnblognormal,
        inflated_loc_probs=inflated_loc_prob
    )
    ind_zind = tfd.Independent(
            distribution=zindmixnblognormal,
            reinterpreted_batch_ndims=0
    )
    return ind_zind.mean()

# 负二项分布版RNA分布平均数推断
def NB_reconstruct(input_dim_rna, h_rna_decoder_z, inikernel):
    NorAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 3, 1e10)
    rna_r = Dense(input_dim_rna, kernel_initializer=inikernel, activation=NorAct, name="rna_r")(h_rna_decoder_z)
    rna_theta = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_theta")(h_rna_decoder_z)
    rna_mean = Lambda(rna_Negbinom_pmf, output_shape=(input_dim_rna,), name="rna_denoised")([rna_r, rna_theta])
    return rna_mean


# 零膨胀负二项分布版RNA分布平均数推断
def ZINB_reconstruct(input_dim_rna, h_rna_decoder_z, inikernel):
    NorAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 3, 1e10)
    rna_r = Dense(input_dim_rna, kernel_initializer=inikernel, activation=NorAct, name="rna_r")(h_rna_decoder_z)
    rna_theta = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_theta")(h_rna_decoder_z)
    rna_zerorate = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_zerorate")(h_rna_decoder_z)
    # r 是离散度参数，theta 是成功概率
    rna_mean = Lambda(rna_Inflatednegbinom_pmf, output_shape=(input_dim_rna,), name="rna_denoised")([rna_r, rna_theta, rna_zerorate])
    return rna_mean


# 零膨胀泊松分布版RNA分布平均数推断
def ZIP_reconstruct(input_dim_rna, h_rna_decoder_z, inikernel):
    rna_lambda_ = Dense(input_dim_rna, kernel_initializer=inikernel, activation="relu", name="rna_lambda_")(h_rna_decoder_z)
    rna_zerorate = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_zerorate")(h_rna_decoder_z)
    rna_mean = Lambda(rna_Inflatedpoisson, output_shape=(input_dim_rna,), name="rna_denoised")([rna_lambda_, rna_zerorate])
    return rna_mean

# IZIP
def IZIP_reconstruct(input_dim, h_rna_decoder_z, inikernel):
    rna_lambda_ = Dense(input_dim, kernel_initializer=inikernel, activation="relu", name="rna_lambda_")(h_rna_decoder_z)
    rna_zerorate = Dense(input_dim, kernel_initializer=inikernel, activation="sigmoid", name="rna_zerorate")(h_rna_decoder_z)
    rna_mean = Lambda(rna_Indinflatedpoisson, output_shape=(input_dim,), name="rna_denoised")([rna_lambda_, rna_zerorate])
    return rna_mean


# 混合泊松分布与负二项分布版的RNA分布的平均数推断
def Mix_P_NB_reconstruct(input_dim_rna, h_rna_decoder_z, inikernel):
    NorAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 3, 1e10)
    rna_lambda_ = Dense(input_dim_rna, kernel_initializer=inikernel, activation="relu", name="rna_lambda_")(h_rna_decoder_z)
    rna_r = Dense(input_dim_rna, kernel_initializer=inikernel, activation=NorAct, name="rna_r")(h_rna_decoder_z)
    rna_theta = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_theta")(h_rna_decoder_z)
    rna_cat = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_cat")(h_rna_decoder_z)
    rna_mean = Lambda(rna_Mixpoissonnb, output_shape=(input_dim_rna,), name="rna_denoised")([rna_lambda_, rna_r, rna_theta, rna_cat])
    return rna_mean


# 零膨胀的混合泊松分布与负二项分布版的RNA分布的平均数推断
def ZIMix_P_NB_reconstruct(input_dim_rna, h_rna_decoder_z, inikernel):
    NorAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 3, 1e10)
    rna_lambda_ = Dense(input_dim_rna, kernel_initializer=inikernel, activation="relu", name="rna_lambda_")(h_rna_decoder_z)
    rna_r = Dense(input_dim_rna, kernel_initializer=inikernel, activation=NorAct, name="rna_r")(h_rna_decoder_z)
    rna_theta = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_theta")(h_rna_decoder_z)
    rna_cat = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_cat")(h_rna_decoder_z)
    rna_zerorate = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_zerorate")(h_rna_decoder_z)
    rna_mean = Lambda(rna_zindmixpoissonnb, output_shape=(input_dim_rna,), name="rna_denoised")([rna_lambda_, rna_r, rna_theta, rna_cat, rna_zerorate])
    return rna_mean


# 混合泊松分布与对数正态分布版的RNA分布的平均数推断
def Mix_P_logNormal_reconstruct(input_dim_rna, h_rna_decoder_z, inikernel):
    rna_lambda_ = Dense(input_dim_rna, kernel_initializer=inikernel, activation="relu", name="rna_lambda_")(h_rna_decoder_z)
    rna_loc = Dense(input_dim_rna, kernel_initializer=inikernel, activation="relu", name="rna_loc")(h_rna_decoder_z)
    rna_scale = Dense(input_dim_rna, kernel_initializer=inikernel, activation="linear", name="rna_scale")(h_rna_decoder_z)
    rna_cat = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_cat")(h_rna_decoder_z)
    rna_mean = Lambda(rna_Mixpoissonlognormal, output_shape=(input_dim_rna,), name="rna_denoised")([rna_lambda_, rna_loc, rna_scale, rna_cat])
    return rna_mean


# 零膨胀的混合泊松分布与对数正态分布版的RNA分布的平均数推断
def ZIMix_P_logNormal_reconstruct(input_dim_rna, h_rna_decoder_z, inikernel):
    rna_lambda_ = Dense(input_dim_rna, kernel_initializer=inikernel, activation="relu", name="rna_lambda_")(h_rna_decoder_z)
    rna_loc = Dense(input_dim_rna, kernel_initializer=inikernel, activation="relu", name="rna_loc")(h_rna_decoder_z)
    rna_scale = Dense(input_dim_rna, kernel_initializer=inikernel, activation="linear", name="rna_scale")(h_rna_decoder_z)
    rna_cat = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_cat")(h_rna_decoder_z)
    rna_zerorate = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_zerorate")(h_rna_decoder_z)
    rna_mean = Lambda(rna_zindmixpoissonlognormal, output_shape=(input_dim_rna,), name="rna_denoised")([rna_lambda_, rna_loc, rna_scale, rna_cat, rna_zerorate])
    return rna_mean


# 独立的零膨胀的混合泊松分布与对数正态分布版的RNA分布的平均数推断
def IZIMix_P_logNormal_reconstruct(input_dim_rna, h_rna_decoder_z, inikernel):
    rna_lambda_ = Dense(input_dim_rna, kernel_initializer=inikernel, activation="relu", name="rna_lambda_")(h_rna_decoder_z)
    rna_loc = Dense(input_dim_rna, kernel_initializer=inikernel, activation="relu", name="rna_loc")(h_rna_decoder_z)
    rna_scale = Dense(input_dim_rna, kernel_initializer=inikernel, activation="linear", name="rna_scale")(h_rna_decoder_z)
    rna_cat = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_cat")(h_rna_decoder_z)
    rna_zerorate = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_zerorate")(h_rna_decoder_z)
    rna_mean = Lambda(rna_indzindmixpoissonlognormal, output_shape=(input_dim_rna,), name="rna_denoised")([rna_lambda_, rna_loc, rna_scale, rna_cat, rna_zerorate])
    return rna_mean


# 独立的零膨胀的混合负二项分布与对数正态分布版的RNA分布的平均数推断
def IZIMix_NB_logNormal_reconstruct(input_dim_rna, h_rna_decoder_z, inikernel):
    NorAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 3, 1e10)
    rna_r = Dense(input_dim_rna, kernel_initializer=inikernel, activation=NorAct, name="rna_r")(h_rna_decoder_z)
    rna_theta = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_theta")(h_rna_decoder_z)
    rna_loc = Dense(input_dim_rna, kernel_initializer=inikernel, activation="relu", name="rna_loc")(h_rna_decoder_z)
    rna_scale = Dense(input_dim_rna, kernel_initializer=inikernel, activation="linear", name="rna_scale")(h_rna_decoder_z)
    rna_cat = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_cat")(h_rna_decoder_z)
    rna_zerorate = Dense(input_dim_rna, kernel_initializer=inikernel, activation="sigmoid", name="rna_zerorate")(h_rna_decoder_z)
    rna_mean = Lambda(rna_indzindmixnblognormal, output_shape=(input_dim_rna,), name="rna_denoised")([rna_r, rna_theta, rna_loc, rna_scale, rna_cat, rna_zerorate])
    return rna_mean


def load_data(filepath, donor_id, assay, gene_mean_min, gene_mean_max, gene_disp_min):
    """
    加载并预处理单细胞数据。

    参数：
    - filepath: 数据文件路径
    - donor_id: 选择的donor_id
    - hvg_var_names: 高度可变基因的变量名列表

    返回：
    - scobj: 预处理后的AnnData对象
    """
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
    构建深度学习模型。

    参数：
    - adata: 输入数据
    - seed: 随机种子
    - mode: 分布模式
    可选参数: NB_mode;
            ZINB_mode;
            ZIP_mode;
            IZIP_mode;
            Mix_P_NB_mode;
            ZIMix_P_NB_mode;
            Mix_P_logNB;
            ZIMix_P_logNB;
            IZIMix_P_logNB;
            IZIMix_NB_logNB

    返回：
    - model: 构建的模型
    """

    inikernel = initializers.glorot_uniform(seed=seed)
    if "X_input" not in scobj.obsm:
        raise ValueError("scobj.obsm['X_input'] 不存在，无法检测特征维度。")
    input_dim = scobj.obsm["X_input"].shape[1]
    input_data = Input(shape=(input_dim,), name='X_input')

    h = input_data
    for units in [2048, 1024]:
        h = Dense(units, kernel_initializer=inikernel)(h)
        h = BatchNormalization()(h)
        h = PReLU()(h)

    h = Dense(bottle_dim, kernel_initializer=inikernel, name="rna_features")(h)
    h = Activation("relu")(h)

    h = Dense(input_dim, kernel_initializer=inikernel, name="rec_dim")(h)
    h = Activation("relu")(h)

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
        print('error')

    model = Model(inputs=input_data, outputs=rna_mean)
    return model


def train_model(model, scobj, lr=0.001, batch_size=32, epochs=3000):  # 原batch_size=32
    """
    训练模型。

    参数：
    - model: 待训练的模型
    - scobj: 包含训练数据的AnnData对象
    - lr: 学习率
    - batch_size: 批次大小
    - epochs: 训练轮数

    返回：
    - history: 训练过程记录
    """
    optimizer = opt.RMSprop(learning_rate=lr, clipvalue=5)
    model.compile(optimizer=optimizer, loss=MeanSquaredError())
    callbacks = [EarlyStopping(monitor="loss", patience=15, verbose=2)]
    history = model.fit(
        x=scobj.obsm["X_input"],
        y=scobj.obsm["rna_nor"],
        epochs=epochs,
        callbacks=callbacks,
        batch_size=batch_size,
        shuffle=True,
        verbose=1
    )
    return history


def denoise_data(model, scobj):
    """
    对数据进行去噪处理。

    参数：
    - model: 训练好的模型
    - scobj: 包含原始数据的AnnData对象

    返回：
    - scobj_denoised: 去噪后的AnnData对象
    """
    temp_denoised_rna = Model(inputs=model.inputs, outputs=model.get_layer("rna_denoised").output).predict([scobj.obsm["X_input"]])
    scobj.obsm["rna_denoised"] = temp_denoised_rna
    scobj_denoised = sc.AnnData(
        temp_denoised_rna,
        obs=scobj.obs,
        var=scobj.var,
        obsm=scobj.obsm,
        layers=scobj.layers,
        uns=scobj.uns,
        varm=scobj.varm
    )
    return scobj_denoised


def calculate_spearman_correlation(scobj):
    """
    计算斯皮尔曼相关性。

    参数：
    - scobj: 包含去噪数据的AnnData对象

    返回：
    - mean_corr: 平均斯皮尔曼相关系数
    """
    temp_denoised_rna = scobj.obsm["rna_denoised"]
    corr_list = []
    for x, y in zip(temp_denoised_rna, scobj.obsm["rna_nor"]):
        corr, _ = spearmanr(x, y)
        corr_list.append(corr)
    mean_corr = np.mean(corr_list)
    return mean_corr

def save_trained_model(model, filepath):
    """
    保存训练后的模型，包括自定义的层和函数。

    参数：
    - model: 训练后的模型
    - filepath: 模型保存的文件路径，例如 'path/to/model.keras'

    返回：
    - 无
    """
    # 移除不受支持的参数，直接调用 model.save()
    model.save(filepath)

def load_trained_model(filepath):
    """
    加载保存的模型，包括自定义的层和函数。

    参数：
    - filepath: 模型文件的路径，例如 'path/to/model.keras'

    返回：
    - model: 加载的模型
    """
    # 创建包含自定义对象的字典
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
        # 如果有其他自定义函数或类，请在此处添加
    }

    # 加载模型，指定自定义对象
    model = load_model(filepath, custom_objects=custom_objects, safe_mode=False)
    return model


def generate_sc_synthetic_data(model, scobj, num_samples=1, deviation_scale=0.1):
    """
    生成围绕均值的小范围随机扰动的合成数据。

    参数：
    - scobj: AnnData 对象，包含原始数据和模型输入
    - model: 训练好的 cellDL 模型
    - num_samples: 每个细胞要生成的合成样本数量
    - deviation_scale: 偏离比例，用于控制随机扰动的大小（默认 0.1 表示±10%）

    返回：
    - synthetic_scobj: 包含合成数据的 AnnData 对象
    """
    # 提取模型输入
    X_input = scobj.obsm["X_input"]
    num_cells, num_genes = X_input.shape

    # 提取模型输出的参数
    lambda_layer = model.get_layer("rna_lambda_")
    zerorate_layer = model.get_layer("rna_zerorate")

    lambda_model = Model(inputs=model.inputs, outputs=lambda_layer.output)
    zerorate_model = Model(inputs=model.inputs, outputs=zerorate_layer.output)

    lambda_values = lambda_model.predict(X_input)
    zerorate_values = zerorate_model.predict(X_input)

    # 计算均值
    mean_values = (1 - zerorate_values) * lambda_values

    # 重复均值
    mean_values_repeated = np.repeat(mean_values, num_samples, axis=0)

    # 添加小的随机扰动
    noise = np.random.uniform(-deviation_scale, deviation_scale, size=mean_values_repeated.shape) * mean_values_repeated
    synthetic_data = mean_values_repeated + noise

    # 确保数据非负
    synthetic_data = np.clip(synthetic_data, a_min=0, a_max=None)

    # 创建合成数据的 obs，并添加原始细胞索引
    synthetic_obs = pd.DataFrame(np.repeat(scobj.obs.values, num_samples, axis=0), columns=scobj.obs.columns)
    synthetic_obs.reset_index(drop=True, inplace=True)
    synthetic_obs['original_cell_index'] = np.repeat(np.arange(num_cells), num_samples)   # ?索引也要添加原始索引吧

    # 创建新的 AnnData 对象
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
