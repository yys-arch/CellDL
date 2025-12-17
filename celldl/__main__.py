import scanpy as sc
from .functions import (data_preprocessing, build_model, train_model, save_trained_model,
                       load_trained_model, denoise_data, generate_sc_synthetic_data)


def main_train():
    scobj = sc.read_h5ad("your dataset")
    scobj = data_preprocessing(scobj)

    model = build_model(scobj, mode='IZIP_mode')
    train_model(model, scobj, epochs=1000)

    save_trained_model(model, 'CellDL_model.keras')

def main_denoise():
    model = load_trained_model('CellDL_model.keras')
    scobj = sc.read_h5ad("your dataset")
    scobj = data_preprocessing(scobj)
    scobj_denoised = denoise_data(model, scobj)
    return scobj_denoised

def main_synthetic():
    model = load_trained_model('CellDL_model.keras')
    scobj = sc.read_h5ad("your dataset")
    scobj = data_preprocessing(scobj)
    scobj_synthetic = generate_sc_synthetic_data(model, scobj)
    return scobj_synthetic


if __name__ == "__main__":
    main_train()
    scobj_denoised = main_denoise()
    scobj_synthetic = main_synthetic()
