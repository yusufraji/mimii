import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import classification_report
import yaml
from pathlib import Path
import pandas as pd
from tqdm import tqdm


cur_dir = Path.cwd()

# load config yaml
with open("config.yaml") as stream:
    config = yaml.safe_load(stream)


def save_fig(fig_dir, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = f'{fig_dir}/{fig_id}.{fig_extension}'
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def model_metrics(y_test, y_pred, decoded, model_name):
    print(f"Decoded classes after applying inverse of label encoder: {decoded}")

    skplt.metrics.plot_confusion_matrix(y_test,
                                        y_pred,
                                        labels=decoded,
                                        title_fontsize='large',
                                        text_fontsize="medium",
                                        cmap='Greens',
                                        figsize=(8,6))
    plt.show()
    
    print("The classification report for the model : \n\n"+ classification_report(y_test, y_pred))


# plots the accuracy and loss against epochs
def plot_history(history, dir, file_name, ae):
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epoch = history.epoch

    plt.figure(figsize=(11,4))
    plt.suptitle(f'History of {file_name}')
    plt.subplot(121)
    plt.plot(epoch, acc, label = 'Training accuracy', color = 'r')
    plt.plot(epoch, val_acc, label = 'Validation accuracy', color = 'b')
    plt.gca().set_ylim(0, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.title('Training and Validation Accuracy')

    plt.subplot(122)
    plt.plot(epoch, loss, label = 'Training loss', color = 'r')
    plt.plot(epoch, val_loss, label = 'Validation loss', color = 'b')
    if not ae:
        plt.gca().set_ylim(0, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.title('Training and Validation Loss')

    save_fig(dir,file_name)
    plt.show()

# fetch dataset
def fetch_dataset(extension="wav", dataset_file_name="dataset_df"):
    print("================================= fetching dataset =================================\n")
    data = pd.DataFrame(columns=['machine_type', 'machine_id', 'machine_type_id', 'normal', 'abnormal', 'db'])
    for db in tqdm(["0dB", "6dB", "min6dB"], desc='fetching db files.', leave=True):
        for folder in tqdm((Path.cwd() / config["dataset_dir"] / db).iterdir(), desc=f'fetching machine files.', leave=False):
            for id in tqdm(folder.iterdir(), desc=f'fetching machine id files.', leave=False):
                normal_glob = (id / 'normal' ).glob(f"*.{extension}")
                abnormal_glob = (id / "abnormal").glob(f"*.{extension}")
                for normal, abnormal in zip(normal_glob, abnormal_glob):
                    values = [folder.name, id.name, f'{folder.name}_{id.name}', normal, abnormal, db]
                    a_dict = dict(zip(list(data.columns), values))
                    data = data.append(a_dict, ignore_index=True)
    data.to_csv(cur_dir / config["dataset_dir"] / f'{dataset_file_name}.csv')
    return data
