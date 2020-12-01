import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.metrics import classification_report




def save_fig(fig_dir, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = f'{fig_dir}' / f'{fig_id}.{fig_extension}'
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def model_metrics(y_test,y_pred,decoded):
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


# plots the accuracy and loss for against epochs
def plot_history(history):
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epoch = history.epoch

    plt.figure(figsize=(11,4))
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
    plt.gca().set_ylim(0, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.title('Training and Validation Loss')

    save_fig(config["results_dir"],"history")
    plt.show()