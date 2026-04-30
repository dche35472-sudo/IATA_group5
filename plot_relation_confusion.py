import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results_TextRepresentation/test_predictions.csv")


rel1_cm = pd.crosstab(df["gold_rel1"], df["pred_rel1"])
rel2_cm = pd.crosstab(df["gold_rel2"], df["pred_rel2"])

print("REL1 confusion matrix:")
print(rel1_cm)
print("\nREL2 confusion matrix:")
print(rel2_cm)

def plot_cm(cm, title, save_name):
    plt.figure(figsize=(7, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(cm.columns)), cm.columns, rotation=45, ha="right")
    plt.yticks(range(len(cm.index)), cm.index)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm.iloc[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_name, dpi=200)
    plt.show()

plot_cm(rel1_cm, "Relation Confusion Matrix (rel1)", "rel1_confusion.png")
plot_cm(rel2_cm, "Relation Confusion Matrix (rel2)", "rel2_confusion.png")