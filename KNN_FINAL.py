import numpy as np
import matplotlib.pyplot as plt

train_data = np.loadtxt(
    "C:/Users/user/OneDrive/Desktop/opencode AIML/.dist/fashion-mnist_train.csv",
    delimiter=",",
    skiprows=1
)

test_data = np.loadtxt(
    "C:/Users/user/OneDrive/Desktop/opencode AIML/.dist/fashion-mnist_test.csv",
    delimiter=",",
    skiprows=1
)

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)

y_train = train_data[:, 0].astype(int)
X_train = train_data[:, 1:].astype(int)

y_test = test_data[:, 0].astype(int)
X_test = test_data[:, 1:].astype(int)

print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)

X_train = X_train / 255.0
X_test = X_test / 255.0

idx = 0
plt.imshow(X_train[idx].reshape(28, 28), cmap="gray")
plt.title(f"Label: {class_names[y_train[idx]]}")
plt.axis("off")
plt.show()

def euclidean_distance(x1, x2):
    diff = x1 - x2
    return np.sqrt(np.sum(diff * diff))


def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

class KNN:
    def __init__(self, k=3, metric="euclidean"):
        self.k = k
        self.metric = metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _distance(self, x1, x2):
        if self.metric == "euclidean":
            return euclidean_distance(x1, x2)
        elif self.metric == "manhattan":
            return manhattan_distance(x1, x2)

    def predict_one(self, x):
        distances = []

        for i in range(len(self.X_train)):
            d = self._distance(x, self.X_train[i])
            distances.append((d, self.y_train[i]))

        distances.sort(key=lambda z: z[0])

        k_labels = [label for _, label in distances[:self.k]]

        predicted = max(set(k_labels), key=k_labels.count)
        confidence = k_labels.count(predicted) / self.k

        return predicted, confidence

    def predict(self, X):
        predictions = []
        confidences = []

        for x in X:
            pred, conf = self.predict_one(x)
            predictions.append(pred)
            confidences.append(conf)

        return np.array(predictions), np.array(confidences)

X_train_small = X_train[:2000]
y_train_small = y_train[:2000]

X_test_small = X_test[:200]
y_test_small = y_test[:200]

knn = KNN(k=3, metric="euclidean")
knn.fit(X_train_small, y_train_small)

y_pred, y_conf = knn.predict(X_test_small)

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

print("Accuracy:", accuracy(y_test_small, y_pred))

i = 0
plt.imshow(X_test_small[i].reshape(28,28), cmap="gray")
plt.title(
    f"Predicted: {class_names[y_pred[i]]}\n"
    f"Confidence: {y_conf[i]:.2f}\n"
    f"Actual: {class_names[y_test_small[i]]}"
)
plt.axis("off")
plt.show()

misclassified = np.where(y_test_small != y_pred)[0]
print("Number of misclassified samples:", len(misclassified))

# Most confident wrong prediction
wrong_sorted = misclassified[np.argsort(-y_conf[misclassified])]
idx = wrong_sorted[0]

plt.imshow(X_test_small[idx].reshape(28,28), cmap="gray")
plt.title(
    f"WRONG but CONFIDENT\n"
    f"Predicted: {class_names[y_pred[idx]]} ({y_conf[idx]:.2f})\n"
    f"Actual: {class_names[y_test_small[idx]]}"
)
plt.axis("off")
plt.show()

def neighbor_distribution(model, x):
    distances = []

    for i in range(len(model.X_train)):
        d = model._distance(x, model.X_train[i])
        distances.append((d, model.y_train[i]))

    distances.sort(key=lambda z: z[0])
    k_labels = [label for _, label in distances[:model.k]]

    counts = {}
    for lbl in k_labels:
        counts[lbl] = counts.get(lbl, 0) + 1

    return counts

dist = neighbor_distribution(knn, X_test_small[i])
print("Neighbor votes:")
for k, v in dist.items():
    print(class_names[k], ":", v)

print("\nEffect of K:")
for k in [1, 3, 5, 7]:
    model = KNN(k=k, metric="euclidean")
    model.fit(X_train_small, y_train_small)
    preds, _ = model.predict(X_test_small)
    print(f"K={k}, Accuracy={accuracy(y_test_small, preds):.3f}")

print("\nDistance Metric Comparison:")
for metric in ["euclidean", "manhattan"]:
    model = KNN(k=3, metric=metric)
    model.fit(X_train_small, y_train_small)
    preds, _ = model.predict(X_test_small)
    print(f"Metric={metric}, Accuracy={accuracy(y_test_small, preds):.3f}")

conf_matrix = np.zeros((10, 10), dtype=int)

for true, pred in zip(y_test_small, y_pred):
    conf_matrix[true][pred] += 1

plt.imshow(conf_matrix, cmap="Blues")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\nClass-wise Accuracy:")
for cls in range(10):
    idxs = np.where(y_test_small == cls)[0]
    acc = np.sum(y_pred[idxs] == cls) / len(idxs)
    print(f"{class_names[cls]:12s}: {acc:.2f}")


import random

def get_k_neighbors(model, x):
    distances = []

    for i in range(len(model.X_train)):
        d = model._distance(x, model.X_train[i])
        distances.append((d, model.X_train[i], model.y_train[i]))

    distances.sort(key=lambda z: z[0])
    return distances[:model.k]


rand_idx = random.randint(0, len(X_test_small) - 1)
query_img = X_test_small[rand_idx]
true_label = y_test_small[rand_idx]

neighbors = get_k_neighbors(knn, query_img)

pred_label, conf = knn.predict_one(query_img)

plt.figure(figsize=(12, 3))

plt.subplot(1, knn.k + 1, 1)
plt.imshow(query_img.reshape(28, 28), cmap="gray")
plt.title(
    f"QUERY\n"
    f"Pred: {class_names[pred_label]}\n"
    f"Conf: {conf:.2f}"
)
plt.axis("off")

for i, (dist, img, label) in enumerate(neighbors):
    plt.subplot(1, knn.k + 1, i + 2)
    plt.imshow(img.reshape(28, 28), cmap="gray")
    plt.title(
        f"{class_names[label]}\n"
        f"d={dist:.2f}"
    )
    plt.axis("off")

plt.suptitle(
    f"True Label: {class_names[true_label]}",
    fontsize=14
)
plt.show()



