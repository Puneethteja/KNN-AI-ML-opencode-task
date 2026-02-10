# KNN Classification on Fashion-MNIST (CSV)

This project implements the **K-Nearest Neighbors (KNN)** algorithm from scratch using **NumPy** and applies it to the **Fashion-MNIST dataset** provided in CSV format.

---

## Project Structure

.dist/
│── fashion-mnist_train.csv
│── fashion-mnist_test.csv
│── KNN_FINAL (1).py
│── README.md


All files are placed in the **same directory (`.dist`)** to avoid file path and loading issues.

---

## Dataset Description

The Fashion-MNIST dataset consists of grayscale images of clothing items.

- Each row represents one image
- First column: Class label (0–9)
- Remaining 784 columns: Pixel values (28×28 image flattened)

### Files Used
- `fashion-mnist_train.csv` — Training dataset
- `fashion-mnist_test.csv` — Testing dataset

---

## Requirements

- Python 3.x
- NumPy

Install NumPy if required:
```bash
pip install numpy
How to Run
Open the .dist folder in VS Code

Open the terminal in the same folder

Run the Python file:

python "KNN_FINAL (1).py"
Dataset Loading
The CSV files are loaded using NumPy:

train_data = np.loadtxt(
    "fashion-mnist_train.csv",
    delimiter=",",
    skiprows=1
)

test_data = np.loadtxt(
    "fashion-mnist_test.csv",
    delimiter=",",
    skiprows=1
)
Since the Python file and CSV files are in the same directory, no absolute file paths are required.

Algorithm Details
Algorithm: K-Nearest Neighbors (KNN)

Distance metric: Euclidean distance

Classification method: Majority voting among nearest neighbors

Common Errors and Solutions
FileNotFoundError
Ensure the .py file and .csv files are in the same folder

Do not use incorrect folder paths

Unicode / Path Errors
Avoid Windows backslashes (\) in file paths

Use relative paths or forward slashes (/)

Purpose
This project is intended for educational and academic use, demonstrating the working of the KNN algorithm without using external machine learning libraries.

Author
Puneeth Darisi