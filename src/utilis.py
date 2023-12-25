import csv

label_to_idx_map = {
    "No_DR": 0,
    "Mild": 1,
    "Moderate": 2,
    "Proliferate_DR": 3,
    "Severe": 4,
}

rev_label_to_index_map = {index: label for label, index in label_to_idx_map.items()}


def label_to_idx(label: str):
    """convert label name to index
    for example: if my dataset consists of 5 labels (AngleMan'Apert'Down'charge''Williams')

    """
    if label not in label_to_idx_map:
        raise ValueError(
            f"Label:{label} not defined. label must be one of {label_to_idx_map.keys()}"
        )

    return label_to_idx_map.get(label)


def idx_to_label(idx: int):
    """similiar as label_to_idx but opposite I.e. take the index and return the string label"""

    if idx not in rev_label_to_index_map:
        raise ValueError(
            f"index not found. index must be one of {rev_label_to_index_map.keys()}"
        )

    return rev_label_to_index_map.get(idx)


def read_as_csv(csv_file):
    image_path = []
    labels = []
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            image_path.append(row[0])
            labels.append(row[1])
    return image_path, labels


if __name__=="__main__":
    images, labels = read_as_csv("data/test.csv")
    
    print(labels)