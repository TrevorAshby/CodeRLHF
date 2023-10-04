import os
def train_val_test_split(dataset_path, x, y, z):
    pass

if __name__ == '__main__':
    x = 0.70 # percentage of train set | MODIFY-ABLE
    y = 0.15 # percentage of validation set | MODIFY-ABLE
    z = 0.15 # percentage of test set | MODIFY-ABLE
    dataset_path = './mini_codenet/data/' # path to dataset that needs to be split | MODIFY-ABLE

    if 1.0 == (x + y + z): # make sure that all percentages sum to 1.0 (representing 100%)
        train_val_test_split(dataset_path, x, y, z)
        print("Complete")
    else:
        print("ERROR: MAKE SURE THAT x, y, AND z PERCENTAGES ADD TO 1.0")