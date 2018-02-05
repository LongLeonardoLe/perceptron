import struct
import perceptron
from array import array

# read data
def read_data(path):
    fdata = open(path, 'rb')
    magic_num, size, rows, cols = struct.unpack(">IIII", fdata.read(16))
    data_array = array("B", fdata.read())
    fdata.close()
    data_list = data_array.tolist()
    for i in range(size):
        data_list[i] /= 255
    return size, rows, cols, data_list


# read labels
def read_label(path):
    flbl = open(path, 'rb')
    magic_num, size = struct.unpack(">II", flbl.read(8))
    label_array = array("b", flbl.read())
    flbl.close()
    label_list = label_array.tolist()
    return size, label_list


# calculate accuracy
def accuracy(predictions, targets, size):
    correct_num = 0
    for i in range(size):
        if predictions[i] == targets[i]:
            correct_num +=1
    return correct_sum/size


if __name__ == "__main__":
    train_size, train_label = read_label("train_label")
    train_size, rows, cols, train_data = read_data("train_data")
    test_size, test_label = read_label("test_label")
    test_size, rows, cols, test_data = read_data("test_data")
    # learning rate = 0.1
    print("Learning rate = 0.1")
    weights = perceptron.train_perceptron(train_data, train_label, train_size, rows, cols, 0.1)
    print("Training accuracy", accuracy(perceptron.predict(weights, train_size, train_data), train_label, train_size))
    print("Testing accuracy", accuracy(perceptron.predict(weights, test_size, test_data), test_label, test_size))
