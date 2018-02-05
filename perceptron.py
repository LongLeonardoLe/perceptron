import random

# train the perceptron
def train_perceptron(data, label, size, rows, cols, eta):
    weights = []

    # random initial weights in [-0.5,0.5]
    for i in range((rows*cols+1)*10*size):
        weights.append(random.uniform(-0.5,0.5))

    print(len(weights))
    # calculate weights
    for i in range(size):
        pos = i*(rows*cols+1) # starting position
        end_pos = pos+(rows*cols) # ending position

        weight_sum = 0
        prediction = 0
        for y in range(0, 10):
            pre_weight_sum = weight_sum
            for j in range(pos, end_pos+1):
                weight_sum += data[j]*weights[j]
            if weight_sum > pre_weight_sum:
                prediction = y

        # update weights
        if prediction != label[i]:
            for m in range(pos, end_pos+1):
                weights[m] += eta*(label[i]-prediction)*data[i]
    
    return weights


# prediction
def predict(weights, size, data):
    predictions = []
    for i in range(size):
        pos = i*(rows*cols+1)
        end_pos = pos+(rows*cols)
        weight_sum = 0
        predictions[i] = 0
        for y in range(0, 10):
            pre_weight_sum = weight_sum
            for j in range(pos, end_pos+1):
                weight_sum += data[j]*weights[j]
            if weight_sum > pre_weight_sum:
                predictions[i] = y
    return predictions
