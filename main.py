import numpy as np
from random import *
import matplotlib.pyplot as plt
from CA import *
from NN import *

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                            Initializors
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
N = 20
TEST_SIZE = 80
RUNS = 100
LEARNING_RATE = 0.001
SHOW_MAP = False
file = 'results.txt'

# If the machine has a GPU, utilize it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                            Neural Network
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def Train(train_x, train_y):
    model = Net() 

    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    loss = 100000
    k = 0
    while loss > 0.01:
        model.zero_grad()
        output = model(train_x)
        loss = F.mse_loss(output, train_y)
        loss.backward()
        optimizer.step()
        if k % 1000 == 0:
            print(k, loss)
        k += 1

    return model

def Test(model, test_x, test_y):
    count = 0
    prediction = model(test_x)
    for i in range(TEST_SIZE):
        if 0 <= math.abs(prediction[i] - test_y[i]): <= 0.5:
            count += 1

    return (count / TEST_SIZE) * 100

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                            Helper Functions
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def get_testing_data(cells):
    input_data = []
    output_data = []

    test_inputs = []
    targets = []

    while len(input_data) < 320:
        x = randint(0, 19)
        y = randint(0, 19)
        if [x, y] not in input_data:
            input_data.append([x, y])
            output_data.append([cells[x, y]])

    for i in range(len(cells)):
        for j in range(len(cells[i])):
            if [i, j] not in input_data:
                test_inputs.append([i, j])
                targets.append([cells[i][j]])

    return torch.Tensor(np.array(input_data)), torch.Tensor(np.array(output_data)), torch.Tensor(np.array(test_inputs)), torch.Tensor(np.array(targets))

def display_results(x, y):
    plt.plot(x, y)
    plt.show()

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                Main
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
if __name__ == "__main__":
    ca = CA(N, N)
    for i in range(1000):
        ca.generate()
        if SHOW_MAP:
            plt.imshow(ca.get_cells(), cmap=plt.cm.gray, interpolation='nearest')
            plt.pause(0.01)
        if ca.check_for_stability():
            plt.title("Stable")
            plt.imshow(ca.get_cells(), cmap=plt.cm.gray, interpolation='nearest')
            plt.show()
            print("Generation: {}".format(i))
            break
    
    train_x, train_y, test_x, test_y = get_testing_data(ca.get_cells())

    runs = []
    efficiency = []
    for i in range(RUNS):
        model = Train(train_x, train_y)
        res = Test(model, test_x, test_y)
        runs.append(i + 1)
        efficiency.append(res)
        print("Run {} efficiency: {}%".format(i, res)) 
    
    form = "%i %.2f \n"
    with open(file,'w') as f:
        for i in range(len(runs)):
            f.write(form % (runs[i],efficiency[i]))


    display_results(runs, efficiency)
