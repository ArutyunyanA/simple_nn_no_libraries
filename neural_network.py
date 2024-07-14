import random
import math

def sigmoid(x):
    """
    Сигмоидная функция активации.
    """
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    """
    Производная сигмоидной функции.
    """
    return x * (1 - x)

def random_weight():
    """
    Генерация случайного веса в диапазоне [-1, 1].
    """
    return random.uniform(-1, 1)

class Neuron:
    def __init__(self, num_inputs):
        """
        Инициализация нейрона с заданным количеством входов.
        """
        self.weights = [random_weight() for _ in range(num_inputs)]  # Инициализация случайных весов
        self.bias = random_weight()  # Инициализация случайного смещения
        self.output = 0  # Выход нейрона после активации
        self.delta = 0  # Ошибка нейрона

    def activate(self, inputs):
        """
        Активация нейрона на основе входных данных.
        """
        total = sum(w * inp for w, inp in zip(self.weights, inputs)) + self.bias  # Линейная комбинация входов и весов
        self.output = sigmoid(total)  # Применение функции активации
        return self.output

    def calculate_delta(self, target=None, downstream_neurons=None, index=None):
        """
        Вычисление ошибки (дельты) нейрона.
        """
        if target is not None:
            error = target - self.output  # Вычисление ошибки при наличии целевого значения
        else:
            error = sum(neuron.weights[index] * neuron.delta for neuron in downstream_neurons)  # Вычисление ошибки при обратном распространении
        self.delta = error * sigmoid_derivative(self.output)  # Применение производной сигмоидной функции

class Layer:
    def __init__(self, num_neurons, num_inputs_per_neuron):
        """
        Инициализация слоя с заданным количеством нейронов и входов на нейрон.
        """
        self.neurons = [Neuron(num_inputs_per_neuron) for _ in range(num_neurons)]  # Создание нейронов слоя

    def forward(self, inputs):
        """
        Прямое распространение входных данных через слой нейронов.
        """
        return [neuron.activate(inputs) for neuron in self.neurons]  # Активация каждого нейрона в слое

    def backward(self, targets=None, downstream_layer=None):
        """
        Обратное распространение ошибки через слой нейронов.
        """
        if targets is not None:
            for i, neuron in enumerate(self.neurons):
                neuron.calculate_delta(target=targets[i])  # Вычисление дельты для каждого нейрона при наличии целевых значений
        else:
            for i, neuron in enumerate(self.neurons):
                neuron.calculate_delta(downstream_neurons=downstream_layer.neurons, index=i)  # Вычисление дельты при обратном распространении

class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        Инициализация нейронной сети с заданными размерами слоев.
        """
        self.layers = []
        for i in range(1, len(layer_sizes)):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i - 1]))  # Создание слоев нейронов

    def forward(self, inputs):
        """
        Прямое распространение входных данных через все слои сети.
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs  # Возвращение выходных данных после прохождения через все слои

    def backward(self, targets):
        """
        Обратное распространение ошибки через все слои сети.
        """
        for i in reversed(range(len(self.layers))):
            if i == len(self.layers) - 1:
                self.layers[i].backward(targets=targets)  # Обратное распространение для последнего слоя сети
            else:
                self.layers[i].backward(downstream_layer=self.layers[i + 1])  # Обратное распространение для остальных слоев

    def update_weights(self, inputs, learning_rate):
        """
        Обновление весов и смещений нейронов сети на основе вычисленных дельт.
        """
        for i, layer in enumerate(self.layers):
            inputs = [neuron.output for neuron in self.layers[i - 1].neurons] if i > 0 else inputs
            for neuron in layer.neurons:
                for j in range(len(neuron.weights)):
                    neuron.weights[j] += learning_rate * neuron.delta * inputs[j]  # Обновление весов нейронов
                neuron.bias += learning_rate * neuron.delta  # Обновление смещений нейронов

    def train(self, training_data, epochs, learning_rate):
        """
        Обучение нейронной сети на основе предоставленных данных.
        """
        for epoch in range(epochs):
            for inputs, targets in training_data:
                self.forward(inputs)
                self.backward(targets)
                self.update_weights(inputs, learning_rate)  # Обновление весов после каждого обучающего примера

if __name__ == "__main__":
    training_data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0])
    ]
    
    nn = NeuralNetwork([2, 2, 1])  # Создание нейронной сети с заданными размерами слоев
    nn.train(training_data, epochs=10000, learning_rate=0.1)  # Обучение сети
    
    for inputs, _ in training_data:
        print(f"Input: {inputs}, Predicted: {nn.forward(inputs)}")  # Вывод предсказаний после обучения
