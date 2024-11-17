import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 定义模型
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

# 超参数组合
hidden_sizes = [32, 64]
learning_rates = [0.001, 0.01]
epochs = 50
batch_size = 2

# 假设logits和标签数据从Lab 1已准备好
logits = torch.tensor([[2.0, 1.0, 0.1], [1.0, 3.0, 0.2]], dtype=torch.float32)
y = torch.tensor([0, 1], dtype=torch.long)

# 创建数据加载器
dataset = TensorDataset(logits, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 保存结果
results = []

# 网格搜索超参数
for hidden_size, learning_rate in itertools.product(hidden_sizes, learning_rates):
    print(f"Testing Hidden Size: {hidden_size}, Learning Rate: {learning_rate}")

    # 初始化模型、损失函数和优化器
    input_size = logits.shape[1]
    output_size = 3
    model = NeuralNetwork(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练过程
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    results.append((hidden_size, learning_rate, avg_loss, accuracy))
    print(f"Hidden Size: {hidden_size}, Learning Rate: {learning_rate}, Loss: {avg_loss:.3f}, Accuracy: {accuracy:.3f}")

# 输出最佳结果
best_result = max(results, key=lambda x: x[3])  # 按准确率排序
print("\nBest Configuration:")
print(f"Hidden Size: {best_result[0]}, Learning Rate: {best_result[1]}, Loss: {best_result[2]:.3f}, Accuracy: {best_result[3]:.3f}")

# 保存结果到文件
with open("hyperparameter_tuning_results.txt", "w") as file:
    file.write("Hyperparameter Tuning Results:\n")
    for result in results:
        file.write(f"Hidden Size: {result[0]}, Learning Rate: {result[1]}, Loss: {result[2]:.3f}, Accuracy: {result[3]:.3f}\n")
    file.write("\nBest Configuration:\n")
    file.write(f"Hidden Size: {best_result[0]}, Learning Rate: {best_result[1]}, Loss: {best_result[2]:.3f}, Accuracy: {best_result[3]:.3f}\n")
