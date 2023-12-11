import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import os
import time

class Train_mix():
    def __init__(self, datapath, lr=0.0001, fullnet_num=128, conv_num=32, deconv_size=(3,3)):
        self.datapath = datapath
        self.lr = lr
        self.fullnet_num = fullnet_num
        self.conv_num = conv_num
        self.deconv_size = deconv_size
        self.train_loss_batch = []
        self.train_accuracy_batch = []
        self.val_loss_batch = []
        self.val_accuracy_batch = []
        self.train_loss_epoch = []
        self.train_accuracy_epoch = []
        self.val_loss_epoch = []
        self.val_accuracy_epoch = []

    def acc_myself(self, y_true, y_pre):
        y_pre = torch.round(y_pre)
        r = torch.eq(y_true, y_pre)
        r = r.to(torch.float32)
        r = torch.sum(r, dim=1)
        d = torch.zeros_like(r, dtype=torch.float32) + 8
        c = torch.eq(r, d)
        c = c.to(torch.float32)

        return torch.divide(torch.sum(c), torch.cast(torch.numel(c), torch.float32))

    def build_model(self, inputs_shape, classes=8):
        bn_axis = 1  # Batch normalization is applied along channel axis

        model = nn.Sequential(
            ConvOffset2D_train(1),  # You need to replace this with the PyTorch equivalent
            nn.Conv2d(1, self.conv_num, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(self.conv_num),
            nn.ReLU(),

            ConvOffset2D_train(32),  # You need to replace this with the PyTorch equivalent
            nn.Conv2d(self.conv_num, self.conv_num * 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(self.conv_num * 2),
            nn.ReLU(),

            ConvOffset2D_train(64),  # You need to replace this with the PyTorch equivalent
            nn.Conv2d(self.conv_num * 2, self.conv_num * 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(self.conv_num * 4),
            nn.ReLU(),

            ConvOffset2D_train(128),  # You need to replace this with the PyTorch equivalent
            nn.Conv2d(self.conv_num * 4, self.conv_num * 8, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(self.conv_num * 8),
            nn.ReLU(),

            ConvOffset2D_train(256),  # You need to replace this with the PyTorch equivalent
            nn.Conv2d(self.conv_num * 8, self.conv_num * 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(self.conv_num * 4),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.conv_num * 4, classes),
            nn.Sigmoid()
        )

        return model

   def start_train(self):
        # Set device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        data = np.load(os.path.join(self.datapath))
        x = torch.tensor(data["arr_0"], dtype=torch.float32)
        y = torch.tensor(data["arr_1"], dtype=torch.float32)

        x = x.unsqueeze(-1)  # Add channel dimension in the last axis because ConvOffset2D_train will reshape the channel to axis=1
        data_shape = x.shape[1:]

        model = self.build_model(data_shape, classes=y.shape[-1])
        model.to(device)
        print(model)

        loss_fn = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-6, nesterov=True)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

        t = time.strftime('%Y-%b-%d_%H-%M-%S')

        batch_size = 32
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(x_test, y_test)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # loop over epochs
        for epoch in range(5000):
            
            # set the model in training mode
            model.train()
            
            # loop over the training set
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # perform a forward pass and calculate the training loss
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                # perform the backpropagation step and update the weight
                loss.backward()
                optimizer.step()  

                # Update losses and accuracies
                self.train_loss_batch.append(loss.item())
                accuracy = self.acc_myself(targets, outputs)
                self.train_accuracy_batch.append(accuracy.item())

            # switch off autograd for evaluation
	        with torch.no_grad():
                
        		# set the model in evaluation mode
        		model.eval()
                
        		# loop over the validation set
        		for inputs, targets in val_loader:
        			
        			inputs, targets = inputs.to(device), targets.to(device)
        			
                    # make the predictions and calculate the validation loss
        			pred = model(x)
                    loss = loss_fn(pred, targets)
                    
        			# Update losses and accuracies
                    self.val_loss_batch.append(loss.item())
                    accuracy = self.acc_myself(targets, outputs)
                    self.val_accuracy_batch.append(accuracy.item())
        			
                    
            if epoch % 100 == 0:
                print(f'Epoch {epoch}/{5000}, Loss: {loss.item()}, Accuracy: {accuracy.item()}')

if __name__ == "__main__":
    trainer = Train_mix(datapath="your_datapath_here")
    trainer.start_train()

    # Plotting losses and accuracies
    plt.plot(trainer.losses_batch, label='Training Loss')
    plt.plot(trainer.accuracy_batch, label='Training Accuracy')
    plt.legend()
    plt.show()