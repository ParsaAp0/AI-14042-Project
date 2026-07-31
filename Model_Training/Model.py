import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pandas.core.series import Series
from PIL import Image

IMAGE_SIZE = 64

class ASLDataset(Dataset):
	def __init__(self, X: Series, Y: Series, transform=None):
		self.X = X.reset_index(drop = True)
		self.Y = Y.reset_index(drop = True)
		# self.dataframe = dataframe.reset_index(drop=True)
		self.transform = transform
		# Create a mapping from text labels (A, B, del...) to integers (0, 1, 2...)
		self.labels = sorted(self.Y.unique())
		self.idx = {label: idx for idx, label in enumerate(self.labels)}
	
	def __len__(self):
		return len(self.X)
	
	def __getitem__(self, idx):
		# Fetch image path and class string
		img_path = self.X[idx]
		class_name = self.Y[idx]
	       	
		       # Load the image lazily (only when requested)
		image = Image.open(img_path).convert('RGB')
		label_idx = self.idx[class_name]
		
		# Apply the resizing and normalization transforms
		if self.transform:
			image = self.transform(image)
					
		return image, torch.tensor(label_idx, dtype=torch.long)

class CNN(nn.Module):
        def __init__(self, num_classes):
                super().__init__()
                self.POOL_SIZE = 2
                self.L1_OUT_CHANNEL  = 32
                self.L1_KERNEL_SIZE  = 3
                self.L2_OUT_CHANNEL  = 64
                self.L2_KERNEL_SIZE  = 3
                self.L3_OUT_CHANNEL  = 128
                self.L3_KERNEL_SIZE  = 3
                self.DROPOUT_P = 0.5
                self.L1_NEURON = 512
                self.FINAL_DIM = IMAGE_SIZE // 8 # x Layers --> 2 ** x
                
                
                self.pool = nn.MaxPool2d(
                        kernel_size = self.POOL_SIZE,
                        stride = self.POOL_SIZE
                )
                self.relu = nn.ReLU()
                
                # Layer 1
                
                self.conv1 = nn.Conv2d(
                        in_channels = 3,
                        out_channels = self.L1_OUT_CHANNEL,
                        kernel_size =self. L1_KERNEL_SIZE,
                        padding = (self.L1_KERNEL_SIZE - 1) // 2
                )
                self.batch_norm1 = nn.BatchNorm2d(self.L1_OUT_CHANNEL)
                
                # Layer  2
                
                self.conv2 = nn.Conv2d(
                        in_channels = self.L1_OUT_CHANNEL,
                        out_channels = self.L2_OUT_CHANNEL,
                        kernel_size = self.L2_KERNEL_SIZE,
                        padding = (self.L2_KERNEL_SIZE - 1) // 2
                )
                self.batch_norm2 = nn.BatchNorm2d(self.L2_OUT_CHANNEL)
                
                # Layer  3
                self.conv3 = nn.Conv2d(
                        in_channels = self.L2_OUT_CHANNEL,
                        out_channels = self.L3_OUT_CHANNEL,
                        kernel_size = self.L3_KERNEL_SIZE,
                        padding = (self.L3_KERNEL_SIZE - 1) // 2
                )
                self.batch_norm3 = nn.BatchNorm2d(self.L3_OUT_CHANNEL)
                
                
                # Fully Connected (Dense) Layers
                self.dropout = nn.Dropout(self.DROPOUT_P)
                # After three MaxPools of 2x2, a 64x64 image becomes 8x8. 
                # So the flattened size is 128 channels * FINAL_DIM * FINAL_DIM
                
                self.fully_connected1 = nn.Linear(self.L3_OUT_CHANNEL * self.FINAL_DIM * self.FINAL_DIM, self.L1_NEURON)
                self.fully_connected2 = nn.Linear(self.L1_NEURON, num_classes)
                
        def forward(self, x: torch.Tensor):
                # L1
                x = self.pool(self.relu(self.batch_norm1(self.conv1(x))))
                
                # L2
                x = self.pool(self.relu(self.batch_norm2(self.conv2(x))))
                
                # L3
                x = self.pool(self.relu(self.batch_norm3(self.conv3(x))))
                
                x = x.view(-1, self.L3_OUT_CHANNEL * self.FINAL_DIM * self.FINAL_DIM)
                
                x = self.dropout(self.relu(self.fully_connected1(x)))
                x = self.fully_connected2(x)
                
                return x