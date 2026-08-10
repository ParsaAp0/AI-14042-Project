import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from Model import ASLDataset, CNN
from pathlib import Path

dataset_classes = sorted(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                        'del', 'nothing', 'space'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df_path = "../Data/train_path_dataframe.csv"
best_model_path = "../models/best.pt"
last_model_path = "../models/last.pt"

IMAGE_SIZE = 64
BATCH_SIZE = 64

def create_image_dataframe(root_dir):
	"""
	Create a DataFrame with image paths and their labels (folder names)
	from a directory structure where each subfolder represents a class.
	
	Args:
		root_dir (str): Path to the root directory containing class subfolders.
	
	Returns:
		pd.DataFrame: DataFrame with columns 'image_path' and 'Label'.
	"""
	data = []
	root_path = Path(root_dir)
	
	# Iterate over each subdirectory (class folder)
	for class_folder in root_path.iterdir():
		if class_folder.is_dir():
			label = class_folder.name  # folder name as label
			# Find all image files (common extensions)
			for img_path in class_folder.glob('*.*'):
				# Check if it's an image file by extension
				if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
					data.append({
						'image_path': str(img_path),
						'label': label
					})
	
	# Create DataFrame
	df = pd.DataFrame(data)
	return df

df = create_image_dataframe("../Data/asl_alphabet_train/asl_alphabet_train")
df.to_csv("../Data/train_path_dataframe.csv", index = False)

class Train_CNN:
        def __init__(self, train_df_path):
                self.train_transforms = transforms.Compose([
                        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                        transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3),  # Randomly change brightness
                        transforms.RandomAffine(degrees=0, translate=(0.3, 0.3)), # Randomly shift the image
                        transforms.ToTensor()
                ])
                
                self.test_transforms = transforms.Compose([
                        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                        transforms.ToTensor(),
                ])
                
                train_path_df = pd.read_csv(train_df_path)
                X_train, X_test, Y_train, Y_test = train_test_split(train_path_df['image_path'], train_path_df['label'], test_size = 0.2, random_state=54)
                X_train, X_valid, Y_train, Y_valid = train_test_split(X_train,  Y_train, test_size = 0.25, random_state=589)
                train_dataset = ASLDataset(X_train, Y_train, transform=self.train_transforms)
                valid_dataset = ASLDataset(X_valid, Y_valid, transform=self.test_transforms)
                test_dataset = ASLDataset(X_test, Y_test, transform=self.test_transforms)
                self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 9)
                self.valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 3)
                self.test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 3)
                self.model = CNN(num_classes=29).to(device)
                
        def learning_process(self,
                        model: CNN,
                        train_loader: DataLoader,
                        valid_loader: DataLoader,
                        criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss(),
                        learning_rate: float = 0.0001,
                        weight_decay=1e-4,
                        epochs: int = 11
                ):
                
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        factor=0.5,
                        patience=1
                )
                train_losses, valid_losses = [], []
                train_accs, valid_accs = [], []
                best_val_loss =10000
                
                # Training
                for epoch in range(epochs):
                        model.train()                        
                        running_loss, correct, total = 0.0, 0, 0
                        for images, labels in train_loader:
                                images, labels = images.to(device), labels.to(device)
                                
                                optimizer.zero_grad()
                                outputs = model(images)
                                loss = criterion(outputs, labels)
                                loss.backward()
                                optimizer.step()
                                
                                running_loss += loss.item()
                                _, predicted = torch.max(outputs.data, 1)
                                total += labels.size(0)
                                correct += (predicted == labels).sum().item()
                                
                        train_losses.append(running_loss / len(train_loader))
                        train_accs.append(100 * correct / total)
                        
                        model.eval()
                        valid_loss, correct, total = 0.0, 0, 0
                        with torch.no_grad():
                                for images, labels in valid_loader:
                                        images, labels = images.to(device), labels.to(device)
                                        
                                        outputs = model(images)
                                        loss = criterion(outputs, labels)
                                        
                                        valid_loss += loss.item()
                                        _, predicted = torch.max(outputs, 1)
                                        total += labels.size(0)
                                        correct += (predicted == labels).sum().item()
                                        
                        valid_losses.append(valid_loss / len(valid_loader))
                        valid_accs.append(100 * correct / total)
                        scheduler.step(valid_loss)

                        current_lr = optimizer.param_groups[0]["lr"]
                        
                        print(f"--------- Epoch {epoch+1}/{epochs} | "
                                f"Train Loss: {train_losses[-1]:.4f}, Acc: {train_accs[-1]:.2f}% | "
                                f"Val Loss: {valid_losses[-1]:.4f}, Acc: {valid_accs[-1]:.2f}%"
                                f"LR: {current_lr:.7f}")
                        
                        if valid_losses[-1] < best_val_loss:
                                best_val_loss = valid_losses[-1]
                                torch.save(model.state_dict(), best_model_path)
                                print(f"---- Best model saved with Val Loss: {best_val_loss:.4f}")
                return train_losses, train_accs, valid_losses, valid_accs
        
        def train(self):
                train_losses, train_accs, valid_losses, valid_accs = self.learning_process(model = self.model, train_loader = self.train_loader, valid_loader = self.valid_loader)
                
        def reset_model(self):
                self.model = CNN(num_classes=29).to(device)
        
        
        def test_model(self, model: nn.Module, test_loader):
                model.eval()
                criterion = nn.CrossEntropyLoss()

                test_loss = 0.0
                correct = 0
                total = 0
                
                all_preds = []
                all_true = []

                with torch.no_grad():
                        for images, labels in test_loader:
                                images, labels = images.to(device), labels.to(device)

                                outputs = model(images)
                                loss = criterion(outputs, labels)

                                test_loss += loss.item()
                                _, predicted = torch.max(outputs.data, 1)
                                total += labels.size(0)
                                correct += (predicted == labels).sum().item()

                                all_preds.extend(predicted.cpu().numpy())
                                all_true.extend(labels.cpu().numpy())

                avg_test_loss = test_loss / len(test_loader)
                test_accuracy = 100 * correct / total

                print("-" * 30)
                print(f"Final Test Loss: {avg_test_loss:.4f}")
                print(f"Final Test Accuracy: {test_accuracy:.2f}%")
                print("-" * 30)

                cm = confusion_matrix(all_true, all_preds)

                row_sums = cm.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                percent = (cm / row_sums) * 100

                color_indices = np.zeros_like(cm, dtype=int)

                text_matrix = np.empty_like(cm, dtype='<U20')
                for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                                p = percent[i, j]
                                text_matrix[i, j] = f"{cm[i, j]}"

                                # Apply color mapping rules
                                if i == j:  # Diagonal cells (correct predictions)
                                        if p > 75:
                                                color_indices[i, j] = 1   # Dark green
                                        elif p > 50:
                                                color_indices[i, j] = 2   # Light green
                                        elif p > 25:
                                                color_indices[i, j] = 3   # Light red
                                        else:
                                                color_indices[i, j] = 4   # Dark red
                                else:  # Off-diagonal cells (misclassifications)
                                        if p > 20:
                                                color_indices[i, j] = 5   # Dark red
                                        elif p > 5:
                                                color_indices[i, j] = 6   # Light red
                                        else:
                                                color_indices[i, j] = 7   # White (no color)

                # Define the mapping of numerical indices to hex colors
                colors_map = {
                        1: '#2ecc71',   # Dark green (diagonal > 75%)
                        2: '#a8e6cf',   # Light green (diagonal 50% - 75%)
                        3: '#ffb3b3',   # Light red (diagonal 25% - 50%)
                        4: '#e74c3c',   # Dark red (diagonal < 25%)
                        5: '#e74c3c',   # Dark red (off-diagonal > 20%)
                        6: '#ffb3b3',   # Light red (off-diagonal 5% - 20%)
                        7: '#ffffff'    # White (off-diagonal < 5%)
                }

                # Build a discrete colorscale for Plotly
                # (z values range between 0.5 and 7.5 to give each integer its own dedicated interval)
                zmin, zmax = 0.5, 7.5
                colorscale = []
                for val in sorted(colors_map.keys()):
                        low = (val - 0.5 - zmin) / (zmax - zmin)
                        high = (val + 0.5 - zmin) / (zmax - zmin)
                        colorscale.append([low, colors_map[val]])
                        colorscale.append([high, colors_map[val]])

                # Plot the heatmap using color_indices as the z-values
                fig = go.Figure(data=go.Heatmap(
                        z=color_indices, x=dataset_classes, y=dataset_classes,
                        colorscale=colorscale,
                        zmin=zmin, zmax=zmax,
                        text=text_matrix, texttemplate='%{text}', textfont={"size": 10, "color": "black"},
                        hovertemplate='True: %{y}<br>Predicted: %{x}<br>%{text}<extra></extra>'
                ))

                fig.update_layout(
                        title='Confusion Matrix: Test Dataset',
                        xaxis_title='Predicted Label', yaxis_title='True Label',
                        width=900, height=700,
                        xaxis={'tickangle': 45}
                )

                fig.show()
        
        def test(self):
                self.test_model(self.model, test_loader=self.test_loader)
                
tc = Train_CNN(train_df_path=train_df_path)
_ = tc.learning_process(tc.model, train_loader=tc.train_loader, valid_loader=tc.test_loader)
tc.test_model(tc.model, tc.test_loader)