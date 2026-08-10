import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
from pathlib import Path

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

path_df = pd.read_csv("../Data/train_path_dataframe.csv")
class_count = path_df['label'].value_counts().reset_index()

fig = px.bar(
	class_count, 
	x='label', 
	y='count',
	title='Number of Samples per Sign Language Class',
	labels={
		'label': 'Sign / Label',
		'count': 'Image Count'
	}
 )

fig.update_layout(
    xaxis_tickangle=-45,
    template='none',
    title_font_size=20,
    bargap=0.2
)

fig.show()

classes = sorted(class_count['label'])

rows = 10
columns = 3
IMG_WIDTH = 300
IMG_HEIGHT = 300

fig = make_subplots(
	rows=rows, 
	cols=columns, 
	subplot_titles=classes,
	horizontal_spacing=0.02,
	vertical_spacing=0.05
)

for i, c_name in enumerate(classes):
	r = (i // columns) + 1
	c = (i % columns) + 1
	
	class_df = path_df[path_df['label'] == c_name]
	random_row = class_df.sample(n=1).iloc[0]
	
	img = Image.open(random_row['image_path'])
	
	# Resize to fill the subplot
	img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
	
	fig.add_trace(go.Image(z=img_resized), row=r, col=c)
	
	# Update axes with auto aspect ratio
	fig.update_xaxes(
		showticklabels=False, 
		showgrid=False, 
		zeroline=False, 
		scaleanchor="y",  # This keeps aspect ratio
		scaleratio=1,     # 1:1 aspect ratio
		row=r, col=c
	)
	fig.update_yaxes(
		showticklabels=False, 
		showgrid=False, 
		zeroline=False, 
		row=r, col=c
	)

fig.update_layout(
	height=rows * IMG_HEIGHT // 2,
	width=columns * IMG_WIDTH // 2,
	title_text="Random Sample Grid From Each ASL Class",
	title_font_size=20,
	showlegend=False
)

fig.show()