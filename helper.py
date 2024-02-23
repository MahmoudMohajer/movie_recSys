import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import onnx
from giza_actions.action import Action, action
from giza_actions.task import task

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

@task(name="Prepare Dataset")
def prepare_dataset():
    print("Prepare dataset...")
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
    with urlopen(url) as zurl:
        with ZipFile(BytesIO(zurl.read())) as zfile:
            zfile.extractall('.')
    
    ratings = pd.read_csv(f'ml-100k/u.data', sep='\t',
                      names=['user_id', 'movie_id', 'rating',
                             'unix_timestamp'])
    
    ratings['user_idx'] = ratings['user_id'] - 1
    ratings['movie_idx'] = ratings['movie_id'] - 1

    print("✅ Datasets prepared successfully")

    return ratings

@task(name="Create Loaders")
def create_loaders(ratings, batch_size):
    print("Create loaders...")
    # Split dataset into train and test sets
    train_data, test_data = train_test_split(ratings[['user_idx', 'movie_idx', 'rating']], test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(torch.LongTensor(train_data['user_idx'].values),
                                  torch.LongTensor(train_data['movie_idx'].values),
                                  torch.FloatTensor(train_data['rating'].values))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(torch.LongTensor(train_data['user_idx'].values),
                                  torch.LongTensor(train_data['movie_idx'].values),
                                  torch.FloatTensor(train_data['rating'].values))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("✅ Loaders created!")
    
    return (train_loader, test_loader)

@task(name="Train model")
def train_model(MF,
                num_users,
                num_movies,
                embedding_size,
                learning_rate, 
                device,
                num_epochs, 
                train_loader):

    print("Train model...")
    # Initialize model, loss function, and optimizer
    model = MF(num_users, num_movies, embedding_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for user_ids_batch, movie_ids_batch, ratings_batch in train_loader:
            user_ids_batch = user_ids_batch.to(device)
            movie_ids_batch = movie_ids_batch.to(device)
            ratings_batch = ratings_batch.to(device)
            optimizer.zero_grad()
            outputs = model(user_ids_batch, movie_ids_batch)
            loss = criterion(outputs, ratings_batch)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    print("✅ Model trained successfully")
        
    return model

@task(name="Test model")
def test_model(model, test_loader, criterion, device):
    print("Test model...")
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0

    with torch.no_grad():  # Disable gradient tracking during evaluation
        for user_ids_batch, item_ids_batch, ratings_batch in test_loader:
            # Move data to the same device as the model
            user_ids_batch = user_ids_batch.to(device)
            item_ids_batch = item_ids_batch.to(device)
            ratings_batch = ratings_batch.to(device)

            # Forward pass
            outputs = model(user_ids_batch, item_ids_batch)

            # Compute loss
            loss = criterion(outputs, ratings_batch)
            test_loss += loss.item() * user_ids_batch.size(0)  # Accumulate the total loss

    # Average the loss over all test batches
    test_loss /= len(test_loader.dataset)

    print(f"Test loss is {test_loss}")
    return test_loss

@task(name="Convert To ONNX")
def convert_to_onnx(model, num_movies, device, file_path):
    # Set the model to evaluation mode
    model.eval()

    # Define input tensors (dummy input)
    user_id = 619
    all_movie_ids = torch.arange(num_movies).to(device)  # Generate all item IDs
    user_ids = torch.full((num_movies,), user_id, dtype=torch.long).to(device)

    # Provide the input shape
    input_shape = (user_ids.size(), all_movie_ids.size())


    # Export the model to ONNX format
    onnx_file_path = file_path 
    torch.onnx.export(model, (user_ids, all_movie_ids), onnx_file_path, input_names=["user_ids", "movie_ids"], output_names=["ratings"], opset_version=11)

    print(f"Model has been converted to ONNX and saved as {onnx_file_path}")
