import torch
import torch
import torch.nn as nn
import torch.optim as optim
from giza_actions.action import Action, action
from giza_actions.task import task

from helper import prepare_dataset, create_loaders
from helper import train_model, test_model, convert_to_onnx


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model and MF means Matrix Factorization
class MF(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size):
        super(MF, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_size)
        self.movie_embeddings = nn.Embedding(num_movies, embedding_size)

    def forward(self, user_ids, movie_ids):
        user_embeds = self.user_embeddings(user_ids)
        movie_embeds = self.movie_embeddings(movie_ids)
        preds = torch.sum(user_embeds * movie_embeds, dim=1)
        return preds.to(device)

@action(name="Action: Convert To ONNX", log_prints=True)
def execution():
    ratings = prepare_dataset()
    # Constants
    num_users = ratings['user_idx'].nunique()
    num_movies = ratings['movie_idx'].nunique()
    embedding_size = 50
    learning_rate = 0.01
    num_epochs = 100 
    batch_size = 64

    train_loader, test_loader = create_loaders(ratings, batch_size)

    
    model = train_model(MF,
                        num_users,
                        num_movies,
                        embedding_size,
                        learning_rate, 
                        device,
                        num_epochs,
                        train_loader)

    criterion = nn.MSELoss()
    test_loss = test_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

    convert_to_onnx(model, num_movies, device, file_path="small-model.onnx")


if __name__ == "__main__":
    action_deploy = Action(entrypoint=execution, name="pytorch-movie_recsys-action")
    action_deploy.serve(name="pytorch-movie_recsys-deployment")