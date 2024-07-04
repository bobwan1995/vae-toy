import os
import argparse
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision.utils import save_image
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils import print
from vae import VAE
from tqdm import tqdm


class Trainer:
    def __init__(self, args):
        self.exp_name = args.exp_name
        self.epochs = args.epochs
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.lr = args.lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = args.image_size
        self.latent_dim = args.latent_dim
        model = VAE(
            image_size=self.image_size,
            latent_dim=self.latent_dim,
        )
        self.model = model.to(self.device)
        if args.ckpt_path:
            # self.model = torch.load(args.ckpt_path, map_location=self.device)
            self.load_pretrained(args.ckpt_path)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.directory = f'output/{self.exp_name}'
        os.makedirs(self.directory, exist_ok=True)
        log_dir = os.path.join(self.directory, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        self.transform = transforms.Compose([
                transforms.Resize(self.image_size, antialias=True),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor()]
        )
        train_dataset = CelebA(args.data_path, transform=self.transform, download=False, split='train')
        test_dataset = CelebA(args.data_path, transform=self.transform, download=False, split='valid')
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.train_batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.test_batch_size, shuffle=False)

    def load_pretrained(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded pretrained weights from {}, Epoch: {}".format(ckpt_path, checkpoint['epoch']))

    def loss_function(self, recon_x, x, mu, log_var):
        recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, 3*self.image_size**2), reduction='sum')
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss, kl_div

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training"):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, log_var = self.model(data)
            recon_loss, kl_div = self.loss_function(recon_batch, data, mu, log_var)
            loss = recon_loss + kl_div
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

            if batch_idx % 1000 == 0:
                print(f'Train Epoch: [{batch_idx}/{len(self.train_loader)}] Loss: {loss.item() / len(data)}')
                self.writer.add_scalar('Loss/train', loss.item() / len(data),
                                       epoch * len(self.train_loader) + batch_idx)
        # torch.save(self.model, f'{self.directory}/vae_model_{epoch}.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
        }, f'{self.directory}/vae_model_{epoch}.pth')

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc="Testing"):
                data = data.to(self.device)
                recon_batch, mu, log_var = self.model(data)
                recon_loss, kl_div = self.loss_function(recon_batch, data, mu, log_var)
                loss = recon_loss + kl_div
                test_loss += loss.item()
                self.writer.add_scalar('Loss/test', loss, epoch * len(self.test_loader) + i)

                if i == 0:
                    save_image(recon_batch.view(self.test_batch_size, 3, self.image_size, self.image_size),
                               os.path.join(self.directory, f'recon_{epoch}.png'))

    def sample(self, epoch):
        with torch.no_grad():
            sample = torch.randn(self.test_batch_size, self.latent_dim).to(self.device)
            sample = self.model.decode(sample).cpu()
            save_image(sample.view(self.test_batch_size, 3, self.image_size, self.image_size),
                       f'{self.directory}/sample_{str(epoch)}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a VAE on CelebA dataset')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--train_batch_size', type=int, default=1, help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1, help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--data_path', type=str, default='./data', help='path to the dataset')
    parser.add_argument('--exp_name', type=str, default='exp', help='exp name')
    parser.add_argument('--ckpt_path', type=str, default='', help='ckpt')
    parser.add_argument('--image_size', type=int, default=150, help='image size')
    parser.add_argument('--latent_dim', type=int, default=128, help='latent dim')
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.sample(0)
    trainer.test(0)
    for epoch in range(trainer.epochs):
        trainer.train(epoch)
        trainer.sample(epoch)
        trainer.test(epoch)
