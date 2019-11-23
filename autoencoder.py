import torch
import torchvision
import model
import util
import os


class Autoencoder:

    def __init__(self, in_path=None, num_epochs=100, batch_size=50, learning_rate=0.0001, name=None, verbose=False):
        """
        This class implements the training procedure for the autoencoders.
        :param in_path: (string) the file path indicating the location of the training data.
        :param num_epochs: (int) the number of epochs.
        :param batch_size: (int) the batch size.
        :param learning_rate: (int) the learning rate for the Adam optimizer.
        :param name: (string) the name of the model (used when saving the parameters to file)
        :param verbose: (boolean) if true, the training process is printed to console
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = torch.cuda.is_available()
        self.in_path = in_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.name = name

        self.net = model.AutoEncoder().cuda() if self.use_cuda else model.AutoEncoder()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.losses = []
        self.verbose = verbose
        self.start_epoch = 1

    def train(self):
        """
        Train the autoencoder.
        """
        training_data = torchvision.datasets.ImageFolder(self.in_path, torchvision.transforms.Compose([
                                                         torchvision.transforms.ToTensor()]))

        data_loader = torch.utils.data.DataLoader(training_data, batch_size=self.batch_size, shuffle=True, num_workers=2)

        criterion = torch.nn.L1Loss().cuda() if self.use_cuda else torch.nn.L1Loss()

        progress_bar = util.ProgressBar()

        if not os.path.exists('saves/'):
            os.makedirs('saves/')

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            print('Epoch {}/{}'.format(epoch, self.num_epochs))
            for i, data in enumerate(data_loader, 1):
                x, _ = data
                x = x.to(self.device)

                output = self.net(x)

                loss = criterion(output, x)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.losses.append(loss)

                if self.verbose and (i % 10 == 0 or i == len(data_loader) - 1):
                    info_str = 'loss: {:.4f}'.format(self.losses[-1])
                    progress_bar.update(max_value=len(data_loader), current_value=i + 1, info=info_str)

            progress_bar.new_line()

            self.save(epoch=epoch, path='saves/' + self.name + '_' + str(epoch) + '.pth')

    def encode(self, x):
        return self.net.encoder(x)

    def decode(self, z):
        return self.net.decoder(z)

    def save(self, epoch, path):
        """
        Saves the autoencoder to the specified file path.
        :param epoch: (int) current epoch
        :param path: (string) file path
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses
        }, path)

    def load(self, path):
        """
        Loads the weights of the autoencoder from the file path.
        :param path: (string) file path
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.losses = checkpoint['losses']
        self.start_epoch = checkpoint['epoch']

    def eval(self):
        """
        Sets the autoencoder to evaluation mode.
        This is not necessary but it saves time for evaluations.
        """
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False
