import torch
import torchvision
import torch.autograd as autograd
import model
import autoencoder
import util
import os


class Siamese:

    def __init__(self, in_path_photo=None, in_path_oil=None, autoencoder_path=None, num_epochs=100, batch_size=50,
                 learning_rate=0.0002, recon_loss_weight=10, penalty_coef=10, verbose=True):
        """
        This class implements the siamese architecture.
        :param in_path_photo: (string) the file path indicating the location of the training data for photographs.
        :param in_path_oil: the file path indicating the location of the training data for oil.
        :param autoencoder_path: (string) the path where the save files of the autoencoders are stored.
        :param num_epochs: (int) the number of epochs.
        :param batch_size: (int) the batch size.
        :param learning_rate: (int) the learning rate for the Adam optimizer.
        :param recon_loss_weight: (float) the parameter that scales the cycle consistency / reconstruction loss (beta)
        :param penalty_coef: (float) the penalty coefficient for the Wasserstein GAN (lambda)
        :param verbose: (boolean) if true, the training process is printed to console
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = torch.cuda.is_available()
        self.in_path_photo = in_path_photo
        self.in_path_oil = in_path_oil
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.recon_loss_weight = recon_loss_weight
        self.penalty_coef = penalty_coef
        self.verbose = verbose

        self.start_epoch = 1
        self.d_photo_losses = []
        self.d_oil_losses = []
        self.g_photo_losses = []
        self.g_oil_losses = []
        self.photo_rec_losses = []
        self.oil_rec_losses = []

        self.auto_photo = autoencoder.Autoencoder()
        self.auto_photo.load(autoencoder_path + 'autoencoder_photo_20.pth')
        self.auto_photo.eval()
        self.auto_oil = autoencoder.Autoencoder()
        self.auto_oil.load(autoencoder_path + 'autoencoder_oil_20.pth')
        self.auto_oil.eval()

        self.map_v_to_u = model.Map().cuda() if self.use_cuda else model.Map()
        self.map_u_to_v = model.Map().cuda() if self.use_cuda else model.Map()

        self.v_to_u_optimizer = torch.optim.Adam(self.map_v_to_u.parameters(), lr=learning_rate, betas=(0.5, 0.9))
        self.u_to_v_optimizer = torch.optim.Adam(self.map_u_to_v.parameters(), lr=learning_rate, betas=(0.5, 0.9))

        self.discriminator_photo = model.Discriminator().cuda() if self.use_cuda else model.Discriminator()
        self.discriminator_oil = model.Discriminator().cuda() if self.use_cuda else model.Discriminator()

        self.d_photo_optimizer = torch.optim.Adam(self.discriminator_photo.parameters(), lr=learning_rate,
                                                  betas=(0.5, 0.9))
        self.d_oil_optimizer = torch.optim.Adam(self.discriminator_oil.parameters(), lr=learning_rate, betas=(0.5, 0.9))

    def train(self):
        """
        Trains the architecture.
        """
        training_data_photo = torchvision.datasets.ImageFolder(self.in_path_photo, torchvision.transforms.Compose([
                                                               torchvision.transforms.ToTensor()]))

        data_loader_photo = torch.utils.data.DataLoader(training_data_photo, batch_size=self.batch_size, shuffle=True,
                                                        num_workers=2)

        training_data_oil = torchvision.datasets.ImageFolder(self.in_path_oil, torchvision.transforms.Compose([
                                                               torchvision.transforms.ToTensor()]))

        data_loader_oil = torch.utils.data.DataLoader(training_data_oil, batch_size=self.batch_size, shuffle=True,
                                                      num_workers=2)

        l1_criterion = torch.nn.L1Loss().cuda() if self.use_cuda else torch.nn.L1Loss()

        progress_bar = util.ProgressBar()

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            print('Epoch {}/{}'.format(epoch, self.num_epochs))
            for i, (photo_batch, oil_batch) in enumerate(zip(data_loader_photo, data_loader_oil), 1):
                x_photo, _ = photo_batch
                x_oil, _ = oil_batch
                x_photo = x_photo.to(self.device)
                x_oil = x_oil.to(self.device)

                if x_photo.shape[0] != x_oil.shape[0]:
                    continue

                z_oil, x_rec_photo = self._train_photo_to_oil_and_back(x_photo, l1_criterion)

                v = self.auto_photo.encode(x_photo)
                u_dash = self.map_v_to_u(v)
                z_oil = self.auto_oil.decode(u_dash)

                self._update_discriminator(self.discriminator_oil, self.d_oil_optimizer, x_real=x_oil, x_fake=z_oil,
                                           losses=self.d_oil_losses)

                z_photo, x_rec_oil = self._train_oil_to_photo_and_back(x_oil, l1_criterion)

                u = self.auto_oil.encode(x_oil)
                v_dash = self.map_u_to_v(u)
                z_photo = self.auto_photo.decode(v_dash)

                self._update_discriminator(self.discriminator_photo, self.d_photo_optimizer, x_real=x_photo,
                                           x_fake=z_photo, losses=self.d_photo_losses)

                if self.verbose:
                    info_str = 'd_photo: {:.4f}, d_oil: {:.4f}, g_photo: {:.4f}, g_oil: {:.4f}'.\
                               format(self.d_photo_losses[-1], self.d_oil_losses[-1], self.g_photo_losses[-1],
                                      self.g_oil_losses[-1])

                    progress_bar.update(max_value=len(data_loader_oil), current_value=i + 1, info=info_str)

            if not os.path.exists('saves/'):
                os.makedirs('saves/')
            self.save(epoch, path='saves')

            progress_bar.new_line()

    def translate_photo_to_oil(self, x_photo):
        """
        Translates the given image of a photograph
        to an image of an oil paint.
        :param x_photo: (tensor) image of a photograph.
        :return: (tensor) image translated to oil paint.
        """
        v = self.auto_photo.encode(x_photo.detach())
        u_dash = self.map_v_to_u(v.detach())
        z_oil = self.auto_oil.decode(u_dash.detach())
        return z_oil

    def translate_oil_to_photo(self, x_oil):
        """
        Translates the given image of an oil painting
        to an image of a photograph.
        :param x_oil: (tensor) image of an oil painting.
        :return: (tensor) image translated to photograph.
        """
        u = self.auto_oil.encode(x_oil.detach())
        v_dash = self.map_u_to_v(u.detach())
        z_photo = self.auto_photo.decode(v_dash.detach())
        return z_photo

    def _train_photo_to_oil_and_back(self, x_photo, criterion):
        # forward photo
        v = self.auto_photo.encode(x_photo.detach())
        u_dash = self.map_v_to_u(v)

        z_oil = self.auto_oil.decode(u_dash)
        d_fake = self.discriminator_oil(z_oil)
        adversarial_loss = -torch.mean(d_fake)
        self.v_to_u_optimizer.zero_grad()
        adversarial_loss.backward()
        self.v_to_u_optimizer.step()
        self.g_oil_losses.append(adversarial_loss.item())

        u = self.auto_oil.encode(z_oil.detach())
        v_dash = self.map_u_to_v(u)
        x_rec_photo = self.auto_photo.decode(v_dash)
        loss_rec_photo = criterion(x_rec_photo, x_photo)
        loss_rec_photo *= self.recon_loss_weight
        self.u_to_v_optimizer.zero_grad()
        self.v_to_u_optimizer.zero_grad()
        loss_rec_photo.backward()
        self.u_to_v_optimizer.step()
        self.v_to_u_optimizer.step()
        self.photo_rec_losses.append(loss_rec_photo.item())

        return z_oil, x_rec_photo

    def _train_oil_to_photo_and_back(self, x_oil, criterion):
        # forward oil
        u = self.auto_oil.encode(x_oil.detach())
        v_dash = self.map_u_to_v(u)

        z_photo = self.auto_photo.decode(v_dash)
        d_fake = self.discriminator_photo(z_photo)
        adversarial_loss = -torch.mean(d_fake)
        self.u_to_v_optimizer.zero_grad()
        adversarial_loss.backward()
        self.u_to_v_optimizer.step()
        self.g_photo_losses.append(adversarial_loss.item())

        v = self.auto_photo.encode(z_photo.detach())
        u_dash = self.map_v_to_u(v)
        x_rec_oil = self.auto_oil.decode(u_dash)
        loss_rec_oil = criterion(x_rec_oil, x_oil)
        loss_rec_oil *= self.recon_loss_weight
        self.v_to_u_optimizer.zero_grad()
        self.u_to_v_optimizer.zero_grad()
        loss_rec_oil.backward()
        self.v_to_u_optimizer.step()
        self.u_to_v_optimizer.step()
        self.oil_rec_losses.append(loss_rec_oil.item())

        return z_photo, x_rec_oil

    def _update_discriminator(self, discriminator, optimizer, x_real, x_fake, losses):
        """
        Performs a single optimization step for the discriminator.
        :param x_real: (tensor) batch of images from the training set.
        """
        d_real = discriminator(x_real)
        d_fake = discriminator(x_fake)
        gradient_penalty = self._gradient_penalty(discriminator, x_real, x_fake)
        d_loss = torch.mean(d_fake) - torch.mean(d_real) + self.penalty_coef * gradient_penalty

        optimizer.zero_grad()
        d_loss.backward()
        optimizer.step()

        losses.append(d_loss.item())

    def _gradient_penalty(self, discriminator, x_real, x_fake):
        """
        Returns the gradient penalty for the discriminator.
        This function was taken from this GitHub repository: https://github.com/EmilienDupont/wgan-gp
        :param x_real: (tensor) batch of images from the training set.
        :param x_fake: (tensor) batch of generated images.
        :return: (float) gradient penalty.
        """
        alpha = torch.rand(x_real.shape[0], 1, 1, 1)
        alpha = alpha.expand_as(x_real)
        alpha = alpha.to(self.device)

        interpolates = alpha * x_real + (1 - alpha) * x_fake
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        interpolates = interpolates.to(self.device)
        d_interpolates = discriminator(interpolates)

        gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(d_interpolates.shape).to(self.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(x_real.shape[0], -1)

        norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = torch.mean((norm - 1) ** 2)
        return gradient_penalty

    def save(self, epoch, path):
        """
        Saves the parameters of all the trainable networks to file.
        :param epoch: (int) the current epoch.
        :param path: (string) a file path indicating the location where the files should be stored.
        :return:
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.discriminator_photo.state_dict(),
            'optimizer_state_dict': self.d_photo_optimizer.state_dict(),
            'losses': self.d_photo_losses
        }, path + '/discriminator_photo_' + str(epoch) + '.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.discriminator_oil.state_dict(),
            'optimizer_state_dict': self.d_oil_optimizer.state_dict(),
            'losses': self.d_oil_losses
        }, path + '/discriminator_oil_' + str(epoch) + '.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.map_v_to_u.state_dict(),
            'optimizer_state_dict': self.v_to_u_optimizer.state_dict(),
            'g_oil_losses': self.g_oil_losses,
            'oil_rec_losses': self.oil_rec_losses
        }, path + '/map_v_to_u_' + str(epoch) + '.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.map_u_to_v.state_dict(),
            'optimizer_state_dict': self.u_to_v_optimizer.state_dict(),
            'g_photo_losses': self.g_photo_losses,
            'photo_rec_losses': self.photo_rec_losses
        }, path + '/map_u_to_v_' + str(epoch) + '.pth')

    def load(self, epoch, path):
        """
        Loads parameters for all the trainable networks from files.
        :param epoch: (int) the epoch from the saves.
        :param path: (string) a file path indicating the location of the saves.
        :return:
        """
        checkpoint = torch.load(path + '/map_v_to_u_' + str(epoch) + '.pth', map_location=self.device)
        self.map_v_to_u.load_state_dict(checkpoint['model_state_dict'])
        self.v_to_u_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.g_oil_losses = checkpoint['g_oil_losses']
        self.oil_rec_losses = checkpoint['oil_rec_losses']
        self.start_epoch = checkpoint['epoch'] + 1
        checkpoint = torch.load(path + '/map_u_to_v_' + str(epoch) + '.pth', map_location=self.device)
        self.map_u_to_v.load_state_dict(checkpoint['model_state_dict'])
        self.u_to_v_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.g_photo_losses = checkpoint['g_photo_losses']
        self.photo_rec_losses = checkpoint['photo_rec_losses']
        checkpoint = torch.load(path + '/discriminator_photo_' + str(epoch) + '.pth', map_location=self.device)
        self.discriminator_photo.load_state_dict(checkpoint['model_state_dict'])
        self.d_photo_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.d_photo_losses = checkpoint['losses']
        checkpoint = torch.load(path + '/discriminator_oil_' + str(epoch) + '.pth', map_location=self.device)
        self.discriminator_oil.load_state_dict(checkpoint['model_state_dict'])
        self.d_oil_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.d_oil_losses = checkpoint['losses']
