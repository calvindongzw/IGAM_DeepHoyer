#Data input

import torchvision
import torch

import torchvision.transforms as transforms

class CIFAR10_Raw:

	def __init__(self, path, train, download=False, batch_size=4, shuffle=True, num_workers=0, pad=2, image_size=32, flip_rate=1):

		self.padding = (pad, pad, pad, pad)
		self.image_size = image_size

		self.transform = transforms.Compose(
						[transforms.ToTensor()
						,transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
						])
		self.trainset = torchvision.datasets.CIFAR10(root=path, train=train, download=download, transform=self.transform)
		self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)

class CIFAR10_Augmented:

	def __init__(self, path, train, download=False, batch_size=4, shuffle=True, num_workers=0, pad=2, image_size=32, flip_rate=1):

		self.padding = (pad, pad, pad, pad)
		self.image_size = image_size

		self.transform = transforms.Compose(
						[transforms.Pad(self.padding, fill=0, padding_mode='constant')
						,transforms.RandomCrop(self.image_size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
						,transforms.RandomHorizontalFlip(p=flip_rate)
						,transforms.ToTensor()
						,transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
						])
		self.trainset = torchvision.datasets.CIFAR10(root=path, train=train, download=download, transform=self.transform)
		self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)

class CIFAR100_Raw:

	def __init__(self, path, train, download=False, batch_size=4, shuffle=True, num_workers=0, pad=2, image_size=32, flip_rate=1):

		self.padding = (pad, pad, pad, pad)
		self.image_size = image_size

		self.transform = transforms.Compose(
						[transforms.ToTensor()
						,transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
						])
		self.trainset = torchvision.datasets.CIFAR100(root=path, train=train, download=download, transform=self.transform)
		self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)

class CIFAR100_Augmented:

	def __init__(self, path, train, download=False, batch_size=4, shuffle=True, num_workers=0, pad=2, image_size=32, flip_rate=1):

		self.padding = (pad, pad, pad, pad)
		self.image_size = image_size

		self.transform = transforms.Compose(
						[transforms.Pad(self.padding, fill=0, padding_mode='constant')
						,transforms.RandomCrop(self.image_size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
						,transforms.RandomHorizontalFlip(p=flip_rate)
						,transforms.ToTensor()
						,transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
						])
		self.trainset = torchvision.datasets.CIFAR100(root=path, train=train, download=download, transform=self.transform)
		self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)


class MNIST_Raw:

	def __init__(self, path, train, download=False, batch_size=4, shuffle=True, num_workers=0, pad=2, image_size=28, flip_rate=1):

		self.padding = (pad, pad, pad, pad)
		self.image_size = image_size

		self.transform = transforms.Compose([transforms.ToTensor()])
		self.trainset = torchvision.datasets.CIFAR100(root=path, train=train, download=download, transform=self.transform)
		self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)



class MNIST_Augmented:

	def __init__(self, path, train, download=False, batch_size=4, shuffle=True, num_workers=0, pad=2, image_size=28, flip_rate=1):

		self.padding = (pad, pad, pad, pad)
		self.image_size = image_size
		self.transform = transforms.Compose(
						[transforms.Pad(self.padding, fill=0, padding_mode='constant')
						,transforms.RandomCrop(self.image_size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
						,transforms.RandomHorizontalFlip(p=flip_rate)
						,transforms.ToTensor()
						])
		self.trainset = torchvision.datasets.CIFAR100(root=path, train=train, download=download, transform=self.transform)
		self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)









