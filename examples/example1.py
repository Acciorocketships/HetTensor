import torch
from hettensor import HetTensor
import hettensor

def run0():
	agent1 = torch.tensor([[0, 1, 1, 2, 3],[1, 2, 3, 3, 0]]).float().requires_grad_(True)
	agent2 = torch.tensor([[0, 0, 1, 2, 2, 3],[1, 2, 1, 3, 2, 0]]).float().requires_grad_(True)
	agent3 = torch.tensor([[1, 2, 2, 3],[2, 0, 3, 0]]).float().requires_grad_(True)

	t = [[agent1, agent2], [], [agent3]]

	het = HetTensor(t)
	x = het[0,0]
	breakpoint()


def run1():
	x = HetTensor([[torch.tensor([1, 2]), torch.tensor([1]), torch.tensor([3, 4, 5])],
				   [torch.tensor([1, 3]), torch.tensor([2]), torch.tensor([1, 2, 3])]])
	y = HetTensor([[torch.tensor([1, 2]), torch.tensor([0]), torch.tensor([0])],
				   [[], torch.tensor([3]), torch.tensor([3]), torch.tensor([1, 2])]])
	z = HetTensor([torch.tensor([[1,2,3],[4,5,6]]), torch.tensor([[7,8,9]])])
	print("x dim types:", x.dim_type)
	print("y dim types:", y.dim_type)
	print("z dim types:", z.dim_type)


def run2():
	x = HetTensor([[torch.tensor([1, 2]), torch.tensor([1]), torch.tensor([3, 4, 5])],
				   [torch.tensor([1, 3, 4]), torch.tensor([2]), torch.tensor([1, 2])]])
	y = HetTensor([[torch.tensor([1, 2]), torch.tensor([0]), torch.tensor([0])],
				   [[], torch.tensor([3]), torch.tensor([3]), torch.tensor([1, 2])]])
	z = HetTensor([torch.tensor([[1,2,3],[4,5,6]]), torch.tensor([[7,8,9]])])
	breakpoint()
	print("x shape:", x.shape)
	print("y shape:", y.shape)
	print("z shape:", z.shape)
	print("x max shape:", x.max_shape)
	print("y max shape:", y.max_shape)
	print("z max shape:", z.max_shape)


def run3():
	x = HetTensor([[torch.tensor([1, 2]), torch.tensor([1]), torch.tensor([3, 4, 5])],
				   [torch.tensor([1, 3, 4]), torch.tensor([2]), torch.tensor([1, 2])]])
	print(x.unbind())
	print(x.to_dense())


def run4():
	x = HetTensor([[torch.tensor([1, 2]), torch.tensor([1]), torch.tensor([3, 4, 5])],
				   [torch.tensor([1, 3, 4]), torch.tensor([2]), torch.tensor([1, 2])]])
	print("Index with HetTensor output:", x[:,0])
	print("Index with torch.Tensor output:", x[:,:,0])


def run5():
	x = HetTensor([[torch.tensor([1, 2]), torch.tensor([1]), torch.tensor([3, 4, 5])],
				   [torch.tensor([1, 3, 4]), torch.tensor([2]), torch.tensor([1, 2])]])
	print(x.sum(dim=0))
	print(x.sum(dim=2))


def run6():
	x = HetTensor([[torch.tensor([1, 2]), torch.tensor([1]), torch.tensor([3, 4, 5])],
				   [torch.tensor([1, 3, 4]), torch.tensor([2]), torch.tensor([1, 2])]])
	print(x.reduce(dim=0, op="amax"))


def run7():
	x = HetTensor([torch.rand(3, 4)])
	print(x)


def run8():
	x = HetTensor([[torch.tensor([1, 2]), torch.tensor([1]), torch.tensor([3, 4, 5])],
				   [torch.tensor([1, 3, 4]), torch.tensor([2]), torch.tensor([1, 2])]])
	x = x + 1
	print(x)


def run9():
	z = HetTensor([torch.tensor([[1,2,3],[4,5,6]]), torch.tensor([[7,8,9]])])
	y = torch.rand(1,1,3)
	x = y + z
	print(x)


def run10():
	x = HetTensor([torch.tensor([[1,2,3],[4,5,6]]), torch.tensor([[7,8,9]])])
	y = HetTensor([torch.tensor([[0.1], [0.2]]), torch.tensor([[0.3]])])
	z = x + y
	print(x.shape)
	print(y.shape)
	print(z)


def run11():
	x = HetTensor([torch.tensor([[1,2,3],[4,5,6]]), torch.tensor([[7,8,9]])])
	y = torch.tensor([[[0.1, 0.2, 0.3]]])
	z = x * y
	print(x.shape)
	print(y.shape)
	print(z)


def run12():
	z = HetTensor([torch.tensor([[1,2,3],[4,5,6]]), torch.tensor([[7,8,9]])])
	# shape: 2 x [2,1] x 3
	idx = HetTensor([torch.tensor([[0,1], [1,0], [1,1]]), torch.tensor([[0,0]])])

	y = z.scatter(idx, batch_dims=[0], index_dim=1)
	print(y)

def run13():
	z = HetTensor([torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[7, 8, 9]])])
	x = torch.rand(2, 1, 3)
	y = z.apply(x, lambda x, y: torch.cat([x, y.expand(x.shape[0], -1)], dim=-1), batch_dims=[0])
	print(z.shape)
	breakpoint()

def run14():
	z = HetTensor([torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[7, 8, 9]])])
	y = z.apply(None, lambda x: x**2, batch_dims=[0])
	print(y)
	breakpoint()


if __name__ == "__main__":
	run14()