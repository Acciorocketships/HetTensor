import torch
from graphtorch import HetTensor

def run():
	agent1 = torch.tensor([[0, 1, 1, 2, 3],[1, 2, 3, 3, 0]]).float().requires_grad_(True)
	agent2 = torch.tensor([[0, 0, 1, 2, 2, 3],[1, 2, 1, 3, 2, 0]]).float().requires_grad_(True)
	agent3 = torch.tensor([[1, 2, 2, 3],[2, 0, 3, 0]]).float().requires_grad_(True)

	t = [[agent1, agent2], [], [agent3]]

	het = HetTensor(t)
	x = het[:,0]
	breakpoint()


def run2():
	t1 = [[torch.tensor([1,2]), torch.tensor([1]), torch.tensor([3,4,5])],[torch.tensor([1,3,4]),torch.tensor([2]),torch.tensor([1,2])]]
	t2 = [[torch.tensor([1,2]), torch.tensor([0]), torch.tensor([0])],[[],torch.tensor([3]),torch.tensor([3]),torch.tensor([1,2])]]
	t3 = [torch.rand(3,4,5), torch.rand(2,4,2),torch.rand(4,4,1),torch.rand(1,4,2)]
	t4 = {"data": torch.tensor([1,2,0,0,3,3,1,2]), "idxs": torch.tensor([[0,0,0,0,1,1,1,1],[0,0,1,2,1,2,3,3],[0,1,0,0,0,0,0,1]]), "dim_perm": [0,1,2]}
	t = t1
	het = HetTensor(t)
	print(het.dim_type)
	print(het)
	print(het[:,0])


if __name__ == "__main__":
	run()