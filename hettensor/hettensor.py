import torch
import numpy as np
from enum import IntEnum
from types import GeneratorType
from typing import List, Union
import pprint

class DimType(IntEnum):
	stack = 0
	het = 1
	batch = 2

	def __repr__(self):
		return self.name


def select_index(arr, dim, idx):
	idx_list = [slice(None)] * len(arr.shape)
	idx_list[dim] = idx
	return arr.__getitem__(tuple(idx_list))


class HetTensor:

	def __init__(self, tensor_list=None, data=None, idxs=None, dim_perm=None):
		# Construct with EITHER tensor_list OR data, idxs, dim_perm
		if tensor_list is None:
			self.normal_tensor = (idxs.shape[0] == 0)
			if self.normal_tensor:
				self.data = data
				return
		else:
			self.normal_tensor = isinstance(tensor_list, torch.Tensor)
			if self.normal_tensor:
				self.data = tensor_list
				return
		if tensor_list is not None:
			sizes = self.get_sizes_tree(tensor_list)
			dim_type = self.get_dim_type_tree(tensor_list, sizes)
			if all([d == DimType.batch for d in dim_type]):
				self.normal_tensor = True
				self.data = self.stack_dense(tensor_list)
				return
			else:
				dim_perm = np.argsort(dim_type, kind='stable')
				data, idxs = self.encode(tensor_list, sizes, dim_type, dim_perm)
		self.dim_perm, self.dim_unperm = self.rectify_dim_perm(dim_perm)
		self.data = data
		self.idxs = idxs
		self.dim_type = self.get_dim_type(self.idxs, self.dim_perm)
		self.dim_type_new = [self.dim_type[self.dim_perm[i]] for i in range(len(self.dim_type))]
		self.sizes = self.get_sizes(self.idxs, self.data, self.dim_type)
		self.sizes_new = [self.sizes[self.dim_perm[i]] for i in range(len(self.sizes))]
		self.max_sizes = [self.sizes[i] if (self.dim_type[i] != DimType.het) else self.max_size(i) for i in range(len(self.sizes))]
		self.n_stack = sum(map(lambda x: x == DimType.stack, self.dim_type))
		self.n_het = sum(map(lambda x: x == DimType.het, self.dim_type))
		self.n_batch = sum(map(lambda x: x == DimType.batch, self.dim_type))
		self.n_dim = len(self.dim_perm)
		if self.n_batch == self.dim():
			self.normal_tensor = True




	def stack_dense(self, data):
		if isinstance(data, torch.Tensor):
			return data
		try:
			return torch.stack(data, dim=0)
		except:
			return torch.stack([self.stack_dense(d) for d in data])


	def rectify_dim_perm(self, dim_perm):
		# Fixes the dim_perm and dim_unperm if a dimension is removed (due to indexing)
		dim_unperm = np.argsort(dim_perm, kind="stable")
		dim_perm = np.argsort(dim_unperm, kind="stable")
		return (dim_perm, dim_unperm)


	def get_sizes(self, idxs, data, dim_type):
		if idxs.shape[1] == 0:
			list_sizes = [0] * idxs.shape[0]
		else:
			counts = [torch.unique(idxs[d], dim=0, return_counts=True)[1] for d in range(idxs.shape[0])]
			list_sizes = [len(count) for count in counts]
		dense_sizes = list(data.shape)
		new_sizes = list_sizes + dense_sizes[1:]
		sizes = [new_sizes[self.dim_unperm[i]] for i in range(len(new_sizes))]
		sizes = [sizes[i] if (dim_type[i] != DimType.het) else None for i in range(len(sizes))]
		return sizes


	def get_dim_type(self, idxs, dim_perm):
		if idxs.shape[1] == 0:
			return [DimType.batch] * len(dim_perm)
		dim_type_new = [None] * len(dim_perm)
		for d in range(len(dim_perm)):
			if d >= idxs.shape[0]:
				dim_type_new[d] = DimType.batch
				continue
			max_size = torch.amax(idxs[d])+1
			other_dims = list(range(idxs.shape[0]))
			other_dims.remove(d)
			slices = [idxs[other_dims][:,idxs[d]==k] for k in range(max_size)]
			same_shape = all([s.shape == slices[0].shape for s in slices])
			same_data = True
			if same_shape:
				same_data = all([torch.all(slices[0] == s) for s in slices])
			is_batch = (same_shape and same_data)
			is_stack = False
			if not is_batch:
				is_stack = all([self.is_dense(s) for s in slices])
			if is_batch:
				dim_type_new[d] = DimType.batch
			elif is_stack:
				dim_type_new[d] = DimType.stack
			else:
				dim_type_new[d] = DimType.het
		dim_type = [dim_type_new[dim_perm[i]] for i in range(len(dim_type_new))]
		return dim_type


	def is_dense(self, idxs):
		if idxs.numel() == 0:
			return True
		dim_sizes = torch.amax(idxs, dim=1)+1
		expected_shape = torch.prod(dim_sizes)
		unique_idxs = torch.unique(idxs, dim=1)
		is_dense = (unique_idxs.shape[1] == expected_shape)
		return is_dense


	def get_sizes_tree(self, tensor_list: Union[List, torch.Tensor]):
		sizes, _ = self.sizes_helper(tensor_list, dims=[], cur_level=0, next_level=0)
		sizes = [x if (x != 0) else None for x in sizes]
		return sizes


	def sizes_helper(self, tensor_list, dims=[], cur_level=0, next_level=0):
		while len(dims) <= cur_level:
			dims += [-1]
		if isinstance(tensor_list, torch.Tensor):
			end_level = cur_level + tensor_list.dim()
			while len(dims) < end_level:
				dims += [-1]
			next_level = max(next_level, end_level)
			cur_dims = np.array(dims[cur_level:end_level])
			new_dims = np.array(tensor_list.shape)
			cur_dims[cur_dims==-1] = new_dims[cur_dims==-1]
			cur_dims[cur_dims != new_dims] = 0
			dims[cur_level:end_level] = cur_dims
		elif isinstance(tensor_list, list):
			if dims[cur_level] == -1:
				dims[cur_level] = len(tensor_list)
			if dims[cur_level] != len(tensor_list):
				dims[cur_level] = 0
			next_level = cur_level+1
			for t in tensor_list:
				dims, next_level = self.sizes_helper(t, dims, cur_level+1, next_level)
		return dims, next_level


	def get_dim_type_tree(self, tensor_list, sizes):
		batch, _ = self.dim_type_helper(tensor_list, sizes=sizes, batch=[True]*len(sizes), cur_level=0)
		batch = [DimType.het if ((sizes[i] is None) or (not x)) else DimType.batch for (i, x) in enumerate(batch)]
		return batch


	def dim_type_helper(self, tensor_list, sizes, batch, cur_level):
		if isinstance(tensor_list, torch.Tensor):
			end_level = cur_level + tensor_list.dim()
			tensor_sizes = [None]
			for i, level in enumerate(range(cur_level, end_level)):
				if sizes[level] == None:
					tensor_sizes.append(tensor_list.shape[i])
			tensor_sizes = tuple(tensor_sizes)
			return batch, tensor_sizes
		elif isinstance(tensor_list, list):
			tensor_sizes = []
			for t in tensor_list:
				batch, subgraph = self.dim_type_helper(t, sizes, batch, cur_level+1)
				tensor_sizes.append(subgraph)
			tensor_sizes = tuple(tensor_sizes)
			if sizes[cur_level] != None:
				all_equal = (len(set(tensor_sizes)) <= 1)
				batch[cur_level] = all_equal
			return batch, tensor_sizes


	def get_new_dim(self, dim, dim_unperm=None):
		if dim_unperm is None:
			dim_unperm = self.dim_unperm
		if isinstance(dim, GeneratorType):
			dim = list(dim)
		if isinstance(dim, list):
			return [self.get_new_dim(d, dim_unperm=dim_unperm) for d in dim]
		return dim_unperm[dim]


	def get_dense_dim(self, dim, idxs=None, dim_unperm=None):
		if idxs is None:
			idxs = self.idxs
		if dim_unperm is None:
			dim_unperm = self.dim_unperm
		if isinstance(dim, GeneratorType):
			dim = list(dim)
		if isinstance(dim, list):
			return [self.get_dense_dim(d, idxs=idxs) for d in dim]
		# dense_dim = self.get_new_dim(dim) - self.n_het
		dense_dim = self.get_new_dim(dim, dim_unperm=dim_unperm) - idxs.shape[0]
		if dense_dim < 0:
			dense_dim = None
		return dense_dim


	def encode(self, tensor_list, sizes, dim_type, dim_perm):

		n_batch = sum(map(lambda x: x == DimType.batch, dim_type))
		dim_unperm = np.argsort(dim_perm, kind='stable')

		def get_new_dim(dim):
			if isinstance(dim, GeneratorType):
				dim = list(dim)
			if isinstance(dim, list):
				return [get_new_dim(d) for d in dim]
			return dim_unperm[dim]

		def flatten(tensor_list, cur_level=0, par_idx=None, out=None, idx=None):
			if out is None:
				out = []
			if idx is None:
				idx = []
			if par_idx is None:
				par_idx = []
			if isinstance(tensor_list, torch.Tensor):
				end_level = cur_level + tensor_list.dim()
				new_dim = get_new_dim(range(cur_level, end_level))
				perm = np.argsort(new_dim, kind="stable")
				out.append(tensor_list.permute(*perm))
				idx.append(par_idx)
				return out, idx
			elif isinstance(tensor_list, list):
				for i, t in enumerate(tensor_list):
					out, idx = flatten(t, cur_level + 1, par_idx + [i], out, idx)
			return out, idx

		out, idx = flatten(tensor_list)
		list_dim = len(idx[0])
		d = 0
		k = 0
		while d < list_dim:
			if dim_type[k] == DimType.batch:
				idx_d = [i[d] for i in idx]
				perm = np.argsort(idx_d, kind="stable")
				out = [torch.stack([out[perm[i]+j*len(idx)//sizes[d]] for j in range(sizes[d])], dim=-1) for i in range(len(idx)//sizes[d])]
				idx = [idx[i][:d]+idx[i][d+1:] for i in range(len(idx)//sizes[d])]
				list_dim = len(idx[0])
				assert get_new_dim(k) == list_dim + out[0].dim() - 1, "oops, dimension order changed"
			else:
				d += 1
			k += 1

		tensor_dim = out[0].dim() - n_batch
		list_idx = torch.as_tensor(idx)
		tensor_idx = torch.as_tensor([list(t.shape[:tensor_dim]) for t in out])
		tensor_idx_tot = tensor_idx.prod(dim=1)

		list_idx_list = list_idx.repeat_interleave(tensor_idx_tot, dim=0)

		tensor_idx_list = [None]*tensor_idx.shape[1]
		for j in range(tensor_idx.shape[1]):
			unsqueezed = torch.ones_like(tensor_idx)
			unsqueezed[:,j] = tensor_idx[:,j]
			x5 = []
			for i in range(tensor_idx.shape[0]):
				x1 = torch.arange(tensor_idx[i,j])
				x2 = x1.view(*unsqueezed[i,:].tolist())
				x3 = x2.expand(*tensor_idx[i,:].tolist())
				x4 = x3.reshape(-1)
				x5.append(x4)
			tensor_idx_list[j] = torch.cat(x5, dim=0)
		tensor_idx_list = torch.stack(tensor_idx_list, dim=-1)

		idxs = torch.cat([list_idx_list, tensor_idx_list], dim=-1).T
		data = torch.cat([o.reshape(-1, *list(o.shape[tensor_dim:])) for o in out], dim=0)

		non_batch_perm = dim_perm[dim_perm < (len(dim_perm)-n_batch)]
		idxs = idxs[non_batch_perm,:]

		return data, idxs


	def is_tensor(self):
		return all([self.dim_type[i] == DimType.batch for i in range(len(self.dim_type))])


	def max_size(self, dim, data=None, idxs=None, dim_unperm=None):
		data = data if (data is not None) else self.data
		idxs = idxs if (idxs is not None) else self.idxs
		dim_unperm = dim_unperm if (dim_unperm is not None) else self.dim_unperm
		dense_dim = self.get_dense_dim(dim, idxs=idxs, dim_unperm=dim_unperm)
		if dense_dim is not None:
			return data.shape[dense_dim+1]
		return torch.max(idxs[self.get_new_dim(dim, dim_unperm=dim_unperm),:], dim=0).values.item()+1


	def unbind(self):
		if self.normal_tensor:
			return self.data
		if self.is_tensor():
			tensor = self.data.reshape(*self.sizes_new)
			if tensor.shape[0] == 0 and len(tensor.shape) == 1:
				return tensor
			tensor = tensor.permute(*self.dim_unperm)
			return tensor
		else:
			out = [None] * (self.max_size(0))
			for k in range(self.max_size(0)):
				tensor = self[k]
				if isinstance(tensor, HetTensor):
					tensor = tensor.unbind()
				out[k] = tensor
			return out


	def __getitem__(self, item):
		if self.normal_tensor:
			return self.data[item]
		if not isinstance(item, tuple):
			item = (item,)
		data = self.data
		idxs = self.idxs
		dim_perm = self.dim_perm.tolist()
		del_dim = []
		for d, s in enumerate(item):
			if s == slice(None):
				continue
			if isinstance(s, int):
				del_dim.append(d)
			dense_dim = self.get_dense_dim(d)
			if dense_dim is not None:
				data = select_index(data, dense_dim+1, s)
			else:
				dim = self.get_new_dim(d)
				if isinstance(s, slice):
					if s.start is not None:
						mask_low = idxs[dim,:] >= s.start
						mask = mask_low
					if s.stop is not None:
						mask_high = idxs[dim,:] < s.stop
						mask = mask_high
					if (s.start is not None) and (s.stop is not None):
						mask = mask_low & mask_high
					data = data[mask]
					idxs = idxs[:,mask]
				elif isinstance(s, int):
					max_size = self.max_size(d)
					if s >= max_size:
						raise ValueError(f'Index {s} for dimension {d} of size {max_size} is out of bounds.')
					mask = idxs[dim,:] == s
					data = data[mask]
					idxs = idxs[:, mask]
		remaining_dims = list(range(idxs.shape[0]))
		for d in del_dim:
			dim_perm.remove(d)
			new_dim = self.get_new_dim(d)
			if self.get_dense_dim(d) is None:
				remaining_dims.remove(new_dim)
		idxs = idxs[remaining_dims]
		if len(remaining_dims) == 0:
			return data
		out = HetTensor(data=data, idxs=idxs, dim_perm=dim_perm)
		if out.is_tensor():
			return out.unbind()
		else:
			return out


	def __repr__(self):
		return pprint.pformat(self.unbind(), indent=4)


	@property
	def shape(self):
		return torch.Size([s if (s is not None) else -1 for s in self.sizes])


	@property
	def max_shape(self):
		return torch.Size(self.max_sizes)


	def to_sparse(self, data=None, idxs=None, dim_perm=None):
		# Casts to torch sparse format (permuted to the new dimensions, as the dense dims must be last)
		data = data if (data is not None) else self.data
		idxs = idxs if (idxs is not None) else self.idxs
		if dim_perm is None:
			dim_perm = self.dim_perm
			dim_unperm = self.dim_unperm
		else:
			dim_unperm = np.argsort(dim_perm, kind="stable")
		max_shape = [self.max_size(dim_perm[i], data=data, idxs=idxs, dim_unperm=dim_unperm) for i in range(len(dim_perm))]
		sparse = torch.sparse_coo_tensor(idxs, data, max_shape)
		return sparse


	def to_dense(self):
		# Casts to a normal tensor padded with zeros
		sparse = self.to_sparse()
		dense = sparse.to_dense()
		return dense.permute(*self.dim_unperm)


	def reduce(self, dim=None, op="sum"):
		dense_dim = self.get_dense_dim(dim)
		if dense_dim is not None:
			data = self.data.sum(dim=dense_dim+1)
			idxs = self.idxs
			dim_perm = self.dim_perm.tolist()
			dim_perm.remove(dim)
			out = HetTensor(data=data, idxs=idxs, dim_perm=dim_perm)
			return out
		else:
			new_dim = self.get_new_dim(dim)
			idxs = torch.cat([self.idxs[:new_dim], self.idxs[new_dim+1:]], dim=0)
			idxs, sum_idx = torch.unique(idxs, dim=1, return_inverse=True)
			sum_idx_exp = sum_idx.view(len(sum_idx), *([1]*(self.data.dim()-1))).expand(-1, *self.data.shape[1:])
			data = torch.zeros(idxs.shape[1], *self.data.shape[1:], dtype=self.data.dtype)
			data = data.scatter_reduce_(0, sum_idx_exp, self.data, reduce=op, include_self=False)
			dim_perm = self.dim_perm.tolist()
			dim_perm.remove(dim)
			out = HetTensor(data=data, idxs=idxs, dim_perm=dim_perm)
			return out


	def sum(self, dim):
		if self.normal_tensor:
			return self.data.sum(dim=dim)
		if dim is None:
			return torch.sum(self.data)
		return self.reduce(dim=dim, op="sum")


	def mean(self, dim):
		if self.normal_tensor:
			return self.data.mean(dim=dim)
		if dim is None:
			return torch.mean(self.data)
		return self.reduce(dim=dim, op="mean")


	def prod(self, dim):
		if self.normal_tensor:
			return self.data.prod(dim=dim)
		if dim is None:
			return torch.prod(self.data)
		return self.reduce(dim=dim, op="prod")


	def amax(self, dim):
		if self.normal_tensor:
			return self.data.amax(dim=dim)
		if dim is None:
			return torch.amax(self.data)
		return self.reduce(dim=dim, op="amax")


	def amin(self, dim):
		if self.normal_tensor:
			return self.data.amin(dim=dim)
		if dim is None:
			return torch.amin(self.data)
		return self.reduce(dim=dim, op="amin")


	def combine(self, other, op):
		if isinstance(other, HetTensor) and other.normal_tensor:
			other = other.data
		if self.normal_tensor and isinstance(other, HetTensor) and (not other.normal_tensor):
			return other.__add__(self)
		if self.normal_tensor and isinstance(other, torch.Tensor):
			return op(self.data, other)
		if isinstance(other, int) or isinstance(other, float) or (isinstance(other, torch.Tensor) and other.dim()==0):
			data = op(self.data, other)
			return HetTensor(data=data, idxs=self.idxs, dim_perm=self.dim_perm)
		if isinstance(other, torch.Tensor):
			assert self.dim() == other.dim(), "Tensors must have the same number of dimensions"
			perm = [None] * self.n_batch
			k = self.n_batch-1
			for d in reversed(range(other.dim())):
				dense_dim = self.get_dense_dim(d)
				if dense_dim is None:
					assert other.shape[d] == 1, "Dimensions corresponding to Het dimensions must be 1 in order to be broadcast"
					other = other.squeeze(d)
				else:
					perm[dense_dim] = k
					k -= 1
			other = other.squeeze().permute(*perm)
			data = op(self.data, other)
			return HetTensor(data=data, idxs=self.idxs, dim_perm=self.dim_perm)
		if isinstance(other, HetTensor):
			assert self.dim() == other.dim(), "Tensors must have the same number of dimensions"
			assert self.data.dim() == other.data.dim(), "Tensors must have the same number of dense dims"
			self_dim_perm = self.dim_unperm[-self.n_batch:] - (self.dim() - self.n_batch) + 1
			other_dim_perm = other.dim_unperm[-other.n_batch:] - (self.dim() - self.n_batch) + 1
			self_data = self.data.permute(0, *self_dim_perm)
			other_data = other.data.permute(0, *other_dim_perm)
			for d in range(1,self.data.dim()):
				if self.data.shape[d] == 1 and other.data.shape[d] != 1:
					exp_size = [-1] * self.data.dim()
					exp_size[d] = other.data.shape[d]
					self_data = self_data.expand(*exp_size)
				if other.data.shape[d] == 1 and self.data.shape[d] != 1:
					exp_size = [-1] * other.data.dim()
					exp_size[d] = self.data.shape[d]
					other_data = other_data.expand(*exp_size)
			sparse_self = self.to_sparse(data=self_data)
			sparse_other = other.to_sparse(data=other_data)
			result = op(sparse_self, sparse_other)
			result = result.coalesce()
			result_data = result.values()
			result_idxs = result.indices()
			data_unperm = np.argsort(self_dim_perm, kind="stable") + 1
			result_data = result_data.permute(0, *data_unperm)
			out = HetTensor(data=result_data, idxs=result_idxs, dim_perm=self.dim_perm)
			return out


	def __add__(self, other):
		return self.combine(other, lambda x, y: x + y)


	def __radd__(self, other):
		return self + other


	def __mul__(self, other):
		return self.combine(other, lambda x, y: x * y)


	def __rmul__(self, other):
		return self * other


	def dim(self):
		return len(self.sizes)


	def unsqueeze(self, dim):
		if dim == self.dim():
			dim = -1
		if dim == -1:
			data = self.data.unsqueeze(-1)
			dim_perm = np.insert(self.dim_perm, self.dim(), self.dim())
			return HetTensor(data=data, idxs=self.idxs, dim_perm=dim_perm)
		dense_dim = self.get_dense_dim(dim)
		new_dim = self.get_new_dim(dim)
		if dense_dim is not None:
			data = self.data.unsqueeze(dense_dim+1)
			dim_perm = np.insert(self.dim_perm, new_dim, dim)
			return HetTensor(data=data, idxs=self.idxs, dim_perm=dim_perm)
		else:
			dim_perm = self.dim_perm.copy()
			dim_perm[dim_perm >= dim] += 1
			dim_perm = np.insert(dim_perm, new_dim, dim)
			idxs = torch.cat([self.idxs[:new_dim], torch.zeros(1,self.idxs.shape[1]), self.idxs[new_dim:]], dim=0).int()
			return HetTensor(data=self.data, idxs=idxs, dim_perm=dim_perm)


	def scatter(self, idxs, batch_dims=[], index_dim=1, data=None, src_sink=True):
		if data is None:
			data = self
		if len(batch_dims) > 0:
			assert idxs.max_shape[0] == data.max_shape[0]
			dim_size = data.max_shape[batch_dims[0]]
			cat_list = [None] * dim_size
			for i in range(dim_size):
				sub_mask = idxs.idxs[0,:]==i
				sub_idxs = HetTensor(data=idxs.data[sub_mask], idxs=idxs.idxs[1:, sub_mask], dim_perm=idxs.dim_perm[1:])
				if sub_idxs.normal_tensor:
					sub_idxs = sub_idxs.data
				sub_data = select_index(data, batch_dims[0], i)
				sub_index_dim = index_dim if batch_dims[0] >= index_dim else index_dim-1
				sub_scatter = data.scatter(sub_idxs, data=sub_data, index_dim=sub_index_dim)
				cat_list[i] = sub_scatter.unsqueeze(0)
			out = cat(cat_list, batch_dims[0])
			return out
		else:
			assert isinstance(idxs, torch.Tensor)
			if src_sink:
				idxsi = idxs[:, 0]
				idxsj = idxs[:, 1]
			else:
				idxsi = idxs[:, 1]
				idxsj = idxs[:, 0]
			if isinstance(data, torch.Tensor):
				new_data = select_index(data, index_dim, idxsi)
				new_dim_perm = list(range(new_data.dim()+1))
				idxs_elements = running_counts(idxsj)
				idxs_new = torch.stack([idxsj, idxs_elements], dim=0)
				out = HetTensor(data=new_data, idxs=idxs_new, dim_perm=new_dim_perm)
				return out
			elif isinstance(data, HetTensor):
				raise NotImplementedError()


	def apply(self, x, op, batch_dims=[], x_self=None):
		if x_self is None:
			x_self = self
		if len(batch_dims) > 0:
			if isinstance(x_self, HetTensor):
				dim_size = x_self.max_shape[batch_dims[0]]
			else:
				dim_size = x_self.shape[batch_dims[0]]
			cat_list = [None] * dim_size
			for i in range(dim_size):
				sub_self = select_index(x_self, batch_dims[0], i)
				if x is not None:
					sub_x = select_index(x, batch_dims[0], i)
				else:
					sub_x = None
				cat_list[i] = self.apply(sub_x, op, batch_dims[1:], x_self=sub_self)
			new_data = cat(cat_list, dim=batch_dims[0])
			out = HetTensor(data=new_data, idxs=self.idxs, dim_perm=self.dim_perm)
			return out
		else:
			if isinstance(x_self, HetTensor):
				if x is not None:
					new_data = op(x_self.data, x)
				else:
					new_data = op(x_self.data)
				return HetTensor(data=new_data, idxs=x_self.idxs, dim_perm=x_self.dim_perm)
			else:
				if x is not None:
					return op(x_self, x)
				else:
					return op(x_self)


def cat(het_list, dim=0):
	if not any([isinstance(x, HetTensor) for x in het_list]):
		return torch.cat(het_list, dim=dim)
	max_shapes = torch.tensor([x.max_shape[dim] for x in het_list])
	max_shapes_cum = torch.cat([torch.tensor([0]), torch.cumsum(max_shapes, dim=0)], dim=0)
	idxs = [x.idxs.clone() for x in het_list]
	data = [x.data.clone() for x in het_list]
	for i, x in enumerate(idxs):
		x[dim] += max_shapes_cum[i]
	idxs_new = torch.cat(idxs, dim=1)
	data_new = torch.cat(data)
	out = HetTensor(idxs=idxs_new, data=data_new, dim_perm=het_list[0].dim_perm)
	return out


def running_counts(idxs):
	perm = torch.argsort(idxs, stable=True)
	unperm = torch.argsort(perm, stable=True)
	idxsj_sorted = idxs[perm]
	_, counts_idxsj = torch.unique(idxsj_sorted, return_counts=True)
	arange = torch.cat([torch.arange(count) for count in counts_idxsj], dim=0)
	counts = arange[unperm]
	return counts