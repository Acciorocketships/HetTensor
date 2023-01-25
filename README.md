# HetTensor

A wrapper for PyTorch Tensors which enables variable-sized dimensions.

## Installation

```bash
pip install -e HetTensor
```

## Features

### Printing
HetTensors can be constructed with a nested list of torch.Tensor. Printing a HetTensor displays it in this nested form.

```python
from hettensor import HetTensor
import torch

x = HetTensor([[torch.tensor([1,2]), torch.tensor([3])], [torch.tensor([4,5,6])]])
print(x)
```
> [[tensor([1, 2]), tensor([3])], tensor([[4, 5, 6]])]

### Dim Types
HetTensors support any number of nested levels, and any shape of tensors. HetTensor will assign each dimension of the input to one of two categories: batch and het (represented as an enum). Batch dimensions have a fixed size, and indexing along them yields slices of the same shape. On the other hand, indexing along a het dimension will result in slices of different shapes. Equivalently, indexing across other het dimensions will yield varying sizes in a het dimension.
```python
x = HetTensor([[torch.tensor([1,2]), torch.tensor([1]), torch.tensor([3,4,5])],  [torch.tensor([1,3,4]), torch.tensor([2]), torch.tensor([1,2])]])
y = HetTensor([[torch.tensor([1,2]), torch.tensor([0]), torch.tensor([0])],  [[], torch.tensor([3]), torch.tensor([3]), torch.tensor([1,2])]])
z = HetTensor([torch.tensor([[1,2,3],[4,5,6]]),  torch.tensor([[7,8,9]])])
print("x dim types:", x.dim_type)
print("y dim types:", y.dim_type)
print("z dim types:", z.dim_type)
```
> x dim types: [het, het, het] <br>
> y dim types: [het, het, het] <br>
> z dim types: [het, het, batch]

### Shape
Once instantiated, the shape of a HetTensor can be queried with x.shape, where the heterogeneous dimensions are given values of -1. To retrieve the maximum size along the heterogeneous dimensions, used x.max_shape (which represents the shape of the equivalent zero-padded dense tensor).
```python
x = HetTensor([[torch.tensor([1,2]), torch.tensor([1]), torch.tensor([3,4,5])],  [torch.tensor([1,3,4]), torch.tensor([2]), torch.tensor([1,2])]])
y = HetTensor([[torch.tensor([1,2]), torch.tensor([0]), torch.tensor([0])],  [[], torch.tensor([3]), torch.tensor([3]), torch.tensor([1,2])]])
z = HetTensor([torch.tensor([[1,2,3],[4,5,6]]),  torch.tensor([[7,8,9]])])
print("x shape:", x.shape)
print("y shape:", y.shape)
print("z shape:", z.shape)
print("x max shape:", x.max_shape)
print("y max shape:", y.max_shape)
print("z max shape:", z.max_shape)
```
> x shape: torch.Size([-1, -1, -1]) <br>
> y shape: torch.Size([-1, -1, -1]) <br>
> z shape: torch.Size([-1, -1, 3]) <br>
> x max shape: torch.Size([2, 3, 3]) <br>
> y max shape: torch.Size([2, 4, 2]) <br>
> z max shape: torch.Size([2, 2, 3])

### Conversions
A HetTensor can be converted back into a nested list of tensors with x.unbind(), or it can be converted to a padded dense tensor with x.to_dense().
```python
x = HetTensor([[torch.tensor([1,2]), torch.tensor([1]), torch.tensor([3,4,5])],  [torch.tensor([1,3,4]), torch.tensor([2]), torch.tensor([1,2])]])
print(x.unbind())
print(x.to_dense())
```
> [[tensor([1, 2]), tensor([1]), tensor([3, 4, 5])], [tensor([1, 3, 4]), tensor([2]), tensor([1, 2])]] <p>
> tensor([[[1, 2, 0], <br>
>          [1, 0, 0], <br>
>          [3, 4, 5]], <p>
>         [[1, 3, 4],<br>
>          [2, 0, 0], <br>
>          [1, 2, 0]]])

### Indexing
HetTensors can be indexed just like normal tensors. If there are no variable-size dimensions in the result (which occurs when all het dimensions are indexed out), then a normal tensor is returned.
```python
x = HetTensor([[torch.tensor([1,2]), torch.tensor([1]), torch.tensor([3,4,5])],  [torch.tensor([1,3,4]), torch.tensor([2]), torch.tensor([1,2])]])
print("Index with HetTensor output:", x[:,0])
print("Index with torch.Tensor output:", x[:,:,0])
```
> [tensor([1, 2]), tensor([1, 3, 4])] <p>
> tensor([[1, 1, 3], <br>
>         [1, 2, 1]])

### Reduce Ops
HetTensors can be reduced (with sum, mean, prod, amax, amin) along any dimension. As with indexing, if the result is a dense tensor, then a normal tensor is returned.
```python
x = HetTensor([[torch.tensor([1,2]), torch.tensor([1]), torch.tensor([3,4,5])],  [torch.tensor([1,3,4]), torch.tensor([2]), torch.tensor([1,2])]])
print(x.sum(dim=0))
print(x.sum(dim=2))
```
> [tensor([2, 5, 4]), tensor([3]), tensor([4, 6, 5])] <p>
> tensor([[ 3,  1, 12], <br>
>         [ 8,  2,  3]])

### Operations
Operations such as + and * are implemented between two HetTensors, or between HetTensors and normal tensors. Broadcasting is also enabled, so dimensions of size 1 will be repeated across the all elements in the corresponding dimension of the other tensor.
```python
  x = HetTensor([torch.tensor([[1,2,3],[4,5,6]]), torch.tensor([[7,8,9]])])
	y = HetTensor([torch.tensor([[0.1], [0.2]]), torch.tensor([[0.3]])])
	z = x + y
  print(x.shape, y.shape)
	print(z)
  
  x = HetTensor([torch.tensor([[1,2,3],[4,5,6]]), torch.tensor([[7,8,9]])])
	y = torch.tensor([[[0.1, 0.2, 0.3]]])
  print(x.shape, y.shape)
	z = x * y
  
	print(z)
```
> torch.Size([-1, -1, 3]) torch.Size([-1, -1, 1])
> [   tensor([[1.1000, 2.1000, 3.1000], <br>
>             [4.2000, 5.2000, 6.2000]]), <br>
>     tensor([[7.3000, 8.3000, 9.3000]])] <p>
> torch.Size([-1, -1, 3]) torch.Size([1, 1, 3])
> [   tensor([[0.1000, 0.4000, 0.9000], <br>
>             [0.4000, 1.0000, 1.8000]]), <br>
>     tensor([[0.7000, 1.6000, 2.7000]])]
