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
HetTensors support any number of nested levels, and any shape of tensors. HetTensor will assign each dimension of the input to one of three categories: batch, stack, and het (represented as an enum). Batch dimensions have a fixed size, and indexing along them yields outputs of the same shape. Stack dimensions have a fixed size, but indexing along them yields outputs of different shapes. Finally, het dimensions have a variable size.
```python
x = HetTensor([[torch.tensor([1,2]), torch.tensor([1]), torch.tensor([3,4,5])],  [torch.tensor([1,3,4]), torch.tensor([2]), torch.tensor([1,2])]])
y = HetTensor([[torch.tensor([1,2]), torch.tensor([0]), torch.tensor([0])],  [[], torch.tensor([3]), torch.tensor([3]), torch.tensor([1,2])]])
z = HetTensor([torch.tensor([[1,2,3],[4,5,6]]),  torch.tensor([[7,8,9]])])
print("x dim types:", x.dim_type)
print("y dim types:", y.dim_type)
print("z dim types:", z.dim_type)
```
> x dim types: [stack, stack, het] <br>
> y dim types: [stack, het, het] <br>
> z dim types: [stack, het, batch]

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
> x shape: torch.Size([2, 3, -1]) <br>
> y shape: torch.Size([2, -1, -1]) <br>
> z shape: torch.Size([2, -1, 3]) <br>
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
