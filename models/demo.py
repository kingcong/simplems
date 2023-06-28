

import simplems as ms


# x1 = ms.Tensor([1, 2], device=ms.cpu())
# x2 = ms.Tensor([3, 4], device=ms.cpu())
# x3 = ms.Tensor(x2, device=ms.cpu())
# x1.backward(1)
# print(x1.backward(1))

v1 = ms.Tensor([2], dtype="float32", device=ms.cpu())
v2 = ms.ops.exp(v1)
v3 = v2 + 1
v4 = v2 * v3

v5 = ms.NDArray.to(v4, device=ms.cuda())
print(v5.device)

# v4.backward()
# print(v3.grad)
# print(v2)
# print(v4.inputs[0] == v2 and v4.inputs[1] == v3)


