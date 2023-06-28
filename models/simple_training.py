import sys
sys.path.append('./')
from models import *

device = ndl.cpu()

### CIFAR-10 training ###

def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None, epoch=0):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    if opt:
        model.train()
    else:
        model.eval()

    correct = 0.0
    num_samples = 0.0
    i = 0
    total_loss = 0
    for batch in dataloader:
        X, y = batch[0], batch[1]
        X, y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)

        out = model(X)
        correct += np.sum(np.argmax(out.detach().numpy(), axis=1) == y.numpy())
        loss = loss_fn(out, y)
        total_loss += loss.detach().numpy() * y.shape[0]

        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()
        num_samples += y.shape[0]
        print("epoch = " + str(epoch) + ", batch = " + str(i) + ", train accuracy = " + str(correct/num_samples) + ", train loss = " + str(total_loss / num_samples))
        i += 1
    return correct / num_samples, total_loss / num_samples
    

def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)

    print("-------- train_cifar10 start-----------")
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(n_epochs):
        train_acc, train_loss = epoch_general_cifar10(dataloader=dataloader, model=model, opt=opt, epoch=i)
        print("epoch = " + str(i) + ", train accuracy = " + str(train_acc) + ", train loss = " + str(train_loss))
    print("-------- train_cifar10 end-----------")
    return train_acc, train_loss
    


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    test_acc, test_loss = epoch_general_cifar10(dataloader=dataloader, model=model)
    return test_acc, test_loss
    



### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    if opt is None:
        model.eval()
    f = loss_fn()
    avg_loss = []
    avg_acc = 0
    cnt = 0
    n_batch = data.shape[0]
    i = 0
    while i < n_batch:
        if opt:
            opt.reset_grad()
        x, y = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)
        b = y.shape[0]
        y_, h = model(x)
        loss = f(y_, y)
        if opt:
            loss.backward()
            opt.step()
        cnt += b
        avg_loss.append(loss.detach().numpy().item() * b)
        avg_acc += np.sum(y_.detach().numpy().argmax(axis=1) == y.numpy())
        print("step = " + str(i) + ", avg_acc = " + str(avg_acc/cnt) + ", avg_loss  = " + str(np.sum(avg_loss)/cnt))
        i += seq_len

    return avg_acc / cnt, np.sum(avg_loss) / cnt
    


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)

    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for i in range(n_epochs):
        avg_acc, avg_loss = epoch_general_ptb(
            data=data,
            model=model,
            seq_len=seq_len,
            loss_fn=loss_fn,
            opt=opt,
            clip=clip,
            device=device,
            dtype=dtype
        )
        print("epoch = " + str(i) + ", avg_acc = " + str(avg_acc) + ", avg_loss  = " + str(avg_loss))
    return avg_acc, avg_loss
    


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    avg_acc, avg_loss = epoch_general_ptb(
        data=data,
        model=model,
        seq_len=seq_len,
        loss_fn=loss_fn,
        opt=None,
        clip=None,
        device=device,
        dtype=dtype
    )
    return avg_acc, avg_loss


if __name__ == "__main__":
    device = ndl.cpu()
    dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    dataloader = ndl.data.DataLoader(\
            dataset=dataset,
            batch_size=128,
            shuffle=True
            )

    model = ResNet9(device=device, dtype="float32")
    print(model)
    train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
         lr=0.001, weight_decay=0.001)
    test_acc, test_loss = evaluate_cifar10(model, dataloader)
    print("train accuracy = " + str(test_acc) + ", train loss = " + str(test_loss))

    # corpus = ndl.data.Corpus("./data/ptb")
    # seq_len = 40
    # batch_size = 16
    # hidden_size = 100
    # print("------ training begin ---------")
    # train_data = ndl.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
    # model = LanguageModel(1, len(corpus.dictionary), hidden_size, num_layers=2, device=device)
    # train_ptb(model, train_data, seq_len, n_epochs=10, device=device)
