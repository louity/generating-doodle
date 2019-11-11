import numpy as np
import torch


def make_image_point(img):
    '''
    img is just an array of shape (nbr_point, 3)
    to the point parametrization.
    '''
    len_img = len(img[:, 0])
    new_img = np.zeros((len_img, 5))
    new_img[:len_img, :2] = img[:, :2]
    new_img[:len_img-1, 2] = 1-img[:-1, 2]
    new_img[:len_img, 3] = img[:, 2]
    new_img[(len_img-1):, 4] = 1
    new_img[len_img-1, 2:4] = 0
    return(new_img)


# Read data into batch
def make_batch_point(batch_sequences, max_len_out, use_cuda):
    strokes = []
    lengths = []
    indice = 0
    for seq in batch_sequences:
        len_seq = len(seq[:, 0])
        new_seq = np.zeros((max_len_out, 5))
        new_seq[:len_seq, :2] = seq[:, :2]
        new_seq[:len_seq-1, 2] = 1-seq[:-1, 2]
        new_seq[:len_seq, 3] = seq[:, 2]
        new_seq[(len_seq-1):, 4] = 1
        new_seq[len_seq-1, 2:4] = 0
        lengths.append(len(seq[:, 0]))
        # TODO: should replace with lengths.append(len_seq)
        strokes.append(new_seq)
        indice += 1

    if use_cuda:
        batch = torch.from_numpy(np.stack(strokes, 1)).cuda().float()
    else:
        batch = torch.from_numpy(np.stack(strokes, 1)).float()
    return batch, lengths


def make_batch_line(batch_size, data, max_len_out, use_cuda):
    batch_idx = np.random.choice(len(data), batch_size)
    batch_sequences = [data[idx] for idx in batch_idx]
    lines = []
    lengths = []
    indice = 0

    for seq in batch_sequences:
        len_seq = len(seq[:, 0])
        new_seq = np.zeros((max_len_out, 5))
        # copy the 4 coordinates (cx, cy, length, angle)
        new_seq[:len_seq, :4] = seq[:, :4]
        # a one for the end of the sketch
        new_seq[len_seq-1, 4] = 1

        lengths.append(len(seq[:, 0]))
        lines.append(new_seq)

        indice += 1

    batch = torch.from_numpy(np.stack(lines, 1)).float()
    if use_cuda:
        batch = batch.cuda()

    return batch, lengths


# Translate batch to make target
def make_target_point(batch, lengths, hyper_params, max_len_out, use_cuda):
    eos = torch.stack([torch.Tensor([0, 0, 0, 0, 1])]*batch.size()[1]).unsqueeze(0)
    if use_cuda:
        eos = eos.cuda()

    batch = torch.cat([batch, eos], 0)
    mask = torch.zeros(max_len_out+1, batch.size()[1])
    for indice, length in enumerate(lengths):
        mask[:length, indice] = 1
    if use_cuda:
        mask = mask.cuda()
    dx = torch.stack([batch.data[:, :, 0]]*hyper_params.M, 2)
    dy = torch.stack([batch.data[:, :, 1]]*hyper_params.M, 2)
    p1 = batch.data[:, :, 2]
    p2 = batch.data[:, :, 3]
    p3 = batch.data[:, :, 4]
    p = torch.stack([p1, p2, p3], 2)
    return mask, dx, dy, p


def make_target_line(batch, lengths, hyper_params, max_len_out, use_cuda):
    # TODO: not sure to get why End Of Sequence is necessary
    eos = torch.stack([torch.Tensor([0, 0, 0, 0, 1])]*batch.size()[1]).unsqueeze(0)
    if use_cuda:
        eos = eos.cuda()
    # add eos at each element of batch?
    batch = torch.cat([batch, eos], 0)
    mask = torch.zeros(max_len_out+1, batch.size()[1])
    for indice, length in enumerate(lengths):
        mask[:length, indice] = 1
    if use_cuda:
        mask = mask.cuda()
    dx = torch.stack([batch.data[:, :, 0]] * hyper_params.M, dim=2)
    dy = torch.stack([batch.data[:, :, 1]] * hyper_params.M, dim=2)
    r = torch.stack([batch.data[:, :, 2]] * hyper_params.Mr, dim=2)
    phi = torch.stack([batch.data[:, :, 3]] * hyper_params.Mphi, dim=2)
    # TODO: check that that it is self.hyper_params and not self.hp
    p0 = batch.data[:, :, -1]
    return mask, dx, dy, r, phi, p0
