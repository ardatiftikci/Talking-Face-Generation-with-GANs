import torch


def repeat_embeddings(embeddings, repeat_count):
    return torch.stack([torch.stack(repeat_count * [embeddings[i]]) for i in range(embeddings.shape[0])])


def cut_sequence(audio_sequence, stride, padding, audio_sample_size):
    pad_left = torch.zeros(audio_sequence.shape[0], padding // 2, 1)
    pad_right = torch.zeros(audio_sequence.shape[0], padding - padding // 2, 1)
    audio_sequence = torch.cat((pad_left, audio_sequence), 1)
    audio_sequence = torch.cat((audio_sequence, pad_right), 1)
    stacked = audio_sequence.narrow(1, 0, audio_sample_size).unsqueeze(1)
    iterations = (audio_sequence.shape[1] - audio_sample_size) // stride + 1
    for i in range(1, iterations):
        stacked = torch.cat((stacked, audio_sequence.narrow(1, i * stride, audio_sample_size).unsqueeze(1)), dim=1)
    return stacked
