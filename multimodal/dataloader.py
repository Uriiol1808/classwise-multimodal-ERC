import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd

import json

IEMOCAP_VA_MAP = {
    0: (0.80, 0.60),  # happy
    1: (0.20, 0.30),  # sad
    2: (0.50, 0.50),  # neutral
    3: (0.20, 0.80),  # angry
    4: (0.75, 0.80),  # excited
    5: (0.25, 0.55),  # frustrated
}

MELD_VA_MAP = {
    0: (0.00, 0.00),   # neutral
    1: (0.40, 0.67),   # surprise
    2: (-0.64, 0.60),  # fear
    3: (-0.63, -0.27), # sadness
    4: (0.76, 0.48),   # joy
    5: (-0.60, 0.35),  # disgust
    6: (-0.43, 0.67),  # anger
}


def compute_shift_severity(va_sequence):
    """
    Circumplex shift severity for a single dialogue.
    va_sequence: np.array (N, 2) — valence/arousal for each utterance in [0,1]
    Returns: np.array (N,) normalised to [0,1]
             First utterance is always 0 (no previous state).
    """
    severity = np.zeros(len(va_sequence), dtype=np.float32)
    for i in range(1, len(va_sequence)):
        severity[i] = np.linalg.norm(va_sequence[i] - va_sequence[i - 1]) / 2.828
    return severity

class IEMOCAPDataset(Dataset):
    def __init__(self, path, train=True, va_json="data/iemocap_va.json"):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')
        
        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)

        self.va_dict = json.load(open(va_json)) if va_json is not None else {}

    def __getitem__(self, index):
        vid = self.keys[index]
        utt_ids = self.videoIDs[vid]
        labels  = self.videoLabels[vid]
 
        # Build VA sequence: use continuous annotation if available, else Russell
        va_seq = np.array([
            self.va_dict.get(uid, list(IEMOCAP_VA_MAP.get(labels[i], [0.5, 0.5])))
            for i, uid in enumerate(utt_ids)
        ], dtype=np.float32)  # (N, 2)
 
        shift_severity = compute_shift_severity(va_seq)  # (N,)
 
        return torch.FloatTensor(self.videoText[vid]),\
               torch.FloatTensor(self.videoVisual[vid]),\
               torch.FloatTensor(self.videoAudio[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]]),\
               torch.FloatTensor([1]*len(labels)),\
               torch.LongTensor(labels),\
               torch.FloatTensor(shift_severity),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        result = []
        for i in dat:
            if i < 4:
                result.append(pad_sequence(dat[i]))
            elif i < 6:
                result.append(pad_sequence(dat[i], True))
            elif i == 6:
                result.append(pad_sequence(dat[i], True))  # shift_severity (B, T)
            else:
                result.append(dat[i].tolist())
        return result


class MELDDataset(Dataset):
    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.roberta2, self.roberta3, self.roberta4, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, _ = pickle.load(open(path, 'rb'))

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(np.array(self.videoText[vid])),\
            torch.FloatTensor(np.array(self.videoVisual[vid])),\
            torch.FloatTensor(np.array(self.videoAudio[vid])),\
            torch.FloatTensor(self.videoSpeakers[vid]),\
            torch.FloatTensor([1]*len(self.videoLabels[vid])),\
            torch.LongTensor(self.videoLabels[vid]),\
            vid

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<4 else pad_sequence(dat[i], True) if i<6 else dat[i].tolist() for i in dat]