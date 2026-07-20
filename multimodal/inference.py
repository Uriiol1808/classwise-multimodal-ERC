import os
import json
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from dataloader import MELDDataset, IEMOCAPDataset
from model import MaskedNLLLoss, MaskedKLDivLoss, Transformer_Based_Model


MELD_EMOTIONS    = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
IEMOCAP_EMOTIONS = ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated']

MODALITY_COMBOS = [
    ('T+A+V', {'T', 'A', 'V'}),
    ('T+A',   {'T', 'A'}),
    ('T+V',   {'T', 'V'}),
    ('A+V',   {'A', 'V'}),
    ('T',     {'T'}),
    ('A',     {'A'}),
    ('V',     {'V'}),
]


def run_inference(model, loss_function, kl_loss, dataloader, device, cuda,
                  eval_modalities=None):
    if eval_modalities is None:
        eval_modalities = {'T', 'A', 'V'}

    eval_modalities = set(eval_modalities)
    force_drop_modalities = {'T', 'A', 'V'} - eval_modalities

    model.eval()
    preds, labels, masks = [], [], []

    with torch.no_grad():
        for data in dataloader:
            if len(data) == 8:
                textf, visuf, acouf, qmask, umask, label, shift_severity, _ = [
                    d.to(device) if isinstance(d, torch.Tensor) else d
                    for d in data
                ]
            else:
                textf, visuf, acouf, qmask, umask, label = [
                    d.to(device) for d in data[:-1]
                ]
                shift_severity = None

            # Zero raw inputs for ablated modalities here, before the model ever
            # sees them — true input-level ablation regardless of the checkpoint's
            # dropout_strategy. (Don't rely on the model's own dropout_strategy
            # branch for this: 'representation' zeroes only after the cross-modal
            # transformers have already attended over the real signal.)
            if 'T' in force_drop_modalities:
                textf = torch.zeros_like(textf)
            if 'A' in force_drop_modalities:
                acouf = torch.zeros_like(acouf)
            if 'V' in force_drop_modalities:
                visuf = torch.zeros_like(visuf)

            qmask = qmask.permute(1, 0, 2)
            lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1
                       for j in range(len(umask))]

            is_full_tav = (force_drop_modalities == set())
            shift_sev_input = shift_severity if is_full_tav else None

            _, _, _, _, all_prob, _, _, _, _, _, _ = model(
                textf, visuf, acouf, umask, qmask, lengths,
                labels=None,
                shift_severity=shift_sev_input,
                force_drop_modalities=force_drop_modalities,
            )

            lp_   = all_prob.view(-1, all_prob.size(2))
            pred_ = torch.argmax(lp_, dim=1)
            preds.append(pred_.detach().cpu().numpy())
            labels.append(label.view(-1).detach().cpu().numpy())
            masks.append(umask.view(-1).detach().cpu().numpy())

    preds  = np.concatenate(preds)
    labels = np.concatenate(labels)
    masks  = np.concatenate(masks)

    w_f1 = round(f1_score(labels, preds, sample_weight=masks,
                           average='weighted', zero_division=0) * 100, 2)
    acc  = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)

    per_class_f1 = f1_score(labels, preds, sample_weight=masks,
                             average=None, zero_division=0)
    per_class_f1 = [round(float(v) * 100, 2) for v in per_class_f1]

    from sklearn.metrics import recall_score
    per_class_acc = recall_score(labels, preds, sample_weight=masks,
                                  average=None, zero_division=0)
    per_class_acc = [round(float(v) * 100, 2) for v in per_class_acc]

    return {
        'w_f1': w_f1,
        'acc':  acc,
        'per_class_f1':  per_class_f1,
        'per_class_acc': per_class_acc,
        'dropped_modalities': sorted(force_drop_modalities),
        'dropout_strategy':   getattr(model, 'dropout_strategy', 'input'),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',                 required=True)
    parser.add_argument('--no-cuda',                    action='store_true', default=False)
    parser.add_argument('--batch-size',                 type=int,   default=16)
    parser.add_argument('--hidden_dim',                 type=int,   default=1024)
    parser.add_argument('--n_head',                     type=int,   default=8)
    parser.add_argument('--dropout',                    type=float, default=0.3)
    parser.add_argument('--temp',                       type=int,   default=1)
    parser.add_argument('--Dataset',                    default='IEMOCAP')
    parser.add_argument('--D_text',                     type=int,   default=1024)
    parser.add_argument('--D_visual',                   type=int,   default=768)
    parser.add_argument('--D_audio',                    type=int,   default=1024)
    parser.add_argument('--modal-dropout',              type=float, default=0.0)
    parser.add_argument('--dropout-strategy',           type=str,   default='input')
    parser.add_argument('--fusion-mode',                type=str,   default='softmax')
    parser.add_argument('--visual-encoder',             type=str,   default='vit')
    parser.add_argument('--speaker-embeddings',         type=int,   default=1)
    parser.add_argument('--intra-visual-gate',          type=int,   default=0)
    parser.add_argument('--intra-visual-gate-learned',  type=int,   default=1)
    parser.add_argument('--landmark-mlp',               type=int,   default=0)
    parser.add_argument('--disentangle',                type=int,   default=0)
    parser.add_argument('--gamma-orth',                 type=float, default=0.0)
    parser.add_argument('--circumplex-alpha',           type=float, default=0.0)
    parser.add_argument('--va-json',                    type=str,   default='data/iemocap_va.json')
    parser.add_argument('--iemocap-pkl',                type=str,   default='data/iemocap_vit.pkl')
    parser.add_argument('--meld-pkl',                   type=str,   default='data/meld_vit.pkl')
    parser.add_argument('--out',                        default=None)
    args = parser.parse_args()

    cuda   = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if cuda else 'cpu')
    print(f'Device: {"GPU" if cuda else "CPU"}')

    n_speakers    = 9 if args.Dataset == 'MELD' else 2
    n_classes     = 7 if args.Dataset == 'MELD' else 6
    emotion_names = MELD_EMOTIONS if args.Dataset == 'MELD' else IEMOCAP_EMOTIONS

    _LANDMARK_ENCODERS = {'landmarks', 'landmarks_exp', 'action_units'}
    use_landmark_mlp = bool(args.landmark_mlp) and args.visual_encoder in _LANDMARK_ENCODERS

    from dataloader import IEMOCAP_VA_MAP, MELD_VA_MAP
    va_map = None
    if args.circumplex_alpha > 0:
        if args.Dataset == 'IEMOCAP':
            va_map = IEMOCAP_VA_MAP
        elif args.Dataset == 'MELD':
            va_map = MELD_VA_MAP
            
    model = Transformer_Based_Model(
        args.Dataset, args.temp,
        args.D_text, args.D_visual, args.D_audio, args.n_head,
        n_classes=n_classes,
        hidden_dim=args.hidden_dim,
        n_speakers=n_speakers,
        dropout=args.dropout,
        modal_dropout=args.modal_dropout,
        dropout_strategy=args.dropout_strategy,
        fusion_mode=args.fusion_mode,
        use_disentangle=bool(args.disentangle),
        use_speaker_embeddings=bool(args.speaker_embeddings),
        use_intra_visual_gate=bool(args.intra_visual_gate),
        use_intra_visual_gate_learned=bool(args.intra_visual_gate_learned),
        use_landmark_mlp=use_landmark_mlp,
        circumplex_alpha=args.circumplex_alpha,
        va_map=va_map,
    )
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f'Loaded: {args.checkpoint}')

    if args.Dataset == 'MELD':
        testset = MELDDataset(args.meld_pkl, train=False)
    else:
        testset = IEMOCAPDataset(args.iemocap_pkl, train=False, va_json=args.va_json)

    test_loader = DataLoader(testset, batch_size=args.batch_size,
                             collate_fn=testset.collate_fn, num_workers=0)

    loss_function = MaskedNLLLoss()
    kl_loss       = MaskedKLDivLoss()

    results = {
        'checkpoint':     args.checkpoint,
        'dataset':        args.Dataset,
        'visual_encoder': args.visual_encoder,
        'emotions':       emotion_names,
        'combos':         {},
    }

    for name, active in MODALITY_COMBOS:
        zeroed = sorted({'T', 'A', 'V'} - active)
        print(f"  [{name}] zeroed: {zeroed if zeroed else 'none'} ...", end=' ', flush=True)
        t0 = time.perf_counter()
        r  = run_inference(model, loss_function, kl_loss, test_loader,
                           device, cuda, eval_modalities=active)
        print(f"w-F1={r['w_f1']:.2f}  acc={r['acc']:.2f}  ({time.perf_counter()-t0:.1f}s)")
        results['combos'][name] = r

    out_path = args.out or args.checkpoint.replace('.pt', '_inference.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {out_path}")
    print("\nSummary:")
    print(f"  {'Combo':<8}  {'w-F1':>6}  {'Acc':>6}")
    print(f"  {'─'*6:<8}  {'─'*6:>6}  {'─'*6:>6}")
    for name, _ in MODALITY_COMBOS:
        r = results['combos'][name]
        print(f"  {name:<8}  {r['w_f1']:>6.2f}  {r['acc']:>6.2f}")