import random
import numpy as np, argparse, time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import MaskedNLLLoss, MaskedKLDivLoss, Transformer_Based_Model
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import datetime

import matplotlib.pyplot as plt
import wandb

from inference import run_inference, MODALITY_COMBOS
from inference_shift import run_shift_eval


def set_seed(seed: int):
    """Fix all random sources for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic CUDNN ops — slightly slower but fully reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_train_valid_sampler(trainset, valid=0.1, dataset='MELD'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False, pkl_path='data/meld_vit.pkl'):
    trainset = MELDDataset(pkl_path)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset(pkl_path, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    
    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False, pkl_path='data/iemocap_vit.pkl',
                        va_json='data/iemocap_va.json'):
    trainset = IEMOCAPDataset(pkl_path, va_json=va_json)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(pkl_path, train=False, va_json=va_json)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    
    return train_loader, valid_loader, test_loader

def plot_confusion_matrix(labels, preds, masks, class_names, epoch):
    from sklearn.metrics import confusion_matrix
    import numpy as np

    cm = confusion_matrix(labels, preds, sample_weight=masks)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)

    ax.set(xticks=range(len(class_names)),
           yticks=range(len(class_names)),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True label',
           xlabel='Predicted label',
           title=f'Confusion matrix — epoch {epoch}')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    thresh = 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f'{cm_norm[i,j]:.2f}',
                    ha='center', va='center',
                    color='white' if cm_norm[i,j] > thresh else 'black',
                    fontsize=8)
    fig.tight_layout()
    return fig

def train_or_eval_model(model, loss_function, kl_loss, dataloader, epoch, optimizer=None, train=False, gamma_1=1.0, gamma_2=1.0, gamma_3=1.0, gamma_con=0.0, gamma_orth=0.0):
    losses, preds, labels, masks = [], [], [], []
    inference_times = []
    
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        device = torch.device('cuda' if cuda else 'cpu')
        if len(data) == 8:
            textf, visuf, acouf, qmask, umask, label, shift_severity, _ = \
                [d.to(device) if isinstance(d, torch.Tensor) else d for d in data]
        else:
            textf, visuf, acouf, qmask, umask, label = [d.to(device) for d in data[:-1]]
            shift_severity = None
        qmask = qmask.permute(1, 0, 2)
        lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        t0 = time.time()
        log_prob1, log_prob2, log_prob3, all_log_prob, all_prob, \
        kl_log_prob1, kl_log_prob2, kl_log_prob3, kl_all_prob, con_loss, orth_loss = model(textf, visuf, acouf, umask, qmask, lengths, 
                                                                                           labels=label if train else None, shift_severity=shift_severity)
        if not train:
            if cuda: torch.cuda.synchronize()
            inference_times.append((time.time() - t0, int(umask.sum().item())))

        lp_1 = log_prob1.view(-1, log_prob1.size()[2])
        lp_2 = log_prob2.view(-1, log_prob2.size()[2])
        lp_3 = log_prob3.view(-1, log_prob3.size()[2])
        lp_all = all_log_prob.view(-1, all_log_prob.size()[2])
        labels_ = label.view(-1)

        kl_lp_1 = kl_log_prob1.view(-1, kl_log_prob1.size()[2])
        kl_lp_2 = kl_log_prob2.view(-1, kl_log_prob2.size()[2])
        kl_lp_3 = kl_log_prob3.view(-1, kl_log_prob3.size()[2])
        kl_p_all = kl_all_prob.view(-1, kl_all_prob.size()[2])
        
        loss = gamma_1 * loss_function(lp_all, labels_, umask) + \
                gamma_2 * (loss_function(lp_1, labels_, umask) +
                            loss_function(lp_2, labels_, umask) +
                            loss_function(lp_3, labels_, umask)) + \
                gamma_3 * (kl_loss(kl_lp_1, kl_p_all, umask) +
                            kl_loss(kl_lp_2, kl_p_all, umask) +
                            kl_loss(kl_lp_3, kl_p_all, umask))
        
        if con_loss is not None and gamma_con > 0:
            loss = loss + gamma_con * con_loss

        if orth_loss is not None and gamma_orth > 0:
            loss = loss + gamma_orth * orth_loss

        lp_ = all_prob.view(-1, all_prob.size()[2])

        pred_ = torch.argmax(lp_,1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()

            # Optional: clip to prevent blow-ups
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

    if preds!=[]:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), 0.0

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight = masks)*100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight = masks, average='weighted')*100, 2)  
    
    if inference_times:
        total_s   = sum(t for t, _ in inference_times)
        total_utt = sum(n for _, n in inference_times)
        avg_ms_utt = total_s / total_utt * 1000 if total_utt > 0 else 0.0
    else:
        avg_ms_utt = 0.0
 
    return avg_loss, avg_accuracy, labels, preds, masks, avg_fscore, avg_ms_utt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=1024, metavar='hidden_dim', help='output hidden size')
    parser.add_argument('--n_head', type=int, default=8, metavar='n_head', help='number of heads')
    parser.add_argument('--epochs', type=int, default=150, metavar='E', help='number of epochs')
    parser.add_argument('--temp', type=int, default=1, metavar='temp', help='temp')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    parser.add_argument('--Dataset', default='MELD', help='dataset to train and test')

    parser.add_argument('--patience', type=int, default=15)

    parser.add_argument('--speaker-embeddings', type=int, default=1, choices=[0, 1],
                    help='1 = add learned speaker embeddings, 0 = disable them.')
    
    # Embeddings
    parser.add_argument('--D_text',   type=int, default=1024)
    parser.add_argument('--D_audio',  type=int, default=1024)
    parser.add_argument('--visual-encoder', type=str, default='vit')

    # Dropout
    parser.add_argument('--modal-dropout', type=float, default=0.0)
    parser.add_argument('--dropout-strategy', type=str, default='input',
                        choices=['input', 'representation'])

    # Entropy/Softmax/Reliability
    parser.add_argument('--fusion-mode', type=str, default='softmax', 
                        choices=['softmax', 'entropy', 'class_reliability', 
                                 'scalar_reliability', 'transformer_reliability'])
    
    parser.add_argument('--gate-dim', type=int, default=64)
    
    # Intra visual gate
    parser.add_argument('--intra-visual-gate', type=int, default=0, choices=[0, 1])
    parser.add_argument('--intra-visual-gate-learned', type=int, default=0, choices=[0, 1],
                        help='1 = learned gate routing (default) '
                        '0 = fixed equal weighting')
    
    # Landmark MLP
    parser.add_argument('--landmark-mlp', type=int, default=0, choices=[0, 1])

    # Contrastive loss
    parser.add_argument('--gamma-con', type=float, default=0.0,
                        help='Weight for supervised contrastive loss. 0 disables it.')
    parser.add_argument('--con-temperature', type=float, default=0.07,
                        help='Temperature for contrastive loss softmax.')
    
    # Orthogonal Modality Decomposer (OMD)
    parser.add_argument('--disentangle', type=int, default=0, choices=[0, 1],
                        help='1 = enable OMD, 0 = disable.')
    parser.add_argument('--gamma-orth', type=float, default=0.1,
                        help='Weight for the OMD orthogonality loss')
    
    # Wandb
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--plot-tsne', action='store_true', default=False)

    # Circumplex prior
    parser.add_argument('--circumplex-alpha', type=float, default=0.0)
    parser.add_argument('--va-json', type=str, default='data/iemocap_va.json')

    parser.add_argument('--circumplex-tau', type=float, default=0.5)

    # Seed 
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.seed != -1:
        set_seed(args.seed)

    iemocap_path = "/media/ssd2/oriol/IEMOCAP/IEMOCAP_full_release/embeddings/multimodal/new_text"
    meld_path = "/media/ssd2/oriol/MELD/embeddings/multimodal"

    _VISUAL_CONFIGS = {
        'vit':                  {'D_visual': 768,   'meld_pkl': f'{meld_path}/meld_vit.pkl',                        'iemocap_pkl': f'{iemocap_path}/iemocap_vit.pkl'},
        'landmarks':            {'D_visual': 408,   'meld_pkl': f'{meld_path}/meld_landmarks.pkl',                  'iemocap_pkl': f'{iemocap_path}/iemocap_landmarks.pkl'},
        'landmarks_exp':        {'D_visual': 39,    'meld_pkl': f'{meld_path}/meld_landmarks_exp.pkl',              'iemocap_pkl': f'{iemocap_path}/iemocap_landmarks_exp.pkl'},
        'action_units':         {'D_visual': 60,    'meld_pkl': f'{meld_path}/meld_action_units.pkl',               'iemocap_pkl': f'{iemocap_path}/iemocap_action_units.pkl'},
        'vit_landmarks':        {'D_visual': 1176,  'meld_pkl': f'{meld_path}/meld_vit_landmarks.pkl',              'iemocap_pkl': f'{iemocap_path}/iemocap_vit_landmarks.pkl'},
        'vit_landmarks_exp':    {'D_visual': 807,   'meld_pkl': f'{meld_path}/meld_vit_landmarks_exp.pkl',          'iemocap_pkl': f'{iemocap_path}/iemocap_vit_landmarks_exp.pkl'},
        'vit_action_units':     {'D_visual': 828,   'meld_pkl': f'{meld_path}/meld_vit_action_units.pkl',           'iemocap_pkl': f'{iemocap_path}/iemocap_vit_action_units.pkl'},
        'vit_landmarks_deg':    {'D_visual': 1176,  'meld_pkl': '', 'iemocap_pkl': f'{iemocap_path}/iemocap_vit_landmarks_degraded_train.pkl'},
        'vit_action_units_deg': {'D_visual': 828,   'meld_pkl': f'{meld_path}/meld_vit_au_degraded_train.pkl',  'iemocap_pkl': f''},
    }

    today = datetime.datetime.now()
    print(args)
    
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.wandb:
        run = wandb.init(
            entity='oriolmarin18-universitat-aut-noma-de-barcelona',
            project='mySDT',
            name=f"{wandb.util.generate_id()}",
            config=vars(args),
        )

        for key, value in wandb.config.items():
            setattr(args, key, value)

    args.D_visual    = _VISUAL_CONFIGS[args.visual_encoder]['D_visual']
    args.meld_pkl    = _VISUAL_CONFIGS[args.visual_encoder]['meld_pkl']
    args.iemocap_pkl = _VISUAL_CONFIGS[args.visual_encoder]['iemocap_pkl']

    _LANDMARK_ENCODERS = {'landmarks', 'landmarks_exp', 'action_units'}
    use_landmark_mlp = bool(args.landmark_mlp) and args.visual_encoder in _LANDMARK_ENCODERS

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size
    
    D_audio  = args.D_audio
    D_visual = args.D_visual
    D_text   = args.D_text

    D_m = D_audio + D_visual + D_text

    n_speakers = 9 if args.Dataset=='MELD' else 2
    n_classes = 7 if args.Dataset=='MELD' else 6 if args.Dataset=='IEMOCAP' else 1

    va_map = None
    if args.circumplex_alpha > 0:
        if args.Dataset == 'IEMOCAP':
            from dataloader import IEMOCAP_VA_MAP
            va_map = IEMOCAP_VA_MAP
        elif args.Dataset == 'MELD':
            from dataloader import MELD_VA_MAP
            va_map = MELD_VA_MAP

    model = Transformer_Based_Model(
        args.Dataset, args.temp, D_text, D_visual, D_audio, args.n_head,
        n_classes=n_classes,
        hidden_dim=args.hidden_dim,
        n_speakers=n_speakers,
        dropout=args.dropout,
        modal_dropout=args.modal_dropout,
        dropout_strategy=args.dropout_strategy,
        fusion_mode=args.fusion_mode,
        con_temperature=args.con_temperature,
        use_disentangle=bool(args.disentangle),
        use_intra_visual_gate=bool(args.intra_visual_gate),
        use_intra_visual_gate_learned=bool(args.intra_visual_gate_learned),
        use_speaker_embeddings=bool(args.speaker_embeddings),
        use_landmark_mlp=use_landmark_mlp,
        gate_dim=args.gate_dim,
        circumplex_alpha=args.circumplex_alpha,
        va_map=va_map,
        circumplex_tau=args.circumplex_tau)

    total_params = sum(p.numel() for p in model.parameters())
    print('total parameters: {}'.format(total_params))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('training parameters: {}'.format(total_trainable_params))

    device = torch.device('cuda' if cuda else 'cpu')
    model.to(device)
        
    kl_loss = MaskedKLDivLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    if args.Dataset == 'MELD':
        if args.class_weight:
            # neutral, surprise, fear, sadness, joy, disgust, anger
            meld_weights = torch.FloatTensor([
                1.0/0.469507, 1.0/0.119346, 1.0/0.026116, 
                1.0/0.073096, 1.0/0.168369, 1.0/0.026335, 
                1.0/0.117231])
            loss_function = MaskedNLLLoss(meld_weights.to(device) if cuda else meld_weights)
        else:
            loss_function = MaskedNLLLoss()
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.1,
                                                                    batch_size=batch_size,
                                                                    num_workers=0,
                                                                    pkl_path=args.meld_pkl)
    elif args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([
            1.0/0.086747, 1.0/0.144406, 1.0/0.227883, 
            1.0/0.160585, 1.0/0.127711, 1.0/0.252668])
        loss_function = MaskedNLLLoss(loss_weights.to(device) if cuda else loss_weights)
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=0,
                                                                      pkl_path=args.iemocap_pkl,
                                                                      va_json=args.va_json)
    else:
        print("There is no such dataset")

    best_fscore, best_loss, best_label, best_pred, best_mask = None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []

    if cuda and torch.cuda.is_available():
        dev_idx  = torch.cuda.current_device()
        total_mb = torch.cuda.get_device_properties(dev_idx).total_memory / 1024**2
        print(f"GPU: {torch.cuda.get_device_name(dev_idx)}  ({total_mb:.0f} MB total)")
        torch.cuda.reset_peak_memory_stats(dev_idx)
    else:
        dev_idx  = None
        total_mb = 0.0
 
    epoch_times    = []
    training_start = time.time()

    patience_counter = 0
    best_fscore_for_es = None
    for e in range(n_epochs):
        start_time = time.time()

        train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model, loss_function, kl_loss, train_loader, e, optimizer, True, gamma_con = args.gamma_con, gamma_orth=args.gamma_orth)
        valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(model, loss_function, kl_loss, valid_loader, e, gamma_con = args.gamma_con, gamma_orth=args.gamma_orth)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, test_inf_ms = train_or_eval_model(model, loss_function, kl_loss, test_loader, e, gamma_con = args.gamma_con, gamma_orth=args.gamma_orth)
        all_fscore.append(test_fscore)

        if best_fscore == None or best_fscore < test_fscore:
            best_fscore = test_fscore
            best_label, best_pred, best_mask = test_label, test_pred, test_mask
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, f'/media/ssd2/oriol/checkpoints/{args.Dataset}/final/{args.visual_encoder}_{args.fusion_mode}.pt')

        # Early stopping
        if best_fscore_for_es is None or test_fscore > best_fscore_for_es:
            best_fscore_for_es = test_fscore
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping at epoch {e+1} (no improvement for {args.patience} epochs)')
                break

                
        epoch_sec = round(time.time() - start_time, 2)
        epoch_times.append(epoch_sec)
 
        gpu_str = ""
        if cuda and torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated(dev_idx) / 1024**2
            gpu_str = f", gpu_peak: {peak_mb:.0f}MB"

        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
        
        if args.wandb:
            wandb.log({ 
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_fscore": train_fscore,
 
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "test_fscore": test_fscore,

                "best_test_w_f1": best_fscore,
             })

        if (e+1)%10 == 0:
            print(classification_report(best_label, best_pred, sample_weight = best_mask, digits=4))
            print(confusion_matrix(best_label, best_pred, sample_weight = best_mask))


    total_training_s = time.time() - training_start
    avg_epoch_s = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
 
    # Last epoch inference time on test set
    _, _, _, _, _, _, last_inf_ms = train_or_eval_model(
        model, loss_function, kl_loss, test_loader, n_epochs - 1
    )

    print('Test performance..')
    print('F-Score: {}'.format(max(all_fscore)))
    print('F-Score-index: {}'.format(all_fscore.index(max(all_fscore)) + 1))

    print(classification_report(best_label, best_pred, sample_weight = best_mask, digits=4))
    print(confusion_matrix(best_label, best_pred, sample_weight = best_mask))

    print("\nRunning modality evaluation on best model...")
    model.load_state_dict(best_state)
    model.to(device).eval()

    # Modality inference
    combo_results = {}
    for name, active in MODALITY_COMBOS:
        zeroed = sorted({'T', 'A', 'V'} - active)
        print(f"  [{name}] zeroed: {zeroed if zeroed else 'none'} ...", end=' ', flush=True)
        r = run_inference(model, loss_function, kl_loss, test_loader, device, cuda,
                          eval_modalities=active)
        combo_results[name] = r
        print(f"w-F1={r['w_f1']:.2f}  acc={r['acc']:.2f}")

    
    if args.wandb:

        wandb.log({
            f"{name.replace('+', '')}_wf1": r['w_f1']
            for name, r in combo_results.items()
        } | {
            f"{name.replace('+', '')}_acc": r['acc']
            for name, r in combo_results.items()
        })

    # ── Shift / stable split ──────────────────────────────────────────────────
    # Resolve va_map for shift eval: IEMOCAP already loaded above;
    # for MELD fall back to MELD_VA_MAP if available.
    shift_va_map = va_map

    print('\nShift / stable analysis (best model, all modalities):')
    shift_res = run_shift_eval(
        model, test_loader, device, cuda,
        va_map=shift_va_map, dataset=args.Dataset,
    )

    for cond in ('shift', 'stable'):
        r = shift_res['binary'][cond]
        print(f"  {cond:6s}  n={r['n']:4d}  "
              f"acc={r['acc']:.2f}  w-F1={r['w_f1']:.2f}")

    if 'va_buckets' in shift_res:
        print('  VA buckets:')
        for cond in ('stable', 'moderate', 'large'):
            r = shift_res['va_buckets'][cond]
            print(f"    {cond:8s}  n={r['n']:4d}  "
                  f"acc={r['acc']:.2f}  w-F1={r['w_f1']:.2f}")

    if args.wandb:
        shift_log = {
            'shift_acc':  shift_res['binary']['shift']['acc'],
            'shift_wf1':  shift_res['binary']['shift']['w_f1'],
            'shift_n':    shift_res['binary']['shift']['n'],
            'stable_acc': shift_res['binary']['stable']['acc'],
            'stable_wf1': shift_res['binary']['stable']['w_f1'],
            'stable_n':   shift_res['binary']['stable']['n'],
        }
        if 'va_buckets' in shift_res:
            for cond in ('stable', 'moderate', 'large'):
                r = shift_res['va_buckets'][cond]
                shift_log[f'va_{cond}_acc'] = r['acc']
                shift_log[f'va_{cond}_wf1'] = r['w_f1']
                shift_log[f'va_{cond}_n']   = r['n']
        wandb.log(shift_log)

        # if args.plot_tsne:
        #     _log_tsne_to_wandb(model, test_loader, device, cuda, args)

        wandb.finish()