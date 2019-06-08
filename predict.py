from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime

import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd

from utils import *
from modules import *
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=8,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--encoder-hidden', type=int, default=256//4,
                    help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=256//4,
                    help='Number of hidden units.')                    
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--encoder', type=str, default='mlp',
                    help='Type of path encoder model (mlp or cnn).')
parser.add_argument('--decoder', type=str, default='mlp',
                    help='Type of decoder model (mlp, rnn, or sim).')
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')
parser.add_argument('--suffix', type=str, default='_springs5',
                    help='Suffix for training data (e.g. "_charged".')
parser.add_argument('--encoder-dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--edge-types', type=int, default=4,
                    help='The number of edge types to infer.')
parser.add_argument('--workers', type=int, default=0,
                    help='workers .')    
parser.add_argument('--dims', type=int, default=2,
                    help='The number of input dimensions (position + velocity).')
parser.add_argument('--index', type=int, default=-1,
                    help='The number of input dimensions (position + velocity).')
parser.add_argument('--n_out', type=int, default=1,
                    help='Number of atoms in simulation.')   
parser.add_argument('--num-atoms', type=int, default=100,
                    help='Number of atoms in simulation.')                                   
parser.add_argument('--stocks', type=int, default=100,
                    help='The number of input dimensions (position + velocity).')                                           
parser.add_argument('--timesteps', type=int, default=10,
                    help='The number of time steps per sample.')
parser.add_argument('--prediction-steps', type=int, default=5, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--skip-first', action='store_true', default=False,
                    help='Skip first edge type in decoder, i.e. it represents no-edge.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--hard', action='store_true', default=False,
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--prior', action='store_true', default=False,
                    help='Whether to use sparsity prior.')
parser.add_argument('--dynamic-graph', action='store_true', default=True,
                    help='Whether test with dynamically re-computed graph.')

from collections import defaultdict

def nested_dict():
    return defaultdict(nested_dict)

logs = defaultdict(dict)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)
train_idx=args.timesteps
val_idx=args.timesteps
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
if args.dynamic_graph:
    print("Testing with dynamically re-computed graph.")
# Save model and meta-data. Always saves in a new sub-folder.
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    encoder_file = os.path.join(save_folder, 'encoder.pt')
    decoder_file = os.path.join(save_folder, 'decoder.pt')

    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")
# train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(
#     args.batch_size, args.suffix,args.workers)
train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_finance_data(
    args.batch_size, args.suffix,args.workers,args.index,args.stocks,train_idx,val_idx)
# Generate off-diagonal interaction graph
off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)

rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)

if args.encoder == 'mlp':
    encoder = MLPEncoder(args.timesteps * args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)
elif args.encoder == 'cnn':
    encoder = CNNEncoder(args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)
if args.decoder == 'mlp':
    decoder = MLPDecoderP(n_in_node=args.dims,
                        edge_types=args.edge_types,
                        msg_hid=args.decoder_hidden,
                        msg_out=args.decoder_hidden,
                        n_hid=args.decoder_hidden,
                        do_prob=args.decoder_dropout,
                        skip_first=args.skip_first)                         
elif args.decoder == 'rnn':
    decoder = RNNDecoderP(n_in_node=args.dims,
                        edge_types=args.edge_types,
                        n_hid=args.decoder_hidden,
                        do_prob=args.decoder_dropout,
                        skip_first=args.skip_first,
                        n_out=args.n_out)
elif args.decoder == 'sim':
    decoder = SimulationDecoder(loc_max, loc_min, vel_max, vel_min, args.suffix)

if args.load_folder:
    encoder_file = os.path.join(args.load_folder, 'encoder.pt')
    encoder.load_state_dict(torch.load(encoder_file))
    decoder_file = os.path.join(args.load_folder, 'decoder.pt')
    decoder.load_state_dict(torch.load(decoder_file))

    args.save_folder = False

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)

# Linear indices of an upper triangular mx, used for acc calculation
triu_indices = get_triu_offdiag_indices(args.num_atoms)
tril_indices = get_tril_offdiag_indices(args.num_atoms)

if args.prior:
    prior = np.array([0.91, 0.03, 0.03, 0.03])  # TODO: hard coded for now
    print("Using prior")
    print(prior)
    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)

    if args.cuda:
        log_prior = log_prior.cuda()

if args.cuda:
    encoder.cuda()
    decoder.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
    triu_indices = triu_indices.cuda()
    tril_indices = tril_indices.cuda()

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)
fc_out = nn.Linear(args.dims*args.stocks,args.n_out).cuda()
def train(epoch, best_val_loss):
    t = time.time()
    nll_train = []
    acc_train = []
    kl_train = []
    mse_train = []

    encoder.train()
    decoder.train()
    scheduler.step()
    preds = []
    tars = []
    eds =[]
    for batch_idx, (data, relations) in enumerate(train_loader):

        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data), Variable(relations)

        optimizer.zero_grad()

        logits = encoder(data, rel_rec.to_sparse(), rel_send.to_sparse())
        edges = gumbel_softmax(logits, tau=args.temp, hard=args.hard)
        prob = my_softmax(logits, -1)
        eds.append(edges.detach().cpu().numpy())
        if args.decoder == 'rnn':
            output = decoder(data, edges, rel_rec.to_sparse(), rel_send.to_sparse(), 1)
        else:
            output = decoder(data, edges, rel_rec.to_sparse(), rel_send.to_sparse(), 1)

        target = relations[:,1:]
        output = output.transpose(1,2).contiguous().view(args.batch_size,args.timesteps-1,-1)
        output = fc_out(output)
        output = output.squeeze(2)

        # loss_nll = nll_gaussian(output, target, args.var)


        loss_nll = F.mse_loss(output, target)

        if args.prior:
            loss_kl = kl_categorical(prob, log_prior, args.num_atoms)
        else:
            loss_kl = kl_categorical_uniform(prob, args.num_atoms,
                                             args.edge_types)

        loss = loss_nll + loss_kl

        acc = edge_accuracy2(logits, relations)
        acc_train.append(acc)

        loss.backward()
        optimizer.step()

        mse_train.append(F.mse_loss(output, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())
        preds.append(output.detach().cpu().numpy().flatten())
        tars.append(target.detach().cpu().numpy().flatten())

    logs['train']['pred'] = np.concatenate(preds)
    logs['train']['truth'] = np.concatenate(tars)
    logs['edges']['train'] = eds
    nll_val = []
    acc_val = []
    kl_val = []
    mse_val = []

    encoder.eval()
    decoder.eval()
    preds = []
    tars = []
    eds = []
    for batch_idx, (data, relations) in enumerate(valid_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data, volatile=True), Variable(
            relations, volatile=True)

        logits = encoder(data, rel_rec.to_sparse(), rel_send.to_sparse())
        edges = gumbel_softmax(logits, tau=args.temp, hard=True)
        prob = my_softmax(logits, -1)
        eds.append(edges.detach().cpu().numpy())
        # validation output uses teacher forcing
        if args.decoder == 'rnn':
            output = decoder(data, edges, rel_rec.to_sparse(), rel_send.to_sparse(), 1)
        else:
            output = decoder(data, edges, rel_rec, rel_send, 1)
        target = relations[:,1:]
        output = output.transpose(1,2).contiguous().view(args.batch_size,args.timesteps-1,-1)
        output = fc_out(output)
        output = output.squeeze(2)

        # loss_nll = nll_gaussian(output, target, args.var)
        loss_nll = F.mse_loss(output, target)
        # loss_nll = nll_gaussian(output, target, args.var)
        loss_kl = kl_categorical_uniform(prob, args.num_atoms, args.edge_types)

        acc = edge_accuracy2(logits, relations)
        acc_val.append(acc)

        mse_val.append(F.mse_loss(output, target).item())
        nll_val.append(loss_nll.item())
        kl_val.append(loss_kl.item())
        preds.append(output.detach().cpu().numpy().flatten())
        tars.append(target.detach().cpu().numpy().flatten())

    logs['val']['pred'] = np.concatenate(preds)
    logs['val']['truth'] = np.concatenate(tars)
    logs['edges']['val'] = eds
    pickle.dump(logs, open(os.path.join(save_folder,'logs.pkl'),"wb" ))
    # pd.DataFrame(logs).to_pickle(os.path.join(save_folder,'logs.pickle'))
    print('Epoch: {:04d}'.format(epoch),
          'nll_train: {:.10f}'.format(np.mean(nll_train)),
          'kl_train: {:.10f}'.format(np.mean(kl_train)),
          'mse_train: {:.10f}'.format(np.mean(mse_train)),
          'acc_train: {:.10f}'.format(np.mean(acc_train)),
          'nll_val: {:.10f}'.format(np.mean(nll_val)),
          'kl_val: {:.10f}'.format(np.mean(kl_val)),
          'mse_val: {:.10f}'.format(np.mean(mse_val)),
          'acc_val: {:.10f}'.format(np.mean(acc_val)),
          'time: {:.4f}s'.format(time.time() - t))
    if args.save_folder and np.mean(nll_val) < best_val_loss:
        torch.save(encoder.state_dict(), encoder_file)
        torch.save(decoder.state_dict(), decoder_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(nll_train)),
              'kl_train: {:.10f}'.format(np.mean(kl_train)),
              'mse_train: {:.10f}'.format(np.mean(mse_train)),
              'acc_train: {:.10f}'.format(np.mean(acc_train)),
              'nll_val: {:.10f}'.format(np.mean(nll_val)),
              'kl_val: {:.10f}'.format(np.mean(kl_val)),
              'mse_val: {:.10f}'.format(np.mean(mse_val)),
              'acc_val: {:.10f}'.format(np.mean(acc_val)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
    return np.mean(nll_val)

def test2(epoch, best_val_loss):
    t = time.time()
    encoder.eval()
    decoder.eval()
    nll_val = []
    acc_val = []
    kl_val = []
    mse_val = []
    preds = []
    tars = []
    eds = []
    for batch_idx, (data, relations) in enumerate(test_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data, volatile=True), Variable(
            relations, volatile=True)

        logits = encoder(data, rel_rec.to_sparse(), rel_send.to_sparse())
        edges = gumbel_softmax(logits, tau=args.temp, hard=True)
        prob = my_softmax(logits, -1)
        eds.append(edges.detach().cpu().numpy())
        # validation output uses teacher forcing
        if args.decoder == 'rnn':
            output = decoder(data, edges, rel_rec.to_sparse(), rel_send.to_sparse(), 1)
        else:
            output = decoder(data, edges, rel_rec, rel_send, 1)
        target = relations[:,1:]
        output = output.transpose(1,2).contiguous().view(args.batch_size,args.timesteps-1,-1)
        output = fc_out(output)
        output = output.squeeze(2)

        # loss_nll = nll_gaussian(output, target, args.var)
        loss_nll = F.mse_loss(output, target)
        # loss_nll = nll_gaussian(output, target, args.var)
        loss_kl = kl_categorical_uniform(prob, args.num_atoms, args.edge_types)

        acc = edge_accuracy2(logits, relations)
        acc_val.append(acc)

        mse_val.append(F.mse_loss(output, target).item())
        nll_val.append(loss_nll.item())
        kl_val.append(loss_kl.item())
        preds.append(output.detach().cpu().numpy().flatten())
        tars.append(target.detach().cpu().numpy().flatten())

    logs['test']['pred'] = np.concatenate(preds)
    logs['test']['truth'] = np.concatenate(tars)
    logs['edges']['test'] = eds
    pickle.dump(logs, open(os.path.join(save_folder,'logs.pkl'),"wb" ))
    # pd.DataFrame(logs).to_pickle(os.path.join(save_folder,'logs.pickle'))
    print('Epoch: {:04d}'.format(epoch),

          'nll_val: {:.10f}'.format(np.mean(nll_val)),
          'kl_val: {:.10f}'.format(np.mean(kl_val)),
          'mse_val: {:.10f}'.format(np.mean(mse_val)),
          'acc_val: {:.10f}'.format(np.mean(acc_val)),
          'time: {:.4f}s'.format(time.time() - t))
    if args.save_folder and np.mean(nll_val) < best_val_loss:
        torch.save(encoder.state_dict(), encoder_file)
        torch.save(decoder.state_dict(), decoder_file)
        print('Epoch: {:04d}'.format(epoch),
              'nll_val: {:.10f}'.format(np.mean(nll_val)),
              'kl_val: {:.10f}'.format(np.mean(kl_val)),
              'mse_val: {:.10f}'.format(np.mean(mse_val)),
              'acc_val: {:.10f}'.format(np.mean(acc_val)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
    return np.mean(nll_val)

# Train model
t_total = time.time()
best_val_loss = np.inf
best_epoch = 0
for epoch in range(args.epochs):
    val_loss = train(epoch, best_val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch))
if args.save_folder:
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()
    # print(save_folder)
    # log.close()

test2(epoch,best_val_loss)
if log is not None:
    print(save_folder)
    log.close()
