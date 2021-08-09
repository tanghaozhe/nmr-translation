import os
import sys
from common import *
from lib.net.lookahead import *
from lib.net.radam import *
from model import *
from dataset import *

sys.path.append("..")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def do_valid(net, tokenizer, valid_loader):
    valid_probability = []
    valid_truth = []
    valid_length = []
    valid_num = 0

    net.eval()
    start_timer = timer()
    for t, batch in enumerate(valid_loader):
        batch_size = len(batch['index'])
        image  = batch['image' ].cuda()
        token  = batch['token' ].cuda()
        length = batch['length']

        with torch.no_grad():
            logit = data_parallel(net, (image, token, length))
            probability = F.softmax(logit, -1)

        valid_num += batch_size
        valid_probability.append(probability.data.cpu().numpy())
        valid_truth.append(token.data.cpu().numpy())
        valid_length.extend(length)
        print('\r %8d / %d  %s'%(valid_num, len(valid_loader.sampler),time_to_str(timer() - start_timer,'sec')),end='',flush=True)

    probability = np.concatenate(valid_probability)
    predict = probability.argmax(-1)
    truth = np.concatenate(valid_truth)
    length = valid_length

    p = probability[:,:-1].reshape(-1,vocab_size)
    t = truth[:,1:].reshape(-1)

    non_pad = np.where(t!=STOI['[<pad>]'])[0]
    p = p[non_pad]
    t = t[non_pad]
    loss = np_loss_cross_entropy(p, t)

    return [loss]


def run_train():
    out_dir = "./result"
    initial_checkpoint = None
    start_lr = 0.0001
    batch_size = 1

    # setup
    for f in ['checkpoint', 'train', 'valid', 'backup']: os.makedirs(out_dir + '/' + f, exist_ok=True)

    log = Logger()
    log.open(out_dir + '/log.train.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t__file__ = %s\n' % __file__)
    log.write('\tout_dir  = %s\n' % out_dir)
    log.write('\n')

    df_train, df_valid = make_fold()

    tokenizer = load_tokenizer()
    train_dataset = NmrDataset(df_train,tokenizer)
    valid_dataset = NmrDataset(df_valid,tokenizer)

    train_loader = DataLoader(
        train_dataset,
        sampler=SequentialSampler(train_dataset),
        batch_size=batch_size,
        drop_last=True,
        num_workers=12,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )
    valid_loader = DataLoader(
        valid_dataset,
        sampler=SequentialSampler(valid_dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=12,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )

    log.write('** dataset setting **\n')
    log.write('train_dataset : \n%s\n' % (train_dataset))
    log.write('valid_dataset : \n%s\n' % (valid_dataset))
    log.write('\n')


    net = Net().cuda()

    if initial_checkpoint is not None:
        f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        start_iteration = f['iteration']
        start_epoch     = f['epoch']
        state_dict = f['state_dict']
        #decoder_state_dict = f['decoder_state_dict']
        #decoder_state_dict = { k.replace('image_pos','image_pos1'):v for k,v, in decoder_state_dict.items()}

        net.load_state_dict(state_dict, strict=True)  # True

    else:
        start_iteration = 0
        start_epoch = 0

    log.write('** net setting **\n')
    log.write('initial_checkpoint : %s\n' % (initial_checkpoint))
    log.write('\n')


    optimizer = Lookahead(RAdam(filter(lambda p: p.requires_grad,net.parameters()), lr=start_lr), alpha=0.5, k=5)
    # optimizer = RAdam(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr)

    num_iteration = 80000 * 1000
    iter_log = 1000
    iter_valid = 1000
    iter_save = list(range(0, num_iteration, 1000))  # 1*1000

    log.write('optimizer\n  %s\n' % (optimizer))
    log.write('\n')



    ## start training here! ##############################################
    log.write('** start training here! **\n')
    # log.write('   is_mixed_precision = %s \n' % str(is_mixed_precision))
    log.write('   batch_size = %d\n' % (batch_size))
    log.write('   experiment = %s\n' % str(__file__.split('/')[-2:]))
    log.write('                      | VALID  | TRAIN ----------------\n')
    log.write('rate     iter   epoch | loss   | loss  | time          \n')
    log.write('-------------------------------------------------------\n')
             # 0.00000   0.00* 0.00  | 0.000  | 0.000  |  0 hr 00 min



    def message(mode='print'):
        if mode == ('print'):
            asterisk = ' '
            loss = batch_loss
        if mode == ('log'):
            asterisk = '*' if iteration in iter_save else ' '
            loss = train_loss

        text = \
            '%0.5f  %5.4f%s %4.2f  | ' % (rate, iteration / 10000, asterisk, epoch,) + \
            '%4.3f  | ' % (valid_loss[0],) + \
            '%4.3f  | ' % (loss[0],) + \
            '%s' % (time_to_str(timer() - start_timer, 'min'))

        return text

    # ----
    valid_loss = np.zeros(2, np.float32)
    train_loss = np.zeros(3, np.float32)
    batch_loss = np.zeros_like(train_loss)
    sum_train_loss = np.zeros_like(train_loss)
    sum_train = 0
    loss0 = torch.FloatTensor([0]).cuda().sum()

    start_timer = timer()
    iteration = start_iteration
    epoch = start_epoch
    rate = 0
    while iteration < num_iteration:
        for t, batch in enumerate(train_loader):
            if iteration in iter_save:
                if iteration != start_iteration:
                    torch.save({
                        'state_dict': net.state_dict(),
                        'iteration': iteration,
                        'epoch': epoch,
                    }, out_dir + '/checkpoint/%08d_model.pth' % (iteration))
                    pass

            if (iteration % iter_valid == 0):
                if iteration != start_iteration:
                    valid_loss = do_valid(net, tokenizer, valid_loader)  #
                    pass

            if (iteration % iter_log == 0):
                print('\r', end='', flush=True)
                log.write(message(mode='log') + '\n')

            # learning rate schduler
            rate = get_learning_rate(optimizer)

            # one iteration update
            batch_size = len(batch['index'])
            image = batch['image' ].cuda()
            token = batch['token' ].cuda()
            length = batch['length']


            net.train()
            optimizer.zero_grad()

            logit = net(image, token, length)
            loss0 = seq_cross_entropy_loss(logit, token, length)

            (loss0).backward()
            optimizer.step()

            # print statistics  --------
            epoch += 1 / len(train_loader)
            iteration += 1

            batch_loss = np.array([loss0.item()])
            sum_train_loss += batch_loss
            sum_train += 1
            if iteration % 100 == 0:
                train_loss = sum_train_loss / (sum_train + 1e-12)
                sum_train_loss[...] = 0
                sum_train = 0

            print('\r', end='', flush=True)
            print(message(mode='print'), end='', flush=True)


    log.write('\n')


if __name__ == '__main__':
    run_train()



