import os
from model import *
from dataset import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

df = pickle.load(open("../data/df_cgnn_1H.pkl", "rb"))

class Net(Net):
    def forward(self, *args):
        return super(Net, self).forward_argmax_decode(*args)


def predict(net, tokenizer, valid_loader):
    result = []
    for t, batch in enumerate(valid_loader):
        batch_size = len(batch['index'])
        image = batch['image'].cuda()
        net.eval()
        with torch.no_grad():
            k = net(image)
            k = k.data.cpu().numpy()
            label = tokenizer.decode(batch["token"][0].tolist()).replace(' ', '')
            prediction = tokenizer.decode(k[0].tolist()).replace(' ', '')
            result.append([label, prediction])
    return result


def run_predict():
    out_dir = \
        './result/'
    initial_checkpoint = \
        './result/checkpoint/00062000_model.pth'#

    tokenizer = load_tokenizer()
    df_train, df_valid = make_fold()
    df_valid = df_valid[:20]

    valid_dataset = NmrDataset(df_valid, tokenizer)
    valid_loader = DataLoader(
        valid_dataset,
        sampler=SequentialSampler(valid_dataset),
        batch_size=1,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
    )
    net = Net().cuda()
    net.load_state_dict(torch.load(initial_checkpoint)['state_dict'], strict=True)

    result = predict(net, tokenizer, valid_loader)
    result = pd.DataFrame(columns=["label","prediction"], data=result)
    print(result)
    # result.to_csv(out_dir+"result.csv", encoding='utf-8')

if __name__ == '__main__':
    run_predict()


