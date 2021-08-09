from tokenizers import Tokenizer
from common import *
import pickle
from configure import *


def pad_sequence_to_max_length(sequence, max_length, padding_value):
    batch_size =len(sequence)
    pad_sequence = np.full((batch_size,max_length), padding_value, np.int32)
    for b, s in enumerate(sequence):
        L = len(s)
        pad_sequence[b, :L, ...] = s
    return pad_sequence

def load_tokenizer():
    tokenizer = Tokenizer.from_file("../data/tokenizer.json")
    return tokenizer


def make_fold():
    df = pickle.load(open("../data/df_cgnn_1H.pkl", "rb"))
    df.drop_duplicates("molecule_id", 'first', inplace=True)
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(df))
    valid_set_size = int(len(df) * 0.25)
    valid_indices = shuffled_indices[:valid_set_size]
    train_indices = shuffled_indices[valid_set_size:]
    df_train = df.iloc[train_indices]
    df_valid = df.iloc[valid_indices]
    return df_train, df_valid


def null_augment(r):
    image = r['image']
    image = cv2.resize(image, dsize=(image_size,image_size), interpolation=cv2.INTER_LINEAR)
    r['image'] = image
    return r

class NmrDataset(Dataset):
    def __init__(self, df, tokenizer, augment=null_augment):
        super().__init__()
        self.tokenizer = tokenizer
        self.df = df
        self.augment = augment
        self.length = len(self.df)

    def __len__(self):
        return self.length

    def __getitem__(self,index):
        d = self.df.iloc[index]
        image_file = "../data/cgnn_1H/" + d.path
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        smile = d.smile
        output = self.tokenizer.encode(smile)
        token = output.ids
        r = {
            'index': index,
            'image': image,
            'token': token,
        }

        if self.augment is not None: r = self.augment(r)
        return r

def collate_fn(batch, tokenizer):
    collate = defaultdict(list)

    for r in batch:
        for k, v in r.items():
            collate[k].append(v)

    collate['length'] = [len(l) for l in collate['token']]

    token  = [np.array(t,np.int32) for t in collate['token']]
    token  = pad_sequence_to_max_length(token, max_length=max_length, padding_value=tokenizer.token_to_id("[<pad>]"))
    collate['token'] = torch.from_numpy(token).long()

    image = np.stack(collate['image'])
    image = image.astype(np.float32) / 255
    collate['image'] = torch.from_numpy(image).unsqueeze(1).repeat(1,3,1,1)

    return collate
