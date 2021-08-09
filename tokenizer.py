from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import Whitespace, Digits
from tokenizers import pre_tokenizers
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
import pickle

df_cgnn_1H = pickle.load(open('../data/df_cgnn_1H.pkl', "rb"))
smiles = df_cgnn_1H["smile"].values

tokenizer = Tokenizer(BPE(unk_token="[<unk>]", fuse_unk=False))
pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits()])
tokenizer.pre_tokenizer = pre_tokenizer
# max_len = 0
trainer = BpeTrainer(special_tokens=["[<pad>]", "[<start>]", "[<end>]", "[<unk>]"])
tokenizer.train_from_iterator(smiles, trainer=trainer)
tokenizer.post_processor = TemplateProcessing(
    single="[<start>] $A [<end>]",
    special_tokens=[
        ("[<start>]", tokenizer.token_to_id("[<start>]")),
        ("[<end>]", tokenizer.token_to_id("[<end>]")),
    ],
)
tokenizer.save("../data/tokenizer.json")
# sample_smile = "[H]O[C@]1([H])C([H])([H])[C@@]2([H])[C@@](C([H])([H])[H])(C([H])([H])[H])[C@@]2([H])[C@@]2([H])[C@]([H])(C([H])([H])C([H])([H])[C@@]2([H])C([H])([H])[H])[C@]1([H])C([H])([H])[H]"
# output = tokenizer.encode(sample_smile)
# print(tokenizer.get_vocab()['[<pad>]'])  #0
# print(tokenizer.decode(output.ids))

