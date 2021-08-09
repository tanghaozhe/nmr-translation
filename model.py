import sys
sys.path.append("..")
# from common import *
from configure import *
from resnet.resnet_26d import *
from torch.nn.utils.rnn import pack_padded_sequence
from transformer import *


image_dim = 1024
text_dim  = 1024
decoder_dim = 1024
num_layer = 2
num_head = 8
ff_dim = 1024
num_pixel=7*7

 
class SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x * torch.sigmoid(x)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        return grad_output * (sigmoid * (1 + x * (1 - sigmoid)))

F_swish = SwishFunction.apply

class Swish(nn.Module):
    def forward(self, x):
        return F_swish(x)





class Encoder(nn.Module):
    def __init__(self,):
        super(Encoder, self).__init__()

        e = make_resnet_26d()

        self.p1 = nn.Sequential(
            e.conv1,
            e.bn1,
            e.act1,
        )
        self.p2 = nn.Sequential(
            e.maxpool,
            e.layer1,
        )
        self.p3 = e.layer2
        self.p4 = e.layer3
        self.p5 = e.layer4
        del e  # dropped

    def forward(self, image):
        batch_size, C, H, W = image.shape
        x = 2 * image - 1

        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)
        x = self.p5(x)
        return x



class Net(nn.Module):
    def __init__(self,):
        super(Net, self).__init__()
        self.encoder = Encoder()
        self.project = nn.Sequential(
            nn.Conv2d(2048,image_dim, kernel_size=1, bias=None),
            nn.BatchNorm2d(image_dim),
            Swish()
        )

        self.token_embed = nn.Embedding(vocab_size, text_dim)
        self.image_embed = TransformerEncode(image_dim, ff_dim, num_head, num_layer)

        self.text_pos  = PositionEncode1D(text_dim,max_length)
        self.image_pos = PositionEncode2D(image_dim,int(num_pixel**0.5)+1,int(num_pixel**0.5)+1)

        self.transformer = TransformerDecode(decoder_dim, ff_dim, num_head, num_layer)

        self.logit  = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

        # initialization
        self.token_embed.weight.data.uniform_(-0.1, 0.1)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-0.1, 0.1)



    def forward(self, image, token, length):
        device = image.device
        batch_size = len(image)

        image_mask = None #torch.ones((batch_size, num_pixel,num_pixel),dtype=torch.float32 ).to(device)
        #>>> a = torch.empty(3, 3).uniform_(0, 1)  # generate a uniform random matrix with range [0, 1]
        #torch.bernoulli(a)


        image_embed = self.encoder(image)
        image_embed = self.project(image_embed)
        image_embed = self.image_pos(image_embed)
        image_embed = image_embed.permute(0, 2, 3, 1).contiguous()
        image_embed = image_embed.reshape(batch_size, num_pixel, image_dim)
        image_embed = self.image_embed(image_embed,image_mask)

        text_embed = self.token_embed(token)
        text_embed = self.text_pos(text_embed)


        text_mask = 1 - np.triu(np.ones((batch_size, max_length, max_length)), k=1).astype(np.uint8)
        text_mask = torch.autograd.Variable(torch.from_numpy(text_mask)).to(device)
        text_image_mask = None
        x = self.transformer(text_embed, image_embed, text_mask, text_image_mask)
        logit = self.logit(x)
        return logit


    @torch.jit.export
    def forward_argmax_decode(self, image):
        device = image.device
        batch_size = len(image)

        image_embed = self.encoder(image)
        image_embed = self.project(image_embed)
        image_embed = self.image_pos(image_embed)
        image_embed = image_embed.permute(0, 2, 3, 1).contiguous()
        image_embed = image_embed.reshape(batch_size, num_pixel, image_dim)
        image_embed = self.image_embed(image_embed, None)

        token = torch.full((batch_size, max_length), STOI['[<pad>]'],dtype=torch.long).to(device)
        text_pos = self.text_pos.pos
        token[:,0] = STOI['[<start>]']

        for t in range(max_length-1):
            last_token = token [:,:(t+1)]
            text_embed = self.token_embed(last_token)
            text_embed = self.text_pos(text_embed) #text_embed + text_pos[:,:(t+1)] #

            text_mask = 1 - np.triu(np.ones((batch_size, t+1, t+1)), k=1).astype(np.uint8)
            text_mask = torch.autograd.Variable(torch.from_numpy(text_mask)).to(device)

            x = self.transformer(text_embed, image_embed, text_mask, None)
            l = self.logit(x[:,-1])
            k = torch.argmax(l, -1)  # predict max
            token[:, t+1] = k
            if ((k == STOI['[<end>]']) | (k == STOI['[<pad>]'])).all():  break
        predict = token[:, 1:]
        return predict


def seq_cross_entropy_loss(logit, token, length):
    truth = token[:, 1:]
    L = [l - 1 for l in length]
    logit = pack_padded_sequence(logit, L, batch_first=True, enforce_sorted=False).data
    truth = pack_padded_sequence(truth, L, batch_first=True, enforce_sorted=False).data
    loss = F.cross_entropy(logit, truth, ignore_index=STOI['[<pad>]'])
    return loss
