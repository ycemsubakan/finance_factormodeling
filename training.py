import torch
import torch.nn as nn      
import torch.nn.functional as F
import matplotlib.pyplot as plt 

# define the model
class attention_predictor(nn.Module):
    def __init__(self, num_chars, hidden_size):
        super(attention_predictor, self).__init__()
        self.rnn = nn.LSTM(input_size=num_chars, hidden_size=hidden_size, batch_first=True, bidirectional=False)
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=num_chars)

    def forward(self, inp):
        
        # inp m x T x N
        h = self.rnn(inp)[0]
        y = self.output_layer(h)
        return F.softmax(y, dim=-1)  # m x T x N

def plot_figure(chars, returns, ranking, savename):
    plt.figure(figsize=[30, 5], dpi=100)

    # plot returns
    plt.subplot(5,1,1)
    plt.tight_layout()
    plt.plot(returns[:, 0],'r')
    plt.grid()
    plt.title('Return')

    for i in range(1,5):
        plt.subplot(5,1,i+1)
        plt.plot(chars[0, :, ranking[i-1]])
        plt.grid()
        plt.title('Characteristic {}'.format(i))

    #plt.subplots_adjust(bottom=0.4, top=0.5)
    plt.savefig('{}.png'.format(savename))


def plot_returns(returns, ranking, savename):
    plt.figure(figsize=[30, 25], dpi=100)

    # plot returns
    plt.tight_layout()

    for i in range(1,ranking.shape[0]):
        plt.subplot(ranking.shape[0],1,i)
        plt.plot(returns[:, ranking[i-1]])
        total_return = returns[:, ranking[i-1]].sum()
        plt.grid()
        plt.title('Returns for stock {}, total return is {}'.format(i, total_return))

    #plt.subplots_adjust(bottom=0.4, top=0.5)
    plt.savefig('{}.png'.format(savename))





if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load data
    chars = torch.load('Simu/SimuData_p100/c1.t').permute(2, 1, 0).to(device)
    returns = torch.load('Simu/SimuData_p100/r1_1.t').permute(1, 0).to(device)

    # split the data
    T_train = 150

    chars_train = chars[:, :T_train, :]
    returns_train = returns[:T_train, :]

    chars_test = chars[:, T_train:, :]
    returns_test = returns[T_train:, :]

    model = attention_predictor(returns.shape[-1], 100).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    # iterate over training epochs

    EP = 4000
    for ep in range(EP):
    # iterate over the batches
        opt.zero_grad()
           
        weights = model.forward(chars_train)
        mean_return = (weights * returns_train.unsqueeze(0)).mean()
        neg_return = - mean_return
        
        # compute the gradient
        neg_return.backward()

        # take a training step
        opt.step()

        # spit out the current loss and other things on the screen
        print('EP [{}/{}], \
               Average return is {}, Learning rate is {}'.format(ep+1, EP, 
                                                                 mean_return.item(),
                                                                 opt.param_groups[0]['lr']))

    # get the results on the test set
    selected_char_ind = 2

    weights_test = model.forward(chars_test)
    weights_mean_test = weights_test.mean(1)
    ranking_test = weights_mean_test[selected_char_ind].sort(descending=True)
    ten_best_test = ranking_test[1][0:250:10]

    # sort according to the estimated weigts 
    # first sum over time
    weights_mean = weights.mean(1)
    # then sort
    ranking = weights_mean[selected_char_ind].sort(descending=True)
    # plot the 4-best ones
    ten_best = ranking[1][0:250:10]
    plot_returns(returns_train.cpu(), ten_best.cpu(), 'train_ranked')


    #  weights_mean_test[46].sort(descending=True)[1][:10] 
    plot_returns(returns_test.cpu(), ten_best.cpu(), 'test_ranked')

    #plot_figure(chars_train.cpu(), returns_train.cpu(), four_best.cpu(), 'train_ranked')
    #plot_figure(chars_train.cpu(), returns_train.cpu(), torch.arange(1,5), 'train_random')


    
    
