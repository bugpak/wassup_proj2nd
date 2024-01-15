import matplotlib.pyplot as plt

def get_r2_graph(pred_prod, tst_prod, pred_cons, tst_cons, name):
    fig, axes = plt.subplots(2,2,figsize=(13,10))
    
    axes[0][0].scatter(x=pred_prod, y=tst_prod, color='b', label='loss')
    plt.ylabel('real y')
    plt.xlabel('pred y')
    axes[0][1].scatter(x=pred_cons, y=tst_cons, color='r', label='val_loss')
    plt.title('r2score visualization')
    plt.ylabel('real y')
    plt.xlabel('pred y')
    axes[1][0].plot(tst_prod,  color='b', label='real')
    axes[1][0].plot(pred_prod,  color='r', label='pred')
    axes[1][1].plot(tst_cons,  color='b', label='real')
    axes[1][1].plot(pred_cons,  color='r', label='pred')
    plt.legend()
    plt.show()
    plt.savefig('graph/graph_{}.png'.format(name))
    return

def get_graph(history:list,name):
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    
    axes[0].plot(history['trn_loss'], 'b-', label='trn_loss')
    axes[0].plot(history['val_loss'], 'r-', label='val_loss')
    plt.title('monitoring loss')
    plt.xlabel('Epoch')
    axes[1].plot(history['lr'], 'r-', label='lr')
    plt.title('monitoring learning rate')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('graph/loss_graph_{}.png'.format(name))
    return