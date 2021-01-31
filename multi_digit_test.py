from mnist_dl import *
from segmentation import *
from PIL import Image
from tqdm import tqdm

import seaborn as sns

def show_pred(img, ind, digits, preds, IMG_NAME = "res.png"):
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    frame = plt.imshow(img, cmap='gray_r')
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title(ind)
    for i in range(5, 9):
        plt.subplot(2, 4, i)
        frame = plt.imshow(digits[i-5], cmap='gray_r')
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        plt.title(int(preds[i-5]))
    # plt.show()
    plt.savefig(IMG_NAME)

def show_corr(correct):
    fig=plt.figure()
    columns = 4
    rows = 4
    i = 1
    for corr in correct[:16]:
        img, sum, prob = corr
        fig.add_subplot(rows,columns,i)
        title = f'{sum} ({prob:.0f}%)'
        plt.title(title, fontsize=7)
        plt.axis('off')
        plt.imshow(img, cmap='gray_r')
        i+=1
    fig.suptitle('Final Test predictions')
    # plt.show()
    plt.savefig('result.png')

def plot_hist(file, title):
    # modify the default parameters of np.load
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    metrics = np.load(file)
    np.load = np_load_old

    sns.set_style('darkgrid')
    colors = sns.color_palette("tab10")
    sns.distplot(metrics[0], color=colors[0], label="0")
    sns.distplot(metrics[1], color=colors[1], label="1")
    sns.distplot(metrics[2], color=colors[2], label="2")
    sns.distplot(metrics[3], color=colors[3], label="3")
    sns.distplot(metrics[4], color=colors[4], label="4")
    sns.distplot(metrics[5], color=colors[5], label="5")
    sns.distplot(metrics[6], color=colors[6], label="6")
    sns.distplot(metrics[7], color=colors[7], label="7")
    sns.distplot(metrics[8], color=colors[8], label="8")
    sns.distplot(metrics[9], color=colors[9], label="9")
    plt.legend()
    plt.xlabel(title)
    plt.savefig(title+".png")

def test_set(model, imgs, labels):
    num = len(labels)
    transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()])
    num_correct = 0
    correct = []
    incorrect = []
    corr_areas = [ [],[],[],[],[],[],[],[],[],[] ]
    corr_widths = [ [],[],[],[],[],[],[],[],[],[] ]
    for i in tqdm(range(0,num)):
        img = imgs[i]
        digits, areas, widths = segment(img)
        if len(digits) == 4:
            preds = []
            probs = []
            sum = 0
            for j,digit in enumerate(digits):
                digit = transform(Image.fromarray(digit)).unsqueeze(0)
                pred, prob = predict(model, digit, device)
                preds.append(pred)
                probs.append(prob)
                sum+=pred
            
            if sum==labels[i]:
                avg_prob = 0
                for j in range(0,4):
                    pred = int(preds[j])
                    corr_widths[pred].append(widths[j])
                    corr_areas[pred].append(areas[j])
                    avg_prob += probs[j]
                avg_prob /= 4
                correct.append([img, sum, avg_prob])
                num_correct+=1
            else:
                incorrect.append([img, i, digits, preds])

    print("Accuracy :"+ str((num_correct*100)/num))
    return incorrect, correct, corr_areas, corr_widths

if __name__ == "__main__":
    imgs = np.load('Data/data0.npy')
    imgs1 = np.load('Data/data1.npy')
    imgs2 = np.load('Data/data2.npy')
    labels = np.load('Data/lab0.npy')
    labels1 = np.load('Data/lab1.npy')
    labels2 = np.load('Data/lab2.npy')

    imgs = np.append(imgs, imgs1, axis=0)
    imgs = np.append(imgs, imgs2, axis=0)

    labels = np.append(labels, labels1, axis=0)
    labels = np.append(labels, labels2, axis=0)
    
    model = Model(10).to(device)
    model.load_state_dict(torch.load('model.dth'))

    incorrect, correct, corr_areas, corr_widths = test_set(model, imgs, labels)
    # np.save('Data/correct', correct)
    # np.save('Data/incorrect', incorrect)
    # np.save('Data/corr_areas', corr_areas)
    # np.save('Data/corr_widths', corr_widths)


    # Stats
    # plot_hist('corr_areas.npy', 'areas')

    # np_load_old = np.load
    # np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    # incorrect = np.load('Data/incorrect.npy')
    # correct = np.load('Data/correct.npy')
    # np.load = np_load_old
    # show_corr(correct)
    
    #for i in range(0,100):
    #    show_pred(*incorrect[i] , IMG_NAME='Data/'+str(i)+'res.png')