from mnist_dl import *
from segmentation import *
from PIL import Image
from matplotlib import cm
from tqdm import tqdm

if __name__ == "__main__":
    imgs = np.load('Data/data0.npy')
    labels = np.load('Data/lab0.npy')
    num = len(labels)

    model = Model(10).to(device)
    model.load_state_dict(torch.load('model.dth'))

    transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()])

    correct = 0
    for i in tqdm(range(0,num)):
        img = imgs[i]
        digits = segment(img)
        if len(digits) == 4:
            sum = 0
            for digit in digits:
                digit = transform(Image.fromarray(digit)).unsqueeze(0)
                pred = predict(model, digit, device)
                sum+=pred
            
            if sum==labels[i]:
                correct+=1

    print(correct)
    print(num)
    print("ACC :"+ str(correct/num) )