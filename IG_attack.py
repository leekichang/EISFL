import os
import copy
import torch
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms

import ops
import utils
import attacks
import config as cfg
import DataManager.datamanager as dm
from imgMetrics import semantic as sm
from imgMetrics import structural as st

def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)

def sanitize(eps, delta, tensor, sigma=None, clip_norm=2.0, add_noise=True):
    if sigma is None:
        sigma = np.sqrt(2 * np.log(1.25 / delta)) / eps
    
    clip_norm = clip_norm
    
    tensor_norm = torch.norm(tensor, p=2)
    if tensor_norm > clip_norm:
        tensor = tensor * (clip_norm / tensor_norm)
    
    if add_noise:
        # noise = torch.normal(mean=0, std=sigma * clip_norm, size=tensor.shape).to(tensor.device)
        noise = torch.normal(mean=0, std=sigma, size=tensor.shape).to(tensor.device)
        tensor = tensor + noise
    return tensor

def sanitize_gradients(model, eps, delta, clip_norm=2.0):
    for p in model.parameters():
        if p.grad is not None:
            sanitized_grad = sanitize(eps, delta, p.grad.data, clip_norm=clip_norm, add_noise=True)
            p.grad.data = sanitized_grad

args = utils.parse_args()
args.n_user = 1
args.optimizer = "SGD"
# args.dataset = "cifar100"
args.attack_name = 'IG'
args.lr = 1
args.device = 'cuda'
# args.defense = 'GradPrune'
torch.manual_seed(args.seed)
N_SAMPLE = 100

args.model = 'RNN'
args.dataset = 'MITBIH'
# testset  = dm.MNIST()[1]
# testset  = dm.CIFAR10()[1]
testset = dm.MITBIH_Dataset(root='../../../disk1/Kichang/dataset/mitbih_arr', split='test')
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# transform  = transforms.Compose([transforms.ToTensor(),])
#                                  #transforms.Resize(64)])
# testset  = datasets.STL10(root='../../../disk1/Kichang/dataset', split='test'  , download=False, transform=transform)
# testset.targets  = testset.labels

print(testset.data[0].shape)
print(len(testset))

img_idx = torch.LongTensor(range(N_SAMPLE))

criterion = cross_entropy_for_onehot
calc_mse = st.MSE()
calc_psnr = st.PSNR()
calc_ssim = st.SSIM()
calc_tv = st.TotalVariation()
calc_lpips = sm.LPIPS()
results = {'mse':[], 'psnr':[], 'ssim':[], 'lpips':[]}
new_model = utils.build_model(args)
# new_model.apply(weights_init)
new_model.train()
old_model = copy.deepcopy(new_model)
new_model, old_model = new_model.to(args.device), old_model.to(args.device)
patch_shuffle = torch.randperm((cfg.IMGSIZE[args.dataset][-1]//cfg.PATCHSIZE[args.dataset])**2)
pixel_shuffle = torch.randperm(cfg.PATCHSIZE[args.dataset])
img_shuffle = ops.ImageShuffle(patch_size=(cfg.PATCHSIZE[args.dataset], cfg.PATCHSIZE[args.dataset]), patch_shuffle=patch_shuffle, pixel_shuffle=pixel_shuffle)    
ts_shuffle  = ops.TSShuffle(patch_size=(cfg.PATCHSIZE[args.dataset], cfg.PATCHSIZE[args.dataset]), patch_shuffle=patch_shuffle, pixel_shuffle=pixel_shuffle)

# for idx in tqdm(img_idx):
with torch.backends.cudnn.flags(enabled=False):
    for idx in img_idx:
        gt_img, gt_label = testset[idx]
        gt_img = gt_img[None,].to(args.device)
        gt_onehot_label = label_to_onehot(torch.tensor(gt_label).long()[None,], num_classes=cfg.N_CLASS[args.dataset]).to(args.device)
        new_model.load_state_dict(copy.deepcopy(old_model.state_dict()))
        optimizer = utils.build_optimizer(new_model, args)
        optimizer.zero_grad()

        if 'EISFL' in args.defense:
            if args.dataset=='MITBIH':
                gt_img_ = ts_shuffle(gt_img)
                print(gt_img_.shape)
            else:
                gt_img_ = img_shuffle.shuffle(gt_img)
        else:
            gt_img_ = gt_img
        if args.dataset=='MITBIH':
            #gt_img_ = gt_img_[:,0:1,:]
            gt_img_ = gt_img_
            print(gt_img_.shape)
        pred = new_model(gt_img_)
        loss = criterion(pred, gt_onehot_label)
        loss.backward()
        if 'GradPrune' in args.defense:
            parameters = new_model.parameters()
            p = float(args.defense.split('_')[-1])
            if isinstance(parameters, torch.Tensor):
                parameters = [parameters]
                parameters = list(
                    filter(lambda p: p.grad is not None, parameters))
            input_grads = [p.grad.data for p in parameters]
            threshold = [torch.quantile(torch.abs(input_grads[i]), p) for i in range(len(input_grads))]
            for i, p in enumerate(new_model.parameters()):
                p.grad[torch.abs(p.grad) < threshold[i]] = 0
        elif 'DP' in args.defense:
            for p in new_model.parameters():
                if p.grad is not None:
                    tensor = p.grad.data
                    sensitivity = 0.005 # MNIST 0.01, CIFAR10 0.01 
                    clip_norm = 4 # args.clip_norm
                    tensor_norm = torch.norm(tensor, p=2)
                    if tensor_norm > clip_norm:
                        tensor = tensor * (clip_norm / tensor_norm)
                    # sigma = np.sqrt(2 * np.log(1.25/args.delta)) / args.epsilon
                    sigma = sensitivity*(np.sqrt(2*np.log(1.25/args.delta))) / (args.epsilon)
                    # print("SIGMA", sigma)
                    noise = torch.normal(mean=0, std=sigma, size=p.shape).to(tensor.device)
                    sanitized_grad = tensor + noise
                    p.grad.data = sanitized_grad
            print("SIGMA", sigma)
        optimizer.step()
        
        if args.defense == 'eisfl':
            torch.manual_seed(args.seed)
            patch_shuffle = None # torch.randperm((cfg.IMGSIZE[args.dataset][-1]//cfg.PATCHSIZE[args.dataset])**2)
            pixel_shuffle = None # torch.randperm(cfg.PATCHSIZE[args.dataset]**2)
            print(f'Patch shuffle: {patch_shuffle}')
            print(f'Pixel shuffle: {pixel_shuffle}')
            shuffler = ops.WeightShuffle(new_model.state_dict(),
                                        cfg.PATCHSIZE[args.dataset],
                                        patch_shuffle=patch_shuffle,
                                        pixel_shuffle=pixel_shuffle)
            print("Before Shuffle", new_model.state_dict()['mixer_blocks.0.token_mixer.2.net.0.weight'][0, :10])
            print("Before Shuffle (old)", old_model.state_dict()['mixer_blocks.0.token_mixer.2.net.0.weight'][0, :10])
            shuffler.set_new_weight(new_model.state_dict())
            shuffled_weight = shuffler.shuffle(model=args.model)
            print("Shuffled Param", shuffled_weight['mixer_blocks.0.token_mixer.2.net.0.weight'][0,:10])
            new_model.load_state_dict(shuffled_weight)
            print("After Shuffle", new_model.state_dict()['mixer_blocks.0.token_mixer.2.net.0.weight'][0,:10])
            print("After Shuffle (old)", old_model.state_dict()['mixer_blocks.0.token_mixer.2.net.0.weight'][0, :10])
        attacker = attacks.IG(args)
        if args.defense == 'DP':
            attacker.name = f"results/{args.defense}/{attacker.name}_{args.dataset}_{args.epsilon}_{args.delta}"
        elif 'GradPrune' in args.defense:
            attacker.name = f"results/{args.defense}/{attacker.name}_{args.dataset}_{args.defense}"
        else:
            attacker.name = f"results/{args.defense}/{attacker.name}_{args.dataset}"
        os.makedirs(attacker.name, exist_ok=True)
        # attacker.attack(copy.deepcopy(old_model), copy.deepcopy(new_model), gnd_labels=torch.tensor(gt_label).long()[None,])
        attacker.attack(copy.deepcopy(old_model), copy.deepcopy(new_model), gnd_labels=torch.tensor(gt_label).long()[None,], gnd_data=gt_img_)
        
        attack_img  = attacker.history[attacker.best_idx]
        if args.dataset != 'MITBIH':
            attack_img.save(f'{attacker.name}/{idx}_result.png')
        else:
            plt.plot(np.array(attack_img).reshape(-1))
            plt.savefig(f'{attacker.name}/{idx}_result.png')
            plt.close()
            np.save(f'{attacker.name}/{idx}_result.npy', attack_img)

        attack_img  = attacker.history[-1]
        if args.dataset != 'MITBIH':
            attack_img.save(f'{attacker.name}/{idx}_result_last.png')
        else:
            plt.plot(np.array(attack_img).reshape(-1))
            plt.savefig(f'{attacker.name}/{idx}_result_last.png')
            plt.close()
            np.save(f'{attacker.name}/{idx}_result_last.npy', attack_img)
        if args.dataset != 'MITBIH':
            img = transforms.ToPILImage()(gt_img_[0].cpu())
            if args.defense == 'NoDefense':
                img.save(f'{attacker.name}/{idx}.png')
        else:
            plt.plot(gt_img[0].cpu().numpy().reshape(-1))
            plt.savefig(f'{attacker.name}/{idx}.png')
            plt.close()
            np.save(f'{attacker.name}/{idx}.npy', gt_img_[0].cpu().detach().numpy())
        
        if args.dataset != 'MITBIH':
            pred_img   = torch.clamp(transforms.ToTensor()(attacker.history[attacker.best_idx]).unsqueeze(0), 0, 1)
            target_img = torch.clamp(gt_img, 0, 1)
            
            mse   = calc_mse(pred_img, target_img).item()
            psnr  = calc_psnr(pred_img, target_img).item()
            ssim  = calc_ssim(pred_img, target_img, window_size=8).item()
            if args.dataset == 'MNIST':
                lpips = calc_lpips(pred_img.repeat(1,3,1,1), target_img.repeat(1,3,1,1)).item()
            else:
                lpips = calc_lpips(pred_img, target_img).item()
            results['mse'].append(mse)
            results['psnr'].append(psnr)
            results['ssim'].append(ssim)
            results['lpips'].append(lpips)
        else:
            pred_img = torch.FloatTensor(attacker.history[attacker.best_idx]).reshape(1, -1)
            target_img = gt_img[0].cpu().detach().reshape(1, -1)
            print(pred_img[:10], target_img[:10])
            mse = F.mse_loss(pred_img, target_img).item()
            results['mse'].append(mse)
            psnr = 10*np.log10(1/(mse**2))
            results['psnr'].append(psnr)
            print(f"MSE: {mse}, PSNR: {psnr}")
            pass
        
        with open(file=f'{attacker.name}/result.pickle', mode='wb') as f:
            pickle.dump(results, f)