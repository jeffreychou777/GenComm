import torch
import torchvision
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
import argparse
from opencood.data_utils.datasets import build_dataset
from torch.utils.data import DataLoader
from thop import profile

def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    opt = parser.parse_args()
    return opt

def main():
    print('1')
    opt = test_parser()
    print('2')
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print('Loading Model from checkpoint')
    # saved_path = opt.model_dir
    # resume_epoch, model = train_utils.load_saved_model(saved_path, model)
    # print(f"resume from {resume_epoch} epoch.")
    
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    data_loader = DataLoader(opencood_dataset,
                            batch_size=1,
                            num_workers=2,
                            collate_fn=opencood_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)
    # Model
    # 计算FLOPs
    for i, batch_data in enumerate(data_loader):
        print(i)
        batch_data = train_utils.to_device(batch_data, device)
        if i == 0:
            flops, params = profile(model, inputs=(batch_data['ego'],))
            print(f"Total FLOPs: {flops / 1e9:.5f} G")
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total Parameters: {total_params / 1e6:.5f} M")
        break

if __name__ == '__main__':
    main()